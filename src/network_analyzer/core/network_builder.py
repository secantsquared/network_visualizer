import asyncio
import json
import logging
import random
import shutil
import time
from asyncio import Semaphore
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
from aiohttp import ClientSession, TCPConnector
from pyvis.network import Network
from tqdm import tqdm

from ..utils.async_limited import AsyncRateLimiter, RateLimiter
from .config import NetworkConfig


class WikipediaNetworkBuilder:
    """
    Builds a network graph of Wikipedia articles by following internal links.

    This class fetches article links from Wikipedia's MediaWiki API and constructs
    a directed graph using NetworkX. It supports breadth-first expansion from
    seed articles with configurable depth and size limits.
    """

    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        base_url: str = "https://en.wikipedia.org",
    ):
        """
        Initialize the network builder.

        Args:
            config: Configuration object with network building parameters
            base_url: Base URL for Wikipedia (default: English Wikipedia)
        """
        self.config = config or NetworkConfig()

        # Set default filter patterns if not provided
        if self.config.filter_patterns is None:
            self.config.filter_patterns = [
                "(identifier)",
                "(disambiguation)",
                "Category:",
                "Template:",
                "File:",
                "User:",
                "Wikipedia:",
                "Help:",
                "Portal:",
                "List of",
                "Lists of",
            ]

        self.api_endpoint = base_url + "/w/api.php"
        self.session = self._create_session()
        self.graph = nx.DiGraph()
        self.visited: Set[str] = set()

        self.communities = []  # Store detected communities
        self.filtered_count = 0  # Track how many articles were filtered out

        # Rate limiting components
        self.rate_limiter = RateLimiter(self.config.rate_limit_delay)

        # Async components
        self.async_session: Optional[ClientSession] = None
        self.async_rate_limiter = AsyncRateLimiter(self.config.rate_limit_delay)
        self.async_semaphore = Semaphore(self.config.max_concurrent_requests)

        # Setup logging first
        self._setup_logging()

        # Auto-adjust parameters if target_network_size is specified
        if self.config.target_network_size:
            self._auto_adjust_parameters()

    def __del__(self):
        """Ensure cleanup on destruction."""
        if (
            hasattr(self, "async_session")
            and self.async_session
            and not self.async_session.closed
        ):
            # Try to close the session if event loop is still running
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup for later
                    loop.create_task(self._close_async_session())
                else:
                    # Run cleanup synchronously
                    loop.run_until_complete(self._close_async_session())
            except Exception:
                # If all else fails, just close the session synchronously
                try:
                    if (
                        hasattr(self.async_session, "_connector")
                        and self.async_session._connector
                    ):
                        self.async_session._connector._close()
                except Exception:
                    pass

    def _create_session(self) -> requests.Session:
        """Create a configured requests session."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "wiki-network-builder/2.0 (https://github.com/you)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
        )
        return session

    async def _create_async_session(self) -> ClientSession:
        """Create a configured async session with connection pooling."""
        headers = {
            "User-Agent": "wiki-network-builder/2.0 (https://github.com/you)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }

        connector = TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=self.config.max_concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        timeout = aiohttp.ClientTimeout(total=self.config.async_request_timeout)

        return ClientSession(headers=headers, connector=connector, timeout=timeout)

    async def _ensure_async_session(self):
        """Ensure async session is created."""
        if self.async_session is None or self.async_session.closed:
            self.async_session = await self._create_async_session()

    async def _close_async_session(self):
        """Close the async session."""
        if self.async_session and not self.async_session.closed:
            try:
                # Close the session
                await self.async_session.close()
                # Wait longer for underlying connections to close
                await asyncio.sleep(0.25)
                # Force close the connector if it still exists
                if (
                    hasattr(self.async_session, "_connector")
                    and self.async_session._connector
                ):
                    await self.async_session._connector.close()
            except Exception as e:
                self.logger.warning(f"Error closing async session: {e}")
            finally:
                self.async_session = None

    def _setup_logging(self):
        """Configure logging with better formatting."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def _auto_adjust_parameters(self):
        """Automatically adjust parameters to reach target network size."""
        target = self.config.target_network_size

        if target <= 20:
            # Small network
            self.config.max_depth = 1
            self.config.max_articles_to_process = target // 2
            self.config.links_per_article = 10
        elif target <= 100:
            # Medium network
            self.config.max_depth = 2
            self.config.max_articles_to_process = target // 3
            self.config.links_per_article = 15
        elif target <= 500:
            # Large network
            self.config.max_depth = 2
            self.config.max_articles_to_process = target // 4
            self.config.links_per_article = 25
        else:
            # Very large network
            self.config.max_depth = 3
            self.config.max_articles_to_process = target // 5
            self.config.links_per_article = 30

        self.logger.info(
            f"Auto-adjusted for target size {target}: "
            f"depth={self.config.max_depth}, "
            f"articles_to_process={self.config.max_articles_to_process}, "
            f"links_per_article={self.config.links_per_article}"
        )

    def _should_filter_article(self, title: str) -> bool:
        """
        Check if an article should be filtered out based on title patterns.

        Args:
            title: Article title to check

        Returns:
            True if article should be filtered out, False otherwise
        """
        if not title:
            return True

        # Check title length
        if (
            len(title) < self.config.min_title_length
            or len(title) > self.config.max_title_length
        ):
            return True

        # Check for unwanted patterns
        title_lower = title.lower()
        for pattern in self.config.filter_patterns:
            if pattern.lower() in title_lower:
                return True

        # Additional checks for common noise patterns

        # Filter articles that are mostly numbers or symbols
        if title.replace(" ", "").replace("-", "").replace("_", "").isdigit():
            return True

        # Filter articles with too many parentheses (often disambiguations)
        if title.count("(") > 2 or title.count(")") > 2:
            return True

        # Filter years (standalone year articles)
        if title.strip().isdigit() and len(title.strip()) == 4:
            return True

        # Filter specific year-based events (e.g., "2012 Malaysian Grand Prix")
        import re
        if re.match(r'^\d{4}\s+.*', title):
            # Allow some educational/technical topics that start with years
            educational_exceptions = ['software', 'standard', 'specification', 'protocol', 'algorithm', 'method']
            if not any(exception in title_lower for exception in educational_exceptions):
                return True

        # Filter entertainment and sports content
        entertainment_patterns = [
            r'.*\s+film$', r'.*\s+movie$', r'.*\s+album$', r'.*\s+song$',
            r'.*\s+band$', r'.*\s+actor$', r'.*\s+singer$', r'.*\s+athlete$',
            r'.*\s+championship$', r'.*\s+tournament$', r'.*\s+cup$',
            r'.*\s+season$', r'.*\s+series$'
        ]
        
        for pattern in entertainment_patterns:
            if re.search(pattern, title_lower):
                return True

        return False

    def _make_api_request(self, params: Dict) -> Optional[Dict]:
        """
        Make an API request with retry logic.

        Args:
            params: API parameters

        Returns:
            JSON response data or None if failed
        """
        for attempt in range(self.config.retry_attempts):
            try:
                self.rate_limiter.wait()
                resp = self.session.get(
                    self.api_endpoint,
                    params=params,
                    timeout=self.config.request_timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.config.retry_attempts - 1:
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}): {e}. Retrying..."
                    )
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.logger.error(
                        f"Request failed after {self.config.retry_attempts} attempts: {e}"
                    )
                    return None
        return None

    async def _make_async_api_request(self, params: Dict) -> Optional[Dict]:
        """Make an async API request with retry logic and concurrency control."""
        await self._ensure_async_session()

        for attempt in range(self.config.retry_attempts):
            async with self.async_semaphore:  # Control concurrency
                try:
                    await self.async_rate_limiter.wait()

                    async with self.async_session.get(
                        self.api_endpoint, params=params
                    ) as response:
                        response.raise_for_status()
                        return await response.json()

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < self.config.retry_attempts - 1:
                        self.logger.warning(
                            f"Async request failed (attempt {attempt + 1}): {e}. Retrying..."
                        )
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        self.logger.error(
                            f"Async request failed after {self.config.retry_attempts} attempts: {e}"
                        )
                        return None
        return None

    def get_article_links(self, title: str) -> List[str]:
        """
        Fetch internal links from the given article title via MediaWiki API.

        Args:
            title: Wikipedia article title

        Returns:
            List of linked article titles (namespace 0)
        """
        all_links = []
        plcontinue = None

        while len(all_links) < self.config.links_per_article:
            params = {
                "action": "query",
                "titles": title,
                "prop": "links",
                "plnamespace": 0,
                "pllimit": min(50, self.config.links_per_article - len(all_links)),
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            }

            if plcontinue:
                params["plcontinue"] = plcontinue

            data = self._make_api_request(params)
            if not data:
                return all_links

            # Handle API-level errors
            if "error" in data:
                self.logger.warning(
                    f"API error for '{title}': {data['error'].get('info')}"
                )
                return all_links

            # Extract links from response
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                return all_links

            page = pages[0]
            if page.get("missing") or page.get("invalid"):
                self.logger.warning(f"Page not found or invalid: '{title}'")
                return all_links

            links = page.get("links", [])
            all_links.extend(link["title"] for link in links if "title" in link)

            # Check for continuation
            plcontinue = data.get("continue", {}).get("plcontinue")
            if not plcontinue:
                break

        # Filter out unwanted articles
        original_count = len(all_links)
        random.shuffle(all_links)
        filtered_links = [
            link
            for link in all_links[: self.config.links_per_article]
            if not self._should_filter_article(link)
        ]

        filtered_out = original_count - len(filtered_links)
        if filtered_out > 0:
            self.logger.debug(f"Filtered {filtered_out} links from '{title}'")
            self.filtered_count += filtered_out

        return filtered_links

    async def get_article_links_async(self, title: str) -> List[str]:
        """Async version of get_article_links."""
        all_links = []
        plcontinue = None

        while len(all_links) < self.config.links_per_article:
            params = {
                "action": "query",
                "titles": title,
                "prop": "links",
                "plnamespace": 0,
                "pllimit": min(50, self.config.links_per_article - len(all_links)),
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            }

            if plcontinue:
                params["plcontinue"] = plcontinue

            data = await self._make_async_api_request(params)
            if not data:
                return all_links

            # Handle API-level errors
            if "error" in data:
                self.logger.warning(
                    f"API error for '{title}': {data['error'].get('info')}"
                )
                return all_links

            # Extract links from response
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                return all_links

            page = pages[0]
            if page.get("missing") or page.get("invalid"):
                self.logger.warning(f"Page not found or invalid: '{title}'")
                return all_links

            links = page.get("links", [])
            all_links.extend(link["title"] for link in links if "title" in link)

            # Check for continuation
            plcontinue = data.get("continue", {}).get("plcontinue")
            if not plcontinue:
                break

        # Filter out unwanted articles
        original_count = len(all_links)
        filtered_links = [
            link
            for link in all_links[: self.config.links_per_article]
            if not self._should_filter_article(link)
        ]

        filtered_out = original_count - len(filtered_links)
        if filtered_out > 0:
            self.logger.debug(f"Filtered {filtered_out} links from '{title}'")
            self.filtered_count += filtered_out

        return filtered_links

    def build_network_breadth_first(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """
        Build network using breadth-first expansion from seed titles.

        Args:
            seeds: List of starting article titles
            progress_callback: Optional callback function for progress updates

        Returns:
            Constructed NetworkX DiGraph
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING WIKIPEDIA NETWORK")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Max depth: {self.config.max_depth}")
        self.logger.info(
            f"  Articles to process: {self.config.max_articles_to_process}"
        )
        self.logger.info(f"  Links per article: {self.config.links_per_article}")
        self.logger.info(
            f"  Estimated network size: {self.config.max_articles_to_process * self.config.links_per_article // 4}-{self.config.max_articles_to_process * self.config.links_per_article // 2} nodes"
        )

        # Use deque for efficient queue operations
        frontier = deque()

        # Filter seed articles and add valid ones to frontier
        valid_seeds = [seed for seed in seeds if not self._should_filter_article(seed)]
        if len(valid_seeds) < len(seeds):
            filtered_seeds = [
                seed for seed in seeds if self._should_filter_article(seed)
            ]
            self.logger.warning(
                f"Filtered out {len(filtered_seeds)} seed articles: {filtered_seeds}"
            )

        frontier.extend(valid_seeds)
        depth = 0
        articles_processed = 0  # Count of articles we've actually fetched links from

        # Track articles at each depth for visualization
        depth_map = {seed: 0 for seed in valid_seeds}

        # Initialize seed nodes in the graph
        for seed in valid_seeds:
            self.graph.add_node(seed, depth=0, processed=False)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            pbar = tqdm(
                total=self.config.max_articles_to_process, desc="Processing articles"
            )

            while (
                frontier
                and depth <= self.config.max_depth
                and articles_processed < self.config.max_articles_to_process
            ):
                current_level_size = len(frontier)
                self.logger.info(
                    f"Depth {depth}: Processing {current_level_size} articles "
                    f"(Total processed so far: {articles_processed})"
                )

                # Process current depth level
                futures = {}
                articles_to_process = []

                # Only process unvisited articles, up to our limit
                articles_this_round = 0
                for _ in range(current_level_size):
                    if not frontier:
                        break
                    if articles_processed >= self.config.max_articles_to_process:
                        break

                    article = frontier.popleft()
                    if article not in self.visited:
                        articles_to_process.append(article)
                        futures[executor.submit(self.get_article_links, article)] = (
                            article
                        )
                        articles_this_round += 1

                # Collect results
                next_articles = set()
                for future in as_completed(futures):
                    article = futures[future]

                    try:
                        links = future.result()
                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error processing '{article}': {e}"
                        )
                        links = []

                    # Mark as processed
                    self.graph.add_node(article, depth=depth, processed=True)
                    self.visited.add(article)
                    articles_processed += 1
                    pbar.update(1)

                    if articles_processed >= self.config.max_articles_to_process:
                        self.logger.info(
                            f"Reached limit of {self.config.max_articles_to_process} articles to process"
                        )
                        break

                    # Add edges and collect next frontier
                    for target in links:
                        # Skip filtered articles
                        if self._should_filter_article(target):
                            continue

                        # Ensure target node exists with correct depth BEFORE adding edge
                        if target not in self.graph:
                            self.graph.add_node(
                                target, depth=depth + 1, processed=False
                            )
                        # Now add the edge
                        self.graph.add_edge(article, target)

                        if target not in self.visited and target not in depth_map:
                            next_articles.add(target)
                            depth_map[target] = depth + 1

                    if progress_callback:
                        progress_callback(
                            articles_processed, self.config.max_articles_to_process
                        )

                # Add next level to frontier (only if we haven't hit our processing limit)
                if articles_processed < self.config.max_articles_to_process:
                    frontier.extend(next_articles)
                depth += 1

            pbar.close()

        self.logger.info("=" * 60)
        self.logger.info(f"NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Articles processed (fetched links): {articles_processed}")
        self.logger.info(f"  Total nodes in network: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges in network: {self.graph.number_of_edges()}")
        if self.filtered_count > 0:
            self.logger.info(f"  Articles filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    async def build_network_breadth_first_async(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """Async version of breadth-first network building."""
        self.logger.info("=" * 60)
        self.logger.info("BUILDING WIKIPEDIA NETWORK (ASYNC)")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Max depth: {self.config.max_depth}")
        self.logger.info(
            f"  Articles to process: {self.config.max_articles_to_process}"
        )
        self.logger.info(f"  Links per article: {self.config.links_per_article}")
        self.logger.info(
            f"  Max concurrent requests: {self.config.max_concurrent_requests}"
        )
        self.logger.info(
            f"  Estimated network size: {self.config.max_articles_to_process * self.config.links_per_article // 4}-{self.config.max_articles_to_process * self.config.links_per_article // 2} nodes"
        )

        try:
            # Initialize async session
            await self._ensure_async_session()

            # Use deque for efficient queue operations
            frontier = deque()

            # Filter seed articles and add valid ones to frontier
            valid_seeds = [
                seed for seed in seeds if not self._should_filter_article(seed)
            ]
            if len(valid_seeds) < len(seeds):
                filtered_seeds = [
                    seed for seed in seeds if self._should_filter_article(seed)
                ]
                self.logger.warning(
                    f"Filtered out {len(filtered_seeds)} seed articles: {filtered_seeds}"
                )

            frontier.extend(valid_seeds)
            depth = 0
            articles_processed = 0

            # Track articles at each depth for visualization
            depth_map = {seed: 0 for seed in valid_seeds}

            # Initialize seed nodes in the graph
            for seed in valid_seeds:
                self.graph.add_node(seed, depth=0, processed=False)

            pbar = tqdm(
                total=self.config.max_articles_to_process,
                desc="Processing articles (async)",
            )

            while (
                frontier
                and depth <= self.config.max_depth
                and articles_processed < self.config.max_articles_to_process
            ):
                current_level_size = len(frontier)
                self.logger.info(
                    f"Depth {depth}: Processing {current_level_size} articles "
                    f"(Total processed so far: {articles_processed})"
                )

                # Process current depth level with async
                articles_to_process = []
                articles_this_round = 0

                for _ in range(current_level_size):
                    if not frontier:
                        break
                    if articles_processed >= self.config.max_articles_to_process:
                        break

                    article = frontier.popleft()
                    if article not in self.visited:
                        articles_to_process.append(article)
                        articles_this_round += 1

                # Create async tasks for all articles in this batch
                if articles_to_process:
                    tasks = [
                        self.get_article_links_async(article)
                        for article in articles_to_process
                    ]

                    # Execute all tasks concurrently
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    next_articles = set()
                    for article, result in zip(articles_to_process, results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Error processing '{article}': {result}")
                            links = []
                        else:
                            links = result

                        # Mark as processed
                        self.graph.add_node(article, depth=depth, processed=True)
                        self.visited.add(article)
                        articles_processed += 1
                        pbar.update(1)

                        if articles_processed >= self.config.max_articles_to_process:
                            break

                        # Add edges and collect next frontier
                        for target in links:
                            if self._should_filter_article(target):
                                continue

                            # Ensure target node exists with correct depth
                            if target not in self.graph:
                                self.graph.add_node(
                                    target, depth=depth + 1, processed=False
                                )
                            self.graph.add_edge(article, target)

                            if target not in self.visited and target not in depth_map:
                                next_articles.add(target)
                                depth_map[target] = depth + 1

                        if progress_callback:
                            progress_callback(
                                articles_processed, self.config.max_articles_to_process
                            )

                # Add next level to frontier
                if articles_processed < self.config.max_articles_to_process:
                    frontier.extend(next_articles)
                depth += 1

            pbar.close()

        finally:
            # Clean up async session
            await self._close_async_session()

        self.logger.info("=" * 60)
        self.logger.info(f"ASYNC NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Articles processed (fetched links): {articles_processed}")
        self.logger.info(f"  Total nodes in network: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges in network: {self.graph.number_of_edges()}")
        if self.filtered_count > 0:
            self.logger.info(f"  Articles filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    def _add_article_to_graph(
        self, article: str, links: List[str], depth: int = 0
    ) -> int:
        """
        Add an article and its links to the graph.

        This method serves as a centralized point for adding nodes and edges,
        allowing subclasses to override for temporal tracking or other purposes.

        Args:
            article: Article title to add
            links: List of linked article titles
            depth: Depth of the article in the search tree

        Returns:
            Number of new nodes added
        """
        added_count = 0

        # Add the main article if not already present
        if article not in self.graph:
            self.graph.add_node(article, depth=depth, processed=True)
            added_count += 1
        else:
            # Update processed status and depth if this is a better path
            if self.graph.nodes[article].get("depth", float("inf")) > depth:
                self.graph.nodes[article]["depth"] = depth
            self.graph.nodes[article]["processed"] = True

        # Add linked articles and edges
        for link in links:
            if not self._should_filter_article(link):
                # Add linked article if not present
                if link not in self.graph:
                    self.graph.add_node(link, depth=depth + 1, processed=False)
                    added_count += 1
                else:
                    # Update depth if this is a better path
                    current_depth = self.graph.nodes[link].get("depth", float("inf"))
                    if current_depth > depth + 1:
                        self.graph.nodes[link]["depth"] = depth + 1

                # Add edge if not present
                if not self.graph.has_edge(article, link):
                    self.graph.add_edge(article, link)

        return added_count

    def build_network_random_walk(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """
        Build network using random walk from seed articles.

        Args:
            seeds: List of starting article titles
            progress_callback: Optional callback function for progress updates

        Returns:
            Constructed NetworkX DiGraph
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING WIKIPEDIA NETWORK (RANDOM WALK)")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Random walk steps: {self.config.random_walk_steps}")
        self.logger.info(f"  Restart probability: {self.config.restart_probability}")
        self.logger.info(f"  Exploration bias: {self.config.exploration_bias}")
        self.logger.info(f"  Links per article: {self.config.links_per_article}")

        # Filter seed articles
        valid_seeds = [seed for seed in seeds if not self._should_filter_article(seed)]
        if len(valid_seeds) < len(seeds):
            filtered_seeds = [
                seed for seed in seeds if self._should_filter_article(seed)
            ]
            self.logger.warning(
                f"Filtered out {len(filtered_seeds)} seed articles: {filtered_seeds}"
            )

        if not valid_seeds:
            self.logger.error("No valid seed articles provided")
            return self.graph

        # Initialize with seed articles
        for seed in valid_seeds:
            self.graph.add_node(seed, depth=0, processed=False, is_seed=True)

        current_article = random.choice(valid_seeds)
        articles_processed = 0
        step = 0

        # Track articles and their available links for random selection
        article_links = {}  # article -> list of links

        with tqdm(
            total=self.config.random_walk_steps, desc="Random walk steps"
        ) as pbar:
            while (
                step < self.config.random_walk_steps
                and articles_processed < self.config.max_articles_to_process
            ):
                step += 1

                # Decide whether to restart from a seed
                if random.random() < self.config.restart_probability:
                    current_article = random.choice(valid_seeds)
                    self.logger.debug(
                        f"Step {step}: Restarted at seed '{current_article}'"
                    )

                # If we haven't processed this article yet, fetch its links
                if current_article not in self.visited:
                    self.logger.debug(
                        f"Step {step}: Processing new article '{current_article}'"
                    )

                    links = self.get_article_links(current_article)
                    if links:
                        article_links[current_article] = links

                        # Add article and its links using centralized method
                        depth = (
                            self.graph.nodes[current_article].get("depth", 0)
                            if current_article in self.graph
                            else 0
                        )

                        # Use centralized method for temporal tracking
                        self._add_article_to_graph(current_article, links, depth)

                        # Mark as seed if applicable
                        if current_article in valid_seeds:
                            self.graph.nodes[current_article]["is_seed"] = True

                        self.visited.add(current_article)
                        articles_processed += 1

                        # Progress callback
                        if progress_callback:
                            progress_callback(
                                articles_processed, self.config.max_articles_to_process
                            )

                # Select next article for the walk
                # Try to continue from current article's links
                available_links = []
                if current_article in article_links:
                    available_links = [
                        link
                        for link in article_links[current_article]
                        if not self._should_filter_article(link)
                    ]

                # Also consider links from other processed articles (exploration)
                if (
                    len(available_links) == 0
                    or random.random() < self.config.exploration_bias
                ):
                    # Collect all available links from processed articles
                    all_links = []
                    for processed_article in self.visited:
                        if processed_article in article_links:
                            all_links.extend(
                                [
                                    link
                                    for link in article_links[processed_article]
                                    if not self._should_filter_article(link)
                                ]
                            )

                    if all_links:
                        available_links = list(set(all_links))  # Remove duplicates

                # Choose next article
                if available_links:
                    current_article = random.choice(available_links)
                    self.logger.debug(f"Step {step}: Moving to '{current_article}'")
                else:
                    # No available links, restart from a seed
                    current_article = random.choice(valid_seeds)
                    self.logger.debug(
                        f"Step {step}: No links available, restarted at '{current_article}'"
                    )

                pbar.update(1)

        self.logger.info("=" * 60)
        self.logger.info(f"RANDOM WALK NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Random walk steps taken: {step}")
        self.logger.info(f"  Articles processed (fetched links): {articles_processed}")
        self.logger.info(f"  Total nodes in network: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges in network: {self.graph.number_of_edges()}")
        if self.filtered_count > 0:
            self.logger.info(f"  Articles filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    def _choose_next_article(
        self,
        current_article: str,
        article_links: Dict[str, List[str]],
        valid_seeds: List[str],
    ) -> Optional[str]:
        """
        Choose the next article for random walk based on current article's links.

        Args:
            current_article: Current article in the walk
            article_links: Dictionary mapping articles to their links
            valid_seeds: List of valid seed articles

        Returns:
            Next article to visit, or None if no valid options
        """
        if current_article not in article_links or not article_links[current_article]:
            return None

        available_links = article_links[current_article]

        # Apply exploration bias
        if self.config.exploration_bias > 0:
            unvisited_links = [
                link for link in available_links if link not in self.visited
            ]
            visited_links = [link for link in available_links if link in self.visited]

            # Choose between unvisited and visited based on bias
            if unvisited_links and random.random() < self.config.exploration_bias:
                return random.choice(unvisited_links)
            elif visited_links:
                return random.choice(visited_links)
            elif unvisited_links:
                return random.choice(unvisited_links)
            else:
                return None
        else:
            # Pure random selection
            return random.choice(available_links)

    def build_network_dfs(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """
        Build network using depth-first search with backtracking.

        Args:
            seeds: List of starting article titles
            progress_callback: Optional callback function for progress updates

        Returns:
            Constructed NetworkX DiGraph
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING WIKIPEDIA NETWORK (DEPTH-FIRST SEARCH)")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Max branch depth: {self.config.dfs_max_branch_depth}")
        self.logger.info(f"  Branches per node: {self.config.dfs_branches_per_node}")
        self.logger.info(
            f"  Backtrack probability: {self.config.dfs_backtrack_probability}"
        )
        self.logger.info(f"  Links per article: {self.config.links_per_article}")

        # Filter seed articles
        valid_seeds = [seed for seed in seeds if not self._should_filter_article(seed)]
        if len(valid_seeds) < len(seeds):
            filtered_seeds = [
                seed for seed in seeds if self._should_filter_article(seed)
            ]
            self.logger.warning(
                f"Filtered out {len(filtered_seeds)} seed articles: {filtered_seeds}"
            )

        if not valid_seeds:
            self.logger.error("No valid seed articles provided")
            return self.graph

        # Initialize with seed articles
        for seed in valid_seeds:
            self.graph.add_node(seed, depth=0, processed=False, is_seed=True)

        articles_processed = 0

        # DFS stack: (article, current_depth, path_from_seed)
        dfs_stack = [(seed, 0, [seed]) for seed in valid_seeds]

        with tqdm(
            total=self.config.max_articles_to_process, desc="DFS processing"
        ) as pbar:
            while (
                dfs_stack and articles_processed < self.config.max_articles_to_process
            ):
                current_article, current_depth, path = dfs_stack.pop()

                # Skip if already processed or too deep
                if (
                    current_article in self.visited
                    or current_depth > self.config.dfs_max_branch_depth
                ):
                    continue

                # Process current article
                self.logger.debug(
                    f"DFS: Processing '{current_article}' at depth {current_depth}"
                )

                links = self.get_article_links(current_article)
                if links:
                    # Add article as processed
                    self.graph.add_node(
                        current_article,
                        depth=current_depth,
                        processed=True,
                        is_seed=current_article in valid_seeds,
                    )
                    self.visited.add(current_article)
                    articles_processed += 1
                    pbar.update(1)

                    # Add linked articles and prepare for DFS
                    branch_links = links[: self.config.dfs_branches_per_node]
                    next_depth = current_depth + 1

                    for target in branch_links:
                        if not self._should_filter_article(target):
                            # Add target node if it doesn't exist
                            if target not in self.graph:
                                self.graph.add_node(
                                    target,
                                    depth=next_depth,
                                    processed=False,
                                    is_seed=False,
                                )
                            # Add edge
                            self.graph.add_edge(current_article, target)

                            # Add to DFS stack if not visited and within depth limit
                            if (
                                target not in self.visited
                                and next_depth <= self.config.dfs_max_branch_depth
                                and target not in path
                            ):  # Avoid cycles
                                dfs_stack.append((target, next_depth, path + [target]))

                    # Occasionally backtrack to explore other branches
                    if (
                        random.random() < self.config.dfs_backtrack_probability
                        and len(path) > 1
                    ):
                        # Backtrack to a previous node in the path
                        backtrack_node = random.choice(path[:-1])
                        backtrack_depth = path.index(backtrack_node)

                        # Get unexplored links from backtrack node
                        if backtrack_node in self.visited:
                            backtrack_links = self.get_article_links(backtrack_node)
                            unexplored = [
                                link
                                for link in backtrack_links[
                                    : self.config.dfs_branches_per_node
                                ]
                                if (
                                    link not in self.visited
                                    and not self._should_filter_article(link)
                                    and link not in path
                                )
                            ]

                            # Add unexplored links to stack
                            for link in unexplored[:2]:  # Limit backtrack exploration
                                if link not in self.graph:
                                    self.graph.add_node(
                                        link,
                                        depth=backtrack_depth + 1,
                                        processed=False,
                                        is_seed=False,
                                    )
                                self.graph.add_edge(backtrack_node, link)
                                dfs_stack.append(
                                    (
                                        link,
                                        backtrack_depth + 1,
                                        path[: backtrack_depth + 1] + [link],
                                    )
                                )

                        self.logger.debug(
                            f"DFS: Backtracked to '{backtrack_node}' from '{current_article}'"
                        )

                    if progress_callback:
                        progress_callback(
                            articles_processed, self.config.max_articles_to_process
                        )
                else:
                    # No links found, mark as processed
                    self.visited.add(current_article)

        self.logger.info("=" * 60)
        self.logger.info(f"DFS NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Articles processed (fetched links): {articles_processed}")
        self.logger.info(f"  Total nodes in network: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges in network: {self.graph.number_of_edges()}")
        if self.filtered_count > 0:
            self.logger.info(f"  Articles filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    def _calculate_topic_similarity(self, title: str, keywords: List[str]) -> float:
        """
        Calculate similarity between article title and topic keywords.

        Args:
            title: Article title
            keywords: List of topic keywords

        Returns:
            Similarity score between 0 and 1
        """
        if not keywords:
            return 1.0

        title_lower = title.lower()
        title_words = set(title_lower.split())

        # Simple keyword matching approach
        keyword_words = set()
        for keyword in keywords:
            keyword_words.update(keyword.lower().split())

        # Calculate Jaccard similarity
        intersection = len(title_words.intersection(keyword_words))
        union = len(title_words.union(keyword_words))

        if union == 0:
            return 0.0

        jaccard_sim = intersection / union

        # Also check for substring matches
        substring_matches = 0
        for keyword in keywords:
            if keyword.lower() in title_lower:
                substring_matches += 1

        substring_sim = substring_matches / len(keywords)

        # Combine both similarity measures
        return max(jaccard_sim, substring_sim * 0.8)

    def build_network_topic_focused(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """
        Build network using topic-focused crawling.

        Args:
            seeds: List of starting article titles
            progress_callback: Optional callback function for progress updates

        Returns:
            Constructed NetworkX DiGraph
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING WIKIPEDIA NETWORK (TOPIC-FOCUSED CRAWLING)")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Topic keywords: {self.config.topic_keywords}")
        self.logger.info(
            f"  Similarity threshold: {self.config.topic_similarity_threshold}"
        )
        self.logger.info(f"  Diversity weight: {self.config.topic_diversity_weight}")
        self.logger.info(f"  Links per article: {self.config.links_per_article}")

        # Use default keywords if none provided
        if not self.config.topic_keywords:
            # Extract keywords from seed articles
            self.config.topic_keywords = []
            for seed in seeds:
                self.config.topic_keywords.extend(seed.lower().split())
            self.logger.info(
                f"  Using extracted keywords: {self.config.topic_keywords}"
            )

        # Filter seed articles
        valid_seeds = [seed for seed in seeds if not self._should_filter_article(seed)]
        if len(valid_seeds) < len(seeds):
            filtered_seeds = [
                seed for seed in seeds if self._should_filter_article(seed)
            ]
            self.logger.warning(
                f"Filtered out {len(filtered_seeds)} seed articles: {filtered_seeds}"
            )

        if not valid_seeds:
            self.logger.error("No valid seed articles provided")
            return self.graph

        # Initialize with seed articles
        for seed in valid_seeds:
            similarity = self._calculate_topic_similarity(
                seed, self.config.topic_keywords
            )
            self.graph.add_node(
                seed,
                depth=0,
                processed=False,
                is_seed=True,
                topic_similarity=similarity,
            )

        articles_processed = 0

        # Priority queue: (negative_priority, article, depth, similarity)
        # Using negative priority for max-heap behavior
        frontier = []
        for seed in valid_seeds:
            similarity = self._calculate_topic_similarity(
                seed, self.config.topic_keywords
            )
            frontier.append((-similarity, seed, 0, similarity))

        # Track diversity - articles from different topic clusters
        topic_clusters = set()

        with tqdm(
            total=self.config.max_articles_to_process, desc="Topic-focused crawling"
        ) as pbar:
            while frontier and articles_processed < self.config.max_articles_to_process:
                # Sort frontier by priority (highest similarity first)
                frontier.sort()

                # Get the most relevant article
                neg_priority, current_article, current_depth, similarity = frontier.pop(
                    0
                )

                # Skip if already processed
                if current_article in self.visited:
                    continue

                self.logger.debug(
                    f"Topic-focused: Processing '{current_article}' (similarity: {similarity:.3f})"
                )

                links = self.get_article_links(current_article)
                if links:
                    # Add article as processed
                    self.graph.add_node(
                        current_article,
                        depth=current_depth,
                        processed=True,
                        is_seed=current_article in valid_seeds,
                        topic_similarity=similarity,
                    )
                    self.visited.add(current_article)
                    articles_processed += 1
                    pbar.update(1)

                    # Track topic diversity
                    article_cluster = (
                        current_article.split()[0]
                        if current_article.split()
                        else current_article
                    )
                    topic_clusters.add(article_cluster)

                    # Process linked articles
                    next_depth = current_depth + 1
                    candidate_links = []

                    for target in links:
                        if not self._should_filter_article(target):
                            target_similarity = self._calculate_topic_similarity(
                                target, self.config.topic_keywords
                            )

                            # Apply similarity threshold
                            if (
                                target_similarity
                                >= self.config.topic_similarity_threshold
                            ):
                                # Add target node if it doesn't exist
                                if target not in self.graph:
                                    self.graph.add_node(
                                        target,
                                        depth=next_depth,
                                        processed=False,
                                        is_seed=False,
                                        topic_similarity=target_similarity,
                                    )
                                # Add edge
                                self.graph.add_edge(current_article, target)

                                # Calculate diversity bonus
                                target_cluster = (
                                    target.split()[0] if target.split() else target
                                )
                                diversity_bonus = 0.0
                                if target_cluster not in topic_clusters:
                                    diversity_bonus = self.config.topic_diversity_weight

                                # Combined priority: similarity + diversity
                                combined_priority = target_similarity + diversity_bonus
                                candidate_links.append(
                                    (
                                        target,
                                        next_depth,
                                        target_similarity,
                                        combined_priority,
                                    )
                                )

                    # Add candidate links to frontier
                    for target, depth, sim, priority in candidate_links:
                        if target not in self.visited:
                            frontier.append((-priority, target, depth, sim))

                    # Limit frontier size to prevent memory issues
                    if len(frontier) > 1000:
                        frontier.sort()
                        frontier = frontier[:500]  # Keep top 500 candidates

                    if progress_callback:
                        progress_callback(
                            articles_processed, self.config.max_articles_to_process
                        )
                else:
                    # No links found, mark as processed
                    self.visited.add(current_article)

        self.logger.info("=" * 60)
        self.logger.info(f"TOPIC-FOCUSED NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Articles processed (fetched links): {articles_processed}")
        self.logger.info(f"  Total nodes in network: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges in network: {self.graph.number_of_edges()}")
        self.logger.info(f"  Topic clusters discovered: {len(topic_clusters)}")
        if self.filtered_count > 0:
            self.logger.info(f"  Articles filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    def build_network_hub_and_spoke(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """
        Build network using hub-and-spoke approach.
        First builds a small network, identifies hubs, then expands from hubs.

        Args:
            seeds: List of starting article titles
            progress_callback: Optional callback function for progress updates

        Returns:
            Constructed NetworkX DiGraph
        """
        self.logger.info("=" * 60)
        self.logger.info("BUILDING WIKIPEDIA NETWORK (HUB-AND-SPOKE)")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Hub selection method: {self.config.hub_selection_method}")
        self.logger.info(f"  Spokes per hub: {self.config.spokes_per_hub}")
        self.logger.info(f"  Hub depth limit: {self.config.hub_depth_limit}")
        self.logger.info(f"  Links per article: {self.config.links_per_article}")

        # Filter seed articles
        valid_seeds = [seed for seed in seeds if not self._should_filter_article(seed)]
        if len(valid_seeds) < len(seeds):
            filtered_seeds = [
                seed for seed in seeds if self._should_filter_article(seed)
            ]
            self.logger.warning(
                f"Filtered out {len(filtered_seeds)} seed articles: {filtered_seeds}"
            )

        if not valid_seeds:
            self.logger.error("No valid seed articles provided")
            return self.graph

        # Phase 1: Build initial small network using breadth-first
        self.logger.info("Phase 1: Building initial network to identify hubs...")

        # Use a smaller limit for initial network
        initial_limit = min(self.config.max_articles_to_process // 2, 15)
        self.logger.info(f"  Processing {initial_limit} articles to identify hubs")

        # Initialize with seed articles
        for seed in valid_seeds:
            self.graph.add_node(seed, depth=0, processed=False, is_seed=True)

        articles_processed = 0
        frontier = deque(valid_seeds)
        depth = 0

        # Build initial network
        while (
            frontier
            and depth <= self.config.hub_depth_limit
            and articles_processed < initial_limit
        ):
            current_level_size = len(frontier)
            self.logger.info(
                f"  Depth {depth}: Processing {current_level_size} articles"
            )

            next_articles = set()
            for _ in range(current_level_size):
                if not frontier or articles_processed >= initial_limit:
                    break

                article = frontier.popleft()
                if article not in self.visited:
                    links = self.get_article_links(article)

                    # Add article as processed
                    self.graph.add_node(article, depth=depth, processed=True)
                    self.visited.add(article)
                    articles_processed += 1

                    # Add links to graph
                    for target in links[:10]:  # Limit links for initial phase
                        if not self._should_filter_article(target):
                            if target not in self.graph:
                                self.graph.add_node(
                                    target, depth=depth + 1, processed=False
                                )
                            self.graph.add_edge(article, target)

                            if target not in self.visited:
                                next_articles.add(target)

            frontier.extend(next_articles)
            depth += 1

        # Phase 2: Identify hubs
        self.logger.info("Phase 2: Identifying hubs...")

        processed_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("processed", False)
        ]

        if self.config.hub_selection_method == "degree":
            # Select nodes with highest degree
            node_degrees = [(node, self.graph.degree(node)) for node in processed_nodes]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            potential_hubs = [node for node, degree in node_degrees if degree > 2]

        elif self.config.hub_selection_method == "pagerank":
            # Use PageRank to identify important nodes
            try:
                pagerank_scores = nx.pagerank(self.graph.subgraph(processed_nodes))
                hub_candidates = sorted(
                    pagerank_scores.items(), key=lambda x: x[1], reverse=True
                )
                potential_hubs = [
                    node for node, score in hub_candidates if score > 0.01
                ]
            except:
                # Fallback to degree if PageRank fails
                potential_hubs = [
                    node for node in processed_nodes if self.graph.degree(node) > 2
                ]

        elif self.config.hub_selection_method == "betweenness":
            # Use betweenness centrality
            try:
                betweenness_scores = nx.betweenness_centrality(
                    self.graph.subgraph(processed_nodes)
                )
                hub_candidates = sorted(
                    betweenness_scores.items(), key=lambda x: x[1], reverse=True
                )
                potential_hubs = [
                    node for node, score in hub_candidates if score > 0.01
                ]
            except:
                # Fallback to degree if betweenness fails
                potential_hubs = [
                    node for node in processed_nodes if self.graph.degree(node) > 2
                ]
        else:
            # Default to degree-based selection
            potential_hubs = [
                node for node in processed_nodes if self.graph.degree(node) > 2
            ]

        # Select top hubs
        num_hubs = min(
            len(potential_hubs), max(3, self.config.max_articles_to_process // 10)
        )
        selected_hubs = potential_hubs[:num_hubs]

        self.logger.info(f"  Selected {len(selected_hubs)} hubs: {selected_hubs}")

        # Phase 3: Expand from hubs (spoke generation)
        self.logger.info("Phase 3: Expanding spokes from hubs...")

        with tqdm(
            total=self.config.max_articles_to_process - articles_processed,
            desc="Hub-and-spoke expansion",
        ) as pbar:

            for hub in selected_hubs:
                if articles_processed >= self.config.max_articles_to_process:
                    break

                self.logger.info(f"  Expanding from hub: {hub}")

                # Get additional links from this hub
                hub_links = self.get_article_links(hub)

                # Select spokes (links not yet in graph)
                new_spokes = [
                    link
                    for link in hub_links
                    if (
                        link not in self.graph and not self._should_filter_article(link)
                    )
                ][: self.config.spokes_per_hub]

                # Add spokes and expand one level from each
                for spoke in new_spokes:
                    if articles_processed >= self.config.max_articles_to_process:
                        break

                    # Add spoke node
                    spoke_depth = self.graph.nodes[hub].get("depth", 0) + 1
                    self.graph.add_node(
                        spoke, depth=spoke_depth, processed=False, is_spoke=True
                    )
                    self.graph.add_edge(hub, spoke)

                    # Expand one level from spoke
                    spoke_links = self.get_article_links(spoke)

                    # Mark spoke as processed
                    self.graph.nodes[spoke]["processed"] = True
                    self.visited.add(spoke)
                    articles_processed += 1
                    pbar.update(1)

                    # Add spoke's links
                    for target in spoke_links[:5]:  # Limit expansion from spokes
                        if not self._should_filter_article(target):
                            if target not in self.graph:
                                self.graph.add_node(
                                    target, depth=spoke_depth + 1, processed=False
                                )
                            self.graph.add_edge(spoke, target)

                    if progress_callback:
                        progress_callback(
                            articles_processed, self.config.max_articles_to_process
                        )

        # Mark hubs in graph
        for hub in selected_hubs:
            if hub in self.graph:
                self.graph.nodes[hub]["is_hub"] = True

        self.logger.info("=" * 60)
        self.logger.info(f"HUB-AND-SPOKE NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Articles processed (fetched links): {articles_processed}")
        self.logger.info(f"  Total nodes in network: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges in network: {self.graph.number_of_edges()}")
        self.logger.info(f"  Hubs identified: {len(selected_hubs)}")
        if self.filtered_count > 0:
            self.logger.info(f"  Articles filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    def build_network(
        self,
        seeds: List[str],
        progress_callback: Optional[callable] = None,
        method: str = "breadth_first",
    ) -> nx.DiGraph:
        """
        Build network using specified method.

        Args:
            seeds: List of starting article titles
            progress_callback: Optional callback function for progress updates
            method: Network generation method - one of:
                - "breadth_first": Breadth-first expansion (default)
                - "breadth_first_async": Async breadth-first expansion
                - "random_walk": Random walk with restart
                - "dfs": Depth-first search with backtracking
                - "topic_focused": Topic-focused crawling
                - "hub_and_spoke": Hub-and-spoke expansion

        Returns:
            Constructed NetworkX DiGraph
        """
        if method == "breadth_first_async":
            # Run async method in event loop
            if self.config.async_enabled:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                try:
                    return loop.run_until_complete(
                        self.build_network_breadth_first_async(seeds, progress_callback)
                    )
                finally:
                    # Session cleanup is handled by the async method itself
                    pass
            else:
                self.logger.warning(
                    "Async disabled, falling back to sync breadth-first"
                )
                return self.build_network_breadth_first(seeds, progress_callback)
        elif method == "random_walk":
            return self.build_network_random_walk(seeds, progress_callback)
        elif method == "dfs":
            return self.build_network_dfs(seeds, progress_callback)
        elif method == "topic_focused":
            return self.build_network_topic_focused(seeds, progress_callback)
        elif method == "hub_and_spoke":
            return self.build_network_hub_and_spoke(seeds, progress_callback)
        else:
            # Use async if enabled, otherwise sync
            if self.config.async_enabled:
                return self.build_network(
                    seeds, progress_callback, "breadth_first_async"
                )
            else:
                return self.build_network_breadth_first(seeds, progress_callback)

    async def build_network_async(
        self,
        seeds: List[str],
        progress_callback: Optional[callable] = None,
        method: str = "breadth_first",
    ) -> nx.DiGraph:
        """Async version of build_network - all methods run in async context."""
        if method == "breadth_first" or method == "breadth_first_async":
            return await self.build_network_breadth_first_async(
                seeds, progress_callback
            )
        else:
            # For now, other methods run in executor (could be made async later)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.build_network, seeds, progress_callback, method
            )

    def analyze_network(self) -> Dict:
        """
        Analyze the network and return statistics.

        Returns:
            Dictionary containing network metrics
        """
        G = self.graph
        stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "average_degree": (
                sum(dict(G.degree()).values()) / G.number_of_nodes() if G else 0
            ),
        }

        # Top nodes by various centrality measures
        if G.number_of_nodes() > 0:
            # Degree centrality
            degree_cent = nx.degree_centrality(G)
            stats["top_by_degree"] = sorted(
                degree_cent.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # PageRank (if graph is not empty)
            try:
                pagerank = nx.pagerank(G, max_iter=100)
                stats["top_by_pagerank"] = sorted(
                    pagerank.items(), key=lambda x: x[1], reverse=True
                )[:10]
            except:
                stats["top_by_pagerank"] = []

            # Community detection on undirected version
            try:
                from networkx.algorithms.community import greedy_modularity_communities

                self.communities = list(
                    greedy_modularity_communities(G.to_undirected())
                )
                stats["num_communities"] = len(self.communities)
                stats["community_sizes"] = sorted(
                    [len(c) for c in self.communities], reverse=True
                )

                # Store community membership as node attributes
                for i, community in enumerate(self.communities):
                    for node in community:
                        if node in G.nodes():
                            G.nodes[node]["community"] = i

                # Calculate modularity
                try:
                    modularity = nx.algorithms.community.modularity(
                        G.to_undirected(), self.communities
                    )
                    stats["modularity"] = modularity
                except:
                    stats["modularity"] = None

                # Find representative nodes for each community (highest degree)
                community_representatives = []
                for i, community in enumerate(self.communities):
                    if community:
                        # Get degrees for nodes in this community
                        community_degrees = {node: G.degree(node) for node in community}
                        representative = max(
                            community_degrees, key=community_degrees.get
                        )
                        community_representatives.append(
                            (i, representative, len(community))
                        )
                stats["community_representatives"] = community_representatives

            except Exception as e:
                self.logger.warning(f"Community detection failed: {e}")
                stats["num_communities"] = 0
                stats["community_sizes"] = []
                stats["modularity"] = None
                stats["community_representatives"] = []

        return stats

    def print_analysis(self, stats: Optional[Dict] = None):
        """Print network analysis results."""
        if stats is None:
            stats = self.analyze_network()

        print("\n" + "=" * 50)
        print("NETWORK ANALYSIS RESULTS")
        print("=" * 50)

        print(f"\nConfiguration:")
        print(f"  Max depth: {self.config.max_depth}")
        print(
            f"  Articles processed: {len([n for n, d in self.graph.nodes(data=True) if d.get('processed', False)])}"
        )
        print(f"  Links per article: {self.config.links_per_article}")
        print(f"  Articles filtered out: {self.filtered_count}")
        print(
            f"  Filter patterns: {', '.join(self.config.filter_patterns[:5])}{'...' if len(self.config.filter_patterns) > 5 else ''}"
        )

        print(f"\nFinal Network:")
        print(f"  Total nodes: {stats['nodes']}")
        print(f"  Total edges: {stats['edges']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  Average Degree: {stats['average_degree']:.2f}")

        if stats.get("top_by_degree"):
            print(f"\nTop 10 by Degree Centrality:")
            for node, score in stats["top_by_degree"]:
                print(f"  {node}: {score:.4f}")

        if stats.get("top_by_pagerank"):
            print(f"\nTop 10 by PageRank:")
            for node, score in stats["top_by_pagerank"]:
                print(f"  {node}: {score:.4f}")

        if stats.get("num_communities", 0) > 0:
            print(f"\nCommunity Detection:")
            print(f"  Number of communities: {stats['num_communities']}")
            print(f"  Community sizes: {stats['community_sizes'][:10]}")
            if stats.get("modularity") is not None:
                print(f"  Modularity: {stats['modularity']:.4f}")

            # Print community representatives
            if stats.get("community_representatives"):
                print(f"\nCommunity Representatives (highest degree nodes):")
                for comm_id, representative, size in stats["community_representatives"][
                    :10
                ]:
                    print(f"  Community {comm_id} ({size} nodes): {representative}")

    def get_community_colors(self, num_communities: int) -> List[str]:
        """Generate visually distinct colors for communities."""
        if num_communities <= 10:
            # Use predefined colors for small numbers
            colors = [
                "#e74c3c",
                "#3498db",
                "#2ecc71",
                "#f39c12",
                "#9b59b6",
                "#1abc9c",
                "#34495e",
                "#f1c40f",
                "#e67e22",
                "#95a5a6",
            ]
            return colors[:num_communities]
        else:
            # Generate colors using colormap for larger numbers
            cmap = plt.cm.Set3
            colors = [
                mcolors.to_hex(cmap(i / num_communities))
                for i in range(num_communities)
            ]
            return colors

    def visualize_pyvis(
        self,
        output_path: str = "wiki_network.html",
        physics: bool = True,
        color_by: str = "depth",  # "depth" or "community"
        size_by: str = "degree",  # "degree", "betweenness", "pagerank", "closeness", "eigenvector"
        physics_engine: str = "barnes_hut",
        custom_physics_params: Optional[dict] = None,
        **kwargs,
    ):
        """
        Create an interactive visualization using pyvis.

        Args:
            output_path: Output HTML file path
            physics: Enable physics simulation
            color_by: Color nodes by "depth" or "community"
            size_by: Size nodes by centrality measure ("degree", "betweenness", "pagerank", "closeness", "eigenvector")
            physics_engine: Physics engine to use ("barnes_hut", "force_atlas2", "hierarchical", "circular", "organic", "centrality")
            custom_physics_params: Custom physics parameters to override defaults
            **kwargs: Additional pyvis Network parameters
        """
        default_params = {
            "height": "750px",
            "width": "100%",
            "bgcolor": "#ffffff",
            "font_color": "black",
            "directed": True,
            "notebook": False,
            "cdn_resources": "in_line",
        }
        default_params.update(kwargs)

        net = Network(**default_params)

        # Configure physics using ForceDirectedVisualizer if available
        if physics:
            try:
                from ..visualization.force_directed_visualizer import (
                    ForceDirectedVisualizer,
                )

                visualizer = ForceDirectedVisualizer(self.graph)

                # Use force-directed visualizer for enhanced physics
                visualizer.visualize(
                    output_path=output_path,
                    physics_type=physics_engine,
                    color_by=color_by,
                    size_by=size_by,
                    custom_params=custom_physics_params,
                    **default_params,
                )
                return  # Exit early since ForceDirectedVisualizer handles everything

            except ImportError:
                # Fallback to original physics configuration
                net.barnes_hut(
                    gravity=-80000,
                    central_gravity=0.3,
                    spring_length=200,
                    spring_strength=0.001,
                    damping=0.09,
                )

        # Choose coloring scheme
        if color_by == "community" and self.communities:
            # Color by community
            community_colors = self.get_community_colors(len(self.communities))
            color_map = {}
            for i, community in enumerate(self.communities):
                for node in community:
                    color_map[node] = community_colors[i]
        else:
            # Color by depth (original behavior)
            color_map = {
                0: "#e95a5a",  # Red for seeds
                1: "#2ce3ff",  # Teal for depth 1
                2: "#c1d145",  # Blue for depth 2
                3: "#ff83e0",  # Green for depth 3
                4: "#5a57fe",  # Yellow for depth 4
            }

        # Add nodes with properties
        for node, data in self.graph.nodes(data=True):
            depth = data.get("depth", 0)
            community = data.get("community", -1)

            if color_by == "community" and node in color_map:
                color = color_map[node]
                title_info = f"{node}\nDepth: {depth}\nCommunity: {community}\nDegree: {self.graph.degree(node)}"
            else:
                color = color_map.get(depth, "#95a5a6")
                title_info = (
                    f"{node}\nDepth: {depth}\nDegree: {self.graph.degree(node)}"
                )

            # Calculate node size based on degree
            degree = self.graph.degree(node)
            size = min(10 + degree * 2, 50)

            net.add_node(
                node,
                label=node,
                title=title_info,
                color=color,
                size=size,
                font={"size": 12},
            )

        # Add edges
        for src, dst in self.graph.edges():
            net.add_edge(src, dst, color="#95a5a6", width=1)

        # Set options
        net.set_options(
            """
        var options = {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "font": {
                    "size": 12,
                    "face": "arial"
                }
            },
            "edges": {
                "color": {
                    "inherit": false
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "interaction": {
                "hover": true,
                "multiselect": true,
                "navigationButtons": true
            },
            "physics": {
                "stabilization": {
                    "iterations": 100
                }
            }
        }
        """
        )

        # Save visualization
        net.write_html(output_path)
        coloring_info = f" (colored by {color_by})" if color_by == "community" else ""
        self.logger.info(f"Interactive graph saved to {output_path}{coloring_info}")

    def visualize_communities_matplotlib(
        self,
        output_path: str = "communities.png",
        figsize: Tuple[int, int] = (12, 8),
        layout: str = "spring",
    ):
        """
        Create a static visualization of communities using matplotlib.

        Args:
            output_path: Output image file path
            figsize: Figure size (width, height)
            layout: Layout algorithm ("spring", "circular", "kamada_kawai")
        """
        if not self.communities:
            self.logger.warning("No communities detected. Run analyze_network() first.")
            return

        plt.figure(figsize=figsize)

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph.to_undirected(), k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph.to_undirected())
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph.to_undirected())
        else:
            pos = nx.spring_layout(self.graph.to_undirected())

        # Generate colors for communities
        community_colors = self.get_community_colors(len(self.communities))

        # Draw each community with a different color
        for i, community in enumerate(self.communities):
            # Only include nodes that are actually in the graph
            community_nodes = [node for node in community if node in self.graph.nodes()]
            if community_nodes:
                nx.draw_networkx_nodes(
                    self.graph.to_undirected(),
                    pos,
                    nodelist=community_nodes,
                    node_color=community_colors[i],
                    alpha=0.8,
                    node_size=50,
                    label=f"Community {i} ({len(community_nodes)} nodes)",
                )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph.to_undirected(), pos, alpha=0.5, edge_color="gray", width=0.5
        )

        plt.title(f"Community Structure ({len(self.communities)} communities)")

        # Create a custom legend with community names instead of overlaying text on graph
        legend_elements = []
        for i, community in enumerate(self.communities):
            community_nodes = [node for node in community if node in self.graph.nodes()]
            if community_nodes:
                # Find the most connected node in this community as representative
                degree_dict = dict(self.graph.degree())
                representative_node = max(
                    community_nodes, key=lambda x: degree_dict.get(x, 0)
                )

                # Create legend entry with community color and representative node name
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=community_colors[i],
                        markersize=10,
                        label=f"Community {i+1}: {representative_node}",
                    )
                )

        # Position legend outside the plot area
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Community visualization saved to {output_path}")

    def print_community_details(self):
        """Print detailed information about each community."""
        if not self.communities:
            self.logger.warning("No communities detected. Run analyze_network() first.")
            return

        print("\n" + "=" * 50)
        print("DETAILED COMMUNITY ANALYSIS")
        print("=" * 50)

        for i, community in enumerate(self.communities):
            # Filter to only include nodes actually in the graph
            community_nodes = [node for node in community if node in self.graph.nodes()]
            if not community_nodes:
                continue

            print(f"\nCommunity {i} ({len(community_nodes)} nodes):")
            print("-" * 30)

            # Calculate community metrics
            subgraph = self.graph.subgraph(community_nodes).to_undirected()
            internal_edges = subgraph.number_of_edges()

            # Find external edges (edges going out of the community)
            external_edges = 0
            for node in community_nodes:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in community_nodes:
                        external_edges += 1

            density = nx.density(subgraph) if len(community_nodes) > 1 else 0

            print(f"  Internal edges: {internal_edges}")
            print(f"  External edges: {external_edges}")
            print(f"  Density: {density:.3f}")

            # Top nodes by degree within community
            community_degrees = {
                node: self.graph.degree(node) for node in community_nodes
            }
            top_nodes = sorted(
                community_degrees.items(), key=lambda x: x[1], reverse=True
            )[:5]

            print(f"  Top nodes by degree:")
            for node, degree in top_nodes:
                print(f"    {node}: {degree}")

    def add_filter_pattern(self, pattern: str):
        """Add a new filter pattern."""
        if pattern not in self.config.filter_patterns:
            self.config.filter_patterns.append(pattern)
            self.logger.info(f"Added filter pattern: {pattern}")

    def remove_filter_pattern(self, pattern: str):
        """Remove a filter pattern."""
        if pattern in self.config.filter_patterns:
            self.config.filter_patterns.remove(pattern)
            self.logger.info(f"Removed filter pattern: {pattern}")

    def get_filter_patterns(self) -> List[str]:
        """Get current filter patterns."""
        return self.config.filter_patterns.copy()

    def save_graph(self, filepath: str):
        """Save the graph to a file (GraphML format)."""
        path = Path(filepath)
        if path.suffix == ".graphml":
            self._save_graphml_with_datetime_conversion(filepath)
        elif path.suffix == ".json":
            # Save as JSON for custom processing
            data = nx.node_link_data(self.graph)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            # Default to GraphML
            self._save_graphml_with_datetime_conversion(filepath + ".graphml")
        self.logger.info(f"Graph saved to {filepath}")

    def _save_graphml_with_datetime_conversion(self, filepath: str):
        """Save graph to GraphML format, converting datetime objects to strings."""
        from datetime import datetime

        # Create a copy of the graph to avoid modifying the original
        graph_copy = self.graph.copy()

        # Convert datetime objects to strings in node attributes
        for node in graph_copy.nodes():
            node_attrs = graph_copy.nodes[node]
            for attr_name, attr_value in node_attrs.items():
                if isinstance(attr_value, datetime):
                    node_attrs[attr_name] = attr_value.isoformat()

        # Convert datetime objects to strings in edge attributes
        for edge in graph_copy.edges():
            edge_attrs = graph_copy.edges[edge]
            for attr_name, attr_value in edge_attrs.items():
                if isinstance(attr_value, datetime):
                    edge_attrs[attr_name] = attr_value.isoformat()

        # Save the modified graph
        nx.write_graphml(graph_copy, filepath)

    def load_graph(self, filepath: str):
        """Load a previously saved graph."""
        path = Path(filepath)
        if path.suffix == ".graphml":
            self.graph = nx.read_graphml(filepath)
        elif path.suffix == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)
        self.visited = set(self.graph.nodes())
        self.logger.info(f"Graph loaded from {filepath}")

    def analyze_influence_propagation(
        self,
        seed_nodes: List[str] = None,
        model: str = "independent_cascade",
        num_simulations: int = 100,
    ) -> Dict:
        """
        Analyze influence propagation in the network.

        Args:
            seed_nodes: Initial nodes to start propagation from
            model: Propagation model ("independent_cascade" or "linear_threshold")
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with influence propagation results
        """
        try:
            from ..analysis.influence_propagation import (
                InfluencePropagationSimulator,
                PropagationConfig,
                PropagationModel,
            )

            # Use seed nodes or find optimal ones
            if seed_nodes is None:
                # Use top PageRank nodes as seeds
                pagerank = nx.pagerank(self.graph)
                seed_nodes = sorted(pagerank.keys(), key=pagerank.get, reverse=True)[:3]

            # Filter valid seed nodes
            valid_seeds = [node for node in seed_nodes if node in self.graph.nodes()]
            if not valid_seeds:
                self.logger.warning("No valid seed nodes found")
                return {}

            # Configure simulation
            model_enum = (
                PropagationModel.INDEPENDENT_CASCADE
                if model == "independent_cascade"
                else PropagationModel.LINEAR_THRESHOLD
            )
            config = PropagationConfig(
                model=model_enum,
                num_simulations=num_simulations,
                activation_probability=0.15,
                verbose=False,
            )

            # Create simulator
            simulator = InfluencePropagationSimulator(self.graph, config)

            # Run Monte Carlo simulation
            mc_results = simulator.monte_carlo_simulation(valid_seeds)

            # Find optimal seeds for comparison
            optimal_seeds = simulator.find_optimal_seeds(len(valid_seeds), "greedy")
            optimal_results = simulator.monte_carlo_simulation(optimal_seeds)

            # Compare different strategies
            strategy_comparison = simulator.compare_seed_strategies(len(valid_seeds))

            # Analyze network vulnerability
            vulnerability = simulator.analyze_network_vulnerability([1, 2, 3, 5])

            return {
                "selected_seeds": {
                    "nodes": valid_seeds,
                    "mean_influence": mc_results["mean_influence"],
                    "activation_rate": mc_results["mean_activation_rate"],
                    "most_influenced": mc_results["most_frequently_activated"][:10],
                },
                "optimal_seeds": {
                    "nodes": optimal_seeds,
                    "mean_influence": optimal_results["mean_influence"],
                    "activation_rate": optimal_results["mean_activation_rate"],
                    "most_influenced": optimal_results["most_frequently_activated"][
                        :10
                    ],
                },
                "strategy_comparison": strategy_comparison,
                "vulnerability_analysis": vulnerability,
                "model_used": model,
                "num_simulations": num_simulations,
            }

        except ImportError:
            self.logger.error("Influence propagation module not found")
            return {}
        except Exception as e:
            self.logger.error(f"Error in influence propagation analysis: {e}")
            return {}

    def visualize_influence_propagation(
        self,
        seeds: List[str] = None,
        model: str = "independent_cascade",
        output_path: str = "influence_propagation.png",
    ):
        """
        Create visualization of influence propagation.

        Args:
            seeds: Seed nodes for propagation
            model: Propagation model to use
            output_path: Output file path
        """
        try:
            from ..analysis.influence_propagation import (
                InfluencePropagationSimulator,
                PropagationConfig,
                PropagationModel,
            )

            # Use provided seeds or find optimal ones
            if seeds is None:
                pagerank = nx.pagerank(self.graph)
                seeds = sorted(pagerank.keys(), key=pagerank.get, reverse=True)[:3]

            # Filter valid seeds
            valid_seeds = [node for node in seeds if node in self.graph.nodes()]
            if not valid_seeds:
                self.logger.warning("No valid seed nodes for propagation visualization")
                return

            # Configure simulation
            model_enum = (
                PropagationModel.INDEPENDENT_CASCADE
                if model == "independent_cascade"
                else PropagationModel.LINEAR_THRESHOLD
            )
            config = PropagationConfig(
                model=model_enum, activation_probability=0.15, verbose=False
            )

            # Create simulator and run single simulation
            simulator = InfluencePropagationSimulator(self.graph, config)
            result = simulator.simulate_propagation(valid_seeds)

            # Create visualization
            simulator.visualize_propagation(result, output_path)

            self.logger.info(
                f"Influence propagation visualization saved to {output_path}"
            )

        except ImportError:
            self.logger.error("Influence propagation module not found")
        except Exception as e:
            self.logger.error(
                f"Error creating influence propagation visualization: {e}"
            )

    def save_to_history(self, history_dir: str = "outputs/history"):
        """
        Save current output files to a unique run directory within history folder.

        Args:
            history_dir: Directory name for storing history files
        """
        # Create history directory if it doesn't exist
        history_path = Path(history_dir)
        history_path.mkdir(exist_ok=True)

        # Find the next available run number
        run_number = 1
        while True:
            run_dir = history_path / f"run{run_number}"
            if not run_dir.exists():
                break
            run_number += 1

        # Create the run directory
        run_dir.mkdir(exist_ok=True)

        # Generate timestamp for metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # List of files to archive
        files_to_archive = [
            "wiki_network_depth.html",
            "wiki_network_communities.html",
            "communities.png",
            "wiki_network.graphml",
            "influence_propagation.png",
            # Unified network files
            "unified_network_depth_barnes_hut.html",
            "unified_network_depth_circular.html",
            "unified_network_depth_force_atlas2.html",
            "unified_network_communities_barnes_hut.html",
            "unified_network_communities_circular.html",
            "unified_network_communities_force_atlas2.html",
            "unified_communities.png",
            "unified_network.graphml",
        ]

        archived_files = []

        for filename in files_to_archive:
            source_path = Path(filename)
            if source_path.exists():
                # Keep original filename
                dest_path = run_dir / filename

                # Copy file to run directory
                shutil.copy2(source_path, dest_path)
                archived_files.append(str(dest_path))
                self.logger.info(f"Archived {filename} -> {dest_path}")

        # Create a metadata file with timestamp and run info
        metadata_path = run_dir / "run_info.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Run: {run_number}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Files archived: {len(archived_files)}\n")
            f.write(f"Files:\n")
            for file in archived_files:
                f.write(f"  - {Path(file).name}\n")

        if archived_files:
            self.logger.info(
                f"Successfully archived {len(archived_files)} files to {run_dir}/"
            )
            return archived_files, str(run_dir)
        else:
            self.logger.warning("No output files found to archive")
            return [], str(run_dir)
