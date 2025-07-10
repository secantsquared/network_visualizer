"""
Unified Network Builder that can work with multiple data sources.
Extends the original WikipediaNetworkBuilder to support pluggable data sources.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx

from .config import NetworkConfig
from ..data_sources import DataSourceAdapter, create_data_source
from .network_builder import WikipediaNetworkBuilder


class UnifiedNetworkBuilder(WikipediaNetworkBuilder):
    """
    Extended network builder that supports multiple data sources.
    Maintains all original Wikipedia functionality while adding support for
    course datasets and other data sources.
    """

    def __init__(self, config: Optional[NetworkConfig] = None, **kwargs):
        """
        Initialize the unified network builder.

        Args:
            config: Configuration object with network building parameters
            **kwargs: Additional arguments for data source creation
        """
        # Initialize parent class
        super().__init__(config)
        
        # Initialize data source
        self.data_source = self._create_data_source(**kwargs)
        
        # Track source-specific metrics
        self.source_metrics = {
            'source_type': self.data_source.get_source_type(),
            'items_processed': 0,
            'relationships_found': 0
        }
        
        self.logger.info(f"Initialized UnifiedNetworkBuilder with {self.data_source.get_source_type()} data source")
        
        # Set debug level if needed
        if self.logger.level > logging.DEBUG:
            self.logger.setLevel(logging.DEBUG)

    def _create_data_source(self, **kwargs) -> DataSourceAdapter:
        """Create appropriate data source based on config."""
        source_type = self.config.data_source_type
        
        if source_type == "wikipedia":
            return create_data_source(self.config, "wikipedia")
        
        elif source_type == "coursera":
            dataset_path = kwargs.get('coursera_dataset_path') or self.config.coursera_dataset_path
            if not dataset_path:
                raise ValueError("coursera_dataset_path required for Coursera data source")
            return create_data_source(self.config, "coursera", dataset_path=dataset_path)
        
        elif source_type == "reddit":
            # Extract Reddit credentials from kwargs or config
            client_id = kwargs.get('reddit_client_id') or self.config.reddit_client_id
            client_secret = kwargs.get('reddit_client_secret') or self.config.reddit_client_secret
            user_agent = kwargs.get('reddit_user_agent') or self.config.reddit_user_agent
            
            return create_data_source(self.config, "reddit", 
                                    client_id=client_id, 
                                    client_secret=client_secret, 
                                    user_agent=user_agent)
        
        elif source_type == "hybrid":
            # Create multiple sources for hybrid mode
            sources = {}
            
            # Always include Wikipedia
            sources["wikipedia"] = create_data_source(self.config, "wikipedia")
            
            # Add Coursera if dataset path provided
            if kwargs.get('coursera_dataset_path') or self.config.coursera_dataset_path:
                dataset_path = kwargs.get('coursera_dataset_path') or self.config.coursera_dataset_path
                sources["coursera"] = create_data_source(self.config, "coursera", dataset_path=dataset_path)
            
            # Add Reddit if credentials provided
            if (kwargs.get('reddit_client_id') or self.config.reddit_client_id) and \
               (kwargs.get('reddit_client_secret') or self.config.reddit_client_secret) and \
               (kwargs.get('reddit_user_agent') or self.config.reddit_user_agent):
                client_id = kwargs.get('reddit_client_id') or self.config.reddit_client_id
                client_secret = kwargs.get('reddit_client_secret') or self.config.reddit_client_secret
                user_agent = kwargs.get('reddit_user_agent') or self.config.reddit_user_agent
                sources["reddit"] = create_data_source(self.config, "reddit",
                                                     client_id=client_id,
                                                     client_secret=client_secret,
                                                     user_agent=user_agent)
            
            hybrid_source = create_data_source(self.config, "hybrid", sources=sources)
            hybrid_source.set_primary_source(self.config.primary_data_source)
            return hybrid_source
        
        else:
            raise ValueError(f"Unknown data source type: {source_type}")

    def get_item_relationships(self, item: str) -> List[str]:
        """
        Get relationships for an item using the configured data source.
        This replaces get_article_links but maintains compatibility.
        """
        try:
            relationships = self.data_source.get_relationships(item)
            self.source_metrics['relationships_found'] += len(relationships)
            return relationships
        except Exception as e:
            self.logger.error(f"Error getting relationships for '{item}': {e}")
            return []

    async def get_item_relationships_async(self, item: str) -> List[str]:
        """Async version of get_item_relationships."""
        try:
            if hasattr(self.data_source, 'get_relationships_async'):
                relationships = await self.data_source.get_relationships_async(item)
            else:
                relationships = self.data_source.get_relationships(item)
            self.source_metrics['relationships_found'] += len(relationships)
            return relationships
        except Exception as e:
            self.logger.error(f"Error getting relationships for '{item}': {e}")
            return []

    def _should_filter_item(self, item: str) -> bool:
        """Check if item should be filtered using data source rules."""
        return self.data_source.should_filter_item(item)

    def get_item_metadata(self, item: str) -> Dict:
        """Get metadata for an item."""
        return self.data_source.get_item_metadata(item)

    # Override the main network building methods to use the unified approach
    def build_network_breadth_first(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """
        Build network using breadth-first expansion with configurable data source.
        """
        source_type = self.data_source.get_source_type()
        self.logger.info("=" * 60)
        self.logger.info(f"BUILDING NETWORK ({source_type.upper()})")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  Data source: {source_type}")
        self.logger.info(f"  Max depth: {self.config.max_depth}")
        self.logger.info(f"  Items to process: {self.config.max_articles_to_process}")
        self.logger.info(f"  Links per item: {self.config.links_per_article}")

        # Use deque for efficient queue operations
        frontier = deque()

        # Filter seed items and add valid ones to frontier
        valid_seeds = [seed for seed in seeds if not self._should_filter_item(seed)]
        if len(valid_seeds) < len(seeds):
            filtered_seeds = [seed for seed in seeds if self._should_filter_item(seed)]
            self.logger.warning(f"Filtered out {len(filtered_seeds)} seed items: {filtered_seeds}")

        frontier.extend(valid_seeds)
        depth = 0
        items_processed = 0

        # Track items at each depth for visualization
        depth_map = {seed: 0 for seed in valid_seeds}

        # Initialize seed nodes in the graph with metadata
        for seed in valid_seeds:
            metadata = self.get_item_metadata(seed)
            self.graph.add_node(seed, depth=0, processed=False, **metadata)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            pbar = tqdm(total=self.config.max_articles_to_process, desc="Processing items")

            while (
                frontier
                and depth <= self.config.max_depth
                and items_processed < self.config.max_articles_to_process
            ):
                current_level_size = len(frontier)
                self.logger.info(
                    f"Depth {depth}: Processing {current_level_size} items "
                    f"(Total processed so far: {items_processed})"
                )

                # Process current depth level
                futures = {}
                items_to_process = []

                # Only process unvisited items, up to our limit
                items_this_round = 0
                for _ in range(current_level_size):
                    if not frontier:
                        break
                    if items_processed >= self.config.max_articles_to_process:
                        break

                    item = frontier.popleft()
                    if item not in self.visited:
                        items_to_process.append(item)
                        futures[executor.submit(self.get_item_relationships, item)] = item
                        items_this_round += 1

                # Collect results
                next_items = set()
                for future in as_completed(futures):
                    item = futures[future]

                    try:
                        relationships = future.result()
                    except Exception as e:
                        self.logger.error(f"Unexpected error processing '{item}': {e}")
                        relationships = []

                    # Mark as processed and add metadata
                    item_metadata = self.get_item_metadata(item)
                    self.graph.add_node(item, depth=depth, processed=True, **item_metadata)
                    self.visited.add(item)
                    items_processed += 1
                    self.source_metrics['items_processed'] += 1
                    pbar.update(1)

                    if items_processed >= self.config.max_articles_to_process:
                        self.logger.info(
                            f"Reached limit of {self.config.max_articles_to_process} items to process"
                        )
                        break

                    # Add edges and collect next frontier
                    for target in relationships:
                        # Skip filtered items
                        if self._should_filter_item(target):
                            continue

                        # Ensure target node exists with correct depth and metadata
                        if target not in self.graph:
                            target_metadata = self.get_item_metadata(target)
                            self.graph.add_node(
                                target, depth=depth + 1, processed=False, **target_metadata
                            )
                        # Add the edge
                        self.graph.add_edge(item, target)

                        if target not in self.visited and target not in depth_map:
                            next_items.add(target)
                            depth_map[target] = depth + 1

                    if progress_callback:
                        progress_callback(items_processed, self.config.max_articles_to_process)

                # Add next level to frontier
                if items_processed < self.config.max_articles_to_process:
                    frontier.extend(next_items)
                depth += 1

            pbar.close()

        self.logger.info("=" * 60)
        self.logger.info(f"NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Items processed: {items_processed}")
        self.logger.info(f"  Total nodes: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges: {self.graph.number_of_edges()}")
        self.logger.info(f"  Source: {source_type}")
        if self.filtered_count > 0:
            self.logger.info(f"  Items filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    async def build_network_breadth_first_async(
        self, seeds: List[str], progress_callback: Optional[callable] = None
    ) -> nx.DiGraph:
        """Async version of breadth-first network building."""
        source_type = self.data_source.get_source_type()
        self.logger.info("=" * 60)
        self.logger.info(f"BUILDING NETWORK (ASYNC - {source_type.upper()})")
        self.logger.info("=" * 60)

        try:
            # Use deque for efficient queue operations
            frontier = deque()

            # Filter seed items and add valid ones to frontier
            valid_seeds = [seed for seed in seeds if not self._should_filter_item(seed)]
            if len(valid_seeds) < len(seeds):
                filtered_seeds = [seed for seed in seeds if self._should_filter_item(seed)]
                self.logger.warning(f"Filtered out {len(filtered_seeds)} seed items: {filtered_seeds}")

            frontier.extend(valid_seeds)
            depth = 0
            items_processed = 0

            # Track items at each depth for visualization
            depth_map = {seed: 0 for seed in valid_seeds}

            # Initialize seed nodes in the graph with metadata
            for seed in valid_seeds:
                metadata = self.get_item_metadata(seed)
                self.graph.add_node(seed, depth=0, processed=False, **metadata)

            pbar = tqdm(total=self.config.max_articles_to_process, desc="Processing items (async)")

            while (
                frontier
                and depth <= self.config.max_depth
                and items_processed < self.config.max_articles_to_process
            ):
                current_level_size = len(frontier)
                self.logger.info(
                    f"Depth {depth}: Processing {current_level_size} items "
                    f"(Total processed so far: {items_processed})"
                )

                # Process current depth level with async
                items_to_process = []
                for _ in range(current_level_size):
                    if not frontier:
                        break
                    if items_processed >= self.config.max_articles_to_process:
                        break

                    item = frontier.popleft()
                    if item not in self.visited:
                        items_to_process.append(item)

                # Create async tasks for all items in this batch
                if items_to_process:
                    tasks = [
                        self.get_item_relationships_async(item) for item in items_to_process
                    ]

                    # Execute all tasks concurrently
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    next_items = set()
                    for item, result in zip(items_to_process, results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Error processing '{item}': {result}")
                            relationships = []
                        else:
                            relationships = result

                        # Mark as processed and add metadata
                        item_metadata = self.get_item_metadata(item)
                        self.graph.add_node(item, depth=depth, processed=True, **item_metadata)
                        self.visited.add(item)
                        items_processed += 1
                        self.source_metrics['items_processed'] += 1
                        pbar.update(1)

                        if items_processed >= self.config.max_articles_to_process:
                            break

                        # Add edges and collect next frontier
                        for target in relationships:
                            if self._should_filter_item(target):
                                continue

                            # Ensure target node exists with correct depth and metadata
                            if target not in self.graph:
                                target_metadata = self.get_item_metadata(target)
                                self.graph.add_node(
                                    target, depth=depth + 1, processed=False, **target_metadata
                                )
                            self.graph.add_edge(item, target)

                            if target not in self.visited and target not in depth_map:
                                next_items.add(target)
                                depth_map[target] = depth + 1

                        if progress_callback:
                            progress_callback(items_processed, self.config.max_articles_to_process)

                # Add next level to frontier
                if items_processed < self.config.max_articles_to_process:
                    frontier.extend(next_items)
                depth += 1

            pbar.close()

        finally:
            # Clean up async session if it exists
            if hasattr(self, 'async_session') and self.async_session:
                await self._close_async_session()

        self.logger.info("=" * 60)
        self.logger.info(f"ASYNC NETWORK BUILDING COMPLETE")
        self.logger.info(f"  Items processed: {items_processed}")
        self.logger.info(f"  Total nodes: {self.graph.number_of_nodes()}")
        self.logger.info(f"  Total edges: {self.graph.number_of_edges()}")
        self.logger.info(f"  Source: {source_type}")
        if self.filtered_count > 0:
            self.logger.info(f"  Items filtered out: {self.filtered_count}")
        self.logger.info("=" * 60)
        return self.graph

    def build_network(
        self,
        seeds: List[str],
        progress_callback: Optional[callable] = None,
        method: str = "breadth_first",
    ) -> nx.DiGraph:
        """
        Build network using specified method with configurable data source.
        """
        # For breadth-first methods, use the unified implementation
        if method == "breadth_first":
            return self.build_network_breadth_first(seeds, progress_callback)
        elif method == "breadth_first_async":
            if self.config.async_enabled:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    self.build_network_breadth_first_async(seeds, progress_callback)
                )
            else:
                self.logger.warning("Async disabled, falling back to sync breadth-first")
                return self.build_network_breadth_first(seeds, progress_callback)
        else:
            # For other methods, use the parent class implementations
            # but override the relationship fetching method
            original_get_article_links = self.get_article_links
            self.get_article_links = self.get_item_relationships
            
            try:
                if method == "random_walk":
                    return self.build_network_random_walk(seeds, progress_callback)
                elif method == "dfs":
                    return self.build_network_dfs(seeds, progress_callback)
                elif method == "topic_focused":
                    return self.build_network_topic_focused(seeds, progress_callback)
                elif method == "hub_and_spoke":
                    return self.build_network_hub_and_spoke(seeds, progress_callback)
                else:
                    return self.build_network_breadth_first(seeds, progress_callback)
            finally:
                # Restore original method
                self.get_article_links = original_get_article_links

    def switch_data_source(self, source_name: str):
        """Switch to a different data source (for hybrid mode)."""
        if hasattr(self.data_source, 'set_primary_source'):
            self.data_source.set_primary_source(source_name)
            self.logger.info(f"Switched to data source: {source_name}")
        else:
            self.logger.warning("Current data source doesn't support switching")

    def get_available_sources(self) -> List[str]:
        """Get available data sources."""
        if hasattr(self.data_source, 'get_available_sources'):
            return self.data_source.get_available_sources()
        return [self.data_source.get_source_type()]

    def get_source_info(self) -> Dict:
        """Get information about the current data source."""
        return {
            'type': self.data_source.get_source_type(),
            'available_sources': self.get_available_sources(),
            'metrics': self.source_metrics
        }

    def get_learning_path_for_skills(self, skills: List[str], difficulty: str = "Beginner") -> List[str]:
        """Get learning path for specific skills (Coursera source only)."""
        if hasattr(self.data_source, 'get_learning_path_for_skills'):
            return self.data_source.get_learning_path_for_skills(skills, difficulty)
        else:
            self.logger.warning("Learning path feature not available for current data source")
            return []

    def get_courses_by_skill(self, skill: str) -> List[str]:
        """Get courses teaching a specific skill (Coursera source only)."""
        if hasattr(self.data_source, 'get_courses_by_skill'):
            return self.data_source.get_courses_by_skill(skill)
        else:
            self.logger.warning("Course search feature not available for current data source")
            return []