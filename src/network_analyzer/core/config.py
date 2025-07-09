# config.py
import logging
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NetworkConfig:
    """Configuration for Wikipedia network building."""

    # Core parameters
    max_depth: int = 2
    max_articles_to_process: int = 50
    links_per_article: int = 20

    # Algorithm-specific parameters
    random_walk_steps: int = 100
    restart_probability: float = 0.15
    exploration_bias: float = 0.0

    # Performance parameters
    max_workers: int = 8
    rate_limit_delay: float = 0.1
    async_enabled: bool = True
    max_concurrent_requests: int = 20
    connection_pool_size: int = 100
    request_timeout: int = 30
    async_request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Filtering parameters
    filter_patterns: Optional[List[str]] = None
    min_title_length: int = 2
    max_title_length: int = 200

    # Auto-adjustment parameters
    target_network_size: Optional[int] = None

    # DFS parameters
    dfs_max_branch_depth: int = 3
    dfs_branches_per_node: int = 5
    dfs_backtrack_probability: float = 0.1

    # Topic-focused parameters
    topic_keywords: Optional[List[str]] = None
    topic_similarity_threshold: float = 0.3
    topic_diversity_weight: float = 0.2

    # Hub-and-spoke parameters
    hub_selection_method: str = "degree"
    spokes_per_hub: int = 3
    hub_depth_limit: int = 2

    # Data source parameters
    data_source_type: str = "wikipedia"  # "wikipedia", "coursera", "hybrid"
    primary_data_source: str = "wikipedia"  # For hybrid mode
    coursera_dataset_path: Optional[str] = None
    
    # Force-directed visualization parameters
    physics_engine: str = "barnes_hut"  # "barnes_hut", "force_atlas2", "hierarchical", "circular", "organic"
    custom_physics_params: Optional[dict] = None
    adaptive_physics: bool = True  # Auto-adjust parameters based on network size

    def __post_init__(self):
        """Initialize default filter patterns and validate config."""
        if self.filter_patterns is None:
            self.filter_patterns = self._get_default_filters()
        self._validate_config()

    def _get_default_filters(self) -> List[str]:
        """Get default filter patterns."""
        return [
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

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.max_articles_to_process <= 0:
            raise ValueError("max_articles_to_process must be positive")
        # Add more validation as needed

    def auto_adjust_for_target_size(self, target_size: int):
        """Auto-adjust parameters for target network size."""
        if target_size <= 20:
            self.max_depth = 1
            self.max_articles_to_process = target_size // 2
            self.links_per_article = 10
        elif target_size <= 100:
            self.max_depth = 2
            self.max_articles_to_process = target_size // 3
            self.links_per_article = 15
        # ... rest of logic

        logging.info(f"Auto-adjusted for target size {target_size}")
