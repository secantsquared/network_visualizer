"""Data source adapters for different network building sources."""

from .base import DataSourceAdapter
from .wikipedia import WikipediaDataSource
from .coursera import CourseraDataSource
from .hybrid import HybridDataSource
from .reddit import RedditDataSource

def create_data_source(config, source_type: str = "wikipedia", **kwargs) -> DataSourceAdapter:
    """Factory function to create data source adapters."""
    
    if source_type == "wikipedia":
        # Import here to avoid circular imports
        from ..core.network_builder import WikipediaNetworkBuilder
        wikipedia_builder = WikipediaNetworkBuilder(config)
        return WikipediaDataSource(config, wikipedia_builder)
    
    elif source_type == "coursera":
        dataset_path = kwargs.get('dataset_path')
        if not dataset_path:
            raise ValueError("dataset_path required for Coursera data source")
        return CourseraDataSource(config, dataset_path)
    
    elif source_type == "hybrid":
        sources = kwargs.get('sources', {})
        return HybridDataSource(config, sources)
    
    elif source_type == "reddit":
        client_id = kwargs.get('client_id') or config.reddit_client_id
        client_secret = kwargs.get('client_secret') or config.reddit_client_secret
        user_agent = kwargs.get('user_agent') or config.reddit_user_agent
        
        if not all([client_id, client_secret, user_agent]):
            raise ValueError("Reddit credentials (client_id, client_secret, user_agent) are required")
        
        return RedditDataSource(config, client_id, client_secret, user_agent)
    
    else:
        raise ValueError(f"Unknown data source type: {source_type}")

__all__ = [
    "DataSourceAdapter",
    "WikipediaDataSource",
    "CourseraDataSource", 
    "HybridDataSource",
    "RedditDataSource",
    "create_data_source",
]