"""Data source adapters for different network building sources."""

from .base import DataSourceAdapter
from .wikipedia import WikipediaDataSource
from .coursera import CourseraDataSource
from .hybrid import HybridDataSource

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
    
    else:
        raise ValueError(f"Unknown data source type: {source_type}")

__all__ = [
    "DataSourceAdapter",
    "WikipediaDataSource",
    "CourseraDataSource", 
    "HybridDataSource",
    "create_data_source",
]