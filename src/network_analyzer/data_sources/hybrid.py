"""
Hybrid data source that combines multiple data sources.
"""

from typing import List, Dict, Optional
from .base import DataSourceAdapter


class HybridDataSource(DataSourceAdapter):
    """Hybrid data source that can use multiple sources."""
    
    def __init__(self, config, sources: Dict[str, DataSourceAdapter]):
        super().__init__(config)
        self.sources = sources
        self.primary_source = None
        
        # Set primary source based on config or default
        if hasattr(config, 'primary_data_source'):
            self.primary_source = config.primary_data_source
        else:
            self.primary_source = list(sources.keys())[0] if sources else None
    
    def set_primary_source(self, source_name: str):
        """Set the primary data source."""
        if source_name in self.sources:
            self.primary_source = source_name
            self.logger.info(f"Primary data source set to: {source_name}")
        else:
            raise ValueError(f"Unknown data source: {source_name}")
    
    def get_relationships(self, item: str) -> List[str]:
        """Get relationships from primary source with fallback to other sources."""
        relationships = []
        
        # Try primary source first
        if self.primary_source and self.primary_source in self.sources:
            try:
                relationships = self.sources[self.primary_source].get_relationships(item)
                self.logger.debug(f"Primary source '{self.primary_source}' found {len(relationships)} relationships for '{item}'")
            except Exception as e:
                self.logger.warning(f"Primary source '{self.primary_source}' failed for '{item}': {e}")
        
        # If no relationships from primary source, try other sources
        if not relationships:
            for source_name, source in self.sources.items():
                if source_name != self.primary_source:
                    try:
                        fallback_relationships = source.get_relationships(item)
                        if fallback_relationships:
                            self.logger.info(f"Fallback source '{source_name}' found {len(fallback_relationships)} relationships for '{item}'")
                            relationships = fallback_relationships
                            break
                    except Exception as e:
                        self.logger.warning(f"Fallback source '{source_name}' failed for '{item}': {e}")
        
        return relationships
    
    async def get_relationships_async(self, item: str) -> List[str]:
        """Get relationships from primary source with fallback (async)."""
        relationships = []
        
        # Try primary source first
        if self.primary_source and self.primary_source in self.sources:
            try:
                source = self.sources[self.primary_source]
                if hasattr(source, 'get_relationships_async'):
                    relationships = await source.get_relationships_async(item)
                else:
                    relationships = source.get_relationships(item)
                self.logger.debug(f"Primary source '{self.primary_source}' found {len(relationships)} relationships for '{item}'")
            except Exception as e:
                self.logger.warning(f"Primary source '{self.primary_source}' failed for '{item}': {e}")
        
        # If no relationships from primary source, try other sources
        if not relationships:
            for source_name, source in self.sources.items():
                if source_name != self.primary_source:
                    try:
                        if hasattr(source, 'get_relationships_async'):
                            fallback_relationships = await source.get_relationships_async(item)
                        else:
                            fallback_relationships = source.get_relationships(item)
                        if fallback_relationships:
                            self.logger.info(f"Fallback source '{source_name}' found {len(fallback_relationships)} relationships for '{item}'")
                            relationships = fallback_relationships
                            break
                    except Exception as e:
                        self.logger.warning(f"Fallback source '{source_name}' failed for '{item}': {e}")
        
        return relationships
    
    def should_filter_item(self, item: str) -> bool:
        """Check if item should be filtered using primary source rules."""
        if self.primary_source and self.primary_source in self.sources:
            return self.sources[self.primary_source].should_filter_item(item)
        return False
    
    def get_item_metadata(self, item: str) -> Dict:
        """Get metadata from primary source."""
        if self.primary_source and self.primary_source in self.sources:
            return self.sources[self.primary_source].get_item_metadata(item)
        return {}
    
    def get_source_type(self) -> str:
        return f"hybrid_{self.primary_source}"
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        return list(self.sources.keys())
    
    def get_source(self, source_name: str) -> Optional[DataSourceAdapter]:
        """Get a specific data source."""
        return self.sources.get(source_name)