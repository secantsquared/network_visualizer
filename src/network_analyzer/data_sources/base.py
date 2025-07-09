"""
Base class for data source adapters.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict


class DataSourceAdapter(ABC):
    """Abstract base class for data source adapters."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def get_relationships(self, item: str) -> List[str]:
        """Get related items for the given item."""
        pass
    
    @abstractmethod
    def should_filter_item(self, item: str) -> bool:
        """Check if item should be filtered out."""
        pass
    
    @abstractmethod
    def get_item_metadata(self, item: str) -> Dict:
        """Get metadata for the given item."""
        pass
    
    @abstractmethod
    def get_source_type(self) -> str:
        """Return the source type identifier."""
        pass