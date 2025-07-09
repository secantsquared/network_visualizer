"""
Wikipedia data source adapter.
"""

from typing import List, Dict
from .base import DataSourceAdapter


class WikipediaDataSource(DataSourceAdapter):
    """Wikipedia data source adapter - wraps existing Wikipedia functionality."""
    
    def __init__(self, config, wikipedia_builder):
        super().__init__(config)
        self.wikipedia_builder = wikipedia_builder
    
    def get_relationships(self, article: str) -> List[str]:
        """Get linked articles from Wikipedia."""
        return self.wikipedia_builder.get_article_links(article)
    
    async def get_relationships_async(self, article: str) -> List[str]:
        """Get linked articles from Wikipedia (async)."""
        return await self.wikipedia_builder.get_article_links_async(article)
    
    def should_filter_item(self, article: str) -> bool:
        """Check if article should be filtered using Wikipedia rules."""
        return self.wikipedia_builder._should_filter_article(article)
    
    def get_item_metadata(self, article: str) -> Dict:
        """Get metadata for Wikipedia article."""
        return {
            'type': 'wikipedia_article',
            'title': article,
            'source': 'Wikipedia',
            'url': f'https://en.wikipedia.org/wiki/{article.replace(" ", "_")}'
        }
    
    def get_source_type(self) -> str:
        return "wikipedia"