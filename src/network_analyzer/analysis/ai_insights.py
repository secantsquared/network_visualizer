"""
AI-Powered Network Insights Generator

This module provides functionality to generate natural language insights
about network analysis results using Azure OpenAI.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import networkx as nx
from openai import AzureOpenAI
import logging
from dotenv import load_dotenv


@dataclass
class NetworkInsight:
    """Represents a network insight with metadata."""
    
    title: str
    content: str
    insight_type: str  # "overview", "communities", "centrality", "patterns"
    confidence: float
    data_source: str


class NetworkInsightsGenerator:
    """
    Generates natural language insights about network analysis using Azure OpenAI.
    
    This class takes network analysis results and creates engaging narrative
    descriptions of network patterns, communities, and key metrics.
    """
    
    def __init__(self, 
                 azure_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_version: Optional[str] = None,
                 deployment_name: Optional[str] = None):
        """
        Initialize the insights generator.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL (will use .env file if not provided)
            api_key: Azure OpenAI API key (will use .env file if not provided)
            api_version: API version to use (defaults to value in .env or "2024-02-01")
            deployment_name: Name of the deployed model (defaults to value in .env or "gpt-4o")
        """
        self.logger = logging.getLogger(__name__)
        
        # Load .env file first (this will load from project root by default)
        load_dotenv()
        
        # Get credentials - prioritize .env file, then environment variables, then passed parameters
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        if not self.azure_endpoint or not self.api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. Please:\n"
                "1. Copy .env.template to .env\n"
                "2. Add your Azure OpenAI credentials to .env file\n"
                "OR set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables"
            )
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        self.logger.info(f"Initialized NetworkInsightsGenerator with deployment: {deployment_name}")
    
    def generate_network_insights(self, 
                                 network_stats: Dict,
                                 graph: Optional[nx.Graph] = None,
                                 topic: str = "Knowledge Network") -> List[NetworkInsight]:
        """
        Generate comprehensive insights about a network.
        
        Args:
            network_stats: Dictionary containing network analysis results
            graph: Optional NetworkX graph for additional analysis
            topic: Main topic/theme of the network
            
        Returns:
            List of NetworkInsight objects
        """
        insights = []
        
        try:
            # Generate overview insight
            overview_insight = self._generate_overview_insight(network_stats, topic)
            insights.append(overview_insight)
            
            # Generate community analysis if available
            if network_stats.get("num_communities", 0) > 0:
                community_insight = self._generate_community_insight(network_stats, topic)
                insights.append(community_insight)
            
            # Generate centrality insights
            centrality_insight = self._generate_centrality_insight(network_stats, topic)
            insights.append(centrality_insight)
            
            # Generate pattern insights if graph is provided
            if graph:
                pattern_insight = self._generate_pattern_insight(network_stats, graph, topic)
                insights.append(pattern_insight)
                
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            # Return a basic insight if AI generation fails
            fallback_insight = NetworkInsight(
                title="Network Overview",
                content=f"Your {topic} network contains {network_stats.get('nodes', 0)} nodes and {network_stats.get('edges', 0)} edges.",
                insight_type="overview",
                confidence=0.5,
                data_source="fallback"
            )
            insights.append(fallback_insight)
        
        return insights
    
    def _generate_overview_insight(self, stats: Dict, topic: str) -> NetworkInsight:
        """Generate an overview insight about the network."""
        prompt = f"""
        Analyze this network and provide an engaging overview insight in 2-3 sentences.
        
        Network Topic: {topic}
        Network Statistics:
        - Nodes: {stats.get('nodes', 0)}
        - Edges: {stats.get('edges', 0)}
        - Density: {stats.get('density', 0):.4f}
        - Average Degree: {stats.get('average_degree', 0):.2f}
        - Communities: {stats.get('num_communities', 0)}
        
        Write a compelling narrative that explains what these numbers mean in practical terms.
        Focus on the size, connectivity, and overall structure. Make it interesting and accessible.
        """
        
        response = self._call_openai(prompt)
        
        return NetworkInsight(
            title="Network Overview",
            content=response,
            insight_type="overview",
            confidence=0.9,
            data_source="azure_openai"
        )
    
    def _generate_community_insight(self, stats: Dict, topic: str) -> NetworkInsight:
        """Generate insights about community structure."""
        community_reps = stats.get('community_representatives', [])
        community_sizes = stats.get('community_sizes', [])
        modularity = stats.get('modularity', 0)
        
        # Format community representatives for the prompt
        rep_text = ""
        for i, (comm_id, rep_node, size) in enumerate(community_reps[:5]):  # Top 5 communities
            rep_text += f"- Community {comm_id + 1}: {size} nodes, centered around '{rep_node}'\n"
        
        modularity_str = f"{modularity:.3f}" if modularity is not None else "N/A"
        
        prompt = f"""
        Analyze the community structure of this {topic} network and provide insights in 2-3 sentences.
        
        Community Analysis:
        - Total Communities: {stats.get('num_communities', 0)}
        - Modularity Score: {modularity_str}
        - Community Sizes: {community_sizes[:5]}
        
        Top Communities:
        {rep_text}
        
        Explain what these communities likely represent, how well-separated they are (modularity), 
        and what this tells us about the knowledge structure. Make it engaging and insightful.
        """
        
        response = self._call_openai(prompt)
        
        return NetworkInsight(
            title="Community Structure",
            content=response,
            insight_type="communities",
            confidence=0.85,
            data_source="azure_openai"
        )
    
    def _generate_centrality_insight(self, stats: Dict, topic: str) -> NetworkInsight:
        """Generate insights about central/important nodes."""
        top_pagerank = stats.get('top_by_pagerank', [])[:5]
        top_degree = stats.get('top_by_degree', [])[:5]
        
        # Format top nodes
        pagerank_text = ""
        for node, score in top_pagerank:
            pagerank_text += f"- {node} (influence: {score:.3f})\n"
        
        degree_text = ""
        for node, score in top_degree:
            degree_text += f"- {node} (connections: {score:.3f})\n"
        
        prompt = f"""
        Analyze the most influential and connected concepts in this {topic} network.
        
        Most Influential (PageRank):
        {pagerank_text}
        
        Most Connected (Degree Centrality):
        {degree_text}
        
        In 2-3 sentences, explain what these central concepts reveal about the knowledge domain.
        What are the key foundational concepts? What acts as bridges between different areas?
        Make it insightful and explain why these concepts are important.
        """
        
        response = self._call_openai(prompt)
        
        return NetworkInsight(
            title="Key Concepts & Influence",
            content=response,
            insight_type="centrality",
            confidence=0.9,
            data_source="azure_openai"
        )
    
    def _generate_pattern_insight(self, stats: Dict, graph: nx.Graph, topic: str) -> NetworkInsight:
        """Generate insights about network patterns and structure."""
        # Calculate additional metrics
        try:
            clustering = nx.average_clustering(graph.to_undirected()) if graph.number_of_nodes() > 0 else 0
            diameter = None
            avg_path_length = None
            
            # Only calculate path metrics for smaller graphs to avoid performance issues
            if graph.number_of_nodes() < 100 and nx.is_connected(graph.to_undirected()):
                diameter = nx.diameter(graph.to_undirected())
                avg_path_length = nx.average_shortest_path_length(graph.to_undirected())
                
        except Exception:
            clustering = 0
            diameter = None
            avg_path_length = None
        
        diameter_str = str(diameter) if diameter else "N/A (large/disconnected network)"
        avg_path_str = f"{avg_path_length:.2f}" if avg_path_length else "N/A"
        
        prompt = f"""
        Analyze the structural patterns in this {topic} network.
        
        Network Metrics:
        - Clustering Coefficient: {clustering:.3f}
        - Network Diameter: {diameter_str}
        - Average Path Length: {avg_path_str}
        - Density: {stats.get('density', 0):.4f}
        
        In 2-3 sentences, explain what these patterns reveal about how knowledge is structured 
        in this domain. Is it tightly clustered? Are concepts closely related? 
        What does this mean for learning or navigation?
        """
        
        response = self._call_openai(prompt)
        
        return NetworkInsight(
            title="Network Patterns",
            content=response,
            insight_type="patterns",
            confidence=0.8,
            data_source="azure_openai"
        )
    
    def _call_openai(self, prompt: str) -> str:
        """Make a call to Azure OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a network analysis expert who creates engaging, accessible insights about knowledge networks. Be concise, insightful, and avoid jargon."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Azure OpenAI API call failed: {e}")
            return "Unable to generate AI insight at this time."
    
    def format_insights_for_display(self, insights: List[NetworkInsight]) -> str:
        """Format insights for console display."""
        output = []
        output.append("\n" + "=" * 60)
        output.append("🤖 AI-POWERED NETWORK INSIGHTS")
        output.append("=" * 60)
        
        for insight in insights:
            output.append(f"\n📊 {insight.title}")
            output.append("-" * len(insight.title))
            output.append(insight.content)
            
            if insight.confidence < 0.8:
                output.append("⚠️  (Generated with limited data)")
        
        output.append("\n" + "=" * 60)
        output.append("💡 Powered by Azure OpenAI")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def save_insights_to_file(self, insights: List[NetworkInsight], filepath: str):
        """Save insights to a JSON file."""
        insights_data = []
        for insight in insights:
            insights_data.append({
                "title": insight.title,
                "content": insight.content,
                "type": insight.insight_type,
                "confidence": insight.confidence,
                "data_source": insight.data_source
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "insights": insights_data,
                "generated_by": "NetworkInsightsGenerator",
                "model": self.deployment_name
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Insights saved to {filepath}")