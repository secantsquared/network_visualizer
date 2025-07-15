"""
AI-Powered Network Insights Generator

This module provides functionality to generate natural language insights
about network analysis results using OpenAI API (including gpt-4o-mini).
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import networkx as nx
import logging

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class NetworkInsight:
    """Represents a network insight with metadata."""
    
    title: str
    content: str
    insight_type: str  # "overview", "communities", "centrality", "patterns", "research"
    confidence: float
    data_source: str


@dataclass
class ComprehensiveNetworkInsights:
    """Container for comprehensive AI-generated network insights."""
    executive_summary: str
    key_findings: List[str]
    community_descriptions: Dict[str, str]
    bridge_nodes: List[Tuple[str, str]]  # (node, explanation)
    research_suggestions: List[str]
    network_patterns: str
    completeness_assessment: str
    individual_insights: List[NetworkInsight]


class NetworkInsightsGenerator:
    """
    Generates natural language insights about network analysis using OpenAI API.
    
    This class takes network analysis results and creates engaging narrative
    descriptions of network patterns, communities, and key metrics.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini"):
        """
        Initialize the insights generator.
        
        Args:
            api_key: OpenAI API key (will use OPENAI_API_KEY environment variable if not provided)
            model: OpenAI model to use (defaults to "gpt-4o-mini")
        """
        self.logger = logging.getLogger(__name__)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please:\n"
                "1. Set OPENAI_API_KEY environment variable\n"
                "OR pass api_key parameter to NetworkInsightsGenerator()"
            )
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        
        self.logger.info(f"Initialized NetworkInsightsGenerator with model: {model}")
    
    def generate_comprehensive_insights(self, 
                                      graph: nx.Graph, 
                                      communities: List[List[str]] = None, 
                                      network_stats: Dict = None, 
                                      seed_nodes: List[str] = None,
                                      topic: str = "Knowledge Network") -> ComprehensiveNetworkInsights:
        """
        Generate comprehensive AI insights about the network (NEW ENHANCED METHOD).
        
        Args:
            graph: NetworkX graph to analyze
            communities: List of communities (lists of node names)
            network_stats: Dictionary of network statistics
            seed_nodes: Original seed nodes used to build the network
            topic: Main topic/theme of the network
            
        Returns:
            ComprehensiveNetworkInsights object with detailed AI analysis
        """
        self.logger.info("Starting comprehensive AI network analysis...")
        
        # Prepare comprehensive network data for AI analysis
        network_data = self._prepare_comprehensive_network_data(graph, communities, network_stats, seed_nodes)
        
        # Generate comprehensive insights using OpenAI
        insights = self._generate_comprehensive_insights(network_data, topic)
        
        # Also generate individual insights for backwards compatibility
        individual_insights = self.generate_network_insights(network_stats or {}, graph, topic)
        insights.individual_insights = individual_insights
        
        self.logger.info("Comprehensive AI network analysis completed")
        return insights
    
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
    
    def _prepare_comprehensive_network_data(self, graph: nx.Graph, communities: List[List[str]] = None, 
                                           network_stats: Dict = None, seed_nodes: List[str] = None) -> Dict:
        """Prepare comprehensive network data for AI analysis."""
        
        # Basic network statistics
        if network_stats is None:
            network_stats = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
            }
        
        # Node centrality measures
        degree_centrality = nx.degree_centrality(graph)
        try:
            betweenness_centrality = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
            pagerank = nx.pagerank(graph)
        except:
            betweenness_centrality = {}
            pagerank = {}
        
        # Top nodes by different measures
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Community information
        community_info = []
        if communities:
            for i, community in enumerate(communities):
                # Filter to only include nodes actually in the graph
                community_nodes = [node for node in community if node in graph.nodes()]
                if community_nodes:
                    # Find representative node (highest degree in community)
                    rep_node = max(community_nodes, key=lambda x: graph.degree(x))
                    community_info.append({
                        "id": i,
                        "size": len(community_nodes),
                        "representative": rep_node,
                        "sample_nodes": community_nodes[:5]  # First 5 nodes as sample
                    })
        
        # Bridge nodes (high betweenness, connect different communities)
        bridge_candidates = [node for node, score in top_betweenness[:5]]
        
        return {
            "basic_stats": network_stats,
            "seed_nodes": seed_nodes or [],
            "top_degree_nodes": [{"node": node, "score": score} for node, score in top_degree],
            "top_betweenness_nodes": [{"node": node, "score": score} for node, score in top_betweenness],
            "top_pagerank_nodes": [{"node": node, "score": score} for node, score in top_pagerank],
            "communities": community_info,
            "bridge_candidates": bridge_candidates,
            "total_communities": len(communities) if communities else 0
        }
    
    def _generate_comprehensive_insights(self, network_data: Dict, topic: str) -> ComprehensiveNetworkInsights:
        """Generate comprehensive insights using OpenAI API."""
        
        # Create comprehensive prompt
        prompt = self._create_comprehensive_analysis_prompt(network_data, topic)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert network analyst and data scientist specializing in knowledge networks and Wikipedia-based research. Provide clear, actionable insights about network structure and patterns."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2500,
                temperature=0.7
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            return self._parse_comprehensive_insights_response(response_text, network_data)
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return self._create_fallback_comprehensive_insights(network_data)
    
    def _create_comprehensive_analysis_prompt(self, network_data: Dict, topic: str) -> str:
        """Create detailed prompt for comprehensive network analysis."""
        
        seed_nodes_str = ', '.join(network_data['seed_nodes']) if network_data['seed_nodes'] else "Not specified"
        
        top_degree_str = '\n'.join([f"- {item['node']}: {item['score']:.3f}" for item in network_data['top_degree_nodes'][:5]])
        top_betweenness_str = '\n'.join([f"- {item['node']}: {item['score']:.3f}" for item in network_data['top_betweenness_nodes'][:5]])
        top_pagerank_str = '\n'.join([f"- {item['node']}: {item['score']:.3f}" for item in network_data['top_pagerank_nodes'][:5]])
        
        communities_str = '\n'.join([f"Community {comm['id']}: {comm['size']} nodes, representative: {comm['representative']}" for comm in network_data['communities'][:5]])
        
        prompt = f"""
Analyze this Wikipedia-based {topic} knowledge network and provide comprehensive insights:

NETWORK OVERVIEW:
- Nodes: {network_data['basic_stats']['nodes']}
- Edges: {network_data['basic_stats']['edges']}
- Density: {network_data['basic_stats']['density']:.4f}
- Average Degree: {network_data['basic_stats']['average_degree']:.2f}
- Communities Detected: {network_data['total_communities']}

SEED NODES (starting topics):
{seed_nodes_str}

TOP NODES BY DEGREE CENTRALITY (most connected):
{top_degree_str}

TOP NODES BY BETWEENNESS CENTRALITY (bridge nodes):
{top_betweenness_str}

TOP NODES BY PAGERANK (most influential):
{top_pagerank_str}

COMMUNITY STRUCTURE:
{communities_str}

Please provide comprehensive analysis in this exact format:

EXECUTIVE_SUMMARY:
[2-3 sentence overview of what this network represents and its key characteristics]

KEY_FINDINGS:
[List 4-6 bullet points of the most important discoveries about network structure and content]

COMMUNITY_DESCRIPTIONS:
[For each major community, explain what knowledge domain it represents and its significance]

BRIDGE_NODES:
[Identify the most important bridge nodes and explain why they're significant knowledge connectors]

RESEARCH_SUGGESTIONS:
[Suggest 4-5 specific research directions, investigations, or learning paths this network reveals]

NETWORK_PATTERNS:
[Describe the overall network structure, clustering patterns, and what it reveals about knowledge organization in this domain]

COMPLETENESS_ASSESSMENT:
[Assess what might be missing from this network, suggest improvements, and identify potential gaps in knowledge coverage]
"""
        return prompt
    
    def _parse_comprehensive_insights_response(self, response_text: str, network_data: Dict) -> ComprehensiveNetworkInsights:
        """Parse OpenAI response into ComprehensiveNetworkInsights object."""
        
        sections = {}
        current_section = None
        
        # Parse sections from response
        for line in response_text.split('\n'):
            line = line.strip()
            if line.endswith(':') and line.replace('_', '').replace(':', '').upper() in [
                'EXECUTIVE_SUMMARY', 'KEY_FINDINGS', 'COMMUNITY_DESCRIPTIONS', 
                'BRIDGE_NODES', 'RESEARCH_SUGGESTIONS', 'NETWORK_PATTERNS', 
                'COMPLETENESS_ASSESSMENT'
            ]:
                current_section = line.replace(':', '').upper()
                sections[current_section] = []
            elif current_section and line:
                sections[current_section].append(line)
        
        # Extract key findings as list
        key_findings = []
        if 'KEY_FINDINGS' in sections:
            for finding in sections['KEY_FINDINGS']:
                if finding.strip().startswith('-') or finding.strip().startswith('•'):
                    key_findings.append(finding.strip().lstrip('-•').strip())
                elif finding.strip():
                    key_findings.append(finding.strip())
        
        # Extract research suggestions as list
        research_suggestions = []
        if 'RESEARCH_SUGGESTIONS' in sections:
            for suggestion in sections['RESEARCH_SUGGESTIONS']:
                if suggestion.strip().startswith('-') or suggestion.strip().startswith('•'):
                    research_suggestions.append(suggestion.strip().lstrip('-•').strip())
                elif suggestion.strip():
                    research_suggestions.append(suggestion.strip())
        
        # Parse bridge nodes
        bridge_nodes = []
        bridge_candidates = network_data.get('bridge_candidates', [])
        for i, candidate in enumerate(bridge_candidates[:3]):
            explanation = f"High betweenness centrality node connecting different knowledge domains"
            if 'BRIDGE_NODES' in sections and i < len(sections['BRIDGE_NODES']):
                explanation = sections['BRIDGE_NODES'][i]
            bridge_nodes.append((candidate, explanation))
        
        # Parse community descriptions
        community_descriptions = {}
        if 'COMMUNITY_DESCRIPTIONS' in sections and network_data.get('communities'):
            for i, comm in enumerate(network_data['communities'][:5]):
                if i < len(sections['COMMUNITY_DESCRIPTIONS']):
                    desc = sections['COMMUNITY_DESCRIPTIONS'][i]
                    community_descriptions[f"Community {comm['id']} ({comm['representative']})"] = desc
        
        return ComprehensiveNetworkInsights(
            executive_summary='\n'.join(sections.get('EXECUTIVE_SUMMARY', ['Network analysis completed.'])),
            key_findings=key_findings,
            community_descriptions=community_descriptions,
            bridge_nodes=bridge_nodes,
            research_suggestions=research_suggestions,
            network_patterns='\n'.join(sections.get('NETWORK_PATTERNS', ['Network structure analyzed.'])),
            completeness_assessment='\n'.join(sections.get('COMPLETENESS_ASSESSMENT', ['Assessment completed.'])),
            individual_insights=[]  # Will be filled by calling method
        )
    
    def _create_fallback_comprehensive_insights(self, network_data: Dict) -> ComprehensiveNetworkInsights:
        """Create basic comprehensive insights if AI analysis fails."""
        
        stats = network_data['basic_stats']
        
        return ComprehensiveNetworkInsights(
            executive_summary=f"Wikipedia knowledge network with {stats['nodes']} articles and {stats['edges']} connections. Network density: {stats['density']:.4f}",
            key_findings=[
                f"Network contains {stats['nodes']} Wikipedia articles",
                f"Average article has {stats['average_degree']:.1f} connections to other articles",
                f"Detected {network_data['total_communities']} distinct knowledge communities",
                "Network shows typical scale-free characteristics of knowledge domains"
            ],
            community_descriptions={"Main communities": "Knowledge communities detected but detailed analysis unavailable due to API limitations"},
            bridge_nodes=[(node, "Important connector between knowledge domains") for node in network_data['bridge_candidates'][:3]],
            research_suggestions=[
                "Explore high-centrality articles as starting points for research",
                "Investigate connections between different knowledge communities",
                "Analyze article depth and quality patterns within communities",
                "Study information flow pathways through bridge articles"
            ],
            network_patterns="Network exhibits standard scale-free structure typical of knowledge domains with hub articles and specialized clusters",
            completeness_assessment="Network appears reasonably complete for the given scope, but may benefit from broader seed article selection",
            individual_insights=[]
        )
    
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
        """Make a call to OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
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
            self.logger.error(f"OpenAI API call failed: {e}")
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
        output.append("💡 Powered by OpenAI")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def format_comprehensive_insights_for_display(self, insights: ComprehensiveNetworkInsights) -> str:
        """Format comprehensive insights for console display."""
        output = []
        output.append("\n" + "=" * 70)
        output.append("🤖 COMPREHENSIVE AI-POWERED NETWORK INSIGHTS")
        output.append("=" * 70)
        
        # Executive Summary
        output.append(f"\n📋 EXECUTIVE SUMMARY")
        output.append("-" * 20)
        output.append(insights.executive_summary)
        
        # Key Findings
        output.append(f"\n🔍 KEY FINDINGS")
        output.append("-" * 15)
        for finding in insights.key_findings:
            output.append(f"• {finding}")
        
        # Community Descriptions
        if insights.community_descriptions:
            output.append(f"\n🏛️  KNOWLEDGE COMMUNITIES")
            output.append("-" * 25)
            for community, description in insights.community_descriptions.items():
                output.append(f"**{community}**: {description}")
        
        # Bridge Nodes
        if insights.bridge_nodes:
            output.append(f"\n🌉 BRIDGE NODES (Knowledge Connectors)")
            output.append("-" * 40)
            for node, explanation in insights.bridge_nodes:
                output.append(f"• **{node}**: {explanation}")
        
        # Research Suggestions
        output.append(f"\n🔬 RESEARCH SUGGESTIONS")
        output.append("-" * 25)
        for suggestion in insights.research_suggestions:
            output.append(f"• {suggestion}")
        
        # Network Patterns
        output.append(f"\n🕸️  NETWORK PATTERNS")
        output.append("-" * 20)
        output.append(insights.network_patterns)
        
        # Completeness Assessment
        output.append(f"\n✅ COMPLETENESS ASSESSMENT")
        output.append("-" * 30)
        output.append(insights.completeness_assessment)
        
        output.append("\n" + "=" * 70)
        output.append("💡 Powered by OpenAI • Generated for Network Analysis")
        output.append("=" * 70)
        
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
                "model": self.model
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Insights saved to {filepath}")
    
    def save_comprehensive_insights_to_file(self, insights: ComprehensiveNetworkInsights, filepath: str):
        """Save comprehensive insights to a markdown file."""
        
        report = f"""# AI Network Analysis Report

## Executive Summary
{insights.executive_summary}

## Key Findings
{chr(10).join([f"• {finding}" for finding in insights.key_findings])}

## Knowledge Communities
{chr(10).join([f"**{comm}**: {desc}" for comm, desc in insights.community_descriptions.items()])}

## Bridge Nodes (Knowledge Connectors)
{chr(10).join([f"• **{node}**: {explanation}" for node, explanation in insights.bridge_nodes])}

## Research Suggestions
{chr(10).join([f"• {suggestion}" for suggestion in insights.research_suggestions])}

## Network Patterns
{insights.network_patterns}

## Completeness Assessment
{insights.completeness_assessment}

---
*Generated by AI Network Analyzer using OpenAI*
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Comprehensive insights report saved to {filepath}")