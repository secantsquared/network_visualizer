"""
Learning Path Analysis Module

This module provides functionality to generate learning paths from knowledge networks
by analyzing prerequisite relationships, difficulty progression, and optimal learning sequences.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
import json
from dataclasses import dataclass, asdict


@dataclass
class LearningPathNode:
    """Represents a node in a learning path with metadata."""
    
    name: str
    difficulty: float
    prerequisites: List[str]
    centrality_score: float
    depth_level: int
    estimated_time: str
    description: str = ""
    resources: List[str] = None
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = []


@dataclass
class LearningPath:
    """Represents a complete learning path with metadata."""
    
    topic: str
    nodes: List[LearningPathNode]
    total_estimated_time: str
    difficulty_progression: List[float]
    path_type: str
    source_data: str
    
    def to_dict(self) -> Dict:
        """Convert learning path to dictionary for JSON serialization."""
        return {
            "topic": self.topic,
            "nodes": [asdict(node) for node in self.nodes],
            "total_estimated_time": self.total_estimated_time,
            "difficulty_progression": self.difficulty_progression,
            "path_type": self.path_type,
            "source_data": self.source_data
        }


class LearningPathAnalyzer:
    """Analyzes knowledge networks to generate optimal learning paths."""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize the learning path analyzer.
        
        Args:
            graph: NetworkX graph representing the knowledge network
        """
        # Convert directed graph to undirected for analysis
        if isinstance(graph, nx.DiGraph):
            self.graph = graph.to_undirected()
        else:
            self.graph = graph
        self.centrality_cache = {}
        self.difficulty_cache = {}
        
    def calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for all nodes."""
        if not self.centrality_cache:
            self.centrality_cache = {
                'pagerank': nx.pagerank(self.graph),
                'betweenness': nx.betweenness_centrality(self.graph),
                'closeness': nx.closeness_centrality(self.graph),
                'degree': dict(self.graph.degree())
            }
        return self.centrality_cache
    
    def calculate_difficulty_score(self, node: str) -> float:
        """
        Calculate difficulty score for a node based on network properties.
        
        Higher values indicate more advanced/difficult topics.
        """
        if node not in self.difficulty_cache:
            centrality = self.calculate_centrality_measures()
            
            # Normalize centrality measures
            max_degree = max(centrality['degree'].values()) if centrality['degree'] else 1
            max_betweenness = max(centrality['betweenness'].values()) if centrality['betweenness'] else 1
            
            # Calculate difficulty based on multiple factors
            degree_score = centrality['degree'].get(node, 0) / max_degree
            betweenness_score = centrality['betweenness'].get(node, 0) / max_betweenness
            pagerank_score = centrality['pagerank'].get(node, 0)
            
            # High degree + high betweenness = foundational (lower difficulty)
            # High PageRank alone = important but potentially advanced
            foundational_score = (degree_score + betweenness_score) / 2
            importance_score = pagerank_score
            
            # Difficulty is higher for important but not foundational topics
            difficulty = importance_score - (foundational_score * 0.3)
            self.difficulty_cache[node] = max(0.1, min(1.0, difficulty))
            
        return self.difficulty_cache[node]
    
    def find_prerequisite_relationships(self, nodes: List[str]) -> Dict[str, List[str]]:
        """
        Identify prerequisite relationships between nodes based on network structure.
        
        Uses shortest path analysis and centrality to infer prerequisites.
        """
        prerequisites = defaultdict(list)
        centrality = self.calculate_centrality_measures()
        
        for node in nodes:
            # Find nodes that are:
            # 1. Connected to this node
            # 2. Have higher centrality (more foundational)
            # 3. Are within reasonable distance
            
            neighbors = list(self.graph.neighbors(node))
            node_centrality = centrality['pagerank'].get(node, 0)
            
            for neighbor in neighbors:
                neighbor_centrality = centrality['pagerank'].get(neighbor, 0)
                
                # If neighbor has significantly higher centrality, it's likely a prerequisite
                if neighbor_centrality > node_centrality * 1.2:
                    prerequisites[node].append(neighbor)
        
        return dict(prerequisites)
    
    def topological_sort_with_difficulty(self, nodes: List[str], 
                                       prerequisites: Dict[str, List[str]]) -> List[str]:
        """
        Perform topological sort considering both prerequisites and difficulty progression.
        """
        # Create a directed graph for prerequisites
        prereq_graph = nx.DiGraph()
        prereq_graph.add_nodes_from(nodes)
        
        for node, prereqs in prerequisites.items():
            for prereq in prereqs:
                if prereq in nodes:
                    prereq_graph.add_edge(prereq, node)
        
        # Get topological order
        try:
            topo_order = list(nx.topological_sort(prereq_graph))
        except nx.NetworkXError:
            # If there are cycles, fall back to difficulty-based ordering
            topo_order = sorted(nodes, key=lambda x: self.calculate_difficulty_score(x))
        
        return topo_order
    
    def estimate_learning_time(self, node: str, difficulty: float) -> str:
        """Estimate learning time based on difficulty and node properties."""
        base_time = 2  # Base hours per topic
        
        # Adjust based on difficulty
        time_multiplier = 1 + (difficulty * 2)  # 1-3x multiplier
        
        # Adjust based on node connectivity (more connected = more complex)
        degree = self.graph.degree(node)
        connectivity_multiplier = 1 + (min(degree, 10) * 0.1)  # Up to 2x for highly connected
        
        total_hours = base_time * time_multiplier * connectivity_multiplier
        
        if total_hours < 1:
            return "30 minutes"
        elif total_hours < 2:
            return "1 hour"
        elif total_hours < 4:
            return f"{int(total_hours)} hours"
        elif total_hours < 8:
            return f"{int(total_hours)} hours"
        else:
            return f"{int(total_hours/8)} days"
    
    def generate_learning_path(self, topic: str, max_nodes: int = 10, 
                             path_type: str = "comprehensive") -> LearningPath:
        """
        Generate a learning path for a given topic.
        
        Args:
            topic: The target topic/skill
            max_nodes: Maximum number of nodes in the path
            path_type: Type of path ('comprehensive', 'fast_track', 'foundational')
        """
        # Find relevant nodes (connected component containing the topic)
        if topic not in self.graph.nodes():
            # Find the closest match
            topic_candidates = [node for node in self.graph.nodes() 
                              if topic.lower() in node.lower()]
            if not topic_candidates:
                raise ValueError(f"Topic '{topic}' not found in network")
            topic = topic_candidates[0]
        
        # Get connected component
        connected_nodes = list(nx.node_connected_component(self.graph, topic))
        
        # Filter nodes based on path type
        if path_type == "foundational":
            # Focus on high-centrality nodes
            centrality = self.calculate_centrality_measures()
            connected_nodes = sorted(connected_nodes, 
                                   key=lambda x: centrality['pagerank'].get(x, 0), 
                                   reverse=True)[:max_nodes]
        elif path_type == "fast_track":
            # Focus on direct path to target
            paths = []
            centrality = self.calculate_centrality_measures()
            important_nodes = [node for node in connected_nodes 
                             if centrality['pagerank'].get(node, 0) > 0.01]
            
            if len(important_nodes) > max_nodes:
                important_nodes = important_nodes[:max_nodes]
            connected_nodes = important_nodes
        
        # Limit nodes for comprehensive path
        if len(connected_nodes) > max_nodes:
            centrality = self.calculate_centrality_measures()
            connected_nodes = sorted(connected_nodes, 
                                   key=lambda x: centrality['pagerank'].get(x, 0), 
                                   reverse=True)[:max_nodes]
        
        # Find prerequisites
        prerequisites = self.find_prerequisite_relationships(connected_nodes)
        
        # Create learning path nodes
        ordered_nodes = self.topological_sort_with_difficulty(connected_nodes, prerequisites)
        
        path_nodes = []
        total_time = 0
        
        for i, node in enumerate(ordered_nodes):
            difficulty = self.calculate_difficulty_score(node)
            estimated_time = self.estimate_learning_time(node, difficulty)
            
            # Extract hours for total calculation
            if "hour" in estimated_time:
                hours = int(estimated_time.split()[0])
            elif "day" in estimated_time:
                hours = int(estimated_time.split()[0]) * 8
            else:
                hours = 0.5
            
            total_time += hours
            
            path_node = LearningPathNode(
                name=node,
                difficulty=difficulty,
                prerequisites=prerequisites.get(node, []),
                centrality_score=self.calculate_centrality_measures()['pagerank'].get(node, 0),
                depth_level=i,
                estimated_time=estimated_time,
                description=f"Learn about {node}",
                resources=[]
            )
            path_nodes.append(path_node)
        
        # Calculate total time
        if total_time < 8:
            total_time_str = f"{int(total_time)} hours"
        elif total_time < 40:
            total_time_str = f"{int(total_time/8)} days"
        else:
            total_time_str = f"{int(total_time/40)} weeks"
        
        return LearningPath(
            topic=topic,
            nodes=path_nodes,
            total_estimated_time=total_time_str,
            difficulty_progression=[node.difficulty for node in path_nodes],
            path_type=path_type,
            source_data="network_analysis"
        )
    
    def generate_multiple_paths(self, topic: str, 
                              path_types: List[str] = None) -> Dict[str, LearningPath]:
        """Generate multiple learning paths for comparison."""
        if path_types is None:
            path_types = ["foundational", "comprehensive", "fast_track"]
        
        paths = {}
        for path_type in path_types:
            try:
                paths[path_type] = self.generate_learning_path(
                    topic, path_type=path_type
                )
            except Exception as e:
                print(f"Warning: Could not generate {path_type} path: {e}")
        
        return paths
    
    def analyze_path_quality(self, path: LearningPath) -> Dict[str, float]:
        """Analyze the quality of a learning path."""
        nodes = [node.name for node in path.nodes]
        
        # Coverage: how well connected are the nodes
        subgraph = self.graph.subgraph(nodes)
        coverage = nx.number_of_edges(subgraph) / (len(nodes) * (len(nodes) - 1) / 2)
        
        # Progression: how smooth is the difficulty progression
        difficulties = path.difficulty_progression
        progression_score = 1.0
        for i in range(1, len(difficulties)):
            if difficulties[i] < difficulties[i-1]:
                progression_score -= 0.1
        
        # Completeness: presence of foundational concepts
        centrality = self.calculate_centrality_measures()
        foundational_score = sum(centrality['pagerank'].get(node, 0) for node in nodes)
        foundational_score /= len(nodes)
        
        return {
            "coverage": coverage,
            "progression": max(0, progression_score),
            "foundational_strength": foundational_score,
            "overall_quality": (coverage + progression_score + foundational_score) / 3
        }