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
import re
from difflib import SequenceMatcher


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
            # Handle disconnected graphs properly for closeness centrality
            if nx.is_connected(self.graph.to_undirected()):
                closeness = nx.closeness_centrality(self.graph)
            else:
                # For disconnected graphs, calculate closeness centrality per component
                closeness = {}
                for component in nx.connected_components(self.graph.to_undirected()):
                    subgraph = self.graph.subgraph(component)
                    component_closeness = nx.closeness_centrality(subgraph)
                    closeness.update(component_closeness)
            
            # Handle disconnected graphs for betweenness centrality
            if nx.is_connected(self.graph.to_undirected()):
                betweenness = nx.betweenness_centrality(self.graph)
            else:
                # For disconnected graphs, calculate betweenness per component
                betweenness = {}
                for component in nx.connected_components(self.graph.to_undirected()):
                    if len(component) > 2:  # Betweenness only meaningful for graphs with 3+ nodes
                        subgraph = self.graph.subgraph(component)
                        component_betweenness = nx.betweenness_centrality(subgraph)
                        betweenness.update(component_betweenness)
                    else:
                        # For small components, betweenness is 0
                        for node in component:
                            betweenness[node] = 0.0
            
            self.centrality_cache = {
                'pagerank': nx.pagerank(self.graph),
                'betweenness': betweenness,
                'closeness': closeness,
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
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text strings using multiple methods.
        
        Returns a score between 0 and 1, where 1 means highly similar.
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Substring match
        if text1 in text2 or text2 in text1:
            return 0.8
        
        # Sequence similarity
        sequence_sim = SequenceMatcher(None, text1, text2).ratio()
        
        # Word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', text1))
        words2 = set(re.findall(r'\b\w+\b', text2))
        
        if not words1 or not words2:
            return sequence_sim
        
        word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Combined similarity score
        return max(sequence_sim, word_overlap)
    
    def is_topically_relevant(self, node: str, target_topic: str, threshold: float = 0.3) -> bool:
        """
        Check if a node is topically relevant to the target topic.
        
        Args:
            node: The node to check
            target_topic: The target topic for the learning path
            threshold: Minimum similarity score for relevance
        """
        # Filter out obviously unrelated topics first
        if self.is_unrelated_topic(node, target_topic):
            return False
        
        # Calculate direct similarity
        direct_sim = self.calculate_semantic_similarity(node, target_topic)
        if direct_sim >= threshold:
            return True
        
        # Check for domain-specific keywords
        target_keywords = self.extract_domain_keywords(target_topic)
        node_keywords = self.extract_domain_keywords(node)
        
        # If they share domain keywords, consider them relevant
        if target_keywords.intersection(node_keywords):
            return True
        
        # Special case: if target is Excel-related and node has spreadsheet/data keywords, consider relevant
        if ('excel' in target_topic.lower() or 'spreadsheet' in target_topic.lower()):
            excel_related_keywords = {'spreadsheet', 'data', 'analysis', 'chart', 'graph', 
                                    'table', 'pivot', 'formula', 'calculation', 'macro', 
                                    'worksheet', 'workbook', 'cell', 'function', 'vba'}
            if node_keywords.intersection(excel_related_keywords):
                return True
        
        # For very low similarity scores, be more strict
        if direct_sim < 0.15:
            return False
        
        return False
    
    def extract_domain_keywords(self, text: str) -> Set[str]:
        """Extract domain-specific keywords from text."""
        text = text.lower()
        
        # Technology/Software domain keywords
        tech_keywords = {'software', 'program', 'application', 'system', 'computer', 
                        'digital', 'data', 'analysis', 'tool', 'platform', 'interface',
                        'database', 'spreadsheet', 'microsoft', 'office', 'excel', 
                        'programming', 'code', 'algorithm', 'technology', 'formula',
                        'macro', 'chart', 'graph', 'table', 'pivot', 'cell', 'workbook',
                        'worksheet', 'calculation', 'function', 'vba', 'visual', 'basic'}
        
        # Extract keywords present in text
        words = set(re.findall(r'\b\w+\b', text))
        return words.intersection(tech_keywords)
    
    def is_unrelated_topic(self, node: str, target_topic: str) -> bool:
        """Check if a topic is clearly unrelated to the target."""
        node_lower = node.lower()
        target_lower = target_topic.lower()
        
        # Patterns that indicate unrelated topics
        unrelated_patterns = [
            r'\d{4}[–\-]\d{2}.*league',      # Sports leagues like "2017–18 Premier League"
            r'\d{4}[–\-]\d{4}.*league',      # Sports leagues like "2017–2018 Premier League"
            r'\d{4} .*grand prix',           # Racing events
            r'\d{4} .*championship',         # Sport championships  
            r'\d{4} in .*',                  # Year-specific events
            r'.*premier league',             # Premier League seasons
            r'.*championship.*season',       # Championship seasons
            r'.*league.*season',             # League seasons
            r'.*crisis$',                    # Crisis events
            r'.*war$',                       # Wars
            r'.*battle',                     # Battles
            r'.*earthquake',                 # Natural disasters
            r'.*hurricane',                  # Natural disasters
            r'.*politician',                 # Politicians
            r'.*actor',                      # Actors
            r'.*singer',                     # Singers
            r'.*athlete',                    # Athletes
            r'.*film$',                      # Films
            r'.*movie$',                     # Movies
            r'.*album$',                     # Albums
            r'.*song$',                      # Songs
        ]
        
        # Check if node matches unrelated patterns
        for pattern in unrelated_patterns:
            if re.search(pattern, node_lower):
                return True
        
        # Check for obvious domain mismatches
        if 'excel' in target_lower or 'spreadsheet' in target_lower:
            # For Excel/spreadsheet topics, filter out entertainment/sports
            entertainment_keywords = {'movie', 'film', 'song', 'album', 'actor', 
                                    'singer', 'band', 'game', 'sport', 'racing',
                                    'football', 'basketball', 'tennis', 'cricket'}
            node_words = set(re.findall(r'\b\w+\b', node_lower))
            if node_words.intersection(entertainment_keywords):
                return True
        
        return False
    
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
        
        # Filter nodes based on topical relevance first
        relevant_nodes = [node for node in connected_nodes 
                         if self.is_topically_relevant(node, topic)]
        
        # If we have too few relevant nodes, expand the search
        if len(relevant_nodes) < max_nodes // 2:
            # Relax the threshold for more nodes
            relevant_nodes = [node for node in connected_nodes 
                             if self.is_topically_relevant(node, topic, threshold=0.1)]
        
        # If still too few, include some high-centrality nodes
        if len(relevant_nodes) < max_nodes // 2:
            centrality = self.calculate_centrality_measures()
            high_centrality_nodes = sorted(connected_nodes, 
                                         key=lambda x: centrality['pagerank'].get(x, 0), 
                                         reverse=True)[:max_nodes]
            # Keep only those that aren't clearly unrelated
            relevant_nodes = [node for node in high_centrality_nodes 
                             if not self.is_unrelated_topic(node, topic)]
        
        connected_nodes = relevant_nodes
        
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
    
    def validate_topic_coherence(self, path: LearningPath) -> Dict[str, float]:
        """
        Validate the topic coherence of a learning path.
        
        Checks if consecutive topics in the path are semantically related
        and if the overall path maintains topic coherence.
        """
        nodes = [node.name for node in path.nodes]
        
        if len(nodes) < 2:
            return {
                "coherence_score": 1.0,
                "consecutive_similarity": 1.0,
                "topic_drift": 0.0,
                "overall_coherence": 1.0
            }
        
        # Check consecutive topic similarity
        consecutive_similarities = []
        for i in range(len(nodes) - 1):
            similarity = self.calculate_semantic_similarity(nodes[i], nodes[i + 1])
            consecutive_similarities.append(similarity)
        
        avg_consecutive_similarity = sum(consecutive_similarities) / len(consecutive_similarities)
        
        # Check topic drift (how much the path deviates from the original topic)
        topic_similarities = []
        for node in nodes:
            similarity = self.calculate_semantic_similarity(node, path.topic)
            topic_similarities.append(similarity)
        
        # Calculate topic drift (lower is better)
        topic_drift = 1.0 - (sum(topic_similarities) / len(topic_similarities))
        
        # Check for coherence violations (topics that are clearly unrelated)
        coherence_violations = 0
        for i, node in enumerate(nodes):
            if self.is_unrelated_topic(node, path.topic):
                coherence_violations += 1
        
        coherence_score = max(0, 1 - (coherence_violations / len(nodes)))
        
        # Overall coherence score
        overall_coherence = (avg_consecutive_similarity + coherence_score + (1 - topic_drift)) / 3
        
        return {
            "coherence_score": coherence_score,
            "consecutive_similarity": avg_consecutive_similarity,
            "topic_drift": topic_drift,
            "overall_coherence": overall_coherence
        }
    
    def improve_path_coherence(self, path: LearningPath) -> LearningPath:
        """
        Improve the coherence of a learning path by removing or reordering topics.
        """
        original_nodes = path.nodes.copy()
        improved_nodes = []
        
        # Remove clearly unrelated topics
        for node in original_nodes:
            if not self.is_unrelated_topic(node.name, path.topic):
                improved_nodes.append(node)
        
        # If we removed too many nodes, try to add back some relevant ones
        if len(improved_nodes) < len(original_nodes) // 2:
            # Add back nodes that have some relevance
            for node in original_nodes:
                if node not in improved_nodes and self.is_topically_relevant(node.name, path.topic, threshold=0.1):
                    improved_nodes.append(node)
        
        # Reorder nodes to improve coherence
        if len(improved_nodes) > 1:
            # Sort by relevance to the topic
            improved_nodes.sort(key=lambda x: self.calculate_semantic_similarity(x.name, path.topic), reverse=True)
            
            # Then apply prerequisite ordering
            node_names = [node.name for node in improved_nodes]
            prerequisites = self.find_prerequisite_relationships(node_names)
            ordered_names = self.topological_sort_with_difficulty(node_names, prerequisites)
            
            # Reorder nodes according to the new ordering
            name_to_node = {node.name: node for node in improved_nodes}
            improved_nodes = [name_to_node[name] for name in ordered_names if name in name_to_node]
        
        # Update depth levels
        for i, node in enumerate(improved_nodes):
            node.depth_level = i
        
        # Create improved path
        improved_path = LearningPath(
            topic=path.topic,
            nodes=improved_nodes,
            total_estimated_time=path.total_estimated_time,
            difficulty_progression=[node.difficulty for node in improved_nodes],
            path_type=path.path_type + "_improved",
            source_data=path.source_data
        )
        
        return improved_path