"""
Influence Propagation Models for Network Analysis

This module implements various influence propagation models to simulate how
information, ideas, or behaviors spread through networks. It includes:

1. Independent Cascade Model (ICM)
2. Linear Threshold Model (LTM)  
3. Influence Maximization algorithms
4. Visualization tools for propagation results

The models can be used to:
- Simulate information diffusion in social networks
- Find optimal seed nodes for maximum influence
- Analyze network vulnerability to influence campaigns
- Study viral marketing and information spread patterns
"""

import random
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging


class PropagationModel(Enum):
    """Available influence propagation models."""
    INDEPENDENT_CASCADE = "independent_cascade"
    LINEAR_THRESHOLD = "linear_threshold"
    CUSTOM = "custom"


@dataclass
class PropagationConfig:
    """Configuration for influence propagation simulation."""
    model: PropagationModel = PropagationModel.INDEPENDENT_CASCADE
    max_iterations: int = 50
    activation_probability: float = 0.1  # For ICM
    threshold_distribution: str = "uniform"  # "uniform", "degree_based", "random"
    edge_weight_method: str = "uniform"  # "uniform", "degree_based", "random"
    seed_selection_method: str = "random"  # "random", "degree", "pagerank", "betweenness"
    num_simulations: int = 100  # For Monte Carlo simulation
    verbose: bool = False


@dataclass
class PropagationResult:
    """Results from influence propagation simulation."""
    seed_nodes: Set[str]
    activated_nodes: Set[str]
    activation_times: Dict[str, int]
    influence_scores: Dict[str, float]
    total_influence: float
    iterations: int
    model_used: PropagationModel
    final_activation_rate: float


class InfluencePropagationSimulator:
    """
    Simulator for various influence propagation models.
    
    This class provides methods to simulate how influence spreads through
    networks using different models and find optimal seed nodes.
    """
    
    def __init__(self, graph: nx.Graph, config: Optional[PropagationConfig] = None):
        """
        Initialize the influence propagation simulator.
        
        Args:
            graph: NetworkX graph to simulate on
            config: Configuration for propagation simulation
        """
        self.graph = graph.copy()
        self.config = config or PropagationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Ensure graph is undirected for influence propagation
        if self.graph.is_directed():
            self.graph = self.graph.to_undirected()
            
        # Initialize edge weights for propagation
        self._initialize_edge_weights()
        
        # Initialize node thresholds for linear threshold model
        self._initialize_node_thresholds()
    
    def _initialize_edge_weights(self):
        """Initialize edge weights for influence propagation."""
        if self.config.edge_weight_method == "uniform":
            # Uniform weights
            weight = 1.0 / max(dict(self.graph.degree()).values()) if self.graph.nodes() else 0.1
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = weight
                
        elif self.config.edge_weight_method == "degree_based":
            # Weight based on node degrees
            for u, v in self.graph.edges():
                degree_u = self.graph.degree(u)
                degree_v = self.graph.degree(v)
                # Weight inversely proportional to target node degree
                self.graph[u][v]['weight'] = 1.0 / max(degree_v, 1)
                
        elif self.config.edge_weight_method == "random":
            # Random weights
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = random.uniform(0.05, 0.3)
    
    def _initialize_node_thresholds(self):
        """Initialize node thresholds for Linear Threshold Model."""
        if self.config.threshold_distribution == "uniform":
            # Uniform random thresholds
            for node in self.graph.nodes():
                self.graph.nodes[node]['threshold'] = random.uniform(0.1, 0.8)
                
        elif self.config.threshold_distribution == "degree_based":
            # Threshold inversely related to degree (high degree = low threshold)
            max_degree = max(dict(self.graph.degree()).values()) if self.graph.nodes() else 1
            for node in self.graph.nodes():
                degree = self.graph.degree(node)
                # Normalize degree and invert for threshold
                normalized_degree = degree / max_degree
                self.graph.nodes[node]['threshold'] = 0.2 + 0.6 * (1 - normalized_degree)
                
        elif self.config.threshold_distribution == "random":
            # Random thresholds with some structure
            for node in self.graph.nodes():
                self.graph.nodes[node]['threshold'] = random.betavariate(2, 2)
    
    def simulate_independent_cascade(self, seed_nodes: Set[str]) -> PropagationResult:
        """
        Simulate influence propagation using Independent Cascade Model.
        
        In ICM, each activated node gets one chance to activate each of its
        inactive neighbors with some probability.
        
        Args:
            seed_nodes: Initial set of activated nodes
            
        Returns:
            PropagationResult with simulation results
        """
        activated = set(seed_nodes)
        newly_activated = set(seed_nodes)
        activation_times = {node: 0 for node in seed_nodes}
        iteration = 0
        
        while newly_activated and iteration < self.config.max_iterations:
            iteration += 1
            current_newly_activated = set()
            
            # Each newly activated node tries to activate its neighbors
            for node in newly_activated:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in activated:
                        # Activation probability based on edge weight and config
                        edge_weight = self.graph[node][neighbor].get('weight', 0.1)
                        activation_prob = min(edge_weight * self.config.activation_probability, 1.0)
                        
                        if random.random() < activation_prob:
                            activated.add(neighbor)
                            current_newly_activated.add(neighbor)
                            activation_times[neighbor] = iteration
                            
                            if self.config.verbose:
                                self.logger.info(f"Node {neighbor} activated by {node} at iteration {iteration}")
            
            newly_activated = current_newly_activated
        
        # Calculate influence scores
        influence_scores = {}
        for node in self.graph.nodes():
            if node in activated:
                # Score based on activation time (earlier = higher influence)
                time_factor = 1.0 / (activation_times.get(node, 1) + 1)
                degree_factor = self.graph.degree(node) / max(dict(self.graph.degree()).values())
                influence_scores[node] = time_factor * degree_factor
            else:
                influence_scores[node] = 0.0
        
        total_influence = len(activated) / len(self.graph.nodes()) if self.graph.nodes() else 0
        
        return PropagationResult(
            seed_nodes=seed_nodes,
            activated_nodes=activated,
            activation_times=activation_times,
            influence_scores=influence_scores,
            total_influence=total_influence,
            iterations=iteration,
            model_used=PropagationModel.INDEPENDENT_CASCADE,
            final_activation_rate=total_influence
        )
    
    def simulate_linear_threshold(self, seed_nodes: Set[str]) -> PropagationResult:
        """
        Simulate influence propagation using Linear Threshold Model.
        
        In LTM, a node becomes activated when the total influence from its
        activated neighbors exceeds its threshold.
        
        Args:
            seed_nodes: Initial set of activated nodes
            
        Returns:
            PropagationResult with simulation results
        """
        activated = set(seed_nodes)
        activation_times = {node: 0 for node in seed_nodes}
        iteration = 0
        
        while iteration < self.config.max_iterations:
            iteration += 1
            newly_activated = set()
            
            # Check each inactive node
            for node in self.graph.nodes():
                if node not in activated:
                    # Calculate total influence from activated neighbors
                    total_influence = 0.0
                    for neighbor in self.graph.neighbors(node):
                        if neighbor in activated:
                            edge_weight = self.graph[node][neighbor].get('weight', 0.1)
                            total_influence += edge_weight
                    
                    # Activate if influence exceeds threshold
                    threshold = self.graph.nodes[node].get('threshold', 0.5)
                    if total_influence >= threshold:
                        activated.add(node)
                        newly_activated.add(node)
                        activation_times[node] = iteration
                        
                        if self.config.verbose:
                            self.logger.info(f"Node {node} activated at iteration {iteration} (influence: {total_influence:.3f}, threshold: {threshold:.3f})")
            
            # Stop if no new activations
            if not newly_activated:
                break
        
        # Calculate influence scores
        influence_scores = {}
        for node in self.graph.nodes():
            if node in activated:
                # Score based on activation time and centrality
                time_factor = 1.0 / (activation_times.get(node, 1) + 1)
                degree_factor = self.graph.degree(node) / max(dict(self.graph.degree()).values()) if self.graph.nodes() else 0
                influence_scores[node] = time_factor * degree_factor
            else:
                influence_scores[node] = 0.0
        
        total_influence = len(activated) / len(self.graph.nodes()) if self.graph.nodes() else 0
        
        return PropagationResult(
            seed_nodes=seed_nodes,
            activated_nodes=activated,
            activation_times=activation_times,
            influence_scores=influence_scores,
            total_influence=total_influence,
            iterations=iteration,
            model_used=PropagationModel.LINEAR_THRESHOLD,
            final_activation_rate=total_influence
        )
    
    def simulate_propagation(self, seed_nodes: Union[Set[str], List[str]]) -> PropagationResult:
        """
        Simulate influence propagation using the configured model.
        
        Args:
            seed_nodes: Initial set of activated nodes
            
        Returns:
            PropagationResult with simulation results
        """
        if isinstance(seed_nodes, list):
            seed_nodes = set(seed_nodes)
        
        # Ensure seed nodes exist in graph
        seed_nodes = {node for node in seed_nodes if node in self.graph.nodes()}
        
        if not seed_nodes:
            raise ValueError("No valid seed nodes provided")
        
        if self.config.model == PropagationModel.INDEPENDENT_CASCADE:
            return self.simulate_independent_cascade(seed_nodes)
        elif self.config.model == PropagationModel.LINEAR_THRESHOLD:
            return self.simulate_linear_threshold(seed_nodes)
        else:
            raise ValueError(f"Unknown propagation model: {self.config.model}")
    
    def monte_carlo_simulation(self, seed_nodes: Union[Set[str], List[str]]) -> Dict:
        """
        Run multiple simulations and return aggregate results.
        
        Args:
            seed_nodes: Initial set of activated nodes
            
        Returns:
            Dictionary with aggregate statistics
        """
        if isinstance(seed_nodes, list):
            seed_nodes = set(seed_nodes)
        
        results = []
        activation_counts = defaultdict(int)
        
        for _ in range(self.config.num_simulations):
            result = self.simulate_propagation(seed_nodes)
            results.append(result)
            
            # Count how many times each node was activated
            for node in result.activated_nodes:
                activation_counts[node] += 1
        
        # Calculate aggregate statistics
        total_influences = [r.total_influence for r in results]
        activation_rates = [r.final_activation_rate for r in results]
        
        # Calculate activation probabilities
        activation_probabilities = {}
        for node in self.graph.nodes():
            activation_probabilities[node] = activation_counts[node] / self.config.num_simulations
        
        return {
            'seed_nodes': seed_nodes,
            'num_simulations': self.config.num_simulations,
            'mean_influence': np.mean(total_influences),
            'std_influence': np.std(total_influences),
            'mean_activation_rate': np.mean(activation_rates),
            'std_activation_rate': np.std(activation_rates),
            'activation_probabilities': activation_probabilities,
            'most_frequently_activated': sorted(
                activation_probabilities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20],
            'individual_results': results
        }
    
    def find_optimal_seeds(self, k: int, method: str = "greedy") -> List[str]:
        """
        Find optimal seed nodes for maximum influence.
        
        Args:
            k: Number of seed nodes to select
            method: Selection method ("greedy", "random", "degree", "pagerank")
            
        Returns:
            List of optimal seed nodes
        """
        if method == "random":
            return random.sample(list(self.graph.nodes()), min(k, len(self.graph.nodes())))
        
        elif method == "degree":
            # Select nodes with highest degree
            degree_centrality = nx.degree_centrality(self.graph)
            return sorted(degree_centrality.keys(), key=degree_centrality.get, reverse=True)[:k]
        
        elif method == "pagerank":
            # Select nodes with highest PageRank
            pagerank = nx.pagerank(self.graph)
            return sorted(pagerank.keys(), key=pagerank.get, reverse=True)[:k]
        
        elif method == "betweenness":
            # Select nodes with highest betweenness centrality
            betweenness = nx.betweenness_centrality(self.graph)
            return sorted(betweenness.keys(), key=betweenness.get, reverse=True)[:k]
        
        elif method == "greedy":
            # Greedy algorithm for influence maximization
            selected_seeds = []
            remaining_nodes = set(self.graph.nodes())
            
            for i in range(k):
                best_node = None
                best_influence = 0
                
                # Test each remaining node
                for node in remaining_nodes:
                    test_seeds = selected_seeds + [node]
                    
                    # Run a few simulations to estimate influence
                    temp_config = PropagationConfig(
                        model=self.config.model,
                        num_simulations=10,  # Fewer simulations for speed
                        verbose=False
                    )
                    temp_simulator = InfluencePropagationSimulator(self.graph, temp_config)
                    mc_result = temp_simulator.monte_carlo_simulation(test_seeds)
                    
                    if mc_result['mean_influence'] > best_influence:
                        best_influence = mc_result['mean_influence']
                        best_node = node
                
                if best_node:
                    selected_seeds.append(best_node)
                    remaining_nodes.remove(best_node)
                    
                    if self.config.verbose:
                        self.logger.info(f"Selected seed {i+1}: {best_node} (influence: {best_influence:.3f})")
            
            return selected_seeds
        
        else:
            raise ValueError(f"Unknown seed selection method: {method}")
    
    def compare_seed_strategies(self, k: int, methods: List[str] = None) -> Dict:
        """
        Compare different seed selection strategies.
        
        Args:
            k: Number of seed nodes to select
            methods: List of methods to compare
            
        Returns:
            Dictionary comparing different strategies
        """
        if methods is None:
            methods = ["random", "degree", "pagerank", "betweenness"]
        
        results = {}
        
        for method in methods:
            self.logger.info(f"Testing seed selection method: {method}")
            
            try:
                seeds = self.find_optimal_seeds(k, method)
                mc_result = self.monte_carlo_simulation(seeds)
                
                results[method] = {
                    'seeds': seeds,
                    'mean_influence': mc_result['mean_influence'],
                    'std_influence': mc_result['std_influence'],
                    'mean_activation_rate': mc_result['mean_activation_rate'],
                    'std_activation_rate': mc_result['std_activation_rate']
                }
                
            except Exception as e:
                self.logger.warning(f"Failed to test method {method}: {e}")
                results[method] = {'error': str(e)}
        
        return results
    
    def analyze_network_vulnerability(self, attack_sizes: List[int] = None) -> Dict:
        """
        Analyze network vulnerability to influence attacks.
        
        Args:
            attack_sizes: List of attack sizes to test
            
        Returns:
            Dictionary with vulnerability analysis
        """
        if attack_sizes is None:
            attack_sizes = [1, 2, 3, 5, 10]
        
        vulnerability_results = {}
        
        for attack_size in attack_sizes:
            if attack_size > len(self.graph.nodes()):
                continue
                
            # Find optimal attack nodes
            attack_nodes = self.find_optimal_seeds(attack_size, "greedy")
            
            # Simulate attack
            mc_result = self.monte_carlo_simulation(attack_nodes)
            
            vulnerability_results[attack_size] = {
                'attack_nodes': attack_nodes,
                'mean_influence': mc_result['mean_influence'],
                'activation_rate': mc_result['mean_activation_rate'],
                'most_vulnerable_nodes': mc_result['most_frequently_activated'][:10]
            }
        
        return vulnerability_results
    
    def visualize_propagation(self, result: PropagationResult, output_path: str = "propagation.png"):
        """
        Create a visualization of influence propagation results.
        
        Args:
            result: PropagationResult to visualize
            output_path: Output file path
        """
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Color nodes based on activation status and time
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            if node in result.seed_nodes:
                node_colors.append('red')
                node_sizes.append(300)
            elif node in result.activated_nodes:
                # Color based on activation time
                time = result.activation_times.get(node, 0)
                intensity = 1.0 - (time / max(result.activation_times.values()) if result.activation_times else 0)
                node_colors.append(plt.cm.Blues(0.3 + 0.7 * intensity))
                node_sizes.append(200)
            else:
                node_colors.append('lightgray')
                node_sizes.append(100)
        
        # Draw network
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', alpha=0.5, width=0.5)
        
        # Add labels for seed nodes
        seed_labels = {node: node for node in result.seed_nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=seed_labels, font_size=8)
        
        plt.title(f"Influence Propagation - {result.model_used.value}\n"
                 f"Seeds: {len(result.seed_nodes)}, Activated: {len(result.activated_nodes)}, "
                 f"Rate: {result.final_activation_rate:.2%}")
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Seed nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Activated nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Inactive nodes')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Propagation visualization saved to {output_path}")
    
    def save_results(self, results: Union[PropagationResult, Dict], output_path: str):
        """
        Save propagation results to JSON file.
        
        Args:
            results: PropagationResult or dictionary to save
            output_path: Output file path
        """
        if isinstance(results, PropagationResult):
            # Convert PropagationResult to dictionary
            data = {
                'seed_nodes': list(results.seed_nodes),
                'activated_nodes': list(results.activated_nodes),
                'activation_times': results.activation_times,
                'influence_scores': results.influence_scores,
                'total_influence': results.total_influence,
                'iterations': results.iterations,
                'model_used': results.model_used.value,
                'final_activation_rate': results.final_activation_rate
            }
        else:
            data = results
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")


def demo_influence_propagation():
    """Demonstrate influence propagation on a sample network."""
    # Create a sample network
    G = nx.karate_club_graph()
    
    # Configure simulation
    config = PropagationConfig(
        model=PropagationModel.INDEPENDENT_CASCADE,
        activation_probability=0.2,
        num_simulations=50,
        verbose=True
    )
    
    # Create simulator
    simulator = InfluencePropagationSimulator(G, config)
    
    # Find optimal seeds
    optimal_seeds = simulator.find_optimal_seeds(3, "greedy")
    print(f"Optimal seeds: {optimal_seeds}")
    
    # Run simulation
    result = simulator.monte_carlo_simulation(optimal_seeds)
    
    print(f"\nMonte Carlo Results:")
    print(f"Mean influence: {result['mean_influence']:.3f} ± {result['std_influence']:.3f}")
    print(f"Mean activation rate: {result['mean_activation_rate']:.2%} ± {result['std_activation_rate']:.2%}")
    
    # Visualize one simulation
    single_result = result['individual_results'][0]
    simulator.visualize_propagation(single_result, "demo_propagation.png")
    
    # Compare strategies
    comparison = simulator.compare_seed_strategies(3)
    print(f"\nStrategy Comparison:")
    for method, stats in comparison.items():
        if 'error' not in stats:
            print(f"{method}: {stats['mean_influence']:.3f} influence")


if __name__ == "__main__":
    demo_influence_propagation()