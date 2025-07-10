"""
Temporal Network Evolution Analysis

This module provides tools for analyzing how networks evolve over time,
tracking growth patterns, and creating dynamic visualizations of network development.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class NetworkSnapshot:
    """Represents a network state at a specific time."""
    timestamp: datetime
    nodes: Set[str]
    edges: Set[Tuple[str, str]]
    node_attributes: Dict[str, Dict] = field(default_factory=dict)
    edge_attributes: Dict[Tuple[str, str], Dict] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def to_networkx(self) -> nx.Graph:
        """Convert snapshot to NetworkX graph."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in self.nodes:
            attrs = self.node_attributes.get(node, {})
            G.add_node(node, **attrs)
        
        # Add edges with attributes
        for edge in self.edges:
            attrs = self.edge_attributes.get(edge, {})
            G.add_edge(edge[0], edge[1], **attrs)
        
        return G


@dataclass
class TemporalMetrics:
    """Metrics for temporal network analysis."""
    growth_rates: Dict[str, float] = field(default_factory=dict)
    centrality_evolution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    community_stability: Dict[str, float] = field(default_factory=dict)
    network_density_over_time: List[float] = field(default_factory=list)
    clustering_over_time: List[float] = field(default_factory=list)
    influential_nodes_over_time: Dict[str, List[str]] = field(default_factory=dict)


class TemporalNetworkAnalyzer:
    """
    Analyzes temporal evolution of networks with dynamic visualization capabilities.
    
    This class tracks network growth over time, computes evolution metrics,
    and creates animated visualizations showing network development.
    """
    
    def __init__(self, snapshots: List[NetworkSnapshot] = None):
        """
        Initialize the temporal network analyzer.
        
        Args:
            snapshots: List of network snapshots ordered by time
        """
        self.snapshots: List[NetworkSnapshot] = snapshots or []
        self.logger = logging.getLogger(__name__)
        self.metrics = TemporalMetrics()
        
        # Sort snapshots by timestamp
        self.snapshots.sort(key=lambda s: s.timestamp)
        
        # Animation state
        self.animation_data = {}
        self.color_maps = {
            'growth': LinearSegmentedColormap.from_list('growth', ['#3498db', '#e74c3c']),
            'centrality': LinearSegmentedColormap.from_list('centrality', ['#ecf0f1', '#8e44ad']),
            'influence': LinearSegmentedColormap.from_list('influence', ['#f39c12', '#27ae60'])
        }
    
    def add_snapshot(self, snapshot: NetworkSnapshot):
        """Add a new snapshot to the temporal sequence."""
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.timestamp)
        self.logger.info(f"Added snapshot from {snapshot.timestamp}")
    
    def create_snapshot_from_graph(self, graph: nx.Graph, timestamp: datetime = None, 
                                  metadata: Dict = None) -> NetworkSnapshot:
        """Create a snapshot from a NetworkX graph."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract nodes and edges
        nodes = set(graph.nodes())
        edges = set(graph.edges())
        
        # Extract attributes
        node_attrs = {node: dict(graph.nodes[node]) for node in nodes}
        edge_attrs = {edge: dict(graph.edges[edge]) for edge in edges}
        
        return NetworkSnapshot(
            timestamp=timestamp,
            nodes=nodes,
            edges=edges,
            node_attributes=node_attrs,
            edge_attributes=edge_attrs,
            metadata=metadata or {}
        )
    
    def compute_growth_metrics(self) -> Dict[str, float]:
        """Compute network growth metrics over time."""
        if len(self.snapshots) < 2:
            return {}
        
        metrics = {}
        
        # Node growth rate
        node_counts = [len(s.nodes) for s in self.snapshots]
        if len(node_counts) > 1:
            total_growth = (node_counts[-1] - node_counts[0]) / node_counts[0] if node_counts[0] > 0 else 0
            time_span = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()
            metrics['node_growth_rate'] = total_growth / (time_span / 3600) if time_span > 0 else 0  # per hour
        
        # Edge growth rate
        edge_counts = [len(s.edges) for s in self.snapshots]
        if len(edge_counts) > 1:
            total_growth = (edge_counts[-1] - edge_counts[0]) / edge_counts[0] if edge_counts[0] > 0 else 0
            time_span = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()
            metrics['edge_growth_rate'] = total_growth / (time_span / 3600) if time_span > 0 else 0  # per hour
        
        # Density evolution
        densities = []
        for snapshot in self.snapshots:
            graph = snapshot.to_networkx()
            num_nodes = len(graph.nodes())
            num_edges = len(graph.edges())
            if num_nodes > 1:
                max_edges = num_nodes * (num_nodes - 1)
                density = num_edges / max_edges if max_edges > 0 else 0
            else:
                density = 0
            densities.append(density)
        
        metrics['density_trend'] = np.polyfit(range(len(densities)), densities, 1)[0] if len(densities) > 1 else 0
        self.metrics.network_density_over_time = densities
        
        # Clustering evolution
        clustering_coeffs = []
        for snapshot in self.snapshots:
            graph = snapshot.to_networkx().to_undirected()
            if len(graph.nodes()) > 2:
                clustering = nx.average_clustering(graph)
            else:
                clustering = 0
            clustering_coeffs.append(clustering)
        
        metrics['clustering_trend'] = np.polyfit(range(len(clustering_coeffs)), clustering_coeffs, 1)[0] if len(clustering_coeffs) > 1 else 0
        self.metrics.clustering_over_time = clustering_coeffs
        
        self.metrics.growth_rates = metrics
        return metrics
    
    def analyze_node_evolution(self, node: str) -> Dict[str, Union[float, List[float]]]:
        """Analyze how a specific node evolves over time."""
        evolution = {
            'first_appearance': None,
            'degree_evolution': [],
            'centrality_evolution': {},
            'influence_evolution': []
        }
        
        for snapshot in self.snapshots:
            if node in snapshot.nodes:
                if evolution['first_appearance'] is None:
                    evolution['first_appearance'] = snapshot.timestamp
                
                # Degree evolution
                graph = snapshot.to_networkx()
                degree = graph.degree(node) if node in graph else 0
                evolution['degree_evolution'].append(degree)
                
                # Centrality evolution
                if len(graph.nodes()) > 1:
                    try:
                        pagerank = nx.pagerank(graph)
                        betweenness = nx.betweenness_centrality(graph)
                        
                        if 'pagerank' not in evolution['centrality_evolution']:
                            evolution['centrality_evolution']['pagerank'] = []
                        if 'betweenness' not in evolution['centrality_evolution']:
                            evolution['centrality_evolution']['betweenness'] = []
                        
                        evolution['centrality_evolution']['pagerank'].append(pagerank.get(node, 0))
                        evolution['centrality_evolution']['betweenness'].append(betweenness.get(node, 0))
                    except:
                        pass
            else:
                evolution['degree_evolution'].append(0)
                for metric in evolution['centrality_evolution']:
                    evolution['centrality_evolution'][metric].append(0)
        
        return evolution
    
    def detect_influential_nodes_over_time(self, top_k: int = 10) -> Dict[str, List[str]]:
        """Detect most influential nodes at each time step."""
        influential_over_time = {}
        
        for i, snapshot in enumerate(self.snapshots):
            graph = snapshot.to_networkx()
            
            if len(graph.nodes()) == 0:
                influential_over_time[f'snapshot_{i}'] = []
                continue
            
            # Combine multiple centrality measures
            centrality_scores = defaultdict(float)
            
            try:
                # PageRank
                pagerank = nx.pagerank(graph)
                for node, score in pagerank.items():
                    centrality_scores[node] += score
                
                # Degree centrality
                degree_centrality = nx.degree_centrality(graph)
                for node, score in degree_centrality.items():
                    centrality_scores[node] += score
                
                # Betweenness centrality (for smaller graphs)
                if len(graph.nodes()) <= 100:
                    betweenness = nx.betweenness_centrality(graph)
                    for node, score in betweenness.items():
                        centrality_scores[node] += score
            except:
                # Fallback to just degree
                for node in graph.nodes():
                    centrality_scores[node] = graph.degree(node)
            
            # Get top k nodes
            top_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            influential_over_time[f'snapshot_{i}'] = [node for node, _ in top_nodes]
        
        self.metrics.influential_nodes_over_time = influential_over_time
        return influential_over_time
    
    def create_animated_visualization(self, output_path: str = "network_evolution.gif",
                                    layout_method: str = "spring", 
                                    color_by: str = "growth",
                                    fps: int = 2, 
                                    figsize: Tuple[int, int] = (12, 8)) -> str:
        """
        Create an animated visualization of network evolution.
        
        Args:
            output_path: Path to save the animation
            layout_method: Layout algorithm ("spring", "circular", "random")
            color_by: Coloring scheme ("growth", "centrality", "influence")
            fps: Frames per second
            figsize: Figure size
            
        Returns:
            Path to saved animation
        """
        if not self.snapshots:
            raise ValueError("No snapshots available for animation")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get all nodes across all snapshots for consistent positioning
        all_nodes = set()
        for snapshot in self.snapshots:
            all_nodes.update(snapshot.nodes)
        
        # Create master graph for consistent layout
        master_graph = nx.Graph()
        master_graph.add_nodes_from(all_nodes)
        
        # Add edges from all snapshots to get better layout
        for snapshot in self.snapshots:
            for edge in snapshot.edges:
                master_graph.add_edge(edge[0], edge[1])
        
        # Generate layout
        if layout_method == "spring":
            pos = nx.spring_layout(master_graph, k=2, iterations=50)
        elif layout_method == "circular":
            pos = nx.circular_layout(master_graph)
        elif layout_method == "random":
            pos = nx.random_layout(master_graph)
        else:
            pos = nx.spring_layout(master_graph)
        
        # Prepare animation data
        self._prepare_animation_data(color_by)
        
        # Log snapshot information for debugging
        self.logger.info(f"Total snapshots: {len(self.snapshots)}")
        non_empty_count = sum(1 for s in self.snapshots if len(s.nodes) > 0)
        self.logger.info(f"Non-empty snapshots: {non_empty_count}")
        if self.snapshots:
            sizes = [len(s.nodes) for s in self.snapshots]
            self.logger.info(f"Snapshot sizes: {sizes}")
        
        # Filter out empty snapshots for better visualization
        non_empty_snapshots = [s for s in self.snapshots if len(s.nodes) > 0]
        
        if not non_empty_snapshots:
            raise ValueError("No non-empty snapshots available for animation")
        
        # Animation function
        def animate(frame):
            ax.clear()
            
            if frame >= len(non_empty_snapshots):
                return
            
            snapshot = non_empty_snapshots[frame]
            graph = snapshot.to_networkx()
            
            # Get nodes and edges present in this snapshot
            current_nodes = list(graph.nodes())
            current_edges = list(graph.edges())
            
            # Filter positions for current nodes
            current_pos = {node: pos[node] for node in current_nodes if node in pos}
            
            # Node colors and sizes
            node_colors = self._get_node_colors(graph, frame, color_by)
            node_sizes = self._get_node_sizes(graph, frame)
            
            # Draw network
            if current_nodes:
                nx.draw_networkx_nodes(graph, current_pos, node_color=node_colors, 
                                     node_size=node_sizes, alpha=0.8)
            
            if current_edges:
                nx.draw_networkx_edges(graph, current_pos, edge_color='gray', 
                                     alpha=0.5, width=0.5)
            
            # Add labels for important nodes
            important_nodes = self._get_important_nodes(graph, frame, top_k=5)
            if important_nodes:
                important_labels = {node: node for node in important_nodes if node in current_pos}
                nx.draw_networkx_labels(graph, current_pos, labels=important_labels, 
                                      font_size=8, font_weight='bold')
            
            # Title and info
            timestamp = snapshot.timestamp.strftime("%Y-%m-%d %H:%M")
            ax.set_title(f"Network Evolution - {timestamp}\n"
                        f"Nodes: {len(current_nodes)}, Edges: {len(current_edges)}", 
                        fontsize=12, fontweight='bold')
            
            ax.axis('off')
            
            # Add frame counter
            ax.text(0.02, 0.98, f"Frame {frame + 1}/{len(non_empty_snapshots)}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(non_empty_snapshots), 
                                    interval=1000//fps, repeat=True, blit=False)
        
        # Save animation
        try:
            ani.save(output_path, writer='pillow', fps=fps)
            self.logger.info(f"Animation saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save animation: {e}")
            # Try saving as MP4
            try:
                mp4_path = output_path.replace('.gif', '.mp4')
                ani.save(mp4_path, writer='ffmpeg', fps=fps)
                self.logger.info(f"Animation saved to {mp4_path}")
                output_path = mp4_path
            except Exception as e2:
                self.logger.error(f"Failed to save as MP4: {e2}")
                raise
        
        plt.close(fig)
        return output_path
    
    def _prepare_animation_data(self, color_by: str):
        """Prepare data for animation coloring."""
        self.animation_data = {}
        
        if color_by == "growth":
            # Track when each node first appears (only count non-empty snapshots)
            node_appearance = {}
            non_empty_index = 0
            for i, snapshot in enumerate(self.snapshots):
                if len(snapshot.nodes) > 0:
                    for node in snapshot.nodes:
                        if node not in node_appearance:
                            node_appearance[node] = non_empty_index
                    non_empty_index += 1
            self.animation_data['node_appearance'] = node_appearance
        
        elif color_by == "centrality":
            # Compute centrality for each snapshot (skip empty ones)
            centrality_data = []
            for snapshot in self.snapshots:
                graph = snapshot.to_networkx()
                if len(graph.nodes()) > 0:
                    try:
                        pagerank = nx.pagerank(graph)
                        centrality_data.append(pagerank)
                    except:
                        centrality_data.append({})
                else:
                    centrality_data.append({})
            self.animation_data['centrality'] = centrality_data
        
        elif color_by == "influence":
            # Track influence propagation if available
            self.detect_influential_nodes_over_time()
    
    def _get_node_colors(self, graph: nx.Graph, frame: int, color_by: str) -> List:
        """Get node colors for animation frame."""
        nodes = list(graph.nodes())
        colors = []
        
        if color_by == "growth":
            # Color based on when node appeared
            node_appearance = self.animation_data.get('node_appearance', {})
            max_frames = sum(1 for s in self.snapshots if len(s.nodes) > 0)
            max_frames = max(max_frames, 1)  # Avoid division by zero
            
            for node in nodes:
                appearance_frame = node_appearance.get(node, 0)
                age = frame - appearance_frame
                # Newer nodes are redder, older nodes are bluer
                intensity = max(0, 1 - age / max_frames)
                colors.append(self.color_maps['growth'](intensity))
        
        elif color_by == "centrality":
            # Color based on centrality
            centrality_data = self.animation_data.get('centrality', [])
            if frame < len(centrality_data):
                centrality = centrality_data[frame]
                max_centrality = max(centrality.values()) if centrality else 1
                for node in nodes:
                    intensity = centrality.get(node, 0) / max_centrality if max_centrality > 0 else 0
                    colors.append(self.color_maps['centrality'](intensity))
            else:
                colors = ['#3498db'] * len(nodes)
        
        else:  # Default coloring
            colors = ['#3498db'] * len(nodes)
        
        return colors
    
    def _get_node_sizes(self, graph: nx.Graph, frame: int) -> List[int]:
        """Get node sizes for animation frame."""
        nodes = list(graph.nodes())
        sizes = []
        
        # Size based on degree
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for node in nodes:
            degree = degrees.get(node, 0)
            size = 50 + (degree / max_degree) * 300
            sizes.append(size)
        
        return sizes
    
    def _get_important_nodes(self, graph: nx.Graph, frame: int, top_k: int = 5) -> List[str]:
        """Get most important nodes for labeling."""
        if len(graph.nodes()) == 0:
            return []
        
        # Use degree centrality for simplicity
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:top_k]]
    
    def create_evolution_dashboard(self, output_path: str = "evolution_dashboard.png"):
        """Create a comprehensive dashboard showing network evolution."""
        if not self.snapshots:
            raise ValueError("No snapshots available")
        
        # Compute metrics
        self.compute_growth_metrics()
        self.detect_influential_nodes_over_time()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Network Evolution Dashboard", fontsize=16, fontweight='bold')
        
        # Timeline data
        timestamps = [s.timestamp for s in self.snapshots]
        node_counts = [len(s.nodes) for s in self.snapshots]
        edge_counts = [len(s.edges) for s in self.snapshots]
        
        # 1. Node and Edge Growth
        axes[0, 0].plot(timestamps, node_counts, 'b-o', label='Nodes', linewidth=2)
        axes[0, 0].plot(timestamps, edge_counts, 'r-s', label='Edges', linewidth=2)
        axes[0, 0].set_title('Network Growth Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Network Density
        if self.metrics.network_density_over_time:
            axes[0, 1].plot(timestamps, self.metrics.network_density_over_time, 'g-^', linewidth=2)
            axes[0, 1].set_title('Network Density Evolution')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Clustering Coefficient
        if self.metrics.clustering_over_time:
            axes[0, 2].plot(timestamps, self.metrics.clustering_over_time, 'm-d', linewidth=2)
            axes[0, 2].set_title('Clustering Coefficient Evolution')
            axes[0, 2].set_xlabel('Time')
            axes[0, 2].set_ylabel('Clustering Coefficient')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Growth Rates
        if self.metrics.growth_rates:
            growth_metrics = list(self.metrics.growth_rates.keys())
            growth_values = list(self.metrics.growth_rates.values())
            axes[1, 0].bar(growth_metrics, growth_values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[1, 0].set_title('Growth Rate Metrics')
            axes[1, 0].set_ylabel('Rate (per hour)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Node Degree Distribution Evolution
        if len(self.snapshots) > 1:
            # Compare first and last snapshots
            first_graph = self.snapshots[0].to_networkx()
            last_graph = self.snapshots[-1].to_networkx()
            
            if len(first_graph.nodes()) > 0:
                first_degrees = list(dict(first_graph.degree()).values())
                axes[1, 1].hist(first_degrees, bins=20, alpha=0.7, label='Initial', color='blue')
            
            if len(last_graph.nodes()) > 0:
                last_degrees = list(dict(last_graph.degree()).values())
                axes[1, 1].hist(last_degrees, bins=20, alpha=0.7, label='Final', color='red')
            
            axes[1, 1].set_title('Degree Distribution Evolution')
            axes[1, 1].set_xlabel('Degree')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        # 6. Influential Nodes Stability
        if self.metrics.influential_nodes_over_time:
            # Calculate how often top nodes remain in top positions
            stability_scores = []
            snapshots_keys = list(self.metrics.influential_nodes_over_time.keys())
            
            for i in range(1, len(snapshots_keys)):
                prev_top = set(self.metrics.influential_nodes_over_time[snapshots_keys[i-1]][:5])
                curr_top = set(self.metrics.influential_nodes_over_time[snapshots_keys[i]][:5])
                stability = len(prev_top & curr_top) / len(prev_top) if prev_top else 0
                stability_scores.append(stability)
            
            if stability_scores:
                axes[1, 2].plot(range(1, len(stability_scores) + 1), stability_scores, 'o-', linewidth=2)
                axes[1, 2].set_title('Top Nodes Stability')
                axes[1, 2].set_xlabel('Time Step')
                axes[1, 2].set_ylabel('Stability Score')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evolution dashboard saved to {output_path}")
        return output_path
    
    def export_temporal_data(self, output_path: str = "temporal_data.json"):
        """Export temporal analysis data to JSON."""
        data = {
            'snapshots': [],
            'metrics': {
                'growth_rates': self.metrics.growth_rates,
                'network_density_over_time': self.metrics.network_density_over_time,
                'clustering_over_time': self.metrics.clustering_over_time,
                'influential_nodes_over_time': self.metrics.influential_nodes_over_time
            }
        }
        
        # Export snapshot metadata
        for i, snapshot in enumerate(self.snapshots):
            data['snapshots'].append({
                'index': i,
                'timestamp': snapshot.timestamp.isoformat(),
                'num_nodes': len(snapshot.nodes),
                'num_edges': len(snapshot.edges),
                'metadata': snapshot.metadata
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Temporal data exported to {output_path}")
        return output_path


def demo_temporal_analysis():
    """Demonstrate temporal network analysis."""
    import random
    from datetime import datetime, timedelta
    
    # Create sample temporal network
    analyzer = TemporalNetworkAnalyzer()
    
    # Generate evolving network snapshots
    base_time = datetime.now()
    nodes = set()
    edges = set()
    
    for i in range(10):
        # Add some nodes
        new_nodes = {f"node_{j}" for j in range(i * 3, (i + 1) * 3)}
        nodes.update(new_nodes)
        
        # Add some edges
        if len(nodes) > 1:
            node_list = list(nodes)
            for _ in range(min(i * 2, 20)):  # Add more edges over time
                u, v = random.sample(node_list, 2)
                edges.add((u, v))
        
        # Create snapshot
        snapshot = NetworkSnapshot(
            timestamp=base_time + timedelta(hours=i),
            nodes=nodes.copy(),
            edges=edges.copy(),
            metadata={'step': i}
        )
        
        analyzer.add_snapshot(snapshot)
    
    # Analyze evolution
    growth_metrics = analyzer.compute_growth_metrics()
    print("Growth Metrics:", growth_metrics)
    
    # Create visualizations
    analyzer.create_evolution_dashboard("demo_dashboard.png")
    analyzer.create_animated_visualization("demo_evolution.gif")
    
    print("Demo completed - check demo_dashboard.png and demo_evolution.gif")


if __name__ == "__main__":
    demo_temporal_analysis()