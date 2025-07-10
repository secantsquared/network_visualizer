"""
Temporal Network Builder

This module extends the existing network builder to create time-aware networks
that track how networks evolve over time during the building process.
"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from tqdm import tqdm

from .network_builder import WikipediaNetworkBuilder
from .config import NetworkConfig
from ..analysis.temporal_evolution import NetworkSnapshot, TemporalNetworkAnalyzer


class TemporalNetworkBuilder(WikipediaNetworkBuilder):
    """
    Extended network builder that captures temporal evolution during network construction.
    
    This builder creates snapshots at regular intervals during the network building process,
    allowing analysis of how the network grows and evolves over time.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None, 
                 snapshot_interval: int = 10,
                 max_snapshots: int = 50):
        """
        Initialize the temporal network builder.
        
        Args:
            config: Network configuration
            snapshot_interval: Number of nodes to process between snapshots
            max_snapshots: Maximum number of snapshots to keep
        """
        super().__init__(config)
        
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.snapshots: List[NetworkSnapshot] = []
        self.temporal_analyzer = TemporalNetworkAnalyzer()
        
        # Temporal tracking
        self.start_time = None
        self.nodes_processed = 0
        self.processing_timeline = []
        self.node_addition_times = {}
        
        # State for creating snapshots
        self.last_snapshot_count = 0
        
        # Target minimum number of snapshots for good visualization
        self.target_min_snapshots = 15
    
    def _create_snapshot(self, metadata: Dict = None) -> NetworkSnapshot:
        """Create a snapshot of the current network state."""
        current_time = datetime.now()
        
        # Create snapshot
        snapshot = NetworkSnapshot(
            timestamp=current_time,
            nodes=set(self.graph.nodes()),
            edges=set(self.graph.edges()),
            node_attributes={node: dict(self.graph.nodes[node]) for node in self.graph.nodes()},
            edge_attributes={edge: dict(self.graph.edges[edge]) for edge in self.graph.edges()},
            metadata=metadata or {}
        )
        
        # Add to snapshots
        self.snapshots.append(snapshot)
        
        # Limit number of snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        # Update temporal analyzer
        self.temporal_analyzer.add_snapshot(snapshot)
        
        return snapshot
    
    def _get_adaptive_snapshot_interval(self) -> int:
        """Calculate adaptive snapshot interval based on network size."""
        # Estimate final network size from max_articles_to_process
        estimated_final_size = getattr(self.config, 'max_articles_to_process', 100)
        
        # Calculate interval to get target number of snapshots
        if estimated_final_size <= 30:
            # For small networks, create very frequent snapshots
            return max(1, estimated_final_size // 15)  # ~15 snapshots
        elif estimated_final_size <= 100:
            # For medium networks, use moderate interval
            return max(2, estimated_final_size // 20)  # ~20 snapshots
        else:
            # For large networks, use the original interval
            return self.snapshot_interval
    
    def _should_create_snapshot(self) -> bool:
        """Determine if a snapshot should be created."""
        current_node_count = len(self.graph.nodes())
        
        # Skip if network is still empty
        if current_node_count == 0:
            return False
        
        # Use adaptive snapshot interval for better coverage
        adaptive_interval = self._get_adaptive_snapshot_interval()
        
        # Create snapshot if we've processed enough new nodes
        if current_node_count >= self.last_snapshot_count + adaptive_interval:
            return True
        
        # Create snapshot if significant time has passed
        if self.snapshots:
            time_since_last = datetime.now() - self.snapshots[-1].timestamp
            if time_since_last.total_seconds() > 60:  # 1 minute
                return True
        
        return False
    
    def _track_node_addition(self, node: str, depth: int = 0, source: str = None):
        """Track when a node is added to the network."""
        if node not in self.node_addition_times:
            self.node_addition_times[node] = {
                'timestamp': datetime.now(),
                'depth': depth,
                'source': source,
                'order': len(self.node_addition_times)
            }
            
            # Add temporal attributes to the node
            self.graph.nodes[node]['added_at'] = self.node_addition_times[node]['timestamp']
            self.graph.nodes[node]['discovery_order'] = self.node_addition_times[node]['order']
    
    def build_network(self, seeds: List[str], progress_callback: Optional[callable] = None, method: str = "breadth_first") -> nx.Graph:
        """
        Build network with temporal tracking.
        
        Args:
            seeds: List of seed articles
            progress_callback: Optional callback function for progress updates
            method: Network building method
            
        Returns:
            The built network graph
        """
        self.start_time = datetime.now()
        
        # Don't create initial empty snapshot - will be created when first node is added
        
        # Call parent build_network with temporal tracking
        result = super().build_network(seeds, progress_callback, method)
        
        # Create final snapshot
        final_snapshot = self._create_snapshot({
            'type': 'final',
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'build_time': (datetime.now() - self.start_time).total_seconds()
        })
        
        return result
    
    def _add_article_to_graph(self, article: str, links: List[str], depth: int = 0) -> int:
        """Override to add temporal tracking."""
        # Track node addition
        if article not in self.graph:
            self._track_node_addition(article, depth)
        
        # Create snapshot before adding nodes if this is the first addition
        if len(self.graph.nodes()) == 0:
            self._create_snapshot({
                'type': 'initial',
                'current_article': article,
                'depth': depth,
                'nodes_so_far': 0,
                'edges_so_far': 0
            })
        
        # Call parent method
        added_count = super()._add_article_to_graph(article, links, depth)
        
        # Track edge additions
        for link in links:
            if link in self.graph and self.graph.has_edge(article, link):
                # Add temporal attributes to edge
                self.graph.edges[article, link]['added_at'] = datetime.now()
        
        # Create snapshot if needed
        if self._should_create_snapshot():
            self._create_snapshot({
                'type': 'progress',
                'current_article': article,
                'depth': depth,
                'nodes_so_far': len(self.graph.nodes()),
                'edges_so_far': len(self.graph.edges())
            })
            self.last_snapshot_count = len(self.graph.nodes())
        
        return added_count
    
    async def build_network_breadth_first_async(self, seeds: List[str], 
                                               progress_callback: Optional[callable] = None) -> nx.Graph:
        """Override async method to add temporal tracking."""
        # Create initial empty snapshot
        self._create_snapshot({
            'type': 'initial',
            'seeds': seeds,
            'nodes_so_far': 0,
            'edges_so_far': 0
        })
        
        # Create a wrapper progress callback that also creates snapshots
        def temporal_progress_callback(articles_processed: int, total_articles: int = None):
            # Create snapshot if needed
            if self._should_create_snapshot():
                self._create_snapshot({
                    'type': 'progress',
                    'articles_processed': articles_processed,
                    'total_articles': total_articles,
                    'nodes_so_far': len(self.graph.nodes()),
                    'edges_so_far': len(self.graph.edges())
                })
                self.last_snapshot_count = len(self.graph.nodes())
            
            # Call original callback if provided
            if progress_callback:
                progress_callback(articles_processed, total_articles)
        
        # Call parent method with our temporal callback
        result = await super().build_network_breadth_first_async(seeds, temporal_progress_callback)
        
        # Create final snapshot
        self._create_snapshot({
            'type': 'final',
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'build_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        })
        
        return result
    
    def build_network_random_walk(self, seeds: List[str], 
                                 progress_callback: Optional[callable] = None) -> nx.Graph:
        """Override random walk method to add temporal tracking."""
        # Create initial empty snapshot
        self._create_snapshot({
            'type': 'initial',
            'seeds': seeds,
            'method': 'random_walk',
            'nodes_so_far': 0,
            'edges_so_far': 0
        })
        
        # Create a wrapper progress callback that also creates snapshots
        def temporal_progress_callback(articles_processed: int, total_articles: int = None):
            # Create snapshot if needed
            if self._should_create_snapshot():
                self._create_snapshot({
                    'type': 'progress',
                    'method': 'random_walk',
                    'articles_processed': articles_processed,
                    'total_articles': total_articles,
                    'nodes_so_far': len(self.graph.nodes()),
                    'edges_so_far': len(self.graph.edges())
                })
                self.last_snapshot_count = len(self.graph.nodes())
            
            # Call original callback if provided
            if progress_callback:
                progress_callback(articles_processed, total_articles)
        
        # Call parent method with our temporal callback
        result = super().build_network_random_walk(seeds, temporal_progress_callback)
        
        # Create final snapshot
        self._create_snapshot({
            'type': 'final',
            'method': 'random_walk',
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'build_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        })
        
        return result
    
    def build_network_dfs(self, seeds: List[str], 
                         progress_callback: Optional[callable] = None) -> nx.Graph:
        """Override DFS method to add temporal tracking."""
        # Create initial empty snapshot
        self._create_snapshot({
            'type': 'initial',
            'seeds': seeds,
            'method': 'dfs',
            'nodes_so_far': 0,
            'edges_so_far': 0
        })
        
        # Create a wrapper progress callback that also creates snapshots
        def temporal_progress_callback(articles_processed: int, total_articles: int = None):
            # Create snapshot if needed
            if self._should_create_snapshot():
                self._create_snapshot({
                    'type': 'progress',
                    'method': 'dfs',
                    'articles_processed': articles_processed,
                    'total_articles': total_articles,
                    'nodes_so_far': len(self.graph.nodes()),
                    'edges_so_far': len(self.graph.edges())
                })
                self.last_snapshot_count = len(self.graph.nodes())
            
            # Call original callback if provided
            if progress_callback:
                progress_callback(articles_processed, total_articles)
        
        # Call parent method with our temporal callback
        result = super().build_network_dfs(seeds, temporal_progress_callback)
        
        # Create final snapshot
        self._create_snapshot({
            'type': 'final',
            'method': 'dfs',
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'build_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        })
        
        return result
    
    def build_network_topic_focused(self, seeds: List[str], 
                                   progress_callback: Optional[callable] = None) -> nx.Graph:
        """Override topic-focused method to add temporal tracking."""
        # Create initial empty snapshot
        self._create_snapshot({
            'type': 'initial',
            'seeds': seeds,
            'method': 'topic_focused',
            'nodes_so_far': 0,
            'edges_so_far': 0
        })
        
        # Create a wrapper progress callback that also creates snapshots
        def temporal_progress_callback(articles_processed: int, total_articles: int = None):
            # Create snapshot if needed
            if self._should_create_snapshot():
                self._create_snapshot({
                    'type': 'progress',
                    'method': 'topic_focused',
                    'articles_processed': articles_processed,
                    'total_articles': total_articles,
                    'nodes_so_far': len(self.graph.nodes()),
                    'edges_so_far': len(self.graph.edges())
                })
                self.last_snapshot_count = len(self.graph.nodes())
            
            # Call original callback if provided
            if progress_callback:
                progress_callback(articles_processed, total_articles)
        
        # Call parent method with our temporal callback
        result = super().build_network_topic_focused(seeds, temporal_progress_callback)
        
        # Create final snapshot
        self._create_snapshot({
            'type': 'final',
            'method': 'topic_focused',
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'build_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        })
        
        return result
    
    def build_network_hub_and_spoke(self, seeds: List[str], 
                                   progress_callback: Optional[callable] = None) -> nx.Graph:
        """Override hub-and-spoke method to add temporal tracking."""
        # Create initial empty snapshot
        self._create_snapshot({
            'type': 'initial',
            'seeds': seeds,
            'method': 'hub_and_spoke',
            'nodes_so_far': 0,
            'edges_so_far': 0
        })
        
        # Create a wrapper progress callback that also creates snapshots
        def temporal_progress_callback(articles_processed: int, total_articles: int = None):
            # Create snapshot if needed
            if self._should_create_snapshot():
                self._create_snapshot({
                    'type': 'progress',
                    'method': 'hub_and_spoke',
                    'articles_processed': articles_processed,
                    'total_articles': total_articles,
                    'nodes_so_far': len(self.graph.nodes()),
                    'edges_so_far': len(self.graph.edges())
                })
                self.last_snapshot_count = len(self.graph.nodes())
            
            # Call original callback if provided
            if progress_callback:
                progress_callback(articles_processed, total_articles)
        
        # Call parent method with our temporal callback
        result = super().build_network_hub_and_spoke(seeds, temporal_progress_callback)
        
        # Create final snapshot
        self._create_snapshot({
            'type': 'final',
            'method': 'hub_and_spoke',
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'build_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        })
        
        return result
    
    def get_temporal_analysis(self) -> TemporalNetworkAnalyzer:
        """Get the temporal analyzer with all snapshots."""
        return self.temporal_analyzer
    
    def save_to_history(self, history_dir: str = "outputs/history"):
        """
        Save current temporal output files to a unique run directory within history folder.
        
        Args:
            history_dir: Directory name for storing history files
        """
        from pathlib import Path
        import shutil
        
        # Create history directory if it doesn't exist
        history_path = Path(history_dir)
        history_path.mkdir(exist_ok=True, parents=True)
        
        # Find the next available run number
        run_number = 1
        while True:
            run_dir = history_path / f"temporal_run{run_number}"
            if not run_dir.exists():
                break
            run_number += 1
        
        # Create the run directory
        run_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # List of temporal files to archive
        files_to_archive = [
            "temporal_network_dashboard.png",
            "temporal_network_depth.html",
            "temporal_network_communities.html",
            "temporal_communities.png",
            "temporal_network.graphml",
            "temporal_network_evolution.gif",
            "temporal_growth_metrics.gif",
            "temporal_network_data.json",
            "temporal_network_data_extended.json",
            # Also include test files if they exist
            "test_temporal_evolution.gif",
            "test_temporal_dashboard.png",
        ]
        
        archived_files = []
        
        for filename in files_to_archive:
            source_path = Path(filename)
            if source_path.exists():
                # Keep original filename
                dest_path = run_dir / filename
                
                # Copy file to run directory
                try:
                    shutil.copy2(source_path, dest_path)
                    archived_files.append(str(dest_path))
                    self.logger.info(f"Archived {filename} -> {dest_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to archive {filename}: {e}")
        
        # Create a metadata file with timestamp and run info
        metadata_path = run_dir / "temporal_run_info.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Temporal Network Run: {run_number}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total snapshots: {len(self.snapshots)}\n")
            f.write(f"Network size: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges\n")
            f.write(f"Files archived: {len(archived_files)}\n")
            f.write(f"Files:\n")
            for file in archived_files:
                f.write(f"  - {Path(file).name}\n")
        
        if archived_files:
            self.logger.info(
                f"Successfully archived {len(archived_files)} temporal files to {run_dir}/"
            )
            return archived_files, str(run_dir)
        else:
            self.logger.warning("No temporal files found to archive")
            return [], str(run_dir)
    
    def create_evolution_visualization(self, output_path: str = "network_evolution.gif",
                                     **kwargs) -> str:
        """Create animated visualization of network evolution."""
        return self.temporal_analyzer.create_animated_visualization(output_path, **kwargs)
    
    def create_evolution_dashboard(self, output_path: str = "evolution_dashboard.png") -> str:
        """Create evolution dashboard."""
        return self.temporal_analyzer.create_evolution_dashboard(output_path)
    
    def analyze_growth_patterns(self) -> Dict:
        """Analyze growth patterns during network construction."""
        if not self.snapshots:
            return {}
        
        analysis = {
            'total_build_time': (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds(),
            'growth_metrics': self.temporal_analyzer.compute_growth_metrics(),
            'node_addition_pattern': self._analyze_node_addition_pattern(),
            'discovery_depth_analysis': self._analyze_discovery_depths(),
            'temporal_centrality': self._analyze_temporal_centrality()
        }
        
        return analysis
    
    def _analyze_node_addition_pattern(self) -> Dict:
        """Analyze the pattern of node additions."""
        if not self.node_addition_times:
            return {}
        
        # Group nodes by time intervals
        time_intervals = defaultdict(list)
        start_time = min(info['timestamp'] for info in self.node_addition_times.values())
        
        for node, info in self.node_addition_times.items():
            elapsed = (info['timestamp'] - start_time).total_seconds()
            interval = int(elapsed // 10)  # 10-second intervals
            time_intervals[interval].append(node)
        
        # Calculate addition rates
        addition_rates = {}
        for interval, nodes in time_intervals.items():
            addition_rates[interval] = len(nodes)
        
        return {
            'time_intervals': dict(time_intervals),
            'addition_rates': addition_rates,
            'peak_discovery_interval': max(addition_rates.keys(), key=addition_rates.get) if addition_rates else None,
            'total_discovery_time': max(addition_rates.keys()) * 10 if addition_rates else 0
        }
    
    def _analyze_discovery_depths(self) -> Dict:
        """Analyze node discovery by depth."""
        depth_analysis = defaultdict(list)
        
        for node, info in self.node_addition_times.items():
            depth = info.get('depth', 0)
            depth_analysis[depth].append(node)
        
        return {
            'nodes_by_depth': dict(depth_analysis),
            'depth_distribution': {depth: len(nodes) for depth, nodes in depth_analysis.items()},
            'max_depth_reached': max(depth_analysis.keys()) if depth_analysis else 0
        }
    
    def _analyze_temporal_centrality(self) -> Dict:
        """Analyze how centrality measures evolve over time."""
        centrality_evolution = {}
        
        for i, snapshot in enumerate(self.snapshots):
            if len(snapshot.nodes) < 2:
                continue
                
            graph = snapshot.to_networkx()
            
            try:
                # Calculate centrality measures
                degree_centrality = nx.degree_centrality(graph)
                
                # Get top 5 nodes by degree centrality
                top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                
                centrality_evolution[i] = {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'top_nodes': top_nodes,
                    'avg_centrality': sum(degree_centrality.values()) / len(degree_centrality)
                }
            except:
                continue
        
        return centrality_evolution
    
    def export_temporal_data(self, output_path: str = "temporal_network_data.json") -> str:
        """Export comprehensive temporal network data."""
        data = {
            'build_info': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'snapshot_interval': self.snapshot_interval,
                'max_snapshots': self.max_snapshots
            },
            'node_addition_times': {
                node: {
                    'timestamp': info['timestamp'].isoformat(),
                    'depth': info['depth'],
                    'source': info['source'],
                    'order': info['order']
                }
                for node, info in self.node_addition_times.items()
            },
            'growth_analysis': self.analyze_growth_patterns()
        }
        
        # Export via temporal analyzer
        self.temporal_analyzer.export_temporal_data(output_path)
        
        # Also save our extended data
        import json
        extended_path = output_path.replace('.json', '_extended.json')
        with open(extended_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return extended_path
    
    def create_growth_animation_with_metrics(self, output_path: str = "growth_with_metrics.gif") -> str:
        """Create an enhanced animation showing growth with metrics overlay."""
        if not self.snapshots:
            raise ValueError("No snapshots available")
        
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Prepare data
        timestamps = [s.timestamp for s in self.snapshots]
        node_counts = [len(s.nodes) for s in self.snapshots]
        edge_counts = [len(s.edges) for s in self.snapshots]
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            if frame >= len(self.snapshots):
                return
            
            # Left plot: Network visualization
            snapshot = self.snapshots[frame]
            graph = snapshot.to_networkx()
            
            if len(graph.nodes()) > 0:
                pos = nx.spring_layout(graph, k=1, iterations=20)
                
                # Color nodes by discovery order
                node_colors = []
                for node in graph.nodes():
                    order = self.node_addition_times.get(node, {}).get('order', 0)
                    max_order = max((info.get('order', 0) for info in self.node_addition_times.values()), default=0)
                    intensity = order / max_order if max_order > 0 else 0
                    node_colors.append(plt.cm.viridis(intensity))
                
                nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                                     node_size=50, alpha=0.8, ax=ax1)
                nx.draw_networkx_edges(graph, pos, edge_color='gray', 
                                     alpha=0.5, width=0.5, ax=ax1)
            
            ax1.set_title(f"Network Growth - Frame {frame + 1}")
            ax1.axis('off')
            
            # Right plot: Growth metrics
            current_frame = frame + 1
            ax2.plot(range(current_frame), node_counts[:current_frame], 'b-o', label='Nodes')
            ax2.plot(range(current_frame), edge_counts[:current_frame], 'r-s', label='Edges')
            ax2.set_title('Growth Metrics Over Time')
            ax2.set_xlabel('Snapshot')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add current stats
            stats_text = f"Nodes: {len(snapshot.nodes)}\nEdges: {len(snapshot.edges)}\n"
            stats_text += f"Time: {snapshot.timestamp.strftime('%H:%M:%S')}"
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(self.snapshots), 
                                    interval=1000, repeat=True, blit=False)
        
        # Save animation
        try:
            ani.save(output_path, writer='pillow', fps=1)
            self.logger.info(f"Growth animation saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save animation: {e}")
            raise
        
        plt.close(fig)
        return output_path

    def create_evolution_dashboard(self, output_path: str = "temporal_network_dashboard.png") -> str:
        """Create a comprehensive dashboard showing temporal network evolution."""
        return self.temporal_analyzer.create_evolution_dashboard(output_path)

    def create_evolution_visualization(self, output_path: str = "temporal_network_evolution.gif", 
                                     color_by: str = "growth", fps: int = 2) -> str:
        """Create animated visualization of network evolution."""
        return self.temporal_analyzer.create_animated_visualization(
            output_path, color_by=color_by, fps=fps
        )


def demo_temporal_network_building():
    """Demonstrate temporal network building."""
    from .config import NetworkConfig
    
    # Create config with smaller limits for demo
    config = NetworkConfig(
        max_articles_to_process=30,
        max_depth=2,
        links_per_article=5
    )
    
    # Create temporal builder
    builder = TemporalNetworkBuilder(config, snapshot_interval=5)
    
    # Build network
    print("Building temporal network...")
    graph = builder.build_network(["Machine Learning", "Data Science"], method="breadth_first")
    
    # Analyze growth
    print("\nAnalyzing growth patterns...")
    analysis = builder.analyze_growth_patterns()
    
    print(f"Total build time: {analysis['total_build_time']:.2f} seconds")
    print(f"Growth metrics: {analysis['growth_metrics']}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    builder.create_evolution_dashboard("temporal_demo_dashboard.png")
    builder.create_evolution_visualization("temporal_demo_evolution.gif")
    
    # Export data
    builder.export_temporal_data("temporal_demo_data.json")
    
    print("Temporal network demo completed!")
    print("Check the generated files for visualizations and data.")


if __name__ == "__main__":
    demo_temporal_network_building()