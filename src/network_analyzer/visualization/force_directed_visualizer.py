"""
Force-directed layout visualization with different physics engines.
"""

import networkx as nx
from pyvis.network import Network
from typing import Dict, Any, Optional, Tuple
import math
import json


class ForceDirectedVisualizer:
    """
    Enhanced force-directed visualization with multiple physics options.
    """
    
    PHYSICS_PRESETS = {
        "barnes_hut": {
            "name": "Barnes-Hut",
            "description": "Fast approximation for large networks",
            "params": {
                "gravity": -80000,
                "central_gravity": 0.3,
                "spring_length": 200,
                "spring_strength": 0.001,
                "damping": 0.09,
                "avoid_overlap": 0.1
            }
        },
        "force_atlas2": {
            "name": "ForceAtlas2",
            "description": "Gephi-inspired layout for communities",
            "params": {
                "gravity": -50000,
                "central_gravity": 0.01,
                "spring_length": 100,
                "spring_strength": 0.08,
                "damping": 0.4,
                "avoid_overlap": 0.0
            }
        },
        "hierarchical": {
            "name": "Hierarchical",
            "description": "Tree-like structure for directed graphs",
            "params": {
                "gravity": -30000,
                "central_gravity": 0.1,
                "spring_length": 150,
                "spring_strength": 0.05,
                "damping": 0.3,
                "avoid_overlap": 0.2
            }
        },
        "circular": {
            "name": "Circular Force",
            "description": "Circular arrangement with force simulation",
            "params": {
                "gravity": -20000,
                "central_gravity": 0.5,
                "spring_length": 300,
                "spring_strength": 0.01,
                "damping": 0.2,
                "avoid_overlap": 0.3
            }
        },
        "organic": {
            "name": "Organic",
            "description": "Natural, biological-inspired layout",
            "params": {
                "gravity": -60000,
                "central_gravity": 0.2,
                "spring_length": 250,
                "spring_strength": 0.02,
                "damping": 0.15,
                "avoid_overlap": 0.15
            }
        },
        "centrality": {
            "name": "Centrality-Based",
            "description": "Physics optimized for centrality-based node sizing",
            "params": {
                "gravity": -40000,
                "central_gravity": 0.1,
                "spring_length": 200,
                "spring_strength": 0.03,
                "damping": 0.12,
                "avoid_overlap": 0.2
            }
        }
    }
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.node_positions = {}
        
    def _calculate_adaptive_params(self, physics_type: str) -> Dict[str, Any]:
        """Calculate adaptive parameters based on network characteristics."""
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        
        # Get base parameters
        base_params = self.PHYSICS_PRESETS[physics_type]["params"].copy()
        
        # Adaptive scaling based on network size
        if num_nodes > 100:
            # Large networks: reduce forces, increase damping
            base_params["gravity"] *= 0.5
            base_params["spring_strength"] *= 0.7
            base_params["damping"] *= 1.3
        elif num_nodes < 20:
            # Small networks: increase forces for better separation
            base_params["gravity"] *= 1.5
            base_params["spring_strength"] *= 1.2
            base_params["damping"] *= 0.8
            
        # Adaptive scaling based on density
        if density > 0.3:
            # Dense networks: increase repulsion
            base_params["gravity"] *= 1.2
            base_params["avoid_overlap"] *= 1.5
        elif density < 0.1:
            # Sparse networks: reduce repulsion
            base_params["gravity"] *= 0.8
            base_params["spring_length"] *= 1.2
            
        return base_params
        
    def _get_stabilization_config(self, physics_type: str, num_nodes: int) -> Dict[str, Any]:
        """Get stabilization configuration based on physics type and network size."""
        base_iterations = 100
        
        # Adjust iterations based on network size
        if num_nodes > 200:
            iterations = base_iterations * 2
        elif num_nodes > 100:
            iterations = base_iterations * 1.5
        else:
            iterations = base_iterations
            
        # Physics-specific adjustments
        if physics_type == "hierarchical":
            iterations *= 1.5  # Hierarchical needs more time
        elif physics_type == "barnes_hut":
            iterations *= 0.8  # Barnes-Hut converges faster
            
        return {
            "enabled": True,
            "iterations": int(iterations),
            "updateInterval": 50,
            "onlyDynamicEdges": False,
            "fit": True
        }
        
    def _create_physics_config(self, physics_type: str) -> Dict[str, Any]:
        """Create comprehensive physics configuration."""
        if physics_type not in self.PHYSICS_PRESETS:
            raise ValueError(f"Unknown physics type: {physics_type}")
            
        params = self._calculate_adaptive_params(physics_type)
        num_nodes = len(self.graph.nodes())
        
        if physics_type == "hierarchical":
            return {
                "enabled": True,
                "hierarchicalRepulsion": {
                    "nodeDistance": abs(params["gravity"]) / 1000,
                    "centralGravity": params["central_gravity"],
                    "springLength": params["spring_length"],
                    "springConstant": params["spring_strength"],
                    "damping": params["damping"],
                    "avoidOverlap": params["avoid_overlap"]
                },
                "solver": "hierarchicalRepulsion",
                "stabilization": self._get_stabilization_config(physics_type, num_nodes)
            }
        else:
            return {
                "enabled": True,
                "barnesHut": {
                    "gravitationalConstant": params["gravity"],
                    "centralGravity": params["central_gravity"],
                    "springLength": params["spring_length"],
                    "springConstant": params["spring_strength"],
                    "damping": params["damping"],
                    "avoidOverlap": params["avoid_overlap"]
                },
                "solver": "barnesHut",
                "stabilization": self._get_stabilization_config(physics_type, num_nodes)
            }
    
    def visualize(self, 
                  output_path: str,
                  physics_type: str = "barnes_hut",
                  color_by: str = "depth",
                  size_by: str = "degree",
                  custom_params: Optional[Dict[str, Any]] = None,
                  **kwargs) -> None:
        """
        Create force-directed visualization with specified physics.
        
        Args:
            output_path: Path to save HTML file
            physics_type: Physics engine to use
            color_by: Node coloring scheme ("depth" or "community")
            size_by: Node sizing scheme ("degree", "betweenness", "pagerank", "closeness", "eigenvector")
            custom_params: Custom physics parameters to override defaults
            **kwargs: Additional pyvis Network parameters
        """
        # Default network parameters
        default_params = {
            "height": "750px",
            "width": "100%",
            "bgcolor": "#ffffff",
            "font_color": "black",
            "directed": True,
            "notebook": False,
            "cdn_resources": "in_line",
        }
        default_params.update(kwargs)
        
        # Create network
        net = Network(**default_params)
        
        # Set physics configuration
        physics_config = self._create_physics_config(physics_type)
        if custom_params:
            physics_config = self._merge_physics_params(physics_config, custom_params)
            
        # Add nodes with enhanced properties
        self._add_nodes_with_physics_hints(net, color_by, size_by, physics_type)
        
        # Add edges
        self._add_edges_with_physics_hints(net, physics_type)
        
        # Configure physics
        net.set_options(json.dumps({
            "physics": physics_config,
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "font": {
                    "size": 12,
                    "face": "arial"
                }
            },
            "edges": {
                "color": {"inherit": False},
                "smooth": {"type": "continuous"}
            },
            "interaction": {
                "hover": True,
                "multiselect": True,
                "navigationButtons": True,
                "tooltipDelay": 300
            },
            "layout": {
                "randomSeed": 42,  # For reproducible layouts
                "improvedLayout": True
            }
        }))
        
        # Save with physics info in filename
        physics_name = self.PHYSICS_PRESETS[physics_type]["name"]
        base_path = output_path.replace(".html", f"_{physics_type}.html")
        
        net.save_graph(base_path)
        
        # Add physics info to HTML
        self._add_physics_info_to_html(base_path, physics_type, physics_config)
        
        print(f"Force-directed visualization saved: {base_path}")
        print(f"Physics engine: {physics_name}")
        print(f"Nodes: {len(self.graph.nodes())}, Edges: {len(self.graph.edges())}")
    
    def _add_nodes_with_physics_hints(self, net: Network, color_by: str, size_by: str, physics_type: str):
        """Add nodes with physics-aware properties."""
        nodes = list(self.graph.nodes(data=True))
        
        # Calculate centrality measures for sizing
        centrality_measures = self._calculate_centrality_measures()
        
        # Calculate node properties
        degrees = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        # Color scheme
        if color_by == "community":
            communities = self._detect_communities()
            colors = self._generate_community_colors(communities)
        else:
            colors = self._generate_depth_colors(nodes)
            
        for node, data in nodes:
            degree = degrees.get(node, 1)
            
            # Size based on selected centrality measure
            size = self._calculate_node_size(node, size_by, centrality_measures, physics_type)
            
            # Mass affects physics simulation (use centrality for more realistic physics)
            mass = self._calculate_node_mass(node, size_by, centrality_measures)
            
            # Color
            color = colors.get(node, "#3498db")
            
            # Enhanced tooltip
            tooltip = self._create_enhanced_tooltip(node, data, degree, physics_type, centrality_measures)
            
            net.add_node(
                node,
                label=str(node),
                color=color,
                size=size,
                mass=mass,
                title=tooltip,
                physics=True
            )
    
    def _add_edges_with_physics_hints(self, net: Network, physics_type: str):
        """Add edges with physics-aware properties."""
        for source, target, data in self.graph.edges(data=True):
            # Edge strength affects spring simulation
            weight = data.get('weight', 1)
            
            # Adjust edge properties based on physics type
            if physics_type == "hierarchical":
                length = 100 + (1 / weight) * 50 if weight > 0 else 150
                width = 1 + min(weight * 2, 3)
            else:
                length = 200 + (1 / weight) * 100 if weight > 0 else 300
                width = 1 + min(weight, 2)
                
            net.add_edge(
                source,
                target,
                color="#95a5a6",
                width=width,
                length=length,
                physics=True
            )
    
    def _merge_physics_params(self, base_config: Dict[str, Any], 
                            custom_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom parameters with base physics configuration."""
        merged = base_config.copy()
        
        # Determine which solver is being used
        solver = "barnesHut" if "barnesHut" in merged else "hierarchicalRepulsion"
        
        # Merge custom parameters
        if solver in merged:
            merged[solver].update(custom_params)
            
        return merged
    
    def _detect_communities(self) -> Dict[str, int]:
        """Detect communities using modularity optimization."""
        try:
            import community as community_louvain
            return community_louvain.best_partition(self.graph.to_undirected())
        except ImportError:
            # Fallback to simple connected components
            components = list(nx.connected_components(self.graph.to_undirected()))
            communities = {}
            for i, component in enumerate(components):
                for node in component:
                    communities[node] = i
            return communities
    
    def _generate_community_colors(self, communities: Dict[str, int]) -> Dict[str, str]:
        """Generate colors for communities."""
        colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#34495e", "#e67e22", "#95a5a6", "#f1c40f"
        ]
        
        community_colors = {}
        for node, community_id in communities.items():
            color_index = community_id % len(colors)
            community_colors[node] = colors[color_index]
            
        return community_colors
    
    def _generate_depth_colors(self, nodes: list) -> Dict[str, str]:
        """Generate colors based on node depth."""
        colors = {}
        depth_colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#34495e", "#e67e22"
        ]
        
        for node, data in nodes:
            depth = data.get('depth', 0)
            color_index = depth % len(depth_colors)
            colors[node] = depth_colors[color_index]
            
        return colors
    
    def _calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for all nodes."""
        centrality_measures = {}
        
        # Degree centrality
        centrality_measures['degree'] = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (computationally expensive for large graphs)
        if len(self.graph.nodes()) <= 1000:
            # Handle disconnected graphs for betweenness centrality
            if nx.is_connected(self.graph.to_undirected()):
                centrality_measures['betweenness'] = nx.betweenness_centrality(self.graph)
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
                centrality_measures['betweenness'] = betweenness
        else:
            # Use approximation for large graphs
            centrality_measures['betweenness'] = nx.betweenness_centrality(self.graph, k=min(100, len(self.graph.nodes())))
            
        # PageRank centrality
        centrality_measures['pagerank'] = nx.pagerank(self.graph)
        
        # Closeness centrality
        if len(self.graph.nodes()) <= 1000:
            # Handle disconnected graphs properly for closeness centrality
            if nx.is_connected(self.graph.to_undirected()):
                centrality_measures['closeness'] = nx.closeness_centrality(self.graph)
            else:
                # For disconnected graphs, calculate closeness centrality per component
                closeness = {}
                for component in nx.connected_components(self.graph.to_undirected()):
                    subgraph = self.graph.subgraph(component)
                    component_closeness = nx.closeness_centrality(subgraph)
                    closeness.update(component_closeness)
                centrality_measures['closeness'] = closeness
        else:
            # Skip for very large graphs as it's computationally expensive
            centrality_measures['closeness'] = {node: 0.0 for node in self.graph.nodes()}
            
        # Eigenvector centrality
        try:
            # Handle disconnected graphs for eigenvector centrality
            if nx.is_connected(self.graph.to_undirected()):
                centrality_measures['eigenvector'] = nx.eigenvector_centrality(self.graph, max_iter=1000)
            else:
                # For disconnected graphs, calculate eigenvector per component
                eigenvector = {}
                for component in nx.connected_components(self.graph.to_undirected()):
                    if len(component) > 1:  # Eigenvector needs at least 2 nodes
                        subgraph = self.graph.subgraph(component)
                        try:
                            component_eigenvector = nx.eigenvector_centrality(subgraph, max_iter=1000)
                            eigenvector.update(component_eigenvector)
                        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                            # Fallback to degree for this component
                            for node in component:
                                eigenvector[node] = centrality_measures['degree'].get(node, 0.0)
                    else:
                        # Single node component
                        for node in component:
                            eigenvector[node] = 0.0
                centrality_measures['eigenvector'] = eigenvector
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Fallback to degree centrality if eigenvector fails
            centrality_measures['eigenvector'] = centrality_measures['degree']
            
        return centrality_measures
    
    def _calculate_node_size(self, node: str, size_by: str, centrality_measures: Dict[str, Dict[str, float]], physics_type: str) -> float:
        """Calculate node size based on centrality measure with enhanced visual distinction."""
        if size_by not in centrality_measures:
            size_by = 'degree'  # Fallback to degree
            
        centrality_values = centrality_measures[size_by]
        max_centrality = max(centrality_values.values()) if centrality_values else 1
        node_centrality = centrality_values.get(node, 0)
        
        # Normalize centrality score
        normalized_score = node_centrality / max_centrality if max_centrality > 0 else 0
        
        # Apply non-linear scaling for better visual distinction
        # Use square root scaling to emphasize differences while keeping reasonable max size
        scaled_score = normalized_score ** 0.5
        
        # Size based on physics type with wider, more dramatic ranges
        if physics_type == "hierarchical":
            min_size = 8
            max_size = 60  # 7.5x difference
        else:
            min_size = 5
            max_size = 80  # 16x difference
            
        size_range = max_size - min_size
        return min_size + (scaled_score * size_range)
    
    def _calculate_node_mass(self, node: str, size_by: str, centrality_measures: Dict[str, Dict[str, float]]) -> float:
        """Calculate node mass based on centrality measure for physics simulation."""
        if size_by not in centrality_measures:
            size_by = 'degree'  # Fallback to degree
            
        centrality_values = centrality_measures[size_by]
        max_centrality = max(centrality_values.values()) if centrality_values else 1
        node_centrality = centrality_values.get(node, 0)
        
        # Normalize centrality score
        normalized_score = node_centrality / max_centrality if max_centrality > 0 else 0
        
        # Mass affects physics simulation - more central nodes have more mass
        return 1 + (normalized_score * 3)
    
    def _create_enhanced_tooltip(self, node: str, data: Dict[str, Any], 
                                degree: int, physics_type: str, centrality_measures: Dict[str, Dict[str, float]]) -> str:
        """Create enhanced tooltip with physics and centrality information."""
        tooltip_parts = [
            f"{node}",
            f"Degree: {degree}",
            f"Depth: {data.get('depth', 'N/A')}",
            f"Physics: {self.PHYSICS_PRESETS[physics_type]['name']}"
        ]
        
        # Add centrality measures
        for measure_name, measure_values in centrality_measures.items():
            value = measure_values.get(node, 0)
            tooltip_parts.append(f"{measure_name.capitalize()}: {value:.4f}")
        
        if 'title' in data:
            tooltip_parts.append(f"Title: {data['title']}")
            
        return "\n".join(tooltip_parts)
    
    def _add_physics_info_to_html(self, file_path: str, physics_type: str, 
                                 physics_config: Dict[str, Any]):
        """Add physics information to the HTML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        physics_info = f"""
        <div style="position: fixed; top: 10px; right: 10px; background: rgba(255,255,255,0.9); 
                    padding: 10px; border-radius: 5px; font-family: Arial; font-size: 12px;
                    border: 1px solid #ccc; z-index: 1000;">
            <b>Physics Engine:</b> {self.PHYSICS_PRESETS[physics_type]['name']}<br>
            <b>Description:</b> {self.PHYSICS_PRESETS[physics_type]['description']}<br>
            <b>Nodes:</b> {len(self.graph.nodes())}<br>
            <b>Edges:</b> {len(self.graph.edges())}
        </div>
        """
        
        # Insert physics info before closing body tag
        content = content.replace('</body>', f'{physics_info}</body>')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    @classmethod
    def list_physics_options(cls) -> None:
        """List available physics options with descriptions."""
        print("Available Force-Directed Physics Options:")
        print("=" * 50)
        
        for key, preset in cls.PHYSICS_PRESETS.items():
            print(f"{key}: {preset['name']}")
            print(f"   Description: {preset['description']}")
            print()
    
    @classmethod
    def get_physics_info(cls, physics_type: str) -> Dict[str, Any]:
        """Get information about a specific physics type."""
        if physics_type not in cls.PHYSICS_PRESETS:
            raise ValueError(f"Unknown physics type: {physics_type}")
            
        return cls.PHYSICS_PRESETS[physics_type]