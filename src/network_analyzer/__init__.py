"""
Network Analyzer - A tool for building and analyzing knowledge networks.

This package provides tools for:
- Building Wikipedia and Coursera course networks
- Analyzing network structure and communities
- Visualizing networks with interactive tools
- Influence propagation analysis
- Temporal network evolution analysis
"""

__version__ = "0.1.0"
__author__ = "Network Analyzer Team"

from .core.config import NetworkConfig
from .core.network_builder import WikipediaNetworkBuilder
from .core.unified_network_builder import UnifiedNetworkBuilder
from .core.temporal_network_builder import TemporalNetworkBuilder
from .visualization.force_directed_visualizer import ForceDirectedVisualizer
from .analysis.influence_propagation import InfluencePropagationSimulator, PropagationConfig
from .analysis.temporal_evolution import TemporalNetworkAnalyzer, NetworkSnapshot

__all__ = [
    "NetworkConfig",
    "WikipediaNetworkBuilder", 
    "UnifiedNetworkBuilder",
    "TemporalNetworkBuilder",
    "ForceDirectedVisualizer",
    "InfluencePropagationSimulator",
    "PropagationConfig",
    "TemporalNetworkAnalyzer",
    "NetworkSnapshot",
]