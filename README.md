# Network Analyzer

A powerful Python tool for building and analyzing knowledge networks from Wikipedia articles and online course data. The tool can create interactive visualizations and perform advanced network analysis including community detection and influence propagation modeling.

## Features

- **Multi-source Network Building**: Build networks from Wikipedia articles and publicly available Coursera course data from Kaggle
- **Multiple Network Construction Methods**: 
  - Breadth-first search (sync/async)
  - Random walk with restart
  - Depth-first search with backtracking
  - Topic-focused crawling
  - Hub-and-spoke expansion
- **Advanced Analysis**:
  - Community detection
  - Centrality measures (degree, PageRank, betweenness)
  - Network statistics and metrics
  - Influence propagation simulation
- **Interactive Visualizations**:
  - Web-based network graphs with physics simulation
  - Multiple layout algorithms (Barnes-Hut, Force Atlas2, etc.)
  - Static community plots
  - Influence propagation visualizations
- **Data Export**: Save networks in GraphML format for further analysis
- **Comprehensive Filtering**: Intelligent filtering of irrelevant pages and content

## Installation

### From Source

```bash
git clone https://github.com/username/network-analyzer.git
cd network-analyzer
pip install -e .
```

### Requirements

- Python 3.8+
- NetworkX 3.0+
- Required dependencies are listed in `requirements.txt`

## Quick Start

### Basic Wikipedia Network

```python
from network_analyzer import NetworkConfig, WikipediaNetworkBuilder

# Create configuration
config = NetworkConfig(
    max_depth=2,
    max_articles_to_process=50,
    links_per_article=20
)

# Build network
builder = WikipediaNetworkBuilder(config)
graph = builder.build_network(["Machine Learning", "Data Science"])

# Analyze and visualize
stats = builder.analyze_network()
builder.print_analysis(stats)
builder.visualize_pyvis("network.html")
```

### Multi-source Network (Wikipedia + Coursera)

```python
from network_analyzer import NetworkConfig, UnifiedNetworkBuilder

# Configure for hybrid mode
config = NetworkConfig(
    data_source_type="hybrid",
    primary_data_source="wikipedia",
    coursera_dataset_path="data/coursera_courses_2024.csv"
)

# Build unified network
builder = UnifiedNetworkBuilder(config, 
                               coursera_dataset_path="data/coursera_courses_2024.csv")
graph = builder.build_network(["Python Programming", "Machine Learning"])

# Analyze influence propagation
influence_results = builder.analyze_influence_propagation(
    seed_nodes=["Python Programming"],
    model="independent_cascade",
    num_simulations=100
)
```

## CLI Usage

The package includes command-line interfaces for interactive network building:

```bash
# Wikipedia networks
network-analyzer

# Multi-source networks
network-analyzer-unified
```

## Project Structure

```
network-analyzer/
├── src/
│   └── network_analyzer/
│       ├── core/                 # Core network building functionality
│       │   ├── config.py         # Configuration management
│       │   ├── network_builder.py # Wikipedia network builder
│       │   └── unified_network_builder.py # Multi-source builder
│       ├── data_sources/         # Data source adapters
│       │   ├── base.py          # Abstract base class
│       │   ├── wikipedia.py     # Wikipedia data source
│       │   ├── coursera.py      # Coursera data source
│       │   └── hybrid.py        # Hybrid data source
│       ├── analysis/            # Analysis tools
│       │   └── influence_propagation.py # Influence analysis
│       ├── visualization/       # Visualization tools
│       │   └── force_directed_visualizer.py # Interactive viz
│       └── utils/              # Utility functions
│           └── async_limited.py # Rate limiting
├── scripts/                    # Command-line interfaces
├── tests/                     # Test suite
├── data/                      # Data files
└── outputs/                   # Generated outputs
```

## Configuration

The `NetworkConfig` class provides extensive configuration options:

```python
config = NetworkConfig(
    # Core parameters
    max_depth=2,
    max_articles_to_process=50,
    links_per_article=20,
    
    # Performance
    max_workers=8,
    async_enabled=True,
    max_concurrent_requests=20,
    
    # Algorithm-specific
    random_walk_steps=100,
    restart_probability=0.15,
    topic_similarity_threshold=0.3,
    
    # Data sources
    data_source_type="hybrid",
    primary_data_source="wikipedia",
    coursera_dataset_path="data/coursera_courses_2024.csv"
)
```

## Network Building Methods

### 1. Breadth-First Search
Explores the network level by level, ideal for comprehensive local exploration.

### 2. Random Walk
Stochastic exploration with restart probability, good for discovering unexpected connections.

### 3. Depth-First Search
Deep exploration with backtracking, useful for finding long paths and specialized topics.

### 4. Topic-Focused Crawling
Uses keyword similarity to stay focused on specific topics.

### 5. Hub-and-Spoke
Identifies important nodes (hubs) and expands around them.

## Analysis Features

### Community Detection
Automatically identifies clusters of related topics using modularity optimization.

### Centrality Measures
- **Degree Centrality**: Most connected nodes
- **PageRank**: Authority-based importance
- **Betweenness Centrality**: Bridge nodes

### Influence Propagation
Models how information spreads through the network using:
- Independent Cascade Model
- Linear Threshold Model

## Visualization

### Interactive Networks
- Physics-based layouts with multiple engines
- Node coloring by depth or community
- Interactive exploration with zoom and pan
- Customizable styling

### Static Plots
- Community structure visualization
- Influence propagation heatmaps
- Network statistics charts

## Data Sources

### Wikipedia
- Uses MediaWiki API for real-time data
- Intelligent filtering of disambiguation pages
- Configurable link extraction

### Coursera Courses
- Skill-based relationship mapping
- Course metadata integration
- Learning path analysis
- **Dataset**: Uses the publicly available Coursera course dataset from Kaggle: https://www.kaggle.com/datasets/azraimohamad/coursera-course-data?select=coursera_course_dataset_v3.csv

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this tool in academic work, please cite:

```bibtex
@software{network_analyzer,
  title = {Network Analyzer: A Tool for Knowledge Network Analysis},
  author = {Ryan Boris},
  year = {2025},
  url = {https://github.com/username/network-analyzer}
}
```

## Support

- Documentation: [ReadTheDocs](https://network-analyzer.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/username/network-analyzer/issues)
- Discussions: [GitHub Discussions](https://github.com/username/network-analyzer/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
