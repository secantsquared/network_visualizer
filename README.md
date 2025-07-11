# Network Analyzer

A powerful Python tool for building and analyzing knowledge networks from Wikipedia articles and online course data. The tool can create interactive visualizations and perform advanced network analysis including community detection and influence propagation modeling.

## Features

- **Multi-source Network Building**: Build networks from Wikipedia articles, Reddit communities, and publicly available Coursera course data from Kaggle
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
git clone https://github.com/secantsquared/network-analyzer.git
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

### Learning Path Generation

```python
from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
from network_analyzer.analysis import LearningPathAnalyzer
from network_analyzer.visualization import LearningPathVisualizer

# Build knowledge network
config = NetworkConfig(max_depth=2, max_articles_to_process=20)
builder = WikipediaNetworkBuilder(config)
graph = builder.build_network(["Machine Learning"])

# Generate learning paths
analyzer = LearningPathAnalyzer(graph)
learning_paths = analyzer.generate_multiple_paths("Machine Learning")

# Create visualizations
visualizer = LearningPathVisualizer(learning_paths)
outputs = visualizer.create_all_visualizations(learning_paths, "learning_outputs")

# Display paths
for path_type, path in learning_paths.items():
    print(f"{path_type}: {len(path.nodes)} steps, {path.total_estimated_time}")
    for i, node in enumerate(path.nodes, 1):
        print(f"  {i}. {node.name} ({node.estimated_time})")
```

## CLI Usage

The package includes command-line interfaces for interactive network building:

```bash
# Wikipedia networks
network-analyzer

# Multi-source networks
network-analyzer-unified

# Reddit networks
network-analyzer-reddit

# Learning path generation
network-analyzer-learning-path
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

### Learning Path Generation

Automatically generates optimal learning sequences from knowledge networks:

- **Multiple Path Types**: Foundational, comprehensive, and fast-track learning paths
- **Prerequisite Detection**: Automatically identifies prerequisite relationships
- **Difficulty Progression**: Smart ordering based on topic complexity
- **Time Estimation**: Realistic learning time estimates per topic
- **Quality Metrics**: Comprehensive path analysis and quality scoring

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

### Reddit

- Uses Reddit API (PRAW) for real-time data
- Support for subreddit, user, and discussion networks
- Configurable time filters and data limits
- **Setup**: Requires Reddit API credentials (see setup instructions below)

### Coursera Courses

- Skill-based relationship mapping
- Course metadata integration
- Learning path analysis
- **Dataset**: Uses the publicly available [Coursera course dataset](https://www.kaggle.com/datasets/azraimohamad/coursera-course-data?select=coursera_course_dataset_v3.csv) from Kaggle

## Reddit API Setup

To use Reddit data sources, you need Reddit API credentials:

1. **Create a Reddit App**:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Choose "script" for personal use
   - Fill in the required fields

2. **Set Your Credentials** (choose one method):

   **Option 1: Environment Variables**
   ```bash
   export REDDIT_CLIENT_ID="your_client_id"
   export REDDIT_CLIENT_SECRET="your_client_secret"
   export REDDIT_USER_AGENT="your_app_name:v1.0 (by /u/yourusername)"
   ```

   **Option 2: .env File**
   ```bash
   # Create .env file in project root
   cp .env.template .env
   # Edit .env file with your credentials
   ```

   **Option 3: Shell Profile**
   ```bash
   # Add to ~/.bashrc, ~/.zshrc, etc.
   export REDDIT_CLIENT_ID="your_client_id"
   export REDDIT_CLIENT_SECRET="your_client_secret"
   export REDDIT_USER_AGENT="your_app_name:v1.0 (by /u/yourusername)"
   ```

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
  url = {https://github.com/secantsquared/network-analyzer}
}
```

## Support

- Documentation: [ReadTheDocs](https://network-visualizer.readthedocs.io/en/latest/)
- Issues: [GitHub Issues](https://github.com/secantsquared/network_visualizer/issues)
- Discussions: [GitHub Discussions](https://github.com/secantsquared/network_visualizer/discussions)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
