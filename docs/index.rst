Network Analyzer Documentation
==============================

Welcome to Network Analyzer, a powerful Python tool for building and analyzing knowledge networks from Wikipedia articles and online course data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   installation
   configuration
   api
   examples
   contributing

Overview
--------

Network Analyzer provides a comprehensive suite of tools for:

* **Multi-source Network Building**: Build networks from Wikipedia articles and publicly available Coursera course data
* **Multiple Network Construction Methods**: Breadth-first search, random walk, depth-first search, and more
* **Advanced Analysis**: Community detection, centrality measures, influence propagation simulation
* **Interactive Visualizations**: Web-based network graphs with physics simulation
* **Data Export**: Save networks in GraphML format for further analysis

Key Features
------------

Network Building Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Breadth-First Search**: Explores the network level by level, ideal for comprehensive local exploration
* **Random Walk**: Stochastic exploration with restart probability, good for discovering unexpected connections
* **Depth-First Search**: Deep exploration with backtracking, useful for finding long paths
* **Topic-Focused Crawling**: Uses keyword similarity to stay focused on specific topics
* **Hub-and-Spoke**: Identifies important nodes (hubs) and expands around them

Analysis Capabilities
~~~~~~~~~~~~~~~~~~~

* **Community Detection**: Automatically identifies clusters of related topics using modularity optimization
* **Centrality Measures**: Degree centrality, PageRank, betweenness centrality
* **Network Statistics**: Comprehensive metrics and analysis
* **Influence Propagation**: Models how information spreads through the network

Visualization Options
~~~~~~~~~~~~~~~~~~~

* **Interactive Networks**: Physics-based layouts with multiple engines
* **Static Plots**: Community structure visualization and influence propagation heatmaps
* **Customizable Styling**: Node coloring by depth or community
* **Export Formats**: Multiple output formats for further analysis

Data Sources
~~~~~~~~~~~~

* **Wikipedia**: Uses MediaWiki API for real-time data with intelligent filtering
* **Coursera Courses**: Skill-based relationship mapping using publicly available Kaggle dataset
* **Hybrid Mode**: Combines multiple data sources for comprehensive analysis

Quick Start
-----------

Basic Wikipedia Network
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Multi-source Network
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Installation
------------

From Source
~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/username/network-analyzer.git
    cd network-analyzer
    pip install -e .

Requirements
~~~~~~~~~~~~

* Python 3.8+
* NetworkX 3.0+
* See requirements.txt for full dependency list

CLI Usage
---------

The package includes command-line interfaces for interactive network building:

.. code-block:: bash

    # Wikipedia networks
    network-analyzer

    # Multi-source networks
    network-analyzer-unified

Project Structure
-----------------

.. code-block:: text

    network-analyzer/
    ├── src/
    │   └── network_analyzer/
    │       ├── core/                 # Core network building functionality
    │       ├── data_sources/         # Data source adapters
    │       ├── analysis/            # Analysis tools
    │       ├── visualization/       # Visualization tools
    │       └── utils/              # Utility functions
    ├── scripts/                    # Command-line interfaces
    ├── tests/                     # Test suite
    ├── data/                      # Data files
    └── outputs/                   # Generated outputs

License
-------

MIT License - see LICENSE file for details.

Citation
--------

If you use this tool in academic work, please cite:

.. code-block:: bibtex

    @software{network_analyzer,
      title = {Network Analyzer: A Tool for Knowledge Network Analysis},
      author = {Ryan Boris},
      year = {2025},
      url = {https://github.com/username/network-analyzer}
    }

Support
-------

* Issues: `GitHub Issues <https://github.com/username/network-analyzer/issues>`_
* Discussions: `GitHub Discussions <https://github.com/username/network-analyzer/discussions>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`