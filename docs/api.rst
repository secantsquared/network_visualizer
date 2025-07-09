API Reference
=============

This section provides detailed documentation for Network Analyzer's API.

Core Classes
------------

NetworkConfig
~~~~~~~~~~~~~

.. autoclass:: network_analyzer.NetworkConfig
   :members:
   :undoc-members:
   :show-inheritance:

WikipediaNetworkBuilder
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: network_analyzer.WikipediaNetworkBuilder
   :members:
   :undoc-members:
   :show-inheritance:

UnifiedNetworkBuilder
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: network_analyzer.UnifiedNetworkBuilder
   :members:
   :undoc-members:
   :show-inheritance:

Data Sources
------------

Base Data Source
~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.data_sources.base
   :members:
   :undoc-members:
   :show-inheritance:

Wikipedia Data Source
~~~~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.data_sources.wikipedia
   :members:
   :undoc-members:
   :show-inheritance:

Coursera Data Source
~~~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.data_sources.coursera
   :members:
   :undoc-members:
   :show-inheritance:

Hybrid Data Source
~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.data_sources.hybrid
   :members:
   :undoc-members:
   :show-inheritance:

Analysis Tools
--------------

Influence Propagation
~~~~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.analysis.influence_propagation
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

Force Directed Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.visualization.force_directed_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Async Utilities
~~~~~~~~~~~~~~

.. automodule:: network_analyzer.utils.async_limited
   :members:
   :undoc-members:
   :show-inheritance:

Core Module API
---------------

network_analyzer.core.config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.core.config
   :members:
   :undoc-members:
   :show-inheritance:

network_analyzer.core.network_builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.core.network_builder
   :members:
   :undoc-members:
   :show-inheritance:

network_analyzer.core.unified_network_builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: network_analyzer.core.unified_network_builder
   :members:
   :undoc-members:
   :show-inheritance:

Data Types and Structures
-------------------------

Network Data
~~~~~~~~~~~~

The network data is represented using NetworkX graphs:

.. code-block:: python

    import networkx as nx
    
    # Network Analyzer returns NetworkX Graph objects
    graph = builder.build_network(["Machine Learning"])
    
    # Standard NetworkX operations are available
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    
    # Access node attributes
    for node, attrs in graph.nodes(data=True):
        print(f"Node: {node}")
        print(f"  Depth: {attrs.get('depth', 'N/A')}")
        print(f"  Source: {attrs.get('source', 'N/A')}")

Analysis Results
~~~~~~~~~~~~~~~

Analysis methods return structured dictionaries:

.. code-block:: python

    stats = builder.analyze_network()
    
    # Available statistics
    print(f"Basic metrics: {stats['basic_metrics']}")
    print(f"Centrality: {stats['centrality']}")
    print(f"Communities: {stats['communities']}")
    print(f"Density: {stats['density']}")

Influence Propagation Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    results = builder.analyze_influence_propagation(
        seed_nodes=["Machine Learning"],
        model="independent_cascade"
    )
    
    # Available results
    print(f"Final infected count: {results['final_infected_count']}")
    print(f"Propagation steps: {results['propagation_steps']}")
    print(f"Influence graph: {results['influence_graph']}")

Configuration Options
--------------------

The ``NetworkConfig`` class accepts the following parameters:

Core Network Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``max_depth``
     - int
     - Maximum depth to explore from seed nodes (default: 2)
   * - ``max_articles_to_process``
     - int
     - Maximum number of articles to process (default: 50)
   * - ``links_per_article``
     - int
     - Number of links to follow per article (default: 20)
   * - ``method``
     - str
     - Exploration method ("breadth_first", "random_walk", etc.)

Performance Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``async_enabled``
     - bool
     - Enable asynchronous processing (default: True)
   * - ``max_workers``
     - int
     - Number of worker threads (default: 4)
   * - ``max_concurrent_requests``
     - int
     - Maximum concurrent API requests (default: 10)
   * - ``request_delay``
     - float
     - Delay between requests in seconds (default: 0.1)

Data Source Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``data_source_type``
     - str
     - Data source type ("wikipedia", "coursera", "hybrid")
   * - ``primary_data_source``
     - str
     - Primary data source for hybrid mode
   * - ``coursera_dataset_path``
     - str
     - Path to Coursera dataset CSV file

Exception Handling
------------------

Network Analyzer defines custom exceptions:

.. automodule:: network_analyzer.core.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Common exceptions:

.. code-block:: python

    from network_analyzer.core.exceptions import (
        NetworkAnalyzerError,
        ConfigurationError,
        DataSourceError,
        NetworkBuildError
    )
    
    try:
        builder = WikipediaNetworkBuilder(config)
        graph = builder.build_network(["Invalid Topic"])
    except NetworkBuildError as e:
        print(f"Network build failed: {e}")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")

Method Reference
----------------

Common Network Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Build network
    graph = builder.build_network(seed_topics)
    
    # Analyze network
    stats = builder.analyze_network()
    
    # Visualize network
    builder.visualize_pyvis("output.html")
    builder.visualize_communities("communities.png")
    
    # Save network
    builder.save_network("network.graphml")
    
    # Get network statistics
    basic_stats = builder.get_basic_statistics()
    centrality = builder.calculate_centrality()
    communities = builder.detect_communities()

Analysis Methods
~~~~~~~~~~~~~~~

.. code-block:: python

    # Community detection
    communities = builder.detect_communities(
        algorithm="louvain",
        resolution=1.0
    )
    
    # Centrality calculation
    centrality = builder.calculate_centrality(
        measures=["degree", "pagerank", "betweenness"]
    )
    
    # Influence propagation
    influence = builder.analyze_influence_propagation(
        seed_nodes=["Machine Learning"],
        model="independent_cascade",
        num_simulations=100
    )

Visualization Methods
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Interactive visualization
    builder.visualize_pyvis(
        output_file="network.html",
        layout="barnes_hut",
        physics_enabled=True,
        node_size_method="degree"
    )
    
    # Static community plot
    builder.visualize_communities(
        output_file="communities.png",
        layout="spring",
        node_size=300,
        font_size=12
    )
    
    # Influence propagation visualization
    builder.visualize_influence_propagation(
        results=influence_results,
        output_file="influence.png"
    )

Extending Network Analyzer
--------------------------

Custom Data Sources
~~~~~~~~~~~~~~~~~~

To create a custom data source, inherit from ``BaseDataSource``:

.. code-block:: python

    from network_analyzer.data_sources.base import BaseDataSource
    
    class CustomDataSource(BaseDataSource):
        def __init__(self, config):
            super().__init__(config)
        
        def get_related_topics(self, topic):
            # Implement your data source logic
            return related_topics
        
        def get_topic_metadata(self, topic):
            # Return metadata about the topic
            return metadata

Custom Analysis Methods
~~~~~~~~~~~~~~~~~~~~~~

Add custom analysis methods by extending the builder classes:

.. code-block:: python

    from network_analyzer import WikipediaNetworkBuilder
    
    class CustomNetworkBuilder(WikipediaNetworkBuilder):
        def custom_analysis(self):
            # Implement your custom analysis
            return results

Threading and Async Support
---------------------------

Network Analyzer supports both synchronous and asynchronous operations:

.. code-block:: python

    # Synchronous operation
    config = NetworkConfig(async_enabled=False)
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Machine Learning"])
    
    # Asynchronous operation
    config = NetworkConfig(async_enabled=True, max_workers=8)
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Machine Learning"])

Error Handling Best Practices
-----------------------------

.. code-block:: python

    import logging
    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    from network_analyzer.core.exceptions import NetworkAnalyzerError
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        config = NetworkConfig(
            max_depth=2,
            max_articles_to_process=50
        )
        builder = WikipediaNetworkBuilder(config)
        graph = builder.build_network(["Machine Learning"])
        
    except NetworkAnalyzerError as e:
        logging.error(f"Network Analyzer error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

Performance Considerations
-------------------------

For optimal performance:

1. **Use Async**: Enable async processing for I/O-bound operations
2. **Adjust Workers**: Set ``max_workers`` based on your system
3. **Enable Caching**: Use caching to avoid repeated API calls
4. **Monitor Memory**: Large networks can consume significant memory
5. **Batch Operations**: Process multiple topics in batches when possible

Example performance configuration:

.. code-block:: python

    config = NetworkConfig(
        async_enabled=True,
        max_workers=8,
        max_concurrent_requests=20,
        cache_enabled=True,
        cache_expire_after=3600
    )