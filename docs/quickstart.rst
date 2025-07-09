Quick Start
===========

This guide will help you get started with Network Analyzer quickly.

Basic Usage
-----------

1. Import the Core Classes
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder

2. Create a Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        max_depth=2,                    # How deep to explore
        max_articles_to_process=50,     # Maximum number of articles
        links_per_article=20,           # Links to follow per article
        method="breadth_first"          # Exploration method
    )

3. Build Your First Network
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create a builder instance
    builder = WikipediaNetworkBuilder(config)
    
    # Build network starting from seed topics
    graph = builder.build_network(["Machine Learning", "Data Science"])
    
    # The graph is a NetworkX graph object
    print(f"Network has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

4. Analyze the Network
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get comprehensive network statistics
    stats = builder.analyze_network()
    
    # Print analysis results
    builder.print_analysis(stats)
    
    # Access specific metrics
    print(f"Network density: {stats['density']:.3f}")
    print(f"Average clustering: {stats['average_clustering']:.3f}")

5. Visualize the Network
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create interactive HTML visualization
    builder.visualize_pyvis("my_network.html")
    
    # Create static community plot
    builder.visualize_communities("communities.png")

Complete Example
----------------

Here's a complete example that builds, analyzes, and visualizes a network:

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    def main():
        # Configure the network builder
        config = NetworkConfig(
            max_depth=2,
            max_articles_to_process=30,
            links_per_article=15,
            method="breadth_first",
            async_enabled=True,
            max_workers=4
        )
        
        # Create builder and build network
        builder = WikipediaNetworkBuilder(config)
        print("Building network...")
        graph = builder.build_network(["Artificial Intelligence", "Machine Learning"])
        
        # Analyze network
        print("Analyzing network...")
        stats = builder.analyze_network()
        builder.print_analysis(stats)
        
        # Visualize
        print("Creating visualizations...")
        builder.visualize_pyvis("ai_network.html")
        builder.visualize_communities("ai_communities.png")
        
        # Save network
        builder.save_network("ai_network.graphml")
        print("Network saved as ai_network.graphml")
        
        # Print top nodes by centrality
        print("\nTop 5 nodes by PageRank:")
        for node, score in sorted(stats['pagerank'].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {node}: {score:.3f}")
    
    if __name__ == "__main__":
        main()

Multi-Source Networks
---------------------

Network Analyzer can also build networks from multiple data sources:

.. code-block:: python

    from network_analyzer import NetworkConfig, UnifiedNetworkBuilder
    
    # Configure for hybrid mode
    config = NetworkConfig(
        data_source_type="hybrid",
        primary_data_source="wikipedia",
        coursera_dataset_path="data/coursera_courses_2024.csv",
        max_depth=2,
        max_articles_to_process=40
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
    
    print(f"Influence spread: {influence_results['final_infected_count']} nodes")

Command Line Interface
---------------------

Network Analyzer provides interactive CLI tools:

**Basic Wikipedia Networks:**

.. code-block:: bash

    network-analyzer

**Multi-source Networks:**

.. code-block:: bash

    network-analyzer-unified

The CLI will guide you through:

1. Choosing exploration method
2. Setting parameters
3. Selecting seed topics
4. Configuring output options

Different Exploration Methods
-----------------------------

Network Analyzer supports several exploration methods:

Breadth-First Search
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(method="breadth_first")
    # Explores level by level - good for comprehensive local coverage

Random Walk
~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        method="random_walk",
        random_walk_steps=100,
        restart_probability=0.15
    )
    # Stochastic exploration - good for discovering unexpected connections

Depth-First Search
~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(method="depth_first")
    # Deep exploration with backtracking - good for finding long paths

Topic-Focused Crawling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        method="topic_focused",
        topic_similarity_threshold=0.3
    )
    # Stays focused on similar topics using keyword matching

Hub and Spoke
~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(method="hub_and_spoke")
    # Identifies important nodes and expands around them

Performance Optimization
-----------------------

For better performance with large networks:

.. code-block:: python

    config = NetworkConfig(
        async_enabled=True,              # Enable async processing
        max_workers=8,                   # Number of worker threads
        max_concurrent_requests=20,      # Concurrent API requests
        cache_enabled=True               # Enable response caching
    )

Output Files
-----------

Network Analyzer creates several output files:

* **{name}.html**: Interactive network visualization
* **{name}.graphml**: Network data in GraphML format
* **{name}_communities.png**: Community structure plot
* **run_info.txt**: Detailed run information and statistics

Next Steps
----------

* Learn about :doc:`configuration` options
* Explore the :doc:`api` documentation
* Check out more :doc:`examples`
* Read about :doc:`contributing` to the project