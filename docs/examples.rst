Examples
========

This section provides practical examples of using Network Analyzer for different use cases.

Basic Examples
--------------

Simple Wikipedia Network
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    # Create a simple configuration
    config = NetworkConfig(
        max_depth=2,
        max_articles_to_process=30,
        links_per_article=15
    )
    
    # Build network
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Machine Learning"])
    
    # Print basic info
    print(f"Network has {graph.number_of_nodes()} nodes")
    print(f"Network has {graph.number_of_edges()} edges")
    
    # Analyze and visualize
    stats = builder.analyze_network()
    builder.print_analysis(stats)
    builder.visualize_pyvis("ml_network.html")

Multiple Seed Topics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    config = NetworkConfig(
        max_depth=2,
        max_articles_to_process=50,
        links_per_article=20
    )
    
    # Build network from multiple starting points
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network([
        "Artificial Intelligence",
        "Machine Learning",
        "Deep Learning",
        "Neural Networks"
    ])
    
    # Analyze the network
    stats = builder.analyze_network()
    
    # Print top nodes by different centrality measures
    print("Top 5 nodes by PageRank:")
    for node, score in sorted(stats['pagerank'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {score:.4f}")
    
    print("\nTop 5 nodes by Degree:")
    for node, score in sorted(stats['degree'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {score}")

Advanced Examples
-----------------

Async Network Building
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    import asyncio
    import time
    
    async def build_network_async():
        config = NetworkConfig(
            max_depth=3,
            max_articles_to_process=100,
            links_per_article=25,
            async_enabled=True,
            max_workers=8,
            max_concurrent_requests=30
        )
        
        builder = WikipediaNetworkBuilder(config)
        
        start_time = time.time()
        graph = builder.build_network(["Computer Science", "Mathematics"])
        end_time = time.time()
        
        print(f"Built network in {end_time - start_time:.2f} seconds")
        print(f"Network has {graph.number_of_nodes()} nodes")
        
        # Analyze and save
        stats = builder.analyze_network()
        builder.save_network("cs_math_network.graphml")
        builder.visualize_pyvis("cs_math_network.html")
        
        return graph, stats
    
    # Run the async function
    graph, stats = asyncio.run(build_network_async())

Random Walk Exploration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    config = NetworkConfig(
        method="random_walk",
        max_depth=3,
        max_articles_to_process=80,
        random_walk_steps=150,
        restart_probability=0.1,
        num_walks=5
    )
    
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Physics"])
    
    # Analyze the random walk network
    stats = builder.analyze_network()
    
    print("Random Walk Network Analysis:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    print(f"  Average clustering: {stats['average_clustering']:.4f}")
    print(f"  Network diameter: {stats.get('diameter', 'N/A')}")
    
    # Visualize with physics simulation
    builder.visualize_pyvis(
        "physics_network.html",
        physics_enabled=True,
        layout="barnes_hut"
    )

Topic-Focused Crawling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    config = NetworkConfig(
        method="topic_focused",
        max_depth=2,
        max_articles_to_process=60,
        topic_similarity_threshold=0.4,
        focus_keywords=["machine learning", "neural network", "algorithm"]
    )
    
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Machine Learning"])
    
    # Analyze topic focus
    stats = builder.analyze_network()
    
    print("Topic-Focused Network:")
    print(f"  Focused on: {config.focus_keywords}")
    print(f"  Similarity threshold: {config.topic_similarity_threshold}")
    print(f"  Nodes found: {graph.number_of_nodes()}")
    
    # Create focused visualization
    builder.visualize_communities("focused_communities.png")

Multi-Source Examples
---------------------

Hybrid Wikipedia-Coursera Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, UnifiedNetworkBuilder
    
    config = NetworkConfig(
        data_source_type="hybrid",
        primary_data_source="wikipedia",
        coursera_dataset_path="data/coursera_courses_2024.csv",
        max_depth=2,
        max_articles_to_process=60,
        cross_source_similarity_threshold=0.3
    )
    
    builder = UnifiedNetworkBuilder(
        config, 
        coursera_dataset_path="data/coursera_courses_2024.csv"
    )
    
    # Build hybrid network
    graph = builder.build_network([
        "Python Programming",
        "Data Science",
        "Machine Learning"
    ])
    
    # Analyze multi-source network
    stats = builder.analyze_network()
    
    print("Hybrid Network Analysis:")
    print(f"  Total nodes: {graph.number_of_nodes()}")
    print(f"  Total edges: {graph.number_of_edges()}")
    
    # Identify nodes by source
    wikipedia_nodes = [n for n, d in graph.nodes(data=True) 
                      if d.get('source') == 'wikipedia']
    coursera_nodes = [n for n, d in graph.nodes(data=True) 
                     if d.get('source') == 'coursera']
    
    print(f"  Wikipedia nodes: {len(wikipedia_nodes)}")
    print(f"  Coursera nodes: {len(coursera_nodes)}")
    
    # Visualize with source-based coloring
    builder.visualize_pyvis("hybrid_network.html")

Influence Propagation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, UnifiedNetworkBuilder
    
    config = NetworkConfig(
        data_source_type="hybrid",
        primary_data_source="wikipedia",
        coursera_dataset_path="data/coursera_courses_2024.csv",
        max_depth=2,
        max_articles_to_process=80
    )
    
    builder = UnifiedNetworkBuilder(
        config,
        coursera_dataset_path="data/coursera_courses_2024.csv"
    )
    
    # Build network
    graph = builder.build_network([
        "Python Programming",
        "Machine Learning",
        "Data Analysis"
    ])
    
    # Analyze influence propagation
    influence_results = builder.analyze_influence_propagation(
        seed_nodes=["Python Programming"],
        model="independent_cascade",
        num_simulations=200,
        activation_probability=0.1
    )
    
    print("Influence Propagation Results:")
    print(f"  Seed nodes: {influence_results['seed_nodes']}")
    print(f"  Final infected: {influence_results['final_infected_count']}")
    print(f"  Propagation steps: {influence_results['propagation_steps']}")
    
    # Visualize influence propagation
    builder.visualize_influence_propagation(
        influence_results,
        "influence_propagation.png"
    )

Analysis Examples
-----------------

Community Detection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    config = NetworkConfig(
        max_depth=3,
        max_articles_to_process=100,
        links_per_article=30,
        community_algorithm="louvain",
        resolution=1.2
    )
    
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network([
        "Computer Science",
        "Mathematics",
        "Physics",
        "Biology"
    ])
    
    # Detect communities
    communities = builder.detect_communities()
    
    print("Community Detection Results:")
    print(f"  Number of communities: {len(communities)}")
    
    # Print top communities by size
    community_sizes = [(i, len(community)) for i, community in enumerate(communities)]
    community_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for i, (comm_id, size) in enumerate(community_sizes[:5]):
        print(f"  Community {comm_id}: {size} nodes")
        # Show first few nodes in community
        sample_nodes = list(communities[comm_id])[:3]
        print(f"    Sample nodes: {', '.join(sample_nodes)}")
    
    # Visualize communities
    builder.visualize_communities("communities_large.png")

Centrality Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    config = NetworkConfig(
        max_depth=2,
        max_articles_to_process=60,
        centrality_measures=["degree", "pagerank", "betweenness", "closeness"]
    )
    
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Artificial Intelligence"])
    
    # Calculate all centrality measures
    stats = builder.analyze_network()
    
    # Compare centrality rankings
    measures = ["degree", "pagerank", "betweenness", "closeness"]
    
    print("Top 5 nodes by different centrality measures:")
    for measure in measures:
        print(f"\n{measure.capitalize()}:")
        centrality_data = stats[measure]
        top_nodes = sorted(centrality_data.items(), 
                          key=lambda x: x[1], reverse=True)[:5]
        
        for node, score in top_nodes:
            print(f"  {node}: {score:.4f}")

Network Comparison
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    def compare_networks(topics1, topics2, name1, name2):
        config = NetworkConfig(
            max_depth=2,
            max_articles_to_process=50,
            links_per_article=20
        )
        
        # Build two networks
        builder1 = WikipediaNetworkBuilder(config)
        graph1 = builder1.build_network(topics1)
        stats1 = builder1.analyze_network()
        
        builder2 = WikipediaNetworkBuilder(config)
        graph2 = builder2.build_network(topics2)
        stats2 = builder2.analyze_network()
        
        # Compare networks
        print(f"Network Comparison: {name1} vs {name2}")
        print(f"{'Metric':<20} {'Network 1':<15} {'Network 2':<15}")
        print("-" * 50)
        
        metrics = [
            ("Nodes", graph1.number_of_nodes(), graph2.number_of_nodes()),
            ("Edges", graph1.number_of_edges(), graph2.number_of_edges()),
            ("Density", stats1['density'], stats2['density']),
            ("Avg Clustering", stats1['average_clustering'], stats2['average_clustering']),
            ("Communities", len(stats1['communities']), len(stats2['communities']))
        ]
        
        for metric, val1, val2 in metrics:
            print(f"{metric:<20} {val1:<15} {val2:<15}")
        
        return graph1, graph2, stats1, stats2
    
    # Compare AI vs Biology networks
    ai_topics = ["Artificial Intelligence", "Machine Learning"]
    bio_topics = ["Biology", "Genetics", "Evolution"]
    
    g1, g2, s1, s2 = compare_networks(ai_topics, bio_topics, "AI", "Biology")

Visualization Examples
---------------------

Custom Visualization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    config = NetworkConfig(
        max_depth=2,
        max_articles_to_process=40,
        visualization_layout="force_atlas2",
        node_size_method="pagerank",
        color_scheme="community"
    )
    
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Data Science"])
    
    # Create custom visualization
    builder.visualize_pyvis(
        "custom_network.html",
        layout="force_atlas2",
        physics_enabled=True,
        node_size_method="pagerank",
        edge_smooth=True,
        show_buttons=True
    )

Static Network Plots
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    
    config = NetworkConfig(
        max_depth=2,
        max_articles_to_process=30
    )
    
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(["Machine Learning"])
    
    # Create static visualization using matplotlib
    stats = builder.analyze_network()
    
    # Plot degree distribution
    degrees = [d for n, d in graph.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=20, edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.savefig('degree_distribution.png')
    plt.close()
    
    # Plot centrality correlation
    pagerank = stats['pagerank']
    degree = stats['degree']
    
    # Get common nodes
    common_nodes = set(pagerank.keys()) & set(degree.keys())
    pr_values = [pagerank[node] for node in common_nodes]
    deg_values = [degree[node] for node in common_nodes]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(deg_values, pr_values, alpha=0.6)
    plt.xlabel('Degree Centrality')
    plt.ylabel('PageRank')
    plt.title('Centrality Correlation')
    plt.savefig('centrality_correlation.png')
    plt.close()

Performance Examples
--------------------

Batch Processing
~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    import time
    
    def batch_process_topics(topic_groups):
        config = NetworkConfig(
            max_depth=2,
            max_articles_to_process=40,
            async_enabled=True,
            max_workers=6
        )
        
        results = {}
        
        for group_name, topics in topic_groups.items():
            print(f"Processing {group_name}...")
            start_time = time.time()
            
            builder = WikipediaNetworkBuilder(config)
            graph = builder.build_network(topics)
            stats = builder.analyze_network()
            
            end_time = time.time()
            
            results[group_name] = {
                'graph': graph,
                'stats': stats,
                'processing_time': end_time - start_time
            }
            
            print(f"  Completed in {end_time - start_time:.2f} seconds")
            print(f"  Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return results
    
    # Define topic groups
    topic_groups = {
        'AI': ['Artificial Intelligence', 'Machine Learning'],
        'Science': ['Physics', 'Chemistry', 'Biology'],
        'Technology': ['Computer Science', 'Software Engineering'],
        'Mathematics': ['Mathematics', 'Statistics', 'Calculus']
    }
    
    # Process all groups
    results = batch_process_topics(topic_groups)
    
    # Compare results
    print("\nBatch Processing Summary:")
    for group_name, data in results.items():
        graph = data['graph']
        time_taken = data['processing_time']
        print(f"{group_name}: {graph.number_of_nodes()} nodes in {time_taken:.2f}s")

Memory Management
~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    import gc
    import psutil
    import os
    
    def memory_efficient_processing():
        config = NetworkConfig(
            max_depth=1,  # Reduced depth
            max_articles_to_process=20,  # Smaller batch size
            cache_enabled=False  # Disable cache to save memory
        )
        
        topics = ["Machine Learning", "Deep Learning", "Neural Networks"]
        
        for topic in topics:
            print(f"Processing {topic}...")
            
            # Check memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Build network
            builder = WikipediaNetworkBuilder(config)
            graph = builder.build_network([topic])
            
            # Quick analysis
            stats = builder.analyze_network()
            print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Save results
            builder.save_network(f"{topic.replace(' ', '_')}.graphml")
            
            # Check memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"  Memory used: {memory_after - memory_before:.2f} MB")
            
            # Clean up
            del builder, graph, stats
            gc.collect()
    
    memory_efficient_processing()

Error Handling Examples
----------------------

Robust Network Building
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from network_analyzer import NetworkConfig, WikipediaNetworkBuilder
    from network_analyzer.core.exceptions import NetworkAnalyzerError
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def robust_network_building(topics, max_retries=3):
        config = NetworkConfig(
            max_depth=2,
            max_articles_to_process=50,
            request_delay=0.2  # Slower requests to avoid rate limiting
        )
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} to build network for {topics}")
                
                builder = WikipediaNetworkBuilder(config)
                graph = builder.build_network(topics)
                
                # Validate the network
                if graph.number_of_nodes() == 0:
                    raise ValueError("Empty network generated")
                
                logger.info(f"Successfully built network with {graph.number_of_nodes()} nodes")
                return graph, builder
                
            except NetworkAnalyzerError as e:
                logger.error(f"Network Analyzer error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        return None, None
    
    # Use robust building
    try:
        graph, builder = robust_network_building(["Machine Learning"])
        if graph:
            stats = builder.analyze_network()
            builder.visualize_pyvis("robust_network.html")
    except Exception as e:
        logger.error(f"Failed to build network after all retries: {e}")

This comprehensive set of examples demonstrates the versatility and power of Network Analyzer across different use cases and scenarios.