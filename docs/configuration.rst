Configuration
=============

Network Analyzer uses the ``NetworkConfig`` class to configure all aspects of network building and analysis.

Basic Configuration
-------------------

.. code-block:: python

    from network_analyzer import NetworkConfig

    config = NetworkConfig(
        max_depth=2,
        max_articles_to_process=50,
        links_per_article=20
    )

Core Parameters
---------------

Network Size and Depth
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Default
     - Description
   * - ``max_depth``
     - 2
     - Maximum depth to explore from seed nodes
   * - ``max_articles_to_process``
     - 50
     - Maximum number of articles to process
   * - ``links_per_article``
     - 20
     - Number of links to follow per article

Exploration Method
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Default
     - Description
   * - ``method``
     - "breadth_first"
     - Exploration algorithm to use
   * - ``seed_selection_strategy``
     - "user_defined"
     - How to select initial seed nodes

Available exploration methods:

* **breadth_first**: Level-by-level exploration
* **random_walk**: Stochastic exploration with restart
* **depth_first**: Deep exploration with backtracking
* **topic_focused**: Keyword similarity-based exploration
* **hub_and_spoke**: Expand around important nodes

Performance Settings
--------------------

Concurrency and Threading
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        async_enabled=True,              # Enable async processing
        max_workers=8,                   # Number of worker threads
        max_concurrent_requests=20,      # Concurrent API requests
        request_delay=0.1               # Delay between requests (seconds)
    )

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Default
     - Description
   * - ``async_enabled``
     - True
     - Enable asynchronous processing
   * - ``max_workers``
     - 4
     - Number of worker threads
   * - ``max_concurrent_requests``
     - 10
     - Maximum concurrent API requests
   * - ``request_delay``
     - 0.1
     - Delay between requests (seconds)

Caching
~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        cache_enabled=True,
        cache_expire_after=3600,        # Cache expiry in seconds
        cache_backend="sqlite"          # Cache backend type
    )

Algorithm-Specific Settings
---------------------------

Random Walk Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        method="random_walk",
        random_walk_steps=100,           # Number of steps per walk
        restart_probability=0.15,       # Probability of restart
        num_walks=10                    # Number of walks to perform
    )

Topic-Focused Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        method="topic_focused",
        topic_similarity_threshold=0.3,  # Minimum similarity score
        focus_keywords=["machine learning", "AI"],  # Focus keywords
        similarity_method="cosine"       # Similarity calculation method
    )

Hub and Spoke Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        method="hub_and_spoke",
        hub_threshold=0.8,              # Threshold for hub identification
        spoke_expansion_factor=2.0,     # How much to expand around hubs
        centrality_measure="pagerank"   # Centrality measure for hub detection
    )

Data Source Configuration
------------------------

Wikipedia Settings
~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        data_source_type="wikipedia",
        wikipedia_language="en",         # Wikipedia language
        exclude_categories=[             # Categories to exclude
            "disambiguation",
            "list",
            "portal"
        ],
        min_article_length=500          # Minimum article length
    )

Coursera Settings
~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        data_source_type="coursera",
        coursera_dataset_path="data/coursera_courses_2024.csv",
        skill_similarity_threshold=0.4,  # Skill similarity threshold
        include_prerequisites=True,      # Include prerequisite relationships
        course_rating_threshold=4.0     # Minimum course rating
    )

Hybrid Mode
~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        data_source_type="hybrid",
        primary_data_source="wikipedia",
        coursera_dataset_path="data/coursera_courses_2024.csv",
        cross_source_similarity_threshold=0.3,  # Cross-source similarity
        source_weight_wikipedia=0.7,    # Weight for Wikipedia sources
        source_weight_coursera=0.3      # Weight for Coursera sources
    )

Filtering and Quality Control
-----------------------------

Content Filtering
~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        filter_enabled=True,
        min_article_length=500,         # Minimum article length
        max_article_length=100000,      # Maximum article length
        exclude_patterns=[              # Patterns to exclude
            r"^List of",
            r"^Category:",
            r"disambig"
        ]
    )

Link Quality
~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        min_link_frequency=2,           # Minimum link frequency
        exclude_external_links=True,    # Exclude external links
        link_weight_threshold=0.1       # Minimum link weight
    )

Output Configuration
--------------------

Visualization Settings
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        output_dir="outputs",           # Output directory
        visualization_layout="barnes_hut",  # Layout algorithm
        node_size_method="degree",      # Node sizing method
        color_scheme="depth",           # Node coloring scheme
        show_labels=True,              # Show node labels
        physics_enabled=True           # Enable physics simulation
    )

Available layouts:

* **barnes_hut**: Fast force-directed layout
* **force_atlas2**: Force Atlas 2 algorithm
* **hierarchical**: Hierarchical layout
* **circular**: Circular layout
* **organic**: Organic layout

File Output
~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        save_graphml=True,             # Save GraphML file
        save_gexf=False,               # Save GEXF file
        save_edgelist=False,           # Save edge list
        save_statistics=True,          # Save statistics file
        output_format="html"           # Primary output format
    )

Analysis Settings
-----------------

Community Detection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        community_algorithm="louvain",  # Community detection algorithm
        resolution=1.0,                # Resolution parameter
        min_community_size=3           # Minimum community size
    )

Available algorithms:

* **louvain**: Louvain algorithm
* **leiden**: Leiden algorithm
* **modularity**: Modularity optimization
* **infomap**: Infomap algorithm

Centrality Measures
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        centrality_measures=[           # Centrality measures to calculate
            "degree",
            "pagerank",
            "betweenness",
            "closeness",
            "eigenvector"
        ],
        pagerank_alpha=0.85,           # PageRank damping factor
        betweenness_normalized=True    # Normalize betweenness centrality
    )

Configuration Examples
---------------------

Small, Fast Network
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        max_depth=1,
        max_articles_to_process=20,
        links_per_article=10,
        async_enabled=True,
        max_workers=2
    )

Large, Comprehensive Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        max_depth=3,
        max_articles_to_process=200,
        links_per_article=30,
        async_enabled=True,
        max_workers=8,
        max_concurrent_requests=30,
        cache_enabled=True
    )

Research-Focused Network
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    config = NetworkConfig(
        method="topic_focused",
        topic_similarity_threshold=0.4,
        max_depth=2,
        max_articles_to_process=100,
        centrality_measures=["degree", "pagerank", "betweenness"],
        community_algorithm="leiden",
        save_graphml=True,
        save_statistics=True
    )

Configuration Validation
------------------------

The configuration is automatically validated when creating a ``NetworkConfig`` instance:

.. code-block:: python

    try:
        config = NetworkConfig(
            max_depth=-1,  # Invalid: negative depth
            max_articles_to_process=0  # Invalid: zero articles
        )
    except ValueError as e:
        print(f"Configuration error: {e}")

Environment Variables
--------------------

Some settings can be configured via environment variables:

.. code-block:: bash

    export NETWORK_ANALYZER_CACHE_DIR=/tmp/cache
    export NETWORK_ANALYZER_OUTPUT_DIR=/path/to/outputs
    export NETWORK_ANALYZER_LOG_LEVEL=INFO

Configuration Files
------------------

You can also load configuration from YAML files:

.. code-block:: python

    config = NetworkConfig.from_yaml("config.yaml")

Example ``config.yaml``:

.. code-block:: yaml

    max_depth: 2
    max_articles_to_process: 50
    links_per_article: 20
    method: breadth_first
    async_enabled: true
    max_workers: 4
    output_dir: outputs
    visualization_layout: barnes_hut

Best Practices
--------------

1. **Start Small**: Begin with small networks (depth=1, few articles) to understand the behavior
2. **Use Async**: Enable async processing for better performance with larger networks
3. **Cache Results**: Enable caching to avoid repeated API calls
4. **Monitor Resources**: Adjust ``max_workers`` based on your system capabilities
5. **Validate Configuration**: Always validate your configuration before running large jobs

Next Steps
----------

* Learn about the :doc:`api` documentation
* Explore :doc:`examples` for different use cases
* Check out the :doc:`quickstart` guide for basic usage