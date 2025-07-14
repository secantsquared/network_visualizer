"""
Unified main interface that supports both Wikipedia and Coursera/Kaggle data sources.
"""

import requests_cache
from pathlib import Path
from datetime import datetime

from network_analyzer.core.config import NetworkConfig
from network_analyzer.core.unified_network_builder import UnifiedNetworkBuilder

# Configure caching
requests_cache.install_cache(
    "unified_cache",
    backend="sqlite",
    expire_after=86400,
    allowable_methods=["GET"],
    allowable_codes=[200],
)


def interactive_cli():
    """
    Interactive command-line interface for unified network generation.
    """
    print("=" * 60)
    print("UNIFIED NETWORK BUILDER - INTERACTIVE CLI")
    print("=" * 60)

    # Data source selection
    print("\nSelect a data source:")
    print("  1. Wikipedia (Article networks)")
    print("  2. Coursera (Course/skill networks)")
    print("  3. Hybrid (Both sources available)")

    while True:
        source_choice = input("\nEnter your choice (1-3): ").strip()
        if source_choice in ["1", "2", "3"]:
            break
        print("Invalid choice. Please enter 1-3.")

    # Configure data source
    config = NetworkConfig()
    builder_kwargs = {}

    if source_choice == "1":
        config.data_source_type = "wikipedia"
        config.primary_data_source = "wikipedia"
        print("\n✓ Wikipedia data source selected")
        
    elif source_choice == "2":
        config.data_source_type = "coursera"
        config.primary_data_source = "coursera"
        
        # Get dataset path
        default_path = "./coursera_courses_2024.csv"
        dataset_path = input(f"\nEnter Coursera dataset path [{default_path}]: ").strip()
        if not dataset_path:
            dataset_path = default_path
        
        # Check if file exists
        if not Path(dataset_path).exists():
            print(f"⚠️  Warning: Dataset file not found at {dataset_path}")
            print("Please download the dataset from:")
            print("https://www.kaggle.com/datasets/azraimohamad/coursera-course-data")
            return
        
        config.coursera_dataset_path = dataset_path
        builder_kwargs['coursera_dataset_path'] = dataset_path
        print(f"✓ Coursera data source selected: {dataset_path}")
        
    else:  # Hybrid
        config.data_source_type = "hybrid"
        
        # Get dataset path for Coursera
        default_path = "./coursera_courses_2024.csv"
        dataset_path = input(f"\nEnter Coursera dataset path [{default_path}]: ").strip()
        if not dataset_path:
            dataset_path = default_path
        
        if Path(dataset_path).exists():
            config.coursera_dataset_path = dataset_path
            builder_kwargs['coursera_dataset_path'] = dataset_path
            print(f"✓ Coursera dataset loaded: {dataset_path}")
        else:
            print(f"⚠️  Warning: Coursera dataset not found at {dataset_path}")
            print("Only Wikipedia source will be available in hybrid mode")
        
        # Choose primary source
        print("\nChoose primary data source for hybrid mode:")
        print("  1. Wikipedia (default)")
        print("  2. Coursera")
        
        primary_choice = input("Enter choice (1-2) [1]: ").strip()
        if primary_choice == "2" and config.coursera_dataset_path:
            config.primary_data_source = "coursera"
            print("✓ Primary source: Coursera")
        else:
            config.primary_data_source = "wikipedia"
            print("✓ Primary source: Wikipedia")

    # Method selection
    methods = {
        "1": ("breadth_first", "Breadth-First Search (Level-by-level expansion)"),
        "2": (
            "breadth_first_async",
            "Async Breadth-First Search (High-performance parallel)",
        ),
        "3": ("random_walk", "Random Walk (Stochastic exploration)"),
        "4": ("dfs", "Depth-First Search (Deep path exploration)"),
        "5": ("topic_focused", "Topic-Focused Crawling (Similarity-based)"),
        "6": ("hub_and_spoke", "Hub-and-Spoke (Hub identification + expansion)"),
    }

    print("\nSelect a network generation method:")
    for key, (method, description) in methods.items():
        print(f"  {key}. {description}")

    while True:
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice in methods:
            selected_method, method_description = methods[choice]
            print(f"\nSelected: {method_description}")
            break
        print("Invalid choice. Please enter 1-6.")

    # Seed input based on data source
    print("\n" + "-" * 40)
    if config.data_source_type == "coursera" or config.primary_data_source == "coursera":
        print("SEED COURSES/SKILLS")
        print("-" * 40)
        default_seeds = ["Machine Learning", "Python Programming"]
        seeds_input = input(
            f"Enter seed courses/skills (comma-separated) [{', '.join(default_seeds)}]: "
        ).strip()
    else:
        print("SEED ARTICLES")
        print("-" * 40)
        default_seeds = ["Data Science", "Machine Learning"]
        seeds_input = input(
            f"Enter seed articles (comma-separated) [{', '.join(default_seeds)}]: "
        ).strip()

    if seeds_input:
        seeds = [seed.strip() for seed in seeds_input.split(",")]
    else:
        seeds = default_seeds

    print(f"Using seeds: {seeds}")

    # Common configuration
    print(f"\n" + "-" * 40)
    print(f"CONFIGURATION FOR {method_description.upper()}")
    print("-" * 40)

    # Common parameters
    max_articles = input(
        f"Max items to process [{config.max_articles_to_process}]: "
    ).strip()
    if max_articles.isdigit():
        config.max_articles_to_process = int(max_articles)

    links_per_article = input(
        f"Links per item [{config.links_per_article}]: "
    ).strip()
    if links_per_article.isdigit():
        config.links_per_article = int(links_per_article)

    # Method-specific parameters
    if selected_method == "breadth_first":
        max_depth = input(f"Max depth [{config.max_depth}]: ").strip()
        if max_depth.isdigit():
            config.max_depth = int(max_depth)

    elif selected_method == "breadth_first_async":
        max_depth = input(f"Max depth [{config.max_depth}]: ").strip()
        if max_depth.isdigit():
            config.max_depth = int(max_depth)

        max_concurrent = input(
            f"Max concurrent requests [{config.max_concurrent_requests}]: "
        ).strip()
        if max_concurrent.isdigit():
            config.max_concurrent_requests = int(max_concurrent)

    elif selected_method == "random_walk":
        random_walk_steps = input(
            f"Random walk steps [{config.random_walk_steps}]: "
        ).strip()
        if random_walk_steps.isdigit():
            config.random_walk_steps = int(random_walk_steps)

        restart_probability = input(
            f"Restart probability [{config.restart_probability}]: "
        ).strip()
        if restart_probability.replace(".", "").isdigit():
            config.restart_probability = float(restart_probability)

        exploration_bias = input(
            f"Exploration bias [{config.exploration_bias}]: "
        ).strip()
        if exploration_bias.replace(".", "").isdigit():
            config.exploration_bias = float(exploration_bias)

    elif selected_method == "dfs":
        dfs_max_branch_depth = input(
            f"Max branch depth [{config.dfs_max_branch_depth}]: "
        ).strip()
        if dfs_max_branch_depth.isdigit():
            config.dfs_max_branch_depth = int(dfs_max_branch_depth)

        dfs_branches_per_node = input(
            f"Branches per node [{config.dfs_branches_per_node}]: "
        ).strip()
        if dfs_branches_per_node.isdigit():
            config.dfs_branches_per_node = int(dfs_branches_per_node)

        dfs_backtrack_probability = input(
            f"Backtrack probability [{config.dfs_backtrack_probability}]: "
        ).strip()
        if dfs_backtrack_probability.replace(".", "").isdigit():
            config.dfs_backtrack_probability = float(dfs_backtrack_probability)

    elif selected_method == "topic_focused":
        # Topic keywords
        print("Topic keywords help focus crawling on specific topics.")
        keywords_input = input(
            f"Topic keywords (comma-separated) [{', '.join(config.topic_keywords or [])}]: "
        ).strip()
        if keywords_input:
            config.topic_keywords = [kw.strip() for kw in keywords_input.split(",")]

        topic_similarity_threshold = input(
            f"Similarity threshold [{config.topic_similarity_threshold}]: "
        ).strip()
        if topic_similarity_threshold.replace(".", "").isdigit():
            config.topic_similarity_threshold = float(topic_similarity_threshold)

        topic_diversity_weight = input(
            f"Diversity weight [{config.topic_diversity_weight}]: "
        ).strip()
        if topic_diversity_weight.replace(".", "").isdigit():
            config.topic_diversity_weight = float(topic_diversity_weight)

    elif selected_method == "hub_and_spoke":
        # Hub selection method
        print("Hub selection methods:")
        print("  1. degree - Select by node degree")
        print("  2. pagerank - Select by PageRank score")
        print("  3. betweenness - Select by betweenness centrality")
        
        hub_methods = {"1": "degree", "2": "pagerank", "3": "betweenness"}
        current_method = {"degree": "1", "pagerank": "2", "betweenness": "3"}.get(
            config.hub_selection_method, "1"
        )
        
        hub_choice = input(f"Hub selection method [{current_method}]: ").strip()
        if hub_choice in hub_methods:
            config.hub_selection_method = hub_methods[hub_choice]

        spokes_per_hub = input(
            f"Spokes per hub [{config.spokes_per_hub}]: "
        ).strip()
        if spokes_per_hub.isdigit():
            config.spokes_per_hub = int(spokes_per_hub)

        hub_depth_limit = input(
            f"Hub depth limit [{config.hub_depth_limit}]: "
        ).strip()
        if hub_depth_limit.isdigit():
            config.hub_depth_limit = int(hub_depth_limit)

    # Create output directory structure at the beginning
    output_base = Path("outputs/history")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Find the next available run number
    run_number = 1
    while True:
        run_dir = output_base / f"run{run_number}"
        if not run_dir.exists():
            break
        run_number += 1
    
    # Create the run directory
    run_dir.mkdir(exist_ok=True)
    print(f"Output directory created: {run_dir}")

    # Build the network
    print(f"\n" + "=" * 60)
    print("BUILDING NETWORK")
    print("=" * 60)

    try:
        builder = UnifiedNetworkBuilder(config, **builder_kwargs)
        
        # Show source information
        source_info = builder.get_source_info()
        print(f"Using data source: {source_info['type']}")
        if len(source_info['available_sources']) > 1:
            print(f"Available sources: {', '.join(source_info['available_sources'])}")
        
        # Build network
        builder.build_network(seeds, method=selected_method)

        # Analyze and visualize
        stats = builder.analyze_network()
        builder.print_analysis(stats)

        # Show source-specific information
        if config.data_source_type == "coursera":
            print_coursera_specific_info(builder, seeds)

        # Ask about influence propagation analysis
        print(f"\n" + "-" * 40)
        print("INFLUENCE PROPAGATION ANALYSIS")
        print("-" * 40)
        
        run_influence = input("Analyze influence propagation? (y/n) [n]: ").strip().lower()
        if run_influence == "y":
            print("\nSelect influence propagation model:")
            print("  1. Independent Cascade Model (ICM)")
            print("  2. Linear Threshold Model (LTM)")
            
            model_choice = input("Enter choice (1-2) [1]: ").strip()
            model_name = "linear_threshold" if model_choice == "2" else "independent_cascade"
            
            num_sims = input("Number of simulations [100]: ").strip()
            num_simulations = int(num_sims) if num_sims.isdigit() else 100
            
            # Add activation probability input
            act_prob = input("Activation probability (0.0-1.0) [0.15]: ").strip()
            try:
                activation_probability = float(act_prob) if act_prob else 0.15
                activation_probability = max(0.0, min(1.0, activation_probability))  # Clamp to valid range
            except ValueError:
                activation_probability = 0.15
                print("Invalid input, using default activation probability of 0.15")
            
            # Add edge weight method selection
            print("\nEdge weight method:")
            print("  1. uniform - All edges have equal weight")
            print("  2. random - Random weights (0.05-0.3)")
            print("  3. degree_based - Inverse of target degree")
            edge_method = input("Enter choice (1-3) [2]: ").strip()
            if edge_method == "1":
                edge_weight_method = "uniform"
            elif edge_method == "3":
                edge_weight_method = "degree_based"
            else:
                edge_weight_method = "random"  # Default to random for better spread
            
            print(f"\nRunning influence propagation analysis with {model_name} model...")
            try:
                influence_results = builder.analyze_influence_propagation(
                    seed_nodes=seeds,
                    model=model_name,
                    num_simulations=num_simulations,
                    activation_probability=activation_probability,
                    edge_weight_method=edge_weight_method
                )
                
                if influence_results:
                    print("\n" + "=" * 50)
                    print("INFLUENCE PROPAGATION RESULTS")
                    print("=" * 50)
                    
                    # Show selected seeds results
                    selected = influence_results['selected_seeds']
                    print(f"\nSelected Seeds ({', '.join(selected['nodes'])}):")
                    print(f"  Mean influence: {selected['mean_influence']:.3f}")
                    print(f"  Activation rate: {selected['activation_rate']:.2%}")
                    print(f"  Most influenced nodes: {', '.join([node for node, _ in selected['most_influenced'][:5]])}")
                    
                    # Show optimal seeds results
                    optimal = influence_results['optimal_seeds']
                    print(f"\nOptimal Seeds ({', '.join(optimal['nodes'])}):")
                    print(f"  Mean influence: {optimal['mean_influence']:.3f}")
                    print(f"  Activation rate: {optimal['activation_rate']:.2%}")
                    print(f"  Most influenced nodes: {', '.join([node for node, _ in optimal['most_influenced'][:5]])}")
                    
                    # Show strategy comparison
                    print(f"\nStrategy Comparison:")
                    for strategy, results in influence_results['strategy_comparison'].items():
                        if 'error' not in results:
                            print(f"  {strategy}: {results['mean_influence']:.3f} influence")
                    
                    # Show vulnerability analysis
                    print(f"\nNetwork Vulnerability Analysis:")
                    for attack_size, vuln_data in influence_results['vulnerability_analysis'].items():
                        print(f"  {attack_size} attackers: {vuln_data['activation_rate']:.2%} activation rate")
                    
                    # Create influence propagation visualization
                    create_influence_viz = input("\nCreate influence propagation visualization? (y/n) [y]: ").strip().lower()
                    if create_influence_viz != "n":
                        print("Creating influence propagation visualization...")
                        influence_path = str(run_dir / "influence_propagation.png")
                        builder.visualize_influence_propagation(
                            seeds=optimal['nodes'],
                            model=model_name,
                            output_path=influence_path,
                            activation_probability=activation_probability,
                            edge_weight_method=edge_weight_method
                        )
                        print(f"✓ Influence propagation visualization created: {influence_path}")
                
            except Exception as e:
                print(f"❌ Error running influence propagation analysis: {e}")

        # Ask about visualizations
        print(f"\n" + "-" * 40)
        print("VISUALIZATION OPTIONS")
        print("-" * 40)

        create_viz = input("Create visualizations? (y/n) [y]: ").strip().lower()
        if create_viz != "n":
            # Physics engine selection
            print("\nSelect physics engine for visualization:")
            print("  1. Barnes-Hut (fast, good for large networks)")
            print("  2. ForceAtlas2 (community-focused)")
            print("  3. Hierarchical (tree-like structure)")
            print("  4. Circular (circular arrangement)")
            print("  5. Organic (natural layout)")
            print("  6. Centrality-Based (optimized for centrality node sizing)")
            
            physics_choice = input("Enter choice (1-6) [1]: ").strip()
            physics_engines = {
                "1": "barnes_hut",
                "2": "force_atlas2", 
                "3": "hierarchical",
                "4": "circular",
                "5": "organic",
                "6": "centrality"
            }
            
            selected_physics = physics_engines.get(physics_choice, "barnes_hut")
            config.physics_engine = selected_physics
            
            # Node sizing selection
            print("\nSelect node sizing scheme:")
            print("  1. Degree centrality (number of connections)")
            print("  2. Betweenness centrality (bridge nodes)")
            print("  3. PageRank centrality (importance)")
            print("  4. Closeness centrality (shortest paths)")
            print("  5. Eigenvector centrality (connected to important nodes)")
            
            size_choice = input("Enter choice (1-5) [1]: ").strip()
            size_schemes = {
                "1": "degree",
                "2": "betweenness",
                "3": "pagerank",
                "4": "closeness",
                "5": "eigenvector"
            }
            
            selected_size_scheme = size_schemes.get(size_choice, "degree")
            config.size_by = selected_size_scheme
            
            # Optional: Custom physics parameters
            custom_physics = input("Use custom physics parameters? (y/n) [n]: ").strip().lower()
            if custom_physics == "y":
                print("Enter custom physics parameters (or press Enter for defaults):")
                gravity = input("Gravity [-80000]: ").strip()
                spring_length = input("Spring length [200]: ").strip()
                damping = input("Damping [0.09]: ").strip()
                
                custom_params = {}
                if gravity: custom_params["gravity"] = int(gravity)
                if spring_length: custom_params["spring_length"] = int(spring_length)
                if damping: custom_params["damping"] = float(damping)
                
                config.custom_physics_params = custom_params if custom_params else None
            
            print("Creating visualizations...")
            
            # Create visualizations directly in the output directory
            builder.visualize_pyvis(
                str(run_dir / "unified_network_depth.html"), 
                physics=True, 
                color_by="depth",
                size_by=config.size_by,
                physics_engine=config.physics_engine,
                custom_physics_params=config.custom_physics_params
            )
            builder.visualize_pyvis(
                str(run_dir / "unified_network_communities.html"), 
                physics=True, 
                color_by="community",
                size_by=config.size_by,
                physics_engine=config.physics_engine,
                custom_physics_params=config.custom_physics_params
            )
            builder.visualize_communities_matplotlib(str(run_dir / "unified_communities.png"))
            builder.save_graph(str(run_dir / "unified_network.graphml"))

            physics_name = {
                "barnes_hut": "Barnes-Hut",
                "force_atlas2": "ForceAtlas2", 
                "hierarchical": "Hierarchical",
                "circular": "Circular",
                "organic": "Organic"
            }.get(config.physics_engine, "Barnes-Hut")

            print(f"\nNetwork outputs created using {physics_name} physics in: {run_dir}")
            print("  - unified_network_depth.html (interactive - colored by depth)")
            print("  - unified_network_communities.html (interactive - colored by community)")
            print("  - unified_communities.png (static community plot)")
            print("  - unified_network.graphml (network data)")

        # Create final metadata file with all outputs
        metadata_path = run_dir / "run_info.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Find all files created in the run directory
        created_files = []
        for file_path in run_dir.glob("*"):
            if file_path.is_file() and file_path.name != "run_info.txt":
                created_files.append(file_path.name)
        
        with open(metadata_path, "w") as f:
            f.write(f"Run: {run_number}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Method: {selected_method}\n")
            if 'config' in locals() and hasattr(config, 'physics_engine'):
                f.write(f"Physics Engine: {config.physics_engine}\n")
                f.write(f"Node Sizing: {config.size_by}\n")
            f.write(f"Seeds: {', '.join(seeds)}\n")
            f.write(f"Data Source: {config.data_source_type}\n")
            f.write(f"Files created: {len(created_files)}\n")
            f.write(f"Files:\n")
            for file in sorted(created_files):
                f.write(f"  - {file}\n")

        print(f"\n" + "=" * 60)
        print("NETWORK BUILDING COMPLETE!")
        print("=" * 60)
        print(f"All outputs saved to: {run_dir}")
        print(f"Files created: {', '.join(sorted(created_files))}")

        return builder

    except Exception as e:
        print(f"\n❌ Error building network: {e}")
        print("Please check your configuration and try again.")
        return None


def print_coursera_specific_info(builder, seeds):
    """Print Coursera-specific information if available."""
    if not hasattr(builder.data_source, 'get_courses_by_skill'):
        return
    
    print(f"\n" + "-" * 40)
    print("COURSERA-SPECIFIC INFORMATION")
    print("-" * 40)
    
    # Show courses for each seed skill
    for seed in seeds:
        courses = builder.get_courses_by_skill(seed)
        if courses:
            print(f"\nCourses teaching '{seed}':")
            for course in courses[:5]:  # Show top 5
                metadata = builder.get_item_metadata(course)
                rating = metadata.get('rating', 'N/A')
                difficulty = metadata.get('difficulty', 'Unknown')
                print(f"  - {course} (Rating: {rating}, Difficulty: {difficulty})")
            
            if len(courses) > 5:
                print(f"  ... and {len(courses) - 5} more courses")


def demo_data_source_switching():
    """Demonstrate switching between data sources in hybrid mode."""
    print("\n" + "=" * 60)
    print("DATA SOURCE SWITCHING DEMO")
    print("=" * 60)
    
    # Create hybrid configuration
    config = NetworkConfig()
    config.data_source_type = "hybrid"
    config.primary_data_source = "wikipedia"
    config.coursera_dataset_path = "./coursera_courses_2024.csv"
    
    try:
        builder = UnifiedNetworkBuilder(config, 
                                       coursera_dataset_path="./coursera_courses_2024.csv")
        
        print("Available sources:", builder.get_available_sources())
        
        # Build with Wikipedia
        print("\n1. Building with Wikipedia data...")
        builder.switch_data_source("wikipedia")
        wiki_graph = builder.build_network(["Machine Learning"], method="breadth_first")
        print(f"Wikipedia network: {wiki_graph.number_of_nodes()} nodes, {wiki_graph.number_of_edges()} edges")
        
        # Switch to Coursera
        print("\n2. Switching to Coursera data...")
        builder.switch_data_source("coursera")
        course_graph = builder.build_network(["Machine Learning"], method="breadth_first")
        print(f"Coursera network: {course_graph.number_of_nodes()} nodes, {course_graph.number_of_edges()} edges")
        
        return builder
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure you have the Coursera dataset available.")
        return None


def main():
    """Main function - runs the interactive CLI."""
    try:
        interactive_cli()
    except KeyboardInterrupt:
        print("\n\nBuild interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()