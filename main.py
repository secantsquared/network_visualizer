import requests_cache

from config import NetworkConfig
from network_builder import WikipediaNetworkBuilder

# Configure caching
requests_cache.install_cache(
    "wiki_cache",
    backend="sqlite",
    expire_after=86400,
    allowable_methods=["GET"],
    allowable_codes=[200],
)


def interactive_cli():
    """
    Interactive command-line interface for network generation.
    """
    print("=" * 60)
    print("WIKIPEDIA NETWORK BUILDER - INTERACTIVE CLI")
    print("=" * 60)

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

    # Seed articles input
    print("\n" + "-" * 40)
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

    # Create base configuration
    config = NetworkConfig()

    # Method-specific parameter configuration
    print(f"\n" + "-" * 40)
    print(f"CONFIGURATION FOR {method_description.upper()}")
    print("-" * 40)

    # Common parameters
    max_articles = input(
        f"Max articles to process [{config.max_articles_to_process}]: "
    ).strip()
    if max_articles.isdigit():
        config.max_articles_to_process = int(max_articles)

    links_per_article = input(
        f"Links per article [{config.links_per_article}]: "
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

        pool_size = input(
            f"Connection pool size [{config.connection_pool_size}]: "
        ).strip()
        if pool_size.isdigit():
            config.connection_pool_size = int(pool_size)

    elif selected_method == "random_walk":
        walk_steps = input(f"Random walk steps [{config.random_walk_steps}]: ").strip()
        if walk_steps.isdigit():
            config.random_walk_steps = int(walk_steps)

        restart_prob = input(
            f"Restart probability [{config.restart_probability}]: "
        ).strip()
        try:
            config.restart_probability = float(restart_prob)
        except ValueError:
            pass

        exploration_bias = input(
            f"Exploration bias [{config.exploration_bias}]: "
        ).strip()
        try:
            config.exploration_bias = float(exploration_bias)
        except ValueError:
            pass

    elif selected_method == "dfs":
        branch_depth = input(
            f"Max branch depth [{config.dfs_max_branch_depth}]: "
        ).strip()
        if branch_depth.isdigit():
            config.dfs_max_branch_depth = int(branch_depth)

        branches_per_node = input(
            f"Branches per node [{config.dfs_branches_per_node}]: "
        ).strip()
        if branches_per_node.isdigit():
            config.dfs_branches_per_node = int(branches_per_node)

        backtrack_prob = input(
            f"Backtrack probability [{config.dfs_backtrack_probability}]: "
        ).strip()
        try:
            config.dfs_backtrack_probability = float(backtrack_prob)
        except ValueError:
            pass

    elif selected_method == "topic_focused":
        keywords_input = input(
            "Topic keywords (comma-separated) [extracted from seeds]: "
        ).strip()
        if keywords_input:
            config.topic_keywords = [kw.strip() for kw in keywords_input.split(",")]

        similarity_threshold = input(
            f"Similarity threshold [{config.topic_similarity_threshold}]: "
        ).strip()
        try:
            config.topic_similarity_threshold = float(similarity_threshold)
        except ValueError:
            pass

        diversity_weight = input(
            f"Diversity weight [{config.topic_diversity_weight}]: "
        ).strip()
        try:
            config.topic_diversity_weight = float(diversity_weight)
        except ValueError:
            pass

    elif selected_method == "hub_and_spoke":
        hub_methods = {"1": "degree", "2": "pagerank", "3": "betweenness"}
        print("Hub selection method:")
        for key, method in hub_methods.items():
            print(f"  {key}. {method}")

        hub_choice = input(f"Choose hub method (1-3) [1-degree]: ").strip()
        if hub_choice in hub_methods:
            config.hub_selection_method = hub_methods[hub_choice]

        spokes_per_hub = input(f"Spokes per hub [{config.spokes_per_hub}]: ").strip()
        if spokes_per_hub.isdigit():
            config.spokes_per_hub = int(spokes_per_hub)

        hub_depth_limit = input(f"Hub depth limit [{config.hub_depth_limit}]: ").strip()
        if hub_depth_limit.isdigit():
            config.hub_depth_limit = int(hub_depth_limit)

    # Build the network
    print(f"\n" + "=" * 60)
    print("BUILDING NETWORK")
    print("=" * 60)

    builder = WikipediaNetworkBuilder(config)
    builder.build_network(seeds, method=selected_method)

    # Analyze and visualize
    stats = builder.analyze_network()
    builder.print_analysis(stats)

    # Ask about visualizations
    print(f"\n" + "-" * 40)
    print("VISUALIZATION OPTIONS")
    print("-" * 40)

    create_viz = input("Create visualizations? (y/n) [y]: ").strip().lower()
    if create_viz != "n":
        print("Creating visualizations...")

        # Create all visualizations
        builder.visualize_pyvis(
            "wiki_network_depth.html", physics=True, color_by="depth"
        )
        builder.visualize_pyvis(
            "wiki_network_communities.html", physics=True, color_by="community"
        )
        builder.visualize_communities_matplotlib("communities.png")
        builder.save_graph("wiki_network.graphml")

        # Save to history
        archived_files, run_dir = builder.save_to_history()

        print("\nNetwork outputs created:")
        print("  - wiki_network_depth.html (interactive - colored by depth)")
        print("  - wiki_network_communities.html (interactive - colored by community)")
        print("  - communities.png (static community plot)")
        print("  - wiki_network.graphml (network data)")

        if archived_files:
            print(f"\nHistory saved to: {run_dir}")
            print(f"  Contains {len(archived_files)} files and run metadata")

    print(f"\n" + "=" * 60)
    print("NETWORK BUILDING COMPLETE!")
    print("=" * 60)

    return builder


def main():
    """
    Main function - runs the interactive CLI by default.
    To run the old demo version, use main_demo() instead.
    """
    interactive_cli()


if __name__ == "__main__":
    main()
