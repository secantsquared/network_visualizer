"""
Temporal Network Analysis CLI

Interactive command-line interface for temporal network analysis with
dynamic visualization capabilities.
"""

import requests_cache
from datetime import datetime

from network_analyzer.core.config import NetworkConfig
from network_analyzer.core.temporal_network_builder import TemporalNetworkBuilder
from network_analyzer.analysis.temporal_evolution import TemporalNetworkAnalyzer

# Configure caching
requests_cache.install_cache(
    "temporal_wiki_cache",
    backend="sqlite",
    expire_after=86400,
    allowable_methods=["GET"],
    allowable_codes=[200],
)


def interactive_temporal_cli():
    """
    Interactive CLI for temporal network analysis.
    """
    print("=" * 70)
    print("TEMPORAL NETWORK ANALYZER - INTERACTIVE CLI")
    print("=" * 70)
    print("Build networks over time and analyze their evolution")
    print()

    # Method selection
    methods = {
        "1": ("breadth_first", "Breadth-First Search (Level-by-level expansion)"),
        "2": ("breadth_first_async", "Async Breadth-First Search (High-performance parallel)"),
        "3": ("random_walk", "Random Walk (Stochastic exploration)"),
        "4": ("dfs", "Depth-First Search (Deep path exploration)"),
        "5": ("topic_focused", "Topic-Focused Crawling (Similarity-based)"),
        "6": ("hub_and_spoke", "Hub-and-Spoke (Hub identification + expansion)"),
    }

    print("Select a network generation method:")
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
    print("\n" + "-" * 50)
    print("SEED ARTICLES")
    print("-" * 50)

    default_seeds = ["Machine Learning", "Data Science", "Artificial Intelligence"]
    seeds_input = input(
        f"Enter seed articles (comma-separated) [{', '.join(default_seeds)}]: "
    ).strip()

    if seeds_input:
        seeds = [seed.strip() for seed in seeds_input.split(",")]
    else:
        seeds = default_seeds

    print(f"Using seeds: {seeds}")

    # Temporal configuration
    print("\n" + "-" * 50)
    print("TEMPORAL CONFIGURATION")
    print("-" * 50)

    snapshot_interval = input("Snapshot interval (nodes between snapshots) [10]: ").strip()
    if snapshot_interval.isdigit():
        snapshot_interval = int(snapshot_interval)
    else:
        snapshot_interval = 10

    max_snapshots = input("Maximum snapshots to keep [50]: ").strip()
    if max_snapshots.isdigit():
        max_snapshots = int(max_snapshots)
    else:
        max_snapshots = 50

    # Create base configuration
    config = NetworkConfig()

    # Network parameters
    print("\n" + "-" * 50)
    print("NETWORK PARAMETERS")
    print("-" * 50)

    max_articles = input(
        f"Max articles to process [{config.max_articles_to_process}]: "
    ).strip()
    if max_articles.isdigit():
        config.max_articles_to_process = int(max_articles)

    max_depth = input(f"Max depth [{config.max_depth}]: ").strip()
    if max_depth.isdigit():
        config.max_depth = int(max_depth)

    links_per_article = input(
        f"Links per article [{config.links_per_article}]: "
    ).strip()
    if links_per_article.isdigit():
        config.links_per_article = int(links_per_article)

    # Visualization options
    print("\n" + "-" * 50)
    print("VISUALIZATION OPTIONS")
    print("-" * 50)

    create_animation = input("Create animated visualization? (y/n) [y]: ").strip().lower()
    create_animation = create_animation != "n"

    if create_animation:
        animation_options = {
            "1": ("growth", "Color by node discovery time"),
            "2": ("centrality", "Color by centrality measures"),
            "3": ("influence", "Color by influence scores")
        }
        
        print("\nAnimation coloring scheme:")
        for key, (scheme, description) in animation_options.items():
            print(f"  {key}. {description}")
        
        color_choice = input("Choose coloring scheme (1-3) [1]: ").strip()
        if color_choice in animation_options:
            color_scheme = animation_options[color_choice][0]
        else:
            color_scheme = "growth"
        
        fps = input("Animation FPS [2]: ").strip()
        if fps.isdigit():
            fps = int(fps)
        else:
            fps = 2
    else:
        color_scheme = "growth"
        fps = 2

    # Build the temporal network
    print("\n" + "=" * 70)
    print("BUILDING TEMPORAL NETWORK")
    print("=" * 70)

    builder = TemporalNetworkBuilder(
        config, 
        snapshot_interval=snapshot_interval,
        max_snapshots=max_snapshots
    )
    
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    graph = builder.build_network(seeds, method=selected_method)
    
    end_time = datetime.now()
    build_duration = (end_time - start_time).total_seconds()
    
    print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Build duration: {build_duration:.2f} seconds")
    print(f"Snapshots captured: {len(builder.snapshots)}")

    # Analyze the network
    print("\n" + "=" * 70)
    print("ANALYZING TEMPORAL EVOLUTION")
    print("=" * 70)

    # Get basic network stats
    stats = builder.analyze_network()
    builder.print_analysis(stats)

    # Analyze growth patterns
    growth_analysis = builder.analyze_growth_patterns()
    
    print("\n" + "-" * 50)
    print("TEMPORAL GROWTH ANALYSIS")
    print("-" * 50)
    
    print(f"Total build time: {growth_analysis['total_build_time']:.2f} seconds")
    
    if 'growth_metrics' in growth_analysis:
        growth_metrics = growth_analysis['growth_metrics']
        print(f"Node growth rate: {growth_metrics.get('node_growth_rate', 0):.4f} nodes/hour")
        print(f"Edge growth rate: {growth_metrics.get('edge_growth_rate', 0):.4f} edges/hour")
        print(f"Density trend: {growth_metrics.get('density_trend', 0):.6f}")
        print(f"Clustering trend: {growth_metrics.get('clustering_trend', 0):.6f}")
    
    if 'discovery_depth_analysis' in growth_analysis:
        depth_analysis = growth_analysis['discovery_depth_analysis']
        print(f"Maximum depth reached: {depth_analysis.get('max_depth_reached', 0)}")
        print(f"Depth distribution: {depth_analysis.get('depth_distribution', {})}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    try:
        # Create evolution dashboard
        print("Creating evolution dashboard...")
        dashboard_path = builder.create_evolution_dashboard("temporal_network_dashboard.png")
        print(f"✓ Dashboard saved: {dashboard_path}")

        # Create standard network visualizations
        print("Creating standard network visualizations...")
        builder.visualize_pyvis("temporal_network_depth.html", physics=True, color_by="depth")
        builder.visualize_pyvis("temporal_network_communities.html", physics=True, color_by="community")
        builder.visualize_communities_matplotlib("temporal_communities.png")
        builder.save_graph("temporal_network.graphml")
        print("✓ Standard visualizations created")

        # Create animated visualization
        if create_animation:
            print(f"Creating animated visualization (this may take a while)...")
            try:
                animation_path = builder.create_evolution_visualization(
                    "temporal_network_evolution.gif", 
                    color_by=color_scheme, 
                    fps=fps
                )
                print(f"✓ Animation saved: {animation_path}")
            except Exception as e:
                print(f"✗ Animation failed: {e}")
                print("  Try installing pillow: pip install pillow")

        # Create growth animation with metrics
        print("Creating growth animation with metrics...")
        try:
            metrics_animation = builder.create_growth_animation_with_metrics(
                "temporal_growth_metrics.gif"
            )
            print(f"✓ Growth metrics animation saved: {metrics_animation}")
        except Exception as e:
            print(f"✗ Growth animation failed: {e}")

        # Export temporal data
        print("Exporting temporal data...")
        data_path = builder.export_temporal_data("temporal_network_data.json")
        print(f"✓ Temporal data exported: {data_path}")

        # Archive to history
        print("Archiving results...")
        archived_files, run_dir = builder.save_to_history()
        print(f"✓ Results archived to: {run_dir}")

    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("TEMPORAL NETWORK ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\nGenerated files:")
    print("  📊 temporal_network_dashboard.png - Evolution dashboard")
    print("  🌐 temporal_network_depth.html - Interactive network (by depth)")
    print("  🌐 temporal_network_communities.html - Interactive network (by community)")
    print("  📈 temporal_communities.png - Community structure")
    print("  📋 temporal_network.graphml - Network data")
    
    if create_animation:
        print("  🎬 temporal_network_evolution.gif - Animated evolution")
    
    print("  🎬 temporal_growth_metrics.gif - Growth metrics animation")
    print("  📝 temporal_network_data.json - Temporal analysis data")
    
    print(f"\nNetwork summary:")
    print(f"  • {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"  • {len(builder.snapshots)} snapshots captured")
    print(f"  • {build_duration:.2f} seconds build time")
    
    return builder


def quick_temporal_demo():
    """Run a quick demonstration of temporal network analysis."""
    print("=" * 70)
    print("QUICK TEMPORAL NETWORK DEMO")
    print("=" * 70)
    
    # Create a small network for demo
    config = NetworkConfig(
        max_articles_to_process=20,
        max_depth=2,
        links_per_article=5
    )
    
    builder = TemporalNetworkBuilder(config, snapshot_interval=5, max_snapshots=10)
    
    print("Building small demonstration network...")
    graph = builder.build_network(["Python", "Machine Learning"], method="breadth_first")
    
    print(f"Built network with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    print(f"Captured {len(builder.snapshots)} snapshots")
    
    # Create basic visualizations
    print("Creating visualizations...")
    builder.create_evolution_dashboard("demo_temporal_dashboard.png")
    builder.create_evolution_visualization("demo_temporal_evolution.gif", fps=1)
    
    print("Demo completed! Check demo_temporal_dashboard.png and demo_temporal_evolution.gif")


def main():
    """Main function with menu options."""
    print("=" * 70)
    print("TEMPORAL NETWORK ANALYZER")
    print("=" * 70)
    
    options = {
        "1": ("Interactive Analysis", interactive_temporal_cli),
        "2": ("Quick Demo", quick_temporal_demo),
        "3": ("Exit", exit)
    }
    
    print("\nSelect an option:")
    for key, (description, _) in options.items():
        print(f"  {key}. {description}")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in options:
            description, func = options[choice]
            print(f"\nSelected: {description}")
            if func != exit:
                func()
            break
        print("Invalid choice. Please enter 1-3.")


if __name__ == "__main__":
    main()
