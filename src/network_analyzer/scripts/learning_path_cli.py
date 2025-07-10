"""
Learning Path CLI

Interactive command-line interface for generating and visualizing learning paths
from knowledge networks.
"""

import argparse
import requests_cache
import sys
import os
from typing import Dict, List, Optional

from network_analyzer.core.config import NetworkConfig
from network_analyzer.core.network_builder import WikipediaNetworkBuilder
from network_analyzer.core.unified_network_builder import UnifiedNetworkBuilder
from network_analyzer.data_sources.reddit import RedditDataSource
from network_analyzer.analysis.learning_path import LearningPathAnalyzer
from network_analyzer.visualization.learning_path_visualizer import LearningPathVisualizer


def setup_cache():
    """Configure request caching."""
    requests_cache.install_cache(
        "learning_path_cache",
        backend="sqlite",
        expire_after=86400,
        allowable_methods=["GET"],
        allowable_codes=[200],
    )


def interactive_cli():
    """Interactive CLI for learning path generation."""
    print("=" * 70)
    print("LEARNING PATH GENERATOR - INTERACTIVE CLI")
    print("=" * 70)
    
    # Data source selection
    print("\nSelect your data source:")
    print("  1. Wikipedia (knowledge concepts)")
    print("  2. Reddit (community discussions)")
    print("  3. Hybrid (Wikipedia + Coursera)")
    print("  4. Use existing network file")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            break
        print("Invalid choice. Please enter 1-4.")
    
    data_source = {
        "1": "wikipedia",
        "2": "reddit", 
        "3": "hybrid",
        "4": "file"
    }[choice]
    
    # Topic input
    print("\n" + "-" * 50)
    print("LEARNING TOPIC")
    print("-" * 50)
    
    topic = input("Enter the topic you want to learn about: ").strip()
    if not topic:
        print("Topic is required!")
        return
    
    # Build or load network
    print(f"\n" + "=" * 50)
    print("BUILDING KNOWLEDGE NETWORK")
    print("=" * 50)
    
    graph = None
    
    if data_source == "wikipedia":
        config = NetworkConfig(
            max_depth=2,
            max_articles_to_process=30,
            links_per_article=15
        )
        
        builder = WikipediaNetworkBuilder(config)
        graph = builder.build_network([topic], method="breadth_first")
        
    elif data_source == "reddit":
        # Reddit setup
        try:
            from network_analyzer.data_sources.reddit import RedditDataSource
            reddit_source = RedditDataSource()
            
            config = NetworkConfig(
                max_depth=2,
                max_articles_to_process=25,
                data_source_type="reddit"
            )
            
            # Use Reddit CLI builder logic
            print("Building Reddit network...")
            # This would need to be implemented similar to reddit_cli.py
            print("Reddit integration coming soon!")
            return
            
        except Exception as e:
            print(f"Error setting up Reddit: {e}")
            return
    
    elif data_source == "hybrid":
        # Check for Coursera data
        coursera_path = "data/coursera_courses_2024.csv"
        if not os.path.exists(coursera_path):
            print(f"Coursera dataset not found at {coursera_path}")
            print("Please download from: https://www.kaggle.com/datasets/azraimohamad/coursera-course-data")
            return
        
        config = NetworkConfig(
            data_source_type="hybrid",
            primary_data_source="wikipedia",
            coursera_dataset_path=coursera_path,
            max_depth=2,
            max_articles_to_process=25
        )
        
        builder = UnifiedNetworkBuilder(config, coursera_dataset_path=coursera_path)
        graph = builder.build_network([topic])
    
    elif data_source == "file":
        import networkx as nx
        
        file_path = input("Enter path to GraphML network file: ").strip()
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        try:
            graph = nx.read_graphml(file_path)
            print(f"Loaded network with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        except Exception as e:
            print(f"Error loading network file: {e}")
            return
    
    if graph is None or len(graph.nodes) == 0:
        print("Failed to build network or network is empty!")
        return
    
    print(f"Network built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Learning path generation
    print(f"\n" + "=" * 50)
    print("GENERATING LEARNING PATHS")
    print("=" * 50)
    
    analyzer = LearningPathAnalyzer(graph)
    
    # Path type selection
    print("\nSelect learning path types to generate:")
    print("  1. Foundational (focus on core concepts)")
    print("  2. Comprehensive (detailed coverage)")
    print("  3. Fast Track (efficient path)")
    print("  4. All paths (compare options)")
    
    while True:
        path_choice = input("\nEnter your choice (1-4): ").strip()
        if path_choice in ["1", "2", "3", "4"]:
            break
        print("Invalid choice. Please enter 1-4.")
    
    path_types = {
        "1": ["foundational"],
        "2": ["comprehensive"],
        "3": ["fast_track"],
        "4": ["foundational", "comprehensive", "fast_track"]
    }[path_choice]
    
    # Generate learning paths
    learning_paths = {}
    
    for path_type in path_types:
        try:
            print(f"Generating {path_type} learning path...")
            path = analyzer.generate_learning_path(
                topic=topic,
                max_nodes=15,
                path_type=path_type
            )
            learning_paths[path_type] = path
            
            # Show path summary
            print(f"\n{path_type.upper()} PATH SUMMARY:")
            print(f"  Topics: {len(path.nodes)}")
            print(f"  Estimated time: {path.total_estimated_time}")
            print(f"  Difficulty range: {min(path.difficulty_progression):.2f} - {max(path.difficulty_progression):.2f}")
            
        except Exception as e:
            print(f"Error generating {path_type} path: {e}")
    
    if not learning_paths:
        print("No learning paths could be generated!")
        return
    
    # Path analysis
    print(f"\n" + "-" * 50)
    print("PATH ANALYSIS")
    print("-" * 50)
    
    for path_type, path in learning_paths.items():
        quality = analyzer.analyze_path_quality(path)
        print(f"\n{path_type.upper()} PATH QUALITY:")
        print(f"  Coverage: {quality['coverage']:.2f}")
        print(f"  Progression: {quality['progression']:.2f}")
        print(f"  Foundational strength: {quality['foundational_strength']:.2f}")
        print(f"  Overall quality: {quality['overall_quality']:.2f}")
    
    # Visualization
    print(f"\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    visualizer = LearningPathVisualizer(learning_paths)
    
    # Create output directory
    output_dir = f"learning_path_outputs_{topic.replace(' ', '_').lower()}"
    outputs = visualizer.create_all_visualizations(learning_paths, output_dir)
    
    print(f"\nVisualizations created in: {output_dir}")
    print("\nGenerated files:")
    for viz_type, file_path in outputs.items():
        print(f"  {viz_type}: {file_path}")
    
    # Show learning path details
    print(f"\n" + "=" * 50)
    print("LEARNING PATH DETAILS")
    print("=" * 50)
    
    for path_type, path in learning_paths.items():
        print(f"\n{path_type.upper()} LEARNING PATH:")
        print(f"Topic: {path.topic}")
        print(f"Total time: {path.total_estimated_time}")
        print(f"Path type: {path.path_type}")
        print("\nLearning sequence:")
        
        for i, node in enumerate(path.nodes, 1):
            prereq_str = f" (requires: {', '.join(node.prerequisites)})" if node.prerequisites else ""
            print(f"  {i:2d}. {node.name} - {node.estimated_time} - Difficulty: {node.difficulty:.2f}{prereq_str}")
    
    print(f"\n" + "=" * 70)
    print("LEARNING PATH GENERATION COMPLETE!")
    print("=" * 70)
    
    return learning_paths


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate learning paths from knowledge networks"
    )
    
    # Add command line arguments
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic to generate learning path for"
    )
    
    parser.add_argument(
        "--source",
        choices=["wikipedia", "reddit", "hybrid", "file"],
        default="wikipedia",
        help="Data source for network building"
    )
    
    parser.add_argument(
        "--path-type",
        choices=["foundational", "comprehensive", "fast_track", "all"],
        default="all",
        help="Type of learning path to generate"
    )
    
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=15,
        help="Maximum number of nodes in learning path"
    )
    
    parser.add_argument(
        "--network-file",
        type=str,
        help="Path to existing network file (GraphML format)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="learning_path_outputs",
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Setup caching
    setup_cache()
    
    # Run interactive mode or command-line mode
    if args.interactive or not args.topic:
        interactive_cli()
    else:
        # Command line mode
        print(f"Generating learning path for: {args.topic}")
        print(f"Source: {args.source}")
        print(f"Path type: {args.path_type}")
        
        # Build network based on source
        graph = None
        
        if args.source == "wikipedia":
            config = NetworkConfig(
                max_depth=2,
                max_articles_to_process=30,
                links_per_article=15
            )
            builder = WikipediaNetworkBuilder(config)
            graph = builder.build_network([args.topic], method="breadth_first")
            
        elif args.source == "file" and args.network_file:
            import networkx as nx
            try:
                graph = nx.read_graphml(args.network_file)
                print(f"Loaded network with {len(graph.nodes)} nodes")
            except Exception as e:
                print(f"Error loading network file: {e}")
                return
        
        if graph is None:
            print("Failed to build or load network!")
            return
        
        # Generate learning paths
        analyzer = LearningPathAnalyzer(graph)
        learning_paths = {}
        
        path_types = ["foundational", "comprehensive", "fast_track"] if args.path_type == "all" else [args.path_type]
        
        for path_type in path_types:
            try:
                path = analyzer.generate_learning_path(
                    topic=args.topic,
                    max_nodes=args.max_nodes,
                    path_type=path_type
                )
                learning_paths[path_type] = path
                print(f"Generated {path_type} path with {len(path.nodes)} nodes")
            except Exception as e:
                print(f"Error generating {path_type} path: {e}")
        
        # Create visualizations
        if learning_paths:
            visualizer = LearningPathVisualizer(learning_paths)
            outputs = visualizer.create_all_visualizations(learning_paths, args.output_dir)
            
            print(f"\nVisualizations saved to: {args.output_dir}")
            for viz_type, file_path in outputs.items():
                print(f"  {viz_type}: {file_path}")


if __name__ == "__main__":
    main()