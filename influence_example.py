#!/usr/bin/env python3
"""
Example demonstrating influence propagation on your network data.
"""

from config import NetworkConfig
from unified_network_builder import UnifiedNetworkBuilder

def main():
    """Run influence propagation example."""
    print("=" * 60)
    print("INFLUENCE PROPAGATION EXAMPLE")
    print("=" * 60)
    
    # Create a small network for demonstration
    config = NetworkConfig()
    config.max_articles_to_process = 20
    config.max_depth = 2
    config.links_per_article = 10
    
    builder = UnifiedNetworkBuilder(config)
    
    # Build network
    seeds = ["Machine Learning", "Data Science"]
    print(f"Building network with seeds: {seeds}")
    builder.build_network(seeds, method="breadth_first")
    
    print(f"\nNetwork built: {builder.graph.number_of_nodes()} nodes, {builder.graph.number_of_edges()} edges")
    
    # Run influence propagation analysis
    print("\n" + "-" * 40)
    print("INFLUENCE PROPAGATION ANALYSIS")
    print("-" * 40)
    
    # Test Independent Cascade Model
    print("\n1. Independent Cascade Model:")
    icm_results = builder.analyze_influence_propagation(
        seed_nodes=seeds,
        model="independent_cascade",
        num_simulations=50
    )
    
    if icm_results:
        selected = icm_results['selected_seeds']
        optimal = icm_results['optimal_seeds']
        
        print(f"Selected seeds: {selected['activation_rate']:.1%} activation rate")
        print(f"Optimal seeds: {optimal['activation_rate']:.1%} activation rate")
        
        print("\nStrategy comparison:")
        for strategy, results in icm_results['strategy_comparison'].items():
            if 'error' not in results:
                print(f"  {strategy}: {results['mean_activation_rate']:.1%}")
    
    # Test Linear Threshold Model
    print("\n2. Linear Threshold Model:")
    ltm_results = builder.analyze_influence_propagation(
        seed_nodes=seeds,
        model="linear_threshold",
        num_simulations=50
    )
    
    if ltm_results:
        selected = ltm_results['selected_seeds']
        optimal = ltm_results['optimal_seeds']
        
        print(f"Selected seeds: {selected['activation_rate']:.1%} activation rate")
        print(f"Optimal seeds: {optimal['activation_rate']:.1%} activation rate")
    
    # Create visualizations
    print("\n" + "-" * 40)
    print("CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Visualize ICM propagation
    builder.visualize_influence_propagation(
        seeds=["Machine Learning", "Data Science"],
        model="independent_cascade",
        output_path="icm_propagation.png"
    )
    print("✓ ICM propagation visualization: icm_propagation.png")
    
    # Visualize LTM propagation
    builder.visualize_influence_propagation(
        seeds=["Machine Learning", "Data Science"],
        model="linear_threshold",
        output_path="ltm_propagation.png"
    )
    print("✓ LTM propagation visualization: ltm_propagation.png")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()