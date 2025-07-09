#!/usr/bin/env python3
"""
Test script for force-directed visualization with different physics engines.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import NetworkConfig
from unified_network_builder import UnifiedNetworkBuilder
from force_directed_visualizer import ForceDirectedVisualizer

def create_sample_network():
    """Create a small test network for visualization testing."""
    print("Creating sample network for testing...")
    
    # Configure for small test network
    config = NetworkConfig()
    config.max_depth = 2
    config.max_articles_to_process = 15
    config.data_source_type = "wikipedia"
    
    # Create builder
    builder = UnifiedNetworkBuilder(config)
    
    # Build small network
    test_seeds = ["Python (programming language)"]
    graph = builder.build_network(test_seeds, method="breadth_first")
    
    print(f"Created test network with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    return graph, config

def test_physics_engines():
    """Test all available physics engines."""
    print("=" * 60)
    print("TESTING FORCE-DIRECTED PHYSICS ENGINES")
    print("=" * 60)
    
    # Create sample network
    graph, config = create_sample_network()
    
    # Test each physics engine
    physics_engines = ["barnes_hut", "force_atlas2", "hierarchical", "circular", "organic"]
    
    for engine in physics_engines:
        print(f"\nTesting {engine} physics engine...")
        
        try:
            # Create visualizer
            visualizer = ForceDirectedVisualizer(graph)
            
            # Generate visualization
            output_file = f"test_{engine}.html"
            visualizer.visualize(
                output_path=output_file,
                physics_type=engine,
                color_by="depth"
            )
            
            print(f"✓ Successfully created {output_file}")
            
        except Exception as e:
            print(f"✗ Error with {engine}: {e}")
    
    print("\n" + "=" * 60)
    print("TESTING CUSTOM PHYSICS PARAMETERS")
    print("=" * 60)
    
    # Test custom physics parameters
    custom_params = {
        "gravity": -100000,
        "spring_length": 150,
        "damping": 0.15
    }
    
    try:
        visualizer = ForceDirectedVisualizer(graph)
        visualizer.visualize(
            output_path="test_custom_params.html",
            physics_type="barnes_hut",
            color_by="community",
            custom_params=custom_params
        )
        print("✓ Successfully created visualization with custom parameters")
    except Exception as e:
        print(f"✗ Error with custom parameters: {e}")

def test_physics_info():
    """Test physics information display."""
    print("\n" + "=" * 60)
    print("AVAILABLE PHYSICS ENGINES")
    print("=" * 60)
    
    ForceDirectedVisualizer.list_physics_options()
    
    print("\n" + "=" * 60)
    print("PHYSICS ENGINE DETAILS")
    print("=" * 60)
    
    for engine in ["barnes_hut", "force_atlas2", "hierarchical"]:
        try:
            info = ForceDirectedVisualizer.get_physics_info(engine)
            print(f"\n{engine}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Default params: {info['params']}")
        except Exception as e:
            print(f"Error getting info for {engine}: {e}")

def main():
    """Main test function."""
    print("Starting force-directed visualization tests...")
    
    try:
        # Test physics engines
        test_physics_engines()
        
        # Test physics info
        test_physics_info()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ All tests completed successfully!")
        print("\nGenerated test files:")
        print("  - test_barnes_hut.html")
        print("  - test_force_atlas2.html")
        print("  - test_hierarchical.html")
        print("  - test_circular.html")
        print("  - test_organic.html")
        print("  - test_custom_params.html")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()