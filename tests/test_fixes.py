#!/usr/bin/env python3
"""
Test script to verify the fixes for Coursera data source and hybrid mode.
"""

import logging
from pathlib import Path
from config import NetworkConfig
from unified_network_builder import UnifiedNetworkBuilder

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_coursera_only():
    """Test Coursera data source with skill names as seeds."""
    print("\n" + "="*60)
    print("TESTING COURSERA DATA SOURCE")
    print("="*60)
    
    config = NetworkConfig()
    config.data_source_type = "coursera"
    config.coursera_dataset_path = "./coursera_courses_2024.csv"
    config.max_articles_to_process = 10
    config.max_depth = 1
    
    try:
        builder = UnifiedNetworkBuilder(config, coursera_dataset_path="./coursera_courses_2024.csv")
        
        # Test with skill names as seeds
        seeds = ["Machine Learning", "Python Programming"]
        print(f"Testing with seeds: {seeds}")
        
        graph = builder.build_network(seeds, method="breadth_first")
        
        print(f"Results:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        print(f"  Node list: {list(graph.nodes())[:5]}...")
        
        return graph.number_of_nodes() > len(seeds)
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_hybrid_mode():
    """Test hybrid mode with Wikipedia fallback."""
    print("\n" + "="*60)
    print("TESTING HYBRID MODE")
    print("="*60)
    
    config = NetworkConfig()
    config.data_source_type = "hybrid"
    config.primary_data_source = "wikipedia"
    config.coursera_dataset_path = "./coursera_courses_2024.csv"
    config.max_articles_to_process = 10
    config.max_depth = 1
    
    try:
        builder = UnifiedNetworkBuilder(config, coursera_dataset_path="./coursera_courses_2024.csv")
        
        # Test with Wikipedia articles
        seeds = ["Machine Learning", "Python Programming"]
        print(f"Testing with seeds: {seeds}")
        
        graph = builder.build_network(seeds, method="breadth_first")
        
        print(f"Results:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        print(f"  Node list: {list(graph.nodes())[:5]}...")
        
        return graph.number_of_nodes() > len(seeds)
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing network builder fixes...")
    
    # Check if dataset exists
    if not Path("./coursera_courses_2024.csv").exists():
        print("ERROR: coursera_courses_2024.csv not found")
        return
    
    # Test Coursera only
    coursera_success = test_coursera_only()
    
    # Test hybrid mode
    hybrid_success = test_hybrid_mode()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Coursera only: {'✓ PASSED' if coursera_success else '✗ FAILED'}")
    print(f"Hybrid mode: {'✓ PASSED' if hybrid_success else '✗ FAILED'}")
    
    if coursera_success and hybrid_success:
        print("\n🎉 All tests passed! The fixes are working.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()