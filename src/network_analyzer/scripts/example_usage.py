"""
Example usage of the unified network builder with different data sources.
"""

from config import NetworkConfig
from unified_network_builder import UnifiedNetworkBuilder


def example_wikipedia_usage():
    """Example using Wikipedia data source."""
    print("=" * 50)
    print("EXAMPLE: Wikipedia Data Source")
    print("=" * 50)
    
    config = NetworkConfig()
    config.data_source_type = "wikipedia"
    config.max_articles_to_process = 20
    config.max_depth = 2
    
    builder = UnifiedNetworkBuilder(config)
    
    # Build network
    seeds = ["Machine Learning", "Data Science"]
    graph = builder.build_network(seeds, method="breadth_first")
    
    # Analyze
    stats = builder.analyze_network()
    print(f"Network: {stats['nodes']} nodes, {stats['edges']} edges")
    
    # Create visualization
    builder.visualize_pyvis("wikipedia_example.html", color_by="depth")
    print("Visualization saved to wikipedia_example.html")
    
    return builder


def example_coursera_usage():
    """Example using Coursera data source."""
    print("=" * 50)
    print("EXAMPLE: Coursera Data Source")
    print("=" * 50)
    
    # Note: You need to download the dataset first
    dataset_path = "./coursera_courses_2024.csv"
    
    config = NetworkConfig()
    config.data_source_type = "coursera"
    config.coursera_dataset_path = dataset_path
    config.max_articles_to_process = 15
    config.max_depth = 2
    
    try:
        builder = UnifiedNetworkBuilder(config, coursera_dataset_path=dataset_path)
        
        # Build network with course/skill seeds
        seeds = ["Machine Learning", "Python Programming"]
        graph = builder.build_network(seeds, method="breadth_first")
        
        # Analyze
        stats = builder.analyze_network()
        print(f"Network: {stats['nodes']} nodes, {stats['edges']} edges")
        
        # Show courses for specific skills
        for seed in seeds:
            courses = builder.get_courses_by_skill(seed)
            print(f"\nCourses teaching '{seed}': {len(courses)} found")
            for course in courses[:3]:  # Show top 3
                metadata = builder.get_item_metadata(course)
                print(f"  - {course} (Rating: {metadata.get('rating', 'N/A')})")
        
        # Get learning path
        learning_path = builder.get_learning_path_for_skills(
            ["Python Programming"], difficulty="Beginner"
        )
        print(f"\nBeginner learning path for Python: {len(learning_path)} courses")
        
        # Create visualization
        builder.visualize_pyvis("coursera_example.html", color_by="community")
        print("Visualization saved to coursera_example.html")
        
        return builder
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please download the Coursera dataset from:")
        print("https://www.kaggle.com/datasets/azraimohamad/coursera-course-data")
        return None


def example_hybrid_usage():
    """Example using hybrid data source."""
    print("=" * 50)
    print("EXAMPLE: Hybrid Data Source")
    print("=" * 50)
    
    dataset_path = "./coursera_courses_2024.csv"
    
    config = NetworkConfig()
    config.data_source_type = "hybrid"
    config.primary_data_source = "wikipedia"  # Start with Wikipedia
    config.coursera_dataset_path = dataset_path
    config.max_articles_to_process = 10
    config.max_depth = 1
    
    try:
        builder = UnifiedNetworkBuilder(config, coursera_dataset_path=dataset_path)
        
        print("Available sources:", builder.get_available_sources())
        
        # Build with Wikipedia first
        print("\n1. Building with Wikipedia...")
        seeds = ["Machine Learning"]
        wiki_graph = builder.build_network(seeds, method="breadth_first")
        print(f"Wikipedia network: {wiki_graph.number_of_nodes()} nodes")
        
        # Switch to Coursera
        print("\n2. Switching to Coursera...")
        builder.switch_data_source("coursera")
        
        # Build with same seeds but different source
        course_graph = builder.build_network(seeds, method="breadth_first")
        print(f"Coursera network: {course_graph.number_of_nodes()} nodes")
        
        # Compare the networks
        print("\n3. Comparison:")
        print(f"Wikipedia found {wiki_graph.number_of_nodes()} related articles")
        print(f"Coursera found {course_graph.number_of_nodes()} related courses")
        
        return builder
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the Coursera dataset available.")
        return None


def example_learning_path_generation():
    """Example of generating learning paths for skills."""
    print("=" * 50)
    print("EXAMPLE: Learning Path Generation")
    print("=" * 50)
    
    dataset_path = "./coursera_courses_2024.csv"
    
    config = NetworkConfig()
    config.data_source_type = "coursera"
    config.coursera_dataset_path = dataset_path
    config.max_articles_to_process = 30
    config.max_depth = 2
    
    try:
        builder = UnifiedNetworkBuilder(config, coursera_dataset_path=dataset_path)
        
        # Define learning goals
        target_skills = ["Machine Learning", "Deep Learning", "Data Science"]
        
        print("Target skills:", target_skills)
        
        # Generate learning paths for different levels
        for level in ["Beginner", "Intermediate", "Advanced"]:
            print(f"\n{level} Learning Path:")
            path = builder.get_learning_path_for_skills(target_skills, level)
            
            for i, course in enumerate(path[:5], 1):  # Show top 5
                metadata = builder.get_item_metadata(course)
                rating = metadata.get('rating', 'N/A')
                duration = metadata.get('duration', 'N/A')
                print(f"  {i}. {course}")
                print(f"     Rating: {rating}, Duration: {duration}")
            
            if len(path) > 5:
                print(f"     ... and {len(path) - 5} more courses")
        
        # Build network to show relationships
        print(f"\nBuilding network to show skill relationships...")
        graph = builder.build_network(target_skills, method="topic_focused")
        
        # Create visualization
        builder.visualize_pyvis("learning_path_example.html", color_by="community")
        print("Learning path visualization saved to learning_path_example.html")
        
        return builder
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please download the Coursera dataset to use this feature.")
        return None


def main():
    """Run all examples."""
    print("UNIFIED NETWORK BUILDER EXAMPLES")
    print("=" * 60)
    
    # Example 1: Wikipedia
    wikipedia_builder = example_wikipedia_usage()
    print("\n")
    
    # Example 2: Coursera
    coursera_builder = example_coursera_usage()
    print("\n")
    
    # Example 3: Hybrid
    hybrid_builder = example_hybrid_usage()
    print("\n")
    
    # Example 4: Learning Path Generation
    learning_path_builder = example_learning_path_generation()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    
    # Summary
    examples_run = sum([
        wikipedia_builder is not None,
        coursera_builder is not None,
        hybrid_builder is not None,
        learning_path_builder is not None
    ])
    
    print(f"Successfully ran {examples_run}/4 examples")
    
    if examples_run < 4:
        print("\nTo run all examples, download the Coursera dataset from:")
        print("https://www.kaggle.com/datasets/azraimohamad/coursera-course-data")
        print("and place it as 'coursera_courses_2024.csv' in the current directory.")


if __name__ == "__main__":
    main()