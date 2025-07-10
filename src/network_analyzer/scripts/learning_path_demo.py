"""
Learning Path Demo Script

A comprehensive demonstration of the learning path generation capabilities
for team presentations and showcases.
"""

import os
import time
import requests_cache
from datetime import datetime

from network_analyzer.core.config import NetworkConfig
from network_analyzer.core.network_builder import WikipediaNetworkBuilder
from network_analyzer.analysis.learning_path import LearningPathAnalyzer
from network_analyzer.visualization.learning_path_visualizer import LearningPathVisualizer


def print_header(text: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def print_section(text: str, char: str = "-", width: int = 60):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{text}")
    print(f"{char * width}")


def setup_demo_environment():
    """Set up the demo environment with caching and logging."""
    print_header("LEARNING PATH GENERATOR DEMO", "=", 80)
    print("\n🚀 Setting up demo environment...")
    
    # Configure caching for faster demo
    requests_cache.install_cache(
        "demo_cache",
        backend="sqlite",
        expire_after=86400,
        allowable_methods=["GET"],
        allowable_codes=[200],
    )
    
    # Create demo output directory
    demo_dir = f"demo_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(demo_dir, exist_ok=True)
    
    print(f"✅ Demo environment ready!")
    print(f"📁 Output directory: {demo_dir}")
    
    return demo_dir


def demo_network_building(topics: list, demo_dir: str):
    """Demonstrate network building capabilities."""
    print_header("NETWORK BUILDING DEMONSTRATION")
    
    print(f"🔍 Building knowledge network for topics: {', '.join(topics)}")
    print("📊 Using optimized settings for demo...")
    
    # Configure for demo (smaller, faster network)
    config = NetworkConfig(
        max_depth=2,
        max_articles_to_process=20,
        links_per_article=10,
        async_enabled=True,
        max_concurrent_requests=15
    )
    
    # Build the network
    start_time = time.time()
    builder = WikipediaNetworkBuilder(config)
    graph = builder.build_network(topics, method="breadth_first")
    build_time = time.time() - start_time
    
    print(f"🎯 Network built in {build_time:.2f} seconds")
    print(f"📈 Network statistics:")
    print(f"   • Nodes: {len(graph.nodes)}")
    print(f"   • Edges: {len(graph.edges)}")
    print(f"   • Density: {len(graph.edges) / (len(graph.nodes) * (len(graph.nodes) - 1) / 2):.3f}")
    
    # Save network visualization
    try:
        builder.visualize_pyvis(
            os.path.join(demo_dir, "demo_network.html"),
            physics=True,
            color_by="depth"
        )
        print(f"💾 Network visualization saved: demo_network.html")
    except Exception as e:
        print(f"⚠️  Warning: Could not save network visualization: {e}")
    
    return graph


def demo_learning_path_generation(graph, topic: str, demo_dir: str):
    """Demonstrate learning path generation."""
    print_header("LEARNING PATH GENERATION")
    
    print(f"🎯 Generating learning paths for: {topic}")
    print("🔄 Creating multiple path types for comparison...")
    
    # Initialize analyzer
    analyzer = LearningPathAnalyzer(graph)
    
    # Generate different types of learning paths
    path_types = ["foundational", "comprehensive", "fast_track"]
    learning_paths = {}
    
    for path_type in path_types:
        try:
            print(f"   📝 Generating {path_type} path...")
            path = analyzer.generate_learning_path(
                topic=topic,
                max_nodes=12,
                path_type=path_type
            )
            learning_paths[path_type] = path
            
            # Show brief summary
            print(f"   ✅ {path_type}: {len(path.nodes)} topics, {path.total_estimated_time}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate {path_type} path: {e}")
    
    if not learning_paths:
        print("❌ No learning paths generated!")
        return None
    
    # Analyze path quality
    print_section("PATH QUALITY ANALYSIS")
    
    for path_type, path in learning_paths.items():
        quality = analyzer.analyze_path_quality(path)
        print(f"📊 {path_type.upper()} PATH QUALITY:")
        print(f"   • Coverage: {quality['coverage']:.2f}")
        print(f"   • Progression: {quality['progression']:.2f}")
        print(f"   • Foundational Strength: {quality['foundational_strength']:.2f}")
        print(f"   • Overall Quality: {quality['overall_quality']:.2f}")
    
    return learning_paths


def demo_visualizations(learning_paths: dict, demo_dir: str):
    """Demonstrate visualization capabilities."""
    print_header("VISUALIZATION SHOWCASE")
    
    print("🎨 Creating comprehensive visualizations...")
    print("📊 Generating multiple visualization types...")
    
    # Initialize visualizer
    visualizer = LearningPathVisualizer(learning_paths)
    
    # Create all visualizations
    try:
        outputs = visualizer.create_all_visualizations(learning_paths, demo_dir)
        
        print(f"✅ Created {len(outputs)} visualizations:")
        
        # Group outputs by type
        viz_groups = {
            "timelines": [k for k in outputs.keys() if "timeline" in k],
            "flowcharts": [k for k in outputs.keys() if "flowchart" in k],
            "matrices": [k for k in outputs.keys() if "prerequisites" in k],
            "comparisons": [k for k in outputs.keys() if k in ["difficulty_progression", "dashboard"]],
            "data": [k for k in outputs.keys() if "data" in k]
        }
        
        for group_name, viz_list in viz_groups.items():
            if viz_list:
                print(f"   📈 {group_name.upper()}:")
                for viz in viz_list:
                    filename = os.path.basename(outputs[viz])
                    print(f"      • {filename}")
        
        return outputs
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return {}


def demo_learning_path_details(learning_paths: dict):
    """Show detailed learning path information."""
    print_header("LEARNING PATH DETAILS")
    
    for path_type, path in learning_paths.items():
        print_section(f"{path_type.upper()} LEARNING PATH", "-", 50)
        
        print(f"🎯 Topic: {path.topic}")
        print(f"⏱️  Total Time: {path.total_estimated_time}")
        print(f"📊 Difficulty Range: {min(path.difficulty_progression):.2f} - {max(path.difficulty_progression):.2f}")
        print(f"🔢 Number of Steps: {len(path.nodes)}")
        
        print(f"\n📚 Learning Sequence:")
        for i, node in enumerate(path.nodes, 1):
            prereq_text = ""
            if node.prerequisites:
                prereq_text = f" (requires: {', '.join(node.prerequisites[:2])}{'...' if len(node.prerequisites) > 2 else ''})"
            
            difficulty_emoji = "🟢" if node.difficulty < 0.3 else "🟡" if node.difficulty < 0.7 else "🔴"
            
            print(f"   {i:2d}. {node.name}")
            print(f"       {difficulty_emoji} Difficulty: {node.difficulty:.2f} | ⏱️ Time: {node.estimated_time}{prereq_text}")


def run_complete_demo():
    """Run the complete learning path demo."""
    
    # Setup
    demo_dir = setup_demo_environment()
    
    # Demo topics (can be customized)
    topics = ["Machine Learning", "Data Science"]
    main_topic = topics[0]
    
    try:
        # 1. Network Building Demo
        graph = demo_network_building(topics, demo_dir)
        
        # 2. Learning Path Generation Demo
        learning_paths = demo_learning_path_generation(graph, main_topic, demo_dir)
        
        if learning_paths:
            # 3. Visualization Demo
            outputs = demo_visualizations(learning_paths, demo_dir)
            
            # 4. Detailed Path Analysis
            demo_learning_path_details(learning_paths)
            
            # 5. Summary
            print_header("DEMO SUMMARY")
            print(f"✅ Successfully generated {len(learning_paths)} learning paths")
            print(f"📊 Created {len(outputs)} visualizations")
            print(f"📁 All outputs saved to: {demo_dir}")
            
            print(f"\n🎯 KEY FEATURES DEMONSTRATED:")
            print(f"   • Multi-source network building (Wikipedia)")
            print(f"   • Multiple learning path algorithms")
            print(f"   • Prerequisite relationship detection")
            print(f"   • Difficulty progression analysis")
            print(f"   • Interactive visualizations")
            print(f"   • Comprehensive quality metrics")
            
            print(f"\n🚀 NEXT STEPS:")
            print(f"   • Open the HTML files to explore interactive visualizations")
            print(f"   • Try different topics with: network-analyzer-learning-path")
            print(f"   • Integrate with existing course data or LMS")
            print(f"   • Customize algorithms for specific domains")
            
        else:
            print("❌ Demo failed: No learning paths generated")
            
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print_header("DEMO COMPLETE", "=", 80)


def interactive_demo():
    """Run an interactive demo with user input."""
    print_header("INTERACTIVE LEARNING PATH DEMO")
    
    print("🎯 This demo will showcase the learning path generation capabilities.")
    print("💡 You can customize the topic and see different path types.")
    
    # Get user input
    topic = input("\n🔍 Enter a topic to learn about (or press Enter for 'Machine Learning'): ").strip()
    if not topic:
        topic = "Machine Learning"
    
    print(f"\n🎯 Selected topic: {topic}")
    print("🔄 Starting demo...")
    
    # Setup
    demo_dir = setup_demo_environment()
    
    try:
        # Build network
        graph = demo_network_building([topic], demo_dir)
        
        # Generate paths
        learning_paths = demo_learning_path_generation(graph, topic, demo_dir)
        
        if learning_paths:
            # Show path comparison
            print_section("PATH COMPARISON")
            
            for path_type, path in learning_paths.items():
                print(f"\n📝 {path_type.upper()} PATH:")
                print(f"   • Steps: {len(path.nodes)}")
                print(f"   • Time: {path.total_estimated_time}")
                print(f"   • Difficulty: {min(path.difficulty_progression):.2f} - {max(path.difficulty_progression):.2f}")
                
                # Show first few steps
                print(f"   • First 3 steps:")
                for i, node in enumerate(path.nodes[:3], 1):
                    print(f"     {i}. {node.name} ({node.estimated_time})")
                if len(path.nodes) > 3:
                    print(f"     ... and {len(path.nodes) - 3} more steps")
            
            # Create visualizations
            outputs = demo_visualizations(learning_paths, demo_dir)
            
            print(f"\n🎉 Demo complete! Check the files in: {demo_dir}")
            
        else:
            print("❌ Failed to generate learning paths")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        run_complete_demo()