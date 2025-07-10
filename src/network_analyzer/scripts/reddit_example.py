#!/usr/bin/env python3
"""
Example usage of Reddit data source for network analysis.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

from network_analyzer.core.config import NetworkConfig
from network_analyzer.core.unified_network_builder import UnifiedNetworkBuilder


def main():
    """Main example function."""
    print("Reddit Network Analysis Example")
    print("=" * 50)
    
    # Reddit API credentials - loaded from environment variables or .env file
    reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
    reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'network-analyzer:v1.0 (by /u/your-username)')
    
    if not all([reddit_client_id, reddit_client_secret]):
        print("Error: Reddit API credentials not found!")
        print("\nPlease set your Reddit API credentials using one of these methods:")
        print("\n1. Environment variables:")
        print("   export REDDIT_CLIENT_ID='your_client_id'")
        print("   export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("   export REDDIT_USER_AGENT='your_app_name:v1.0 (by /u/yourusername)'")
        print("\n2. Create a .env file in your project root:")
        print("   REDDIT_CLIENT_ID=your_client_id")
        print("   REDDIT_CLIENT_SECRET=your_client_secret")
        print("   REDDIT_USER_AGENT=your_app_name:v1.0 (by /u/yourusername)")
        print("\n3. Set them in your shell profile (~/.bashrc, ~/.zshrc, etc.)")
        print("\nGet your credentials from: https://www.reddit.com/prefs/apps")
        return
    
    # Example 1: Subreddit Network Analysis
    print("\n1. Building Subreddit Network...")
    config = NetworkConfig(
        data_source_type="reddit",
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=reddit_user_agent,
        reddit_network_type="subreddit",
        reddit_max_posts=50,
        reddit_time_filter="month",
        max_depth=2,
        max_articles_to_process=20,
        links_per_article=10
    )
    
    builder = UnifiedNetworkBuilder(config)
    
    # Build network starting from some popular subreddits
    seed_subreddits = ["Python", "MachineLearning", "datascience"]
    
    try:
        graph = builder.build_network(seed_subreddits)
        
        print(f"Network built successfully!")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        
        # Analyze the network
        stats = builder.analyze_network()
        builder.print_analysis(stats)
        
        # Create visualizations
        print("\nCreating visualizations...")
        builder.visualize_pyvis("reddit_subreddit_network.html")
        builder.visualize_communities("reddit_subreddit_communities.png")
        builder.save_network("reddit_subreddit_network.graphml")
        
        print("Visualizations saved:")
        print("  - reddit_subreddit_network.html (interactive)")
        print("  - reddit_subreddit_communities.png")
        print("  - reddit_subreddit_network.graphml")
        
    except Exception as e:
        print(f"Error building subreddit network: {e}")
        return
    
    # Example 2: User Network Analysis
    print("\n2. Building User Network...")
    config.reddit_network_type = "user"
    config.max_articles_to_process = 10  # Start smaller for user networks
    
    builder = UnifiedNetworkBuilder(config)
    
    # Build network from some active users (replace with real usernames)
    seed_users = ["AutoModerator"]  # Using AutoModerator as it's present in most subreddits
    
    try:
        graph = builder.build_network(seed_users)
        
        print(f"User network built successfully!")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        
        # Create visualization
        builder.visualize_pyvis("reddit_user_network.html")
        print("User network visualization saved: reddit_user_network.html")
        
    except Exception as e:
        print(f"Error building user network: {e}")
    
    # Example 3: Hybrid Network (Reddit + Wikipedia)
    print("\n3. Building Hybrid Network (Reddit + Wikipedia)...")
    hybrid_config = NetworkConfig(
        data_source_type="hybrid",
        primary_data_source="wikipedia",
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=reddit_user_agent,
        reddit_network_type="subreddit",
        max_depth=1,
        max_articles_to_process=15,
        links_per_article=8
    )
    
    hybrid_builder = UnifiedNetworkBuilder(hybrid_config)
    
    try:
        # Start with topics that exist in both Wikipedia and Reddit
        hybrid_seeds = ["Machine Learning", "Python", "Data Science"]
        graph = hybrid_builder.build_network(hybrid_seeds)
        
        print(f"Hybrid network built successfully!")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        
        # Check what sources we have
        print(f"Available sources: {hybrid_builder.get_available_sources()}")
        
        # Create visualization
        hybrid_builder.visualize_pyvis("reddit_hybrid_network.html")
        print("Hybrid network visualization saved: reddit_hybrid_network.html")
        
    except Exception as e:
        print(f"Error building hybrid network: {e}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()