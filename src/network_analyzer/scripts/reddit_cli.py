#!/usr/bin/env python3
"""
Interactive CLI for Reddit network analysis.
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


def get_reddit_credentials():
    """Get Reddit API credentials from user input or environment variables."""
    print("Reddit API Configuration")
    print("-" * 30)
    
    # Try to get from environment first
    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT')
    
    if not client_id:
        print("Reddit Client ID not found in environment.")
        client_id = input("Reddit Client ID: ").strip()
    else:
        print(f"✓ Using Reddit Client ID from environment: {client_id[:8]}...")
    
    if not client_secret:
        print("Reddit Client Secret not found in environment.")
        client_secret = input("Reddit Client Secret: ").strip()
    else:
        print("✓ Using Reddit Client Secret from environment: [HIDDEN]")
    
    if not user_agent:
        print("Reddit User Agent not found in environment.")
        user_agent = input("Reddit User Agent (e.g., 'network-analyzer:v1.0 (by /u/yourusername)'): ").strip()
    else:
        print(f"✓ Using Reddit User Agent from environment: {user_agent}")
    
    if not all([client_id, client_secret, user_agent]):
        print("\nError: All Reddit credentials are required!")
        print("\nTo avoid entering credentials each time, you can:")
        print("1. Set environment variables:")
        print("   export REDDIT_CLIENT_ID='your_client_id'")
        print("   export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("   export REDDIT_USER_AGENT='your_app_name:v1.0 (by /u/yourusername)'")
        print("\n2. Create a .env file in your project root with the above variables")
        print("\nGet your credentials from: https://www.reddit.com/prefs/apps")
        return None, None, None
    
    return client_id, client_secret, user_agent


def get_network_config():
    """Get network configuration from user input."""
    print("\nNetwork Configuration")
    print("-" * 30)
    
    # Network type
    print("Network Types:")
    print("1. Subreddit (relationships between subreddits)")
    print("2. User (relationships between users)")
    print("3. Discussion (relationships between posts/discussions)")
    
    while True:
        choice = input("Choose network type (1-3): ").strip()
        if choice == "1":
            network_type = "subreddit"
            break
        elif choice == "2":
            network_type = "user"
            break
        elif choice == "3":
            network_type = "discussion"
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Time filter for Reddit data
    if network_type == "subreddit":
        print("\nTime Filter Options:")
        print("1. All time")
        print("2. Past year")
        print("3. Past month (recommended)")
        print("4. Past week")
        print("5. Past day")
        
        time_filters = {"1": "all", "2": "year", "3": "month", "4": "week", "5": "day"}
        while True:
            choice = input("Choose time filter (1-5): ").strip()
            if choice in time_filters:
                time_filter = time_filters[choice]
                break
            else:
                print("Invalid choice. Please enter 1-5.")
    else:
        time_filter = "month"
    
    # Network size parameters
    max_depth = int(input("Maximum depth (1-3, recommended: 2): ").strip() or "2")
    max_items = int(input("Maximum items to process (10-50, recommended: 15): ").strip() or "15")
    links_per_item = int(input("Links per item (5-15, recommended: 8): ").strip() or "8")
    
    return network_type, time_filter, max_depth, max_items, links_per_item


def get_seed_items(network_type):
    """Get seed items from user input."""
    print(f"\nSeed {network_type.title()}s")
    print("-" * 30)
    
    if network_type == "subreddit":
        print("Enter subreddit names (without 'r/') separated by commas")
        print("Example: Python, MachineLearning, datascience")
        seeds_input = input("Subreddits: ").strip()
    elif network_type == "user":
        print("Enter usernames (without 'u/') separated by commas")
        print("Example: spez, AutoModerator")
        seeds_input = input("Users: ").strip()
    else:  # discussion
        print("Choose how to specify discussion seeds:")
        print("1. Enter Reddit post URLs")
        print("2. Enter post IDs directly")
        print("3. Get top posts from a subreddit")
        
        while True:
            choice = input("Choose method (1-3): ").strip()
            if choice in ["1", "2", "3"]:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        
        if choice == "1":
            print("\nEnter Reddit post URLs separated by commas")
            print("Example: https://reddit.com/r/Python/comments/abc123/title/")
            seeds_input = input("URLs: ").strip()
            # Extract post IDs from URLs
            import re
            seeds = []
            for url in seeds_input.split(","):
                url = url.strip()
                # Match Reddit post URLs and extract post ID
                match = re.search(r'reddit\.com/r/\w+/comments/([a-zA-Z0-9]+)', url)
                if match:
                    seeds.append(match.group(1))
                else:
                    print(f"Warning: Could not extract post ID from URL: {url}")
            return seeds
            
        elif choice == "2":
            print("\nEnter Reddit post IDs separated by commas")
            print("Example: abc123, def456")
            print("(Post IDs can be found in Reddit URLs after '/comments/')")
            seeds_input = input("Post IDs: ").strip()
            
        else:  # choice == "3"
            subreddit = input("Enter subreddit name to get top posts from: ").strip()
            if not subreddit:
                print("Error: Subreddit name is required")
                return []
            
            print("Time period for top posts:")
            print("1. Past day")
            print("2. Past week") 
            print("3. Past month")
            print("4. Past year")
            print("5. All time")
            
            time_filters = {"1": "day", "2": "week", "3": "month", "4": "year", "5": "all"}
            while True:
                time_choice = input("Choose time period (1-5): ").strip()
                if time_choice in time_filters:
                    time_filter = time_filters[time_choice]
                    break
                print("Invalid choice. Please enter 1-5.")
            
            num_posts = input("Number of top posts to use (default 5): ").strip()
            try:
                num_posts = int(num_posts) if num_posts else 5
                num_posts = min(num_posts, 25)  # Limit to 25 posts
            except ValueError:
                num_posts = 5
            
            # Get top posts from subreddit
            print(f"\nFetching top {num_posts} posts from r/{subreddit}...")
            try:
                # We'll need to create a temporary Reddit instance to fetch posts
                import praw
                import os
                
                # Use credentials from environment
                reddit = praw.Reddit(
                    client_id=os.getenv('REDDIT_CLIENT_ID'),
                    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                    user_agent=os.getenv('REDDIT_USER_AGENT')
                )
                
                subreddit_obj = reddit.subreddit(subreddit)
                top_posts = list(subreddit_obj.top(time_filter=time_filter, limit=num_posts))
                
                seeds = [post.id for post in top_posts]
                print(f"Found {len(seeds)} posts:")
                for i, post in enumerate(top_posts, 1):
                    title = post.title[:60] + "..." if len(post.title) > 60 else post.title
                    print(f"  {i}. {title}")
                
                return seeds
                
            except Exception as e:
                print(f"Error fetching posts: {e}")
                print("Falling back to manual post ID entry...")
                seeds_input = input("Enter post IDs separated by commas: ").strip()
    
    seeds = [seed.strip() for seed in seeds_input.split(",") if seed.strip()]
    return seeds


def main():
    """Main CLI function."""
    print("Reddit Network Analyzer CLI")
    print("=" * 50)
    
    # Get Reddit credentials
    client_id, client_secret, user_agent = get_reddit_credentials()
    if not client_id:
        return
    
    # Get network configuration
    network_type, time_filter, max_depth, max_items, links_per_item = get_network_config()
    
    # Get seed items
    seeds = get_seed_items(network_type)
    if not seeds:
        print("Error: At least one seed item is required!")
        return
    
    # Create configuration
    config = NetworkConfig(
        data_source_type="reddit",
        reddit_client_id=client_id,
        reddit_client_secret=client_secret,
        reddit_user_agent=user_agent,
        reddit_network_type=network_type,
        reddit_time_filter=time_filter,
        reddit_max_posts=50,
        reddit_max_comments=30,
        max_depth=max_depth,
        max_articles_to_process=max_items,
        links_per_article=links_per_item
    )
    
    # Build network
    print(f"\nBuilding {network_type} network...")
    print(f"Seeds: {', '.join(seeds)}")
    print(f"Configuration: depth={max_depth}, items={max_items}, links={links_per_item}")
    print("-" * 50)
    
    try:
        builder = UnifiedNetworkBuilder(config)
        graph = builder.build_network(seeds)
        
        print(f"\nNetwork built successfully!")
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        
        # Analyze network
        print("\nAnalyzing network...")
        stats = builder.analyze_network()
        builder.print_analysis(stats)
        
        # Generate outputs
        print("\nGenerating outputs...")
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"reddit_{network_type}_{timestamp}"
        
        # Save network file
        graphml_file = f"{base_filename}.graphml"
        builder.save_graph(graphml_file)
        print(f"Network saved: {graphml_file}")
        
        # Create interactive visualization
        html_file = f"{base_filename}.html"
        builder.visualize_pyvis(html_file)
        print(f"Interactive visualization: {html_file}")
        
        # Create community visualization (if applicable)
        if graph.number_of_nodes() > 3:
            communities_file = f"{base_filename}_communities.png"
            builder.visualize_communities_matplotlib(communities_file)
            print(f"Community visualization: {communities_file}")
        
        print(f"\nAnalysis complete! Open {html_file} in your browser to explore the network.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()