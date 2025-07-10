"""
Reddit data source adapter using PRAW.
"""

import praw
import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict
from .base import DataSourceAdapter


class RedditDataSource(DataSourceAdapter):
    """Reddit data source adapter using PRAW for network analysis."""
    
    def __init__(self, config, client_id: str, client_secret: str, user_agent: str):
        super().__init__(config)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.network_type = getattr(config, 'reddit_network_type', 'subreddit')
        self.max_posts_per_subreddit = getattr(config, 'reddit_max_posts', 100)
        self.max_comments_per_post = getattr(config, 'reddit_max_comments', 50)
        self.time_filter = getattr(config, 'reddit_time_filter', 'month')
        self.subreddit_cache = {}
        self.user_cache = {}
        
    def get_relationships(self, item: str) -> List[str]:
        """Get related items based on network type."""
        if self.network_type == 'subreddit':
            return self._get_subreddit_relationships(item)
        elif self.network_type == 'user':
            return self._get_user_relationships(item)
        elif self.network_type == 'discussion':
            return self._get_discussion_relationships(item)
        else:
            self.logger.warning(f"Unknown network type: {self.network_type}")
            return []
    
    def _get_subreddit_relationships(self, subreddit_name: str) -> List[str]:
        """Get related subreddits based on user overlap and cross-posts."""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            related_subreddits = set()
            
            # Get top posts from the subreddit
            posts = list(subreddit.top(time_filter=self.time_filter, limit=self.max_posts_per_subreddit))
            
            # Find cross-posted subreddits
            for post in posts:
                if hasattr(post, 'crosspost_parent_list') and post.crosspost_parent_list:
                    for crosspost in post.crosspost_parent_list:
                        if 'subreddit' in crosspost:
                            related_subreddits.add(crosspost['subreddit'])
            
            # Find subreddits mentioned in comments
            for post in posts[:20]:  # Limit to top 20 posts for performance
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:self.max_comments_per_post]:
                        if hasattr(comment, 'body'):
                            # Look for r/subreddit mentions
                            import re
                            subreddit_mentions = re.findall(r'r/([A-Za-z0-9_]+)', comment.body)
                            for mention in subreddit_mentions:
                                if mention.lower() != subreddit_name.lower():
                                    related_subreddits.add(mention)
                except Exception as e:
                    self.logger.debug(f"Error processing comments for post {post.id}: {e}")
                    continue
            
            # Get user overlap (users who post in both subreddits)
            user_subreddits = self._get_user_subreddit_overlap(subreddit_name, posts)
            related_subreddits.update(user_subreddits)
            
            return list(related_subreddits)[:self.config.links_per_article]
            
        except Exception as e:
            self.logger.error(f"Error getting subreddit relationships for {subreddit_name}: {e}")
            return []
    
    def _get_user_subreddit_overlap(self, subreddit_name: str, posts: List) -> Set[str]:
        """Find subreddits where users from this subreddit also post."""
        user_subreddits = set()
        users_processed = 0
        max_users_to_check = 10
        
        for post in posts:
            if users_processed >= max_users_to_check:
                break
                
            try:
                author = post.author
                if author and hasattr(author, 'name'):
                    # Get recent submissions from this user
                    recent_posts = list(author.submissions.new(limit=20))
                    for user_post in recent_posts:
                        if hasattr(user_post, 'subreddit') and user_post.subreddit.display_name.lower() != subreddit_name.lower():
                            user_subreddits.add(user_post.subreddit.display_name)
                    users_processed += 1
            except Exception as e:
                self.logger.debug(f"Error processing user overlap: {e}")
                continue
                
        return user_subreddits
    
    def _get_user_relationships(self, username: str) -> List[str]:
        """Get related users based on interaction patterns."""
        try:
            user = self.reddit.redditor(username)
            related_users = set()
            
            # Get users who replied to this user's comments
            comments = list(user.comments.new(limit=50))
            for comment in comments:
                try:
                    comment.replies.replace_more(limit=0)
                    for reply in comment.replies.list():
                        if hasattr(reply, 'author') and reply.author:
                            related_users.add(reply.author.name)
                except Exception as e:
                    self.logger.debug(f"Error processing comment replies: {e}")
                    continue
            
            # Get users this user replied to
            for comment in comments:
                try:
                    if hasattr(comment, 'parent') and comment.parent():
                        parent = comment.parent()
                        if hasattr(parent, 'author') and parent.author:
                            related_users.add(parent.author.name)
                except Exception as e:
                    self.logger.debug(f"Error processing parent comments: {e}")
                    continue
            
            return list(related_users)[:self.config.links_per_article]
            
        except Exception as e:
            self.logger.error(f"Error getting user relationships for {username}: {e}")
            return []
    
    def _get_discussion_relationships(self, post_id: str) -> List[str]:
        """Get related posts through discussion threads."""
        try:
            submission = self.reddit.submission(id=post_id)
            related_posts = set()
            
            # Get posts mentioned in comments
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:self.max_comments_per_post]:
                if hasattr(comment, 'body'):
                    # Look for Reddit post links
                    import re
                    post_links = re.findall(r'reddit\.com/r/\w+/comments/([a-zA-Z0-9]+)', comment.body)
                    related_posts.update(post_links)
            
            # Get other posts by the same author
            if hasattr(submission, 'author') and submission.author:
                author_posts = list(submission.author.submissions.new(limit=10))
                for post in author_posts:
                    if post.id != post_id:
                        related_posts.add(post.id)
            
            return list(related_posts)[:self.config.links_per_article]
            
        except Exception as e:
            self.logger.error(f"Error getting discussion relationships for post {post_id}: {e}")
            return []
    
    def should_filter_item(self, item: str) -> bool:
        """Check if item should be filtered out."""
        if self.network_type == 'subreddit':
            return self._should_filter_subreddit(item)
        elif self.network_type == 'user':
            return self._should_filter_user(item)
        elif self.network_type == 'discussion':
            return self._should_filter_post(item)
        return False
    
    def _should_filter_subreddit(self, subreddit_name: str) -> bool:
        """Filter out inappropriate or banned subreddits."""
        # Filter common spam/bot subreddits
        spam_subreddits = {
            'spam', 'test', 'testing', 'bot', 'bots', 'shadowban', 'deleted',
            'removed', 'private', 'quarantined'
        }
        
        if subreddit_name.lower() in spam_subreddits:
            return True
        
        # Filter very short or very long names
        if len(subreddit_name) < 2 or len(subreddit_name) > 30:
            return True
        
        # Check if subreddit exists and is accessible
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            # Try to access basic info to check if it's accessible
            _ = subreddit.display_name
            return False
        except Exception:
            return True
    
    def _should_filter_user(self, username: str) -> bool:
        """Filter out deleted/suspended users."""
        if username.lower() in ['deleted', 'removed', '[deleted]', '[removed]']:
            return True
        
        try:
            user = self.reddit.redditor(username)
            # Check if user is accessible
            _ = user.name
            return False
        except Exception:
            return True
    
    def _should_filter_post(self, post_id: str) -> bool:
        """Filter out deleted/removed posts."""
        try:
            submission = self.reddit.submission(id=post_id)
            return submission.removed_by_category is not None
        except Exception:
            return True
    
    def get_item_metadata(self, item: str) -> Dict:
        """Get metadata for the given item."""
        if self.network_type == 'subreddit':
            return self._get_subreddit_metadata(item)
        elif self.network_type == 'user':
            return self._get_user_metadata(item)
        elif self.network_type == 'discussion':
            return self._get_post_metadata(item)
        return {}
    
    def _get_subreddit_metadata(self, subreddit_name: str) -> Dict:
        """Get metadata for subreddit."""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            return {
                'type': 'reddit_subreddit',
                'name': subreddit.display_name,
                'title': subreddit.title,
                'description': subreddit.public_description,
                'subscribers': subreddit.subscribers,
                'created_utc': subreddit.created_utc,
                'source': 'Reddit',
                'url': f'https://reddit.com/r/{subreddit_name}',
                'over18': subreddit.over18 if hasattr(subreddit, 'over18') else False
            }
        except Exception as e:
            self.logger.error(f"Error getting subreddit metadata for {subreddit_name}: {e}")
            return {
                'type': 'reddit_subreddit',
                'name': subreddit_name,
                'source': 'Reddit',
                'url': f'https://reddit.com/r/{subreddit_name}',
                'error': str(e)
            }
    
    def _get_user_metadata(self, username: str) -> Dict:
        """Get metadata for user."""
        try:
            user = self.reddit.redditor(username)
            return {
                'type': 'reddit_user',
                'name': user.name,
                'comment_karma': user.comment_karma,
                'link_karma': user.link_karma,
                'created_utc': user.created_utc,
                'source': 'Reddit',
                'url': f'https://reddit.com/u/{username}',
                'is_employee': user.is_employee if hasattr(user, 'is_employee') else False,
                'is_mod': user.is_mod if hasattr(user, 'is_mod') else False
            }
        except Exception as e:
            self.logger.error(f"Error getting user metadata for {username}: {e}")
            return {
                'type': 'reddit_user',
                'name': username,
                'source': 'Reddit',
                'url': f'https://reddit.com/u/{username}',
                'error': str(e)
            }
    
    def _get_post_metadata(self, post_id: str) -> Dict:
        """Get metadata for post."""
        try:
            submission = self.reddit.submission(id=post_id)
            return {
                'type': 'reddit_post',
                'id': submission.id,
                'title': submission.title,
                'author': submission.author.name if submission.author else '[deleted]',
                'subreddit': submission.subreddit.display_name,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'source': 'Reddit',
                'url': f'https://reddit.com{submission.permalink}',
                'is_self': submission.is_self,
                'over_18': submission.over_18
            }
        except Exception as e:
            self.logger.error(f"Error getting post metadata for {post_id}: {e}")
            return {
                'type': 'reddit_post',
                'id': post_id,
                'source': 'Reddit',
                'error': str(e)
            }
    
    def get_source_type(self) -> str:
        """Return the source type identifier."""
        return f"reddit_{self.network_type}"