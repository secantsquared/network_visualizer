"""
Learning Path Visualization Module

This module provides various visualization methods for learning paths including
interactive timelines, flowcharts, and progress tracking visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pyvis.network import Network
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import json
import os
from datetime import datetime, timedelta

from ..analysis.learning_path import LearningPath, LearningPathNode


class LearningPathVisualizer:
    """Creates various visualizations for learning paths."""
    
    def __init__(self, learning_paths: Dict[str, LearningPath] = None):
        """
        Initialize the visualizer.
        
        Args:
            learning_paths: Dictionary of learning paths to visualize
        """
        self.learning_paths = learning_paths or {}
        self.color_schemes = {
            'difficulty': ['#2E8B57', '#32CD32', '#FFD700', '#FF6347', '#DC143C'],
            'progress': ['#E6E6FA', '#DDA0DD', '#BA55D3', '#9370DB', '#8A2BE2'],
            'category': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }
    
    def create_interactive_timeline(self, path: LearningPath, 
                                  output_file: str = "learning_timeline.html") -> str:
        """
        Create an interactive timeline visualization using Plotly.
        
        Args:
            path: LearningPath object to visualize
            output_file: Output HTML file path
        """
        # Create timeline data
        timeline_data = []
        cumulative_time = 0
        
        for i, node in enumerate(path.nodes):
            # Parse estimated time to hours
            time_str = node.estimated_time
            if "hour" in time_str:
                hours = int(time_str.split()[0])
            elif "day" in time_str:
                hours = int(time_str.split()[0]) * 8
            elif "week" in time_str:
                hours = int(time_str.split()[0]) * 40
            else:
                hours = 1
            
            timeline_data.append({
                'task': node.name,
                'start': cumulative_time,
                'duration': hours,
                'difficulty': node.difficulty,
                'prerequisites': ', '.join(node.prerequisites),
                'description': node.description,
                'depth': node.depth_level
            })
            cumulative_time += hours
        
        # Create Gantt chart
        fig = go.Figure()
        
        for i, task in enumerate(timeline_data):
            color_intensity = task['difficulty']
            color = f'rgba(75, 192, 192, {0.3 + color_intensity * 0.7})'
            
            fig.add_trace(go.Scatter(
                x=[task['start'], task['start'] + task['duration']],
                y=[i, i],
                mode='lines',
                line=dict(color=color, width=20),
                name=task['task'],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Difficulty: %{customdata[1]:.2f}<br>"
                    "Duration: %{customdata[2]} hours<br>"
                    "Prerequisites: %{customdata[3]}<br>"
                    "%{customdata[4]}"
                ),
                customdata=[[task['task'], task['difficulty'], task['duration'], 
                           task['prerequisites'], task['description']]]
            ))
        
        fig.update_layout(
            title=f"Learning Path: {path.topic}",
            xaxis_title="Time (Hours)",
            yaxis_title="Learning Modules",
            height=max(400, len(timeline_data) * 40),
            showlegend=False,
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(timeline_data))),
                ticktext=[task['task'] for task in timeline_data]
            )
        )
        
        # Save to HTML
        fig.write_html(output_file)
        return output_file
    
    def create_flowchart_visualization(self, path: LearningPath, 
                                     output_file: str = "learning_flowchart.html") -> str:
        """
        Create a flowchart visualization showing prerequisites and progression.
        
        Args:
            path: LearningPath object to visualize
            output_file: Output HTML file path
        """
        # Create network graph
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.barnes_hut()
        
        # Add nodes
        for i, node in enumerate(path.nodes):
            # Color based on difficulty
            difficulty_color = self._get_difficulty_color(node.difficulty)
            
            # Size based on centrality
            size = 20 + (node.centrality_score * 100)
            
            net.add_node(
                node.name,
                label=f"{node.name}\n({node.estimated_time})",
                color=difficulty_color,
                size=size,
                title=f"Difficulty: {node.difficulty:.2f}\nTime: {node.estimated_time}\nPrerequisites: {', '.join(node.prerequisites) if node.prerequisites else 'None'}"
            )
        
        # Add edges for prerequisites
        for node in path.nodes:
            for prereq in node.prerequisites:
                if prereq in [n.name for n in path.nodes]:
                    net.add_edge(prereq, node.name, color="rgba(255,255,255,0.5)")
        
        # Add sequential flow edges
        for i in range(len(path.nodes) - 1):
            current = path.nodes[i].name
            next_node = path.nodes[i + 1].name
            net.add_edge(current, next_node, color="rgba(0,255,0,0.8)", width=3)
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            }
          }
        }
        """)
        
        net.save_graph(output_file)
        return output_file
    
    def create_difficulty_progression_chart(self, paths: Dict[str, LearningPath], 
                                          output_file: str = "difficulty_progression.html") -> str:
        """
        Create a chart showing difficulty progression for multiple paths.
        
        Args:
            paths: Dictionary of learning paths
            output_file: Output HTML file path
        """
        fig = go.Figure()
        
        for path_type, path in paths.items():
            steps = list(range(len(path.difficulty_progression)))
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=path.difficulty_progression,
                mode='lines+markers',
                name=path_type.title(),
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Learning Path Difficulty Progression",
            xaxis_title="Learning Step",
            yaxis_title="Difficulty Level",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.write_html(output_file)
        return output_file
    
    def create_comparison_dashboard(self, paths: Dict[str, LearningPath], 
                                  output_file: str = "learning_paths_dashboard.html") -> str:
        """
        Create a comprehensive dashboard comparing multiple learning paths.
        
        Args:
            paths: Dictionary of learning paths
            output_file: Output HTML file path
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Path Comparison', 'Difficulty Distribution', 
                          'Time Requirements', 'Path Length vs Difficulty'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Path comparison bars
        path_names = list(paths.keys())
        node_counts = [len(path.nodes) for path in paths.values()]
        
        fig.add_trace(
            go.Bar(x=path_names, y=node_counts, name="Number of Topics"),
            row=1, col=1
        )
        
        # Difficulty distribution
        for path_type, path in paths.items():
            fig.add_trace(
                go.Box(y=path.difficulty_progression, name=path_type.title()),
                row=1, col=2
            )
        
        # Time requirements
        time_values = []
        for path in paths.values():
            total_time = path.total_estimated_time
            if "hour" in total_time:
                hours = int(total_time.split()[0])
            elif "day" in total_time:
                hours = int(total_time.split()[0]) * 8
            elif "week" in total_time:
                hours = int(total_time.split()[0]) * 40
            else:
                hours = 1
            time_values.append(hours)
        
        fig.add_trace(
            go.Bar(x=path_names, y=time_values, name="Total Hours"),
            row=2, col=1
        )
        
        # Path length vs difficulty
        for path_type, path in paths.items():
            avg_difficulty = np.mean(path.difficulty_progression)
            fig.add_trace(
                go.Scatter(
                    x=[len(path.nodes)],
                    y=[avg_difficulty],
                    mode='markers',
                    marker=dict(size=15),
                    name=path_type.title()
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Learning Paths Comparison Dashboard",
            showlegend=True
        )
        
        fig.write_html(output_file)
        return output_file
    
    def create_progress_tracker(self, path: LearningPath, 
                              completed_nodes: List[str] = None,
                              output_file: str = "progress_tracker.html") -> str:
        """
        Create a progress tracking visualization.
        
        Args:
            path: LearningPath object
            completed_nodes: List of completed node names
            output_file: Output HTML file path
        """
        completed_nodes = completed_nodes or []
        
        # Create progress data
        progress_data = []
        for i, node in enumerate(path.nodes):
            status = "completed" if node.name in completed_nodes else "pending"
            progress_data.append({
                'node': node.name,
                'status': status,
                'difficulty': node.difficulty,
                'order': i,
                'time': node.estimated_time
            })
        
        # Create visualization
        fig = go.Figure()
        
        # Add progress bars
        completed_count = len(completed_nodes)
        total_count = len(path.nodes)
        progress_percent = (completed_count / total_count) * 100
        
        # Overall progress
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=progress_percent,
            title={'text': f"Overall Progress: {path.topic}"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkgreen"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 100], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}},
            domain={'x': [0, 1], 'y': [0.5, 1]}
        ))
        
        # Progress timeline
        colors = ['green' if node['status'] == 'completed' else 'lightblue' 
                 for node in progress_data]
        
        fig.add_trace(go.Bar(
            x=[node['node'] for node in progress_data],
            y=[1] * len(progress_data),
            marker_color=colors,
            name="Progress",
            yaxis='y2'
        ))
        
        fig.update_layout(
            height=600,
            yaxis2=dict(
                title="Topics",
                overlaying='y',
                side='right',
                range=[0, 2]
            ),
            xaxis=dict(title="Learning Topics", tickangle=45)
        )
        
        fig.write_html(output_file)
        return output_file
    
    def create_prerequisite_matrix(self, path: LearningPath, 
                                 output_file: str = "prerequisite_matrix.png") -> str:
        """
        Create a matrix visualization showing prerequisite relationships.
        
        Args:
            path: LearningPath object
            output_file: Output PNG file path
        """
        nodes = [node.name for node in path.nodes]
        n = len(nodes)
        
        # Create adjacency matrix
        matrix = np.zeros((n, n))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for node in path.nodes:
            node_idx = node_to_idx[node.name]
            for prereq in node.prerequisites:
                if prereq in node_to_idx:
                    prereq_idx = node_to_idx[prereq]
                    matrix[prereq_idx, node_idx] = 1
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(nodes, rotation=45, ha='right')
        ax.set_yticklabels(nodes)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 1:
                    ax.text(j, i, '✓', ha='center', va='center', 
                           color='white', fontsize=12, fontweight='bold')
        
        ax.set_title(f'Prerequisite Matrix: {path.topic}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Depends on (Prerequisites)', fontsize=12)
        ax.set_ylabel('Provides foundation for', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _get_difficulty_color(self, difficulty: float) -> str:
        """Get color based on difficulty level."""
        colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347', '#DC143C']
        idx = min(int(difficulty * len(colors)), len(colors) - 1)
        return colors[idx]
    
    def save_learning_path_data(self, paths: Dict[str, LearningPath], 
                               output_file: str = "learning_paths_data.json") -> str:
        """
        Save learning path data to JSON file.
        
        Args:
            paths: Dictionary of learning paths
            output_file: Output JSON file path
        """
        data = {
            "generated_at": datetime.now().isoformat(),
            "paths": {path_type: path.to_dict() for path_type, path in paths.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_file
    
    def create_all_visualizations(self, paths: Dict[str, LearningPath], 
                                output_dir: str = "learning_path_outputs") -> Dict[str, str]:
        """
        Create all available visualizations for the learning paths.
        
        Args:
            paths: Dictionary of learning paths
            output_dir: Directory to save outputs
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        outputs = {}
        
        # Create visualizations for each path
        for path_type, path in paths.items():
            # Timeline
            timeline_file = os.path.join(output_dir, f"{path_type}_timeline.html")
            outputs[f"{path_type}_timeline"] = self.create_interactive_timeline(path, timeline_file)
            
            # Flowchart
            flowchart_file = os.path.join(output_dir, f"{path_type}_flowchart.html")
            outputs[f"{path_type}_flowchart"] = self.create_flowchart_visualization(path, flowchart_file)
            
            # Prerequisite matrix
            matrix_file = os.path.join(output_dir, f"{path_type}_prerequisites.png")
            outputs[f"{path_type}_prerequisites"] = self.create_prerequisite_matrix(path, matrix_file)
        
        # Comparison visualizations
        if len(paths) > 1:
            # Difficulty progression
            diff_file = os.path.join(output_dir, "difficulty_progression.html")
            outputs["difficulty_progression"] = self.create_difficulty_progression_chart(paths, diff_file)
            
            # Dashboard
            dashboard_file = os.path.join(output_dir, "learning_paths_dashboard.html")
            outputs["dashboard"] = self.create_comparison_dashboard(paths, dashboard_file)
        
        # Save data
        data_file = os.path.join(output_dir, "learning_paths_data.json")
        outputs["data"] = self.save_learning_path_data(paths, data_file)
        
        return outputs