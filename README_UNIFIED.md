# Unified Network Builder

A flexible network generation tool that supports multiple data sources, preserving the original Wikipedia functionality while adding support for Coursera course datasets and other sources.

## Features

- **Multi-source Support**: Wikipedia articles, Coursera courses, or hybrid mode
- **Preserved Functionality**: All original Wikipedia features remain intact
- **Learning Path Generation**: Create skill-based learning paths using course data
- **Flexible Configuration**: Switch between data sources at runtime
- **All Original Algorithms**: Breadth-first, random walk, DFS, topic-focused, hub-and-spoke

## Quick Start

### 1. Basic Usage (Wikipedia - Default)

```python
from config import NetworkConfig
from unified_network_builder import UnifiedNetworkBuilder

# Use Wikipedia (same as before)
config = NetworkConfig()
config.data_source_type = "wikipedia"

builder = UnifiedNetworkBuilder(config)
graph = builder.build_network(["Machine Learning", "Data Science"])
```

### 2. Using Coursera Dataset

```python
# Download dataset from: https://www.kaggle.com/datasets/azraimohamad/coursera-course-data

config = NetworkConfig()
config.data_source_type = "coursera"
config.coursera_dataset_path = "./coursera_courses_2024.csv"

builder = UnifiedNetworkBuilder(config, coursera_dataset_path="./coursera_courses_2024.csv")
graph = builder.build_network(["Machine Learning", "Python Programming"])

# Get learning path for skills
learning_path = builder.get_learning_path_for_skills(["Python Programming"], difficulty="Beginner")
```

### 3. Hybrid Mode (Switch Between Sources)

```python
config = NetworkConfig()
config.data_source_type = "hybrid"
config.primary_data_source = "wikipedia"  # Start with Wikipedia
config.coursera_dataset_path = "./coursera_courses_2024.csv"

builder = UnifiedNetworkBuilder(config, coursera_dataset_path="./coursera_courses_2024.csv")

# Build with Wikipedia
wiki_graph = builder.build_network(["Machine Learning"])

# Switch to Coursera
builder.switch_data_source("coursera")
course_graph = builder.build_network(["Machine Learning"])
```

## Interactive CLI

Run the interactive interface to choose your data source:

```bash
python unified_main.py
```

The CLI will guide you through:
1. Choosing data source (Wikipedia/Coursera/Hybrid)
2. Selecting network generation method
3. Configuring parameters
4. Building and visualizing the network

## Data Sources

### Wikipedia (Original)
- **Type**: `wikipedia`
- **Data**: Article titles and internal links
- **Use Case**: Knowledge graphs, topic exploration
- **No setup required**

### Coursera Courses
- **Type**: `coursera`
- **Data**: Course titles, skills, prerequisites, ratings
- **Use Case**: Learning paths, skill development
- **Setup**: Download dataset from Kaggle

### Hybrid Mode
- **Type**: `hybrid`
- **Data**: Both Wikipedia and Coursera
- **Use Case**: Compare different perspectives on topics
- **Setup**: Optional Coursera dataset

## Configuration

All original configuration options remain available. New options:

```python
config = NetworkConfig()

# Data source selection
config.data_source_type = "wikipedia"  # "wikipedia", "coursera", "hybrid"
config.primary_data_source = "wikipedia"  # For hybrid mode
config.coursera_dataset_path = "./coursera_courses_2024.csv"  # Path to dataset
```

## Learning Path Features (Coursera Only)

### Get Courses by Skill
```python
courses = builder.get_courses_by_skill("Python Programming")
```

### Get Learning Path for Skills
```python
path = builder.get_learning_path_for_skills(
    skills=["Machine Learning", "Deep Learning"], 
    difficulty="Beginner"
)
```

### Get Course Metadata
```python
metadata = builder.get_item_metadata("Machine Learning Course")
# Returns: rating, difficulty, duration, skills, organization, etc.
```

## Examples

Run the examples to see all features in action:

```bash
python example_usage.py
```

This will demonstrate:
- Wikipedia network building
- Coursera course networks
- Hybrid mode usage
- Learning path generation

## File Structure

- `unified_main.py` - Interactive CLI interface
- `unified_network_builder.py` - Main unified builder class
- `data_sources.py` - Data source adapters
- `example_usage.py` - Usage examples
- `config.py` - Configuration (updated)
- `network_builder.py` - Original Wikipedia builder (preserved)
- `main.py` - Original interface (preserved)

## Data Source Setup

### Coursera Dataset
1. Go to https://www.kaggle.com/datasets/azraimohamad/coursera-course-data
2. Download the CSV file
3. Place it in the project directory as `coursera_courses_2024.csv`

### Dataset Format
The Coursera dataset should have columns:
- `Title` - Course name
- `Skills` - Skills taught (comma-separated or JSON)
- `Organization` - Course provider
- `Ratings` - Course rating
- `Review count` - Number of reviews
- `Miscellaneous info` - Difficulty, duration, type

## Migration from Original

No changes needed! The original interface remains fully functional:

```python
# This still works exactly as before
from network_builder import WikipediaNetworkBuilder
builder = WikipediaNetworkBuilder()
```

To use new features, simply switch to the unified builder:

```python
# New unified interface
from unified_network_builder import UnifiedNetworkBuilder
builder = UnifiedNetworkBuilder()  # Defaults to Wikipedia
```

## Visualization

All original visualizations work with any data source:

```python
# Works with Wikipedia, Coursera, or hybrid data
builder.visualize_pyvis("network.html", color_by="depth")
builder.visualize_pyvis("communities.html", color_by="community")
builder.visualize_communities_matplotlib("communities.png")
```

## Use Cases

### Wikipedia Mode
- Academic research
- Knowledge graph construction
- Topic exploration
- Content discovery

### Coursera Mode
- Learning path planning
- Skill development roadmaps
- Course recommendation systems
- Educational analytics

### Hybrid Mode
- Compare academic vs. practical perspectives
- Validate learning paths against knowledge domains
- Multi-perspective analysis

## Advanced Usage

### Custom Data Sources
Extend the `DataSourceAdapter` class to add new data sources:

```python
from data_sources import DataSourceAdapter

class CustomDataSource(DataSourceAdapter):
    def get_relationships(self, item: str) -> List[str]:
        # Your custom logic here
        return []
```

### Switching Sources at Runtime
```python
# In hybrid mode
builder.switch_data_source("coursera")
# Build network with Coursera data

builder.switch_data_source("wikipedia")  
# Build network with Wikipedia data
```

## Contributing

When adding new data sources:
1. Extend `DataSourceAdapter` in `data_sources.py`
2. Add configuration options to `NetworkConfig`
3. Update the CLI in `unified_main.py`
4. Add examples to `example_usage.py`

## License

Same as original project.