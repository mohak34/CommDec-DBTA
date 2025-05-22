# Community Detection with Enhanced Temporal Graph Networks (TGN)

## Project Overview

This project implements and extends the Temporal Graph Network (TGN) architecture for community detection in dynamic graphs. The implementation is based on the original TGN paper from Twitter Research: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637).

The key enhancements include:

1. **Disabled Decay Factor**: The decay factor in the DecayTemporalAttention class has been modified (set to zero) to evaluate model performance without temporal decay metrics.
2. **Graph Visualization**: Added comprehensive visualization capabilities to analyze and understand the temporal evolution of communities in dynamic graphs.

## Project Structure

```
CommunityDetection/
├── data/                        # Dataset directory
│   ├── reddit_TGAT.csv          # Reddit dataset
│   └── soc-redditHyperlinks-*.tsv
├── docs/                        # Documentation
│   └── README_TGN.md            # Detailed TGN model documentation
├── notebooks/                   # Jupyter notebooks
│   ├── Enhanced_TGN.ipynb       # Main TGN implementation notebook
│   ├── TGN_with_Visualization.ipynb  # Notebook with disabled decay factor and visualization
│   └── ...                      # Other exploratory notebooks
├── src/                         # Source code
│   ├── enhanced_tgn.py          # Enhanced TGN model implementation
│   ├── dataset_visualization.py # Dataset visualization functions
│   └── graph_visualization.py   # Graph visualization utilities
├── tests/                       # Test scripts
│   └── ...                      # Various test scripts
├── visualizations/              # Output visualizations
│   └── ...                      # Generated visualizations
├── requirements.txt             # Project dependencies
└── setup_path.py                # Script to set up Python path
```

## Installation and Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd CommunityDetection
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add the project to your Python path**:
   ```python
   import setup_path
   ```

## Usage

### Running the Enhanced TGN Model

The main implementation is available in the notebook `notebooks/Enhanced_TGN.ipynb`. This notebook contains the complete implementation of the Temporal Graph Network with the Reddit dataset.

### Visualizing Communities

To visualize communities in the temporal graph:

1. Open `notebooks/TGN_with_Visualization.ipynb`
2. Run the notebook to see visualizations of the Reddit dataset and community structures
3. The visualizations show how communities evolve over time in the dynamic graph

## Key Components

### Enhanced TGN Model

The enhanced TGN model includes a modified DecayTemporalAttention mechanism where the decay factor has been disabled (set to zero) to evaluate the model's performance without temporal decay metrics.

### Visualization Utilities

The project includes comprehensive visualization utilities:

- `visualize_temporal_graph`: Visualizes a temporal graph with time-based edge coloring
- `visualize_community_structure`: Visualizes community structure in the graph
- `visualize_temporal_communities`: Examines network evolution through multiple snapshots

## TGN Model Documentation

For detailed documentation on the Temporal Graph Network model, please see [TGN README](docs/README_TGN.md).

## Dataset

The project uses the Reddit Hyperlinks dataset, which represents connections between subreddit communities. Each edge represents a hyperlink from one subreddit to another, with timestamps indicating when the hyperlink was created.

## License

[Specify your license here]

## Contributors

[Your name and other contributors]
