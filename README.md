# Community Detection and Link Prediction with Temporal Graph Networks

## Project Overview

This project implements two variants of Temporal Graph Network (TGN) architectures for community detection and link prediction in dynamic graphs. The implementation is based on the original TGN paper from Twitter Research: [Temporal Graph Networks for Deep Learning on Dynamic Graphs](https://arxiv.org/abs/2006.10637).

The project provides:

1. **Standard TGN Model**: Complete implementation of the original TGN architecture with configurable temporal attention mechanisms.
2. **Decay TGN Model**: Enhanced TGN with decay-based temporal attention that models how the importance of historical interactions decreases over time.
3. **Comparative Analysis**: Framework for comparing the performance of both models on community detection and link prediction tasks.
4. **Graph Visualization**: Comprehensive visualization capabilities to analyze temporal evolution of communities and network structures.

## Installation and Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mohak34/CommDec-DBTA
   cd CommunityDetection
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Standard TGN Model

The main implementation is available in `src/enhanced_tgn.py`. This file contains the complete implementation of the standard Temporal Graph Network with the Reddit dataset.

### Running the Decay TGN Model

The enhanced Decay TGN model with temporal attention decay is implemented in `src/decay_tgn.py`. This model includes:

- Configurable decay factors for temporal attention
- Memory modules with decay-based temporal attention
- Adaptive decay mechanisms for learning temporal patterns

### Comparative Analysis

To compare both models:

1. Use the standard TGN implementation in `notebooks/Enhanced_TGN.ipynb`
2. Use the Decay TGN implementation with different decay factors via `src/decay_tgn.py`
3. Run experiments with various decay factors to analyze their impact on performance

## Key Components

### Standard TGN Model (`src/enhanced_tgn.py`)

The standard TGN implementation includes:

- Temporal attention mechanisms for modeling dynamic graph interactions
- Memory modules for storing and updating node states over time
- Message passing and aggregation for community detection
- Link prediction capabilities

### Decay TGN Model (`src/decay_tgn.py`)

The Decay TGN model extends the standard TGN with:

- Configurable decay factors for temporal attention weights
- Adaptive decay mechanisms that learn optimal decay rates
- Enhanced memory modules with temporal decay consideration
- Comprehensive decay analysis and visualization tools

### Model Comparison

Both models can be evaluated on:

- **Community Detection**: Identifying clusters of related nodes in temporal graphs
- **Link Prediction**: Predicting future connections between nodes
- **Temporal Pattern Analysis**: Understanding how relationships evolve over time

## Dataset

The project uses the Reddit Hyperlinks dataset, which represents connections between subreddit communities. Each edge represents a hyperlink from one subreddit to another, with timestamps indicating when the hyperlink was created.
