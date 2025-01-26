# Metro Optimization Software

![Metro Optimization](https://via.placeholder.com/800x200.png?text=Metro+Optimization+Software)

## Overview
The **Metro Optimization Software** is designed to simulate, analyze, and optimize subway systems. By leveraging advanced graph theory and machine learning techniques, the software evaluates subway layouts for efficiency in passenger flow and congestion management. 

This project combines concepts from applied mathematics, optimal control, and graph neural networks (GNNs) to develop scalable solutions for urban transit challenges.

## Key Features

- **Graph Representation of Subway Maps**: Models subway systems as graphs using NetworkX, focusing on node classes and graph structure.
- **Passenger Flow Simulation**: Simulates passenger transport to assess congestion and flow.
- **Graph Neural Network (GNN) Integration**: Approximates efficiency scores to bypass computationally expensive simulations.
- **Optimization Algorithms**: Identifies optimal layouts to minimize congestion and maximize efficiency.

## Installation

### Prerequisites
- Python 3.8+
- Recommended: Virtual environment (e.g., `venv` or `conda`)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/parkercarrus/metro.git
   cd metro
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```bash
   streamlit run app.py
   ```

4. Access the app at [localhost:8501](http://localhost:8501).

## Usage

1. **Upload Data**: Import a subway map or graph representation (e.g., CSV, JSON).
2. **Run Simulation**: Use `simulation.py` to test passenger flow scenarios.
3. **Optimize Layout**: Execute optimization scripts to improve system efficiency.
4. **Visualize Results**: Use the Streamlit interface to view metrics and maps interactively.

## Example

```python
from metro.simulation import simulate_flow
from metro.optimization import optimize_layout

# Load your graph
graph = load_graph("data/subway_map.json")

# Simulate passenger flow
simulation_results = simulate_flow(graph)

# Optimize layout
optimized_graph = optimize_layout(graph, simulation_results)
```

## Technologies Used

- **Programming Languages**: Python
- **Libraries**:
  - `NetworkX` for graph analysis
  - `PyTorch Geometric` for GNN-based approximations
  - `Streamlit` for interactive visualization
- **Concepts**: Graph Theory, Machine Learning, Optimization

## Roadmap

- [ ] Enhance GNN training pipeline
- [ ] Add support for real-time passenger data
- [ ] Expand visualizations with heatmaps and flow animations
- [ ] Deploy web app on a scalable cloud platform
