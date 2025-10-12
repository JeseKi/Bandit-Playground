# Bandit Lib

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Python library for simulating and evaluating Multi-Armed Bandit algorithms.

## Introduction

Bandit Lib provides a flexible set of tools for implementing, testing, and comparing different multi-armed bandit strategies. Whether you are exploring Bandit algorithms in academic research, online learning systems, or recommendation systems, this library can help you quickly set up experiments and visualize the results.

## Features

- **Multiple Algorithm Implementations**: Built-in implementations of several classic Bandit algorithms:
  - **Greedy**: The greedy algorithm.
  - **UCB (Upper Confidence Bound)**: The Upper Confidence Bound algorithm.
  - **Thompson Sampling**: Thompson Sampling.
- **Flexible Environment Configuration**:
  - Supports both **static and dynamic environments**. In a dynamic environment, the reward probabilities of the arms can change over time.
  - Customizable number of arms and reward mechanisms.
- **Efficient Batch Training**:
  - Uses `multiprocessing` to run multiple simulations in parallel for robust statistical results.
- **Rich Performance Metrics**:
  - Automatically calculates and records key metrics such as `regret`, `reward`, optimal arm selection rate, etc.
- **Powerful Visualization**:
  - Generates interactive charts using Plotly, making it easy to visualize metrics from a single experiment or compare the performance of multiple algorithms.

## Project Structure

```
bandit/
├── docs/                   # Documentation
├── bandit_lib/             # Core library code
│   ├── agents/             # Bandit agents and algorithm implementations
│   ├── env/                # Simulation environments
│   ├── runner/             # High-level tools for running experiments
│   └── utils/              # Utility functions, including logging and visualization
├── examples/               # Jupyter Notebook examples
└── README.md
```

## Installation
<div style="background:#f0f8ff; border-left:5px solid #4682b4; padding:10px; margin:10px 0;">
  <strong>💡 Note</strong>
  <p>Python: 3.10.18</p>
  <p>OS: Ubuntu 22.04 LTS Desktop</p>
</div>

You can install the dependencies using poetry:

```bash
pip install poetry==2.2.1
poetry install
```

## Quick Start

You can refer to this [document](docs/quick_start.md).

## Examples

The `examples/` directory contains more detailed Jupyter Notebook examples that show how to:
- Run different algorithms and compare their performance.
- Evaluate algorithms in a dynamic environment.
- Customize and save visualization results.

## License

This project is licensed under the [MIT License](LICENSE).
