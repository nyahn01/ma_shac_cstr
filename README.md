# SHAC-CSTR: Soft Hierarchical Actor-Critic for CSTR Control

A reinforcement learning approach for optimal control of Continuous Stirred Tank Reactors (CSTR) under time-varying electricity pricing.

## Overview

This repository implements a modified Soft Hierarchical Actor-Critic (SHAC) algorithm for controlling a CSTR system with the objective of minimizing operational costs while maintaining safe operation within constraint boundaries.

### Key Features

- **Multi-environment training**: Parallel environment sampling for improved data efficiency
- **Constraint handling**: Penalty-based approach for maintaining operational safety
- **Price-aware control**: Integration of time-varying electricity prices in decision making
- **Differentiable simulation**: Physics-based CSTR model using PyTorch for gradient computation

## Project Structure

```
ma_shac_cstr/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── main.py                  # Main training script
├── config.py               # Configuration management (TODO)
├── data/                   # Input data
│   └── consecutive_prices.xlsx
├── source/                 # Core implementation
│   ├── __init__.py
│   ├── agent.py           # RL Agent implementation
│   ├── critic.py          # Value function network
│   ├── policy.py          # Policy network
│   ├── environment.py     # CSTR environment wrapper
│   ├── cstr.py           # Physics-based CSTR model
│   └── memory.py         # Experience replay buffer
├── utils/                 # Utility functions
│   ├── miscellaneous.py  # Helper functions
│   └── plotting.py       # Visualization utilities
├── models/               # Saved model checkpoints
├── notebooks/           # Analysis notebooks
│   ├── analyzeModel_shac2.ipynb
│   └── analyzePrices.ipynb
├── tests/              # Unit tests (TODO)
├── docs/               # Documentation (TODO)
└── results/            # Experimental results (TODO)
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ma_shac_cstr
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the basic training:
```bash
python main.py
```

### Configuration

The main hyperparameters can be modified in `main.py`:
- `num_train_episodes`: Number of training episodes
- `num_environments`: Parallel environments for sampling
- `episode_length`: Length of each episode
- `learning_rate_policy`: Learning rate for policy network
- `learning_rate_critic`: Learning rate for critic network

### Monitoring Training

Training progress is visualized in real-time showing:
- Policy and critic loss curves
- Penalty and reward progression
- Wall-clock time performance

Results are automatically saved to `models/` directory.

## Algorithm Details

### SHAC Implementation

The implementation uses a modified Actor-Critic architecture with:
- **Actor Network**: Policy π(a|s) that outputs continuous actions
- **Critic Network**: Value function V(s) for state evaluation
- **Target Network**: Soft-updated target critic for stable learning
- **GAE**: Generalized Advantage Estimation for variance reduction

### CSTR Environment

The environment models a continuous stirred tank reactor with:
- **State variables**: Concentration (c) and Temperature (T)
- **Control inputs**: Production rate (ρ) and Coolant flow rate (Fc)
- **Constraints**: Safe operating regions for state and storage
- **Objective**: Minimize costs while satisfying constraints

### Key Equations

CSTR dynamics:
```
dc/dt = (1-c)ρ/V - c*k*exp(-N/T)
dT/dt = (Tf-T)ρ/V + c*k*exp(-N/T) - Fc*αc*(T-Tc)
```

Reward function:
```
r = tanh(0.01 * (cost_nominal - cost_actual)) - penalty_constraints
```

## Experimental Results

<!-- TODO: Add experimental results once available -->

### Baseline Performance
- Mean reward before training: [TBD]
- Mean reward after training: [TBD]
- Training time: [TBD]

### Ablation Studies
<!-- TODO: Add ablation studies -->

## Contributing

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run tests:
   ```bash
   pytest tests/
   ```

3. Code formatting:
   ```bash
   black source/ utils/ tests/
   flake8 source/ utils/ tests/
   ```

### Commit Convention

Use conventional commits:
- `feat: add new feature`
- `fix: bug fix`
- `docs: documentation changes`
- `style: code style changes`
- `refactor: code refactoring`
- `test: add tests`
- `chore: maintenance tasks`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

- Author: [Your Name]
- Email: [Your Email]
- Institution: [Your Institution]

## Acknowledgments

- [Any acknowledgments]

## TODO

- [ ] Add comprehensive unit tests
- [ ] Implement configuration management system
- [ ] Add experiment tracking (wandb/mlflow)
- [ ] Create documentation website
- [ ] Add more baseline comparisons
- [ ] Implement hyperparameter tuning
- [ ] Add model interpretability tools