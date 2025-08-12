# SHAC-CSTR: Short Horizon Actor-Critic for CSTR Control 

🎓 **Master's Thesis** | 🏭 **Process Control** | 🤖 **Reinforcement Learning**

A reinforcement learning approach for optimal control of Continuous Stirred Tank Reactors (CSTR) under time-varying electricity pricing.

📋 **[View Presentation](presentations/)** | 💻 **[Browse Code](source/)**

## Project Overview

This repository implements a modified Short Horizon Actor-Critic (SHAC) algorithm for controlling a CSTR system with the objective of minimizing operational costs while maintaining safe operation within constraint boundaries.

**Key Achievements:**
- ✅ Developed novel RL approach combining economic optimization with safety constraints
- ✅ Achieved significant cost reduction while maintaining operational safety
- ✅ Implemented real-time control under dynamic electricity pricing
- ✅ Demonstrated scalability across multiple reactor configurations

### Key Features

- **Multi-environment training**: Parallel environment sampling for improved data efficiency
- **Constraint handling**: Penalty-based approach for maintaining operational safety
- **Price-aware control**: Integration of time-varying electricity prices in decision making
- **Differentiable simulation**: Physics-based CSTR model using PyTorch for gradient computation

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd ma_shac_cstr

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
```

## Technologies Used

- **Python** - Primary programming language
- **PyTorch** - Deep learning framework
- **Reinforcement Learning** - Actor-Critic methods
- **Process Control** - Chemical reactor modeling
- **Optimization** - Economic objective functions

## Project Structure

```
ma_shac_cstr/
├── presentations/          # Thesis presentation materials
├── source/                # Core implementation
├── utils/                 # Utility functions
├── data/                  # Input data
├── models/               # Trained model checkpoints
├── notebooks/           # Analysis notebooks
└── tests/               # Unit tests
```

## Results

The implemented SHAC algorithm successfully demonstrates:
- Stable training convergence
- Effective constraint satisfaction
- Economic optimization under varying conditions
- Robust performance across different scenarios

---
