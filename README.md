# ConDiSim: Conditional Diffusion Models for Simulation-Based Inference

This repository contains the implementation of conditional diffusion models for simulation-based inference tasks.

## Overview

The code implements a diffusion-based approach for posterior inference across multiple benchmark tasks including:
- Two Moons
- Gaussian Linear
- Gaussian Mixture Model
- Bernoulli GLM
- SIR (Susceptible-Infected-Recovered)
- Lotka-Volterra
- SLCP
- Hodgkin-Huxley
- Vilar Oscillator

## Repository Structure

```
.
├── core/                  # Core diffusion model implementation
│   ├── model_architecture.py    # Neural network architecture
│   ├── model_train.py          # Training loop
│   ├── noise_scheduler.py      # Diffusion noise scheduling
│   ├── sampling.py             # Posterior sampling
│   └── train_utils.py          # Training utilities
├── plots/                 # Plotting scripts for each task
├── hh/                    # Hodgkin-Huxley specific code
├── vilar/                 # Vilar oscillator specific code
├── ECDF/                  # SBC diagnostic plots
├── main.py                # Main training script
├── utils.py               # General utilities
└── metrics.py             # Evaluation metrics
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- SciPy
- sbibm (for benchmark tasks)

## Usage

Run the main training script:
```bash
python main.py
```

Generate plots for a specific task:
```bash
python plots/<task_name>.py
```

## Citation

If you use this code, please cite our paper.
