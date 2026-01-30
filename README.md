lhn_AI
A High-Performance Physics-Informed Machine Learning Library for Gravitational Lensing

lhn_AI is a custom-built, high-performance Physics-Informed Machine Learning (PIML) library implemented in C++17 with Python bindings, designed for solving partial differential equations (PDEs) arising in astrophysical simulations, with a primary focus on gravitational lensing.

The library combines physics-based constraints, implicit neural representations, and a forward-mode automatic differentiation engine to efficiently solve the Poisson equation for lensing potential in a mesh-free manner.
Core Contribution
This project introduces a physics-first neural computation framework that:
Directly learns continuous physical fields using neural networks
Computes first- and second-order spatial derivatives analytically during the forward pass
Avoids the memory overhead and instability of reverse-mode backpropagation for high-order PDE constraints
The approach is specifically optimized for gravitational lensing, where accurate Laplacians of the potential field are critical.

Scientific Background
In gravitational lensing, the lensing potential 𝜓(𝑥,𝑦) is governed by the Poisson equation:
∇2𝜓(𝑥,𝑦) = 2𝜅(𝑥,𝑦)
where 𝜅 is the convergence, representing the projected mass density of the lens.
The neural network learns the mapping:
(𝑥,𝑦) => 𝜓(𝑥,𝑦)

From the learned potential, physical observables are obtained via analytical differentiation:
Deflection angle:
𝛼(𝑥,𝑦) = ∇𝜓(𝑥,𝑦)
Reconstructed convergence:
𝜅(𝑥,𝑦) = 0.5*∇2𝜓(𝑥,𝑦)
Einstein rings and other lensing features emerge naturally from ray tracing using the learned potential.

Physics-Informed Learning Framework
The network is trained by minimizing a physics-informed loss:
L = Ldata +λPoisson∥ ∇2ψnet− 2κtrue∥^2 +λgrad∥ ∇ψnet ∥^2
All spatial derivatives are computed using custom forward-mode automatic differentiation, implemented at the C++ level.

Key Features
High-Performance Hybrid Architecture
Core computation written in C++17
Python interface via pybind11
Designed for scientific workloads, not general-purpose ML

Physics-Informed Neural Networks (PINNs)
Explicit PDE constraints embedded in the training objective
Mesh-free formulation using implicit neural representations

Forward-Mode Automatic Differentiation
Exact gradients and Laplacians computed during the forward pass
Memory-efficient and numerically stable for higher-order derivatives
Particularly suited for PDE-constrained learning

SIREN Architecture
Sinusoidal activation functions for modeling high-frequency spatial fields
Superior representation of sharp curvature and fine gravitational structures

Parallelized Training
Vectorized linear algebra via Eigen
OpenMP-based batch training for efficient CPU utilization
(train_batch_poisson.cpp)

Project Structure
lhn_AI/
├── include/lhn/
│   ├── physics/          # PINN core: nodes, layers, trainers
│   ├── core/             # Optimizers and utilities
│   ├── LinearRegression.h
│   └── LogisticRegression.h
├── src/
│   ├── bindings.cpp      # Pybind11 module definitions
│   ├── train_batch_poisson.cpp  # OpenMP-parallel PINN training
│   └── ...
├── lhn_AI/
│   ├── physics.py        # High-level Python API for lensing experiments
│   └── ...
├── extern/               # Third-party dependencies (Eigen, pybind11)
└── tests/                # Benchmarks and validation scripts

Installation
Prerequisites
Python ≥ 3.7
C++ compiler with C++17 support
OpenMP-enabled compiler (recommended)
Eigen3 (included or system-installed)

Build from Source
git clone https://github.com/nghiale0309/LHN_AI.git
cd lhn_AI
pip install -e .

Note: OpenMP support is required to achieve full training performance.

Usage Example: Gravitational Lensing PINN
from lhn_AI.physics import LensingExperiment

experiment = LensingExperiment(
    layers=[2, 64, 64, 1],
    w0=30.0,
    lr=1e-4,
    lambda_lens=1.0,
    lambda_poisson=1.0,
    batch_size=256
)

experiment.train(log_interval=500)

results = experiment.evaluate_grid(xmin=-2, xmax=2, n=256)
results["psi"]        -> lensing potential
results["laplacian"]  -> reconstructed convergence

Additional Modules
The library also includes optimized C++ implementations of:
Linear Regression
Logistic Regression
These modules serve as auxiliary components and benchmarking references.

MIT License
Author: Le Hieu Nghia
Institution: Ho Chi Minh City University of Industry (IUH)

This project is intended for scientific research and educational use in physics-informed machine learning and computational astrophysics.


Running the Project:
This section describes how to build the backend, run the gravitational lensing simulation, and evaluate both physical correctness and computational performance of the library.

1. Environment Setup
Before executing any experiment, the C++ backend must be compiled and installed into the current Python environment.

From the project root directory
pip install -e .
The -e (editable) mode allows modification of Python files without reinstallation.
Any change to C++ source files (.cpp, .h) requires re-running this command to recompile the backend.

2. Troubleshooting
ModuleNotFoundError
Ensure the command pip install -e . was executed in the project root directory
Verify that the active Python environment matches the one used for installation

OpenMP Issues (Windows)
Ensure Visual C++ Redistributable is installed
Confirm that OpenMP flags are enabled in setup.py:
MSVC: /openmp
GCC/Clang: -fopenmp

Low CPU Utilization
Check the environment variable:
OMP_NUM_THREADS=<number_of_cores>

Recommended Workflow
Run the gravitational lensing simulation to verify physical correctness
Execute the benchmark to evaluate computational performance
Modify network or loss parameters and repeat experiments

This workflow is designed for reproducible scientific experiments in physics-informed machine learning and computational astrophysics.