# Spin System

![Python](https://img.shields.io/badge/Python-Scientific%20Computing-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Vectorized%20Simulation-orange)
![Monte Carlo](https://img.shields.io/badge/Method-Monte%20Carlo-brightgreen)
![Ising Model](https://img.shields.io/badge/Physics-Ising%20Model-purple)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20Experimental-lightgrey)


## Overview
**Spin System** is a modular, TensorFlow-based framework for simulating classical spin systems with an emphasis on performance, extensibility, and theoretical rigor. The framework leverages vectorized operations to efficiently handle large lattices and complex interaction topologies, enabling both physical simulations and experimentation with optimization-inspired computational paradigms.

The project is designed for researchers and practitioners working in statistical physics, complex systems, and quantum-inspired optimization.



## Project Objectives
- Provide a flexible and extensible framework for simulating classical spin models.
- Enable efficient large-scale simulations through TensorFlow-based vectorization.
- Support experimentation with spin-based formulations of hard optimization problems.
- Bridge concepts from statistical mechanics, analog computing, and computational complexity.



## Current Capabilities

### Spin Models
- **Ising Model**: Discrete spins  $s_i \in \{-1, 1\}$.
- **Spherical Model**: Continuous unit-vector spins.

### Interaction Structures
- Arbitrary pairwise coupling tensors.
- Built-in interaction schemes:
  - Periodic nearest-neighbor
  - Distance-decaying couplings
  - Curie–Weiss (mean-field)
  - Gaussian random couplings

### Dynamics
- **Metropolis–Hastings Monte Carlo** dynamics.
- Support for temperature schedules and annealing processes.

### Measurements and Observables
- Energy
- Magnetization
- Magnetic susceptibility  
- Real-time observable tracking during simulations



## Theoretical Background

### Spin Systems and Computational Complexity
- Many **NP-Hard** and **NP-Complete** problems can be mapped, in polynomial time, to the problem of finding the ground state of an Ising Hamiltonian.
- An **Ising Machine**, by converging to its ground state, effectively provides solutions to these mapped optimization problems.

### Ising–QUBO Equivalence
The Ising Hamiltonian can be transformed into a **Quadratic Unconstrained Binary Optimization (QUBO)** problem via the variable substitution:

$$
s_i \in \{-1, 1\} \quad \longleftrightarrow \quad x_i \in \{0, 1\}, \quad s_i = 1 - 2x_i
$$

This equivalence allows Ising-based solvers to address a broad class of combinatorial optimization problems.

### Quantum-Inspired and Analog Computing
- Spin system simulations can be interpreted as **quantum-inspired algorithms**, even when implemented on classical hardware.
- An **Ising Machine** is an analog computational device that exploits massively parallel and asynchronous dynamics to evolve toward low-energy states, in contrast to sequential digital algorithms.

### Simulated Annealing
- Under ideal conditions, an inverse-logarithmic temperature schedule guarantees convergence to the global minimum (Geman & Geman).
- This theoretical principle motivates **simulated annealing**, which underpins the primary optimization strategy implemented in this framework.



## Examples

The `examples/` directory contains reference scripts demonstrating the framework’s capabilities.

### 2D Ising Model Simulation
The script `examples/ising.py` performs a temperature sweep on a two-dimensional Ising lattice and records key observables to illustrate phase transition behavior.

#### Magnetization vs. Temperature
- High temperatures (low $\beta$) lead to rapid convergence toward zero magnetization.
- Low temperatures (high $\beta$) exhibit stable, non-zero magnetization states.

![Ising Evolution](examples/images/ising_evolution.png)

#### Critical Exponents
- The simulation reproduces the expected critical behavior of the 2D Ising model.
- A peak in susceptibility and other parameters is observed near the critical inverse temperature:
  $$
  \beta_c \approx 0.44
  $$

![Ising Susceptibility](examples/images/ising_observables_clean.png)



## Intended Audience
- Researchers in statistical and computational physics
- Practitioners exploring quantum-inspired optimization methods
- Students studying spin models, Monte Carlo methods, and complex systems
- Engineers interested in analog and non-von-Neumann computation paradigms



## Contact

**Lucas Gomes de Oliveira Corbanez**  
- Email: lucascorbanez@gmail.com  
- Institutional: lucas.gomes.oliveira@uel.br
