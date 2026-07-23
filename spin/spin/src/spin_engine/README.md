# Spin Engine Walkthrough

The core logic resides in `src/spin_engine`, organized into four main modules. Here is how they interact:

## 1. models
**Purpose**: Defines the physical system, including the lattice structure, spin dimensionality (discrete vs continuous), and the energy function.
- **Interactions**: Holds `spin_state` (the actual configuration) and `interaction_matrix` (the couplings).
- **Key Class**: `BaseSpinSystem` provides the interface. `IsingSystem` and `SphericalSystem` are concrete implementations.

## 2. interactions
**Purpose**: Generates the coupling tensors ($J_{ij}$) that define the connectivity of the system.
- **Interactions**: Used by `models` to initialize the system's Hamiltonian.
- **Key Classes**: `PeriodicNearestNeighborInteraction` (standard d-dimensional lattice), `DecayingInteraction`, `CurieWeissInteraction`.

## 3. dynamics
**Purpose**: Defines the rules for time evolution.
- **Interactions**: Takes a `system` and modifies its `spin_state` over time.
- **Key Classes**:
    - `MetropolisHastings`: Performs Monte Carlo sweeps. It proposes changes (flips or rotations), computes energy differences ($\Delta E$), and accepts/rejects based on thermal probabilities.
    - `Tracker`: Observes the simulation. It is passed to the `sweep` method to record measurements at specified intervals.

## 4. measurements
**Purpose**: Defines observables to be tracked.
- **Interactions**: Instantiated with a `system` reference. Called by `Tracker` during the simulation to compute values like total energy or magnetization.
- **Key Classes**: `Energy`, `Magnetization`, `MagneticSusceptibility`.

## Interaction Flow
1. **Define Interaction**: Generate an interaction matrix (e.g., Nearest Neighbor).
2. **Create Model**: Initialize a System (e.g., Ising) with the lattice size and interaction matrix.
3. **Setup Logic**: Instantiate `MetropolisHastings` dynamics with the system.
4. **Prepare Tracking**: Create a `Tracker` with desired `Measurements`.
5. **Run**: Execute `dynamics.sweep()`, passing the `Tracker`.
