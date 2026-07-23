# AGENTS.md — Spin System

> **Purpose**: This document provides all context an AI coding agent needs to understand, navigate, and contribute to this project.

## Project Identity

**Spin System** (`spin_engine`) is a modular, TensorFlow-based framework for simulating classical spin systems using Monte Carlo methods. The long-term research goal is to explore **Spin Glass phase transitions** and **Topological Phases of Matter**.

- **Author**: Lucas Gomes de Oliveira Corbanez
- **License**: MIT
- **Python**: ≥ 3.9
- **Core dependency**: TensorFlow (all tensor operations, `tf.function` graph compilation, `tf.while_loop` for simulation loops)

---

## Repository Layout

```
spin-system/
├── src/spin_engine/          # Installable Python package (the engine)
│   ├── __init__.py
│   ├── README.md             # Internal walkthrough of module interactions
│   ├── models/               # Physical systems (Hamiltonians + state)
│   │   ├── base.py           # BaseSpinSystem — abstract base (tf.Module)
│   │   ├── ising.py          # IsingSystem — discrete spins {-1, +1}
│   │   ├── spherical.py      # SphericalSystem — continuous unit-vector spins
│   │   ├── wegner.py         # WegnerSystem — Z₂ lattice gauge theory
│   │   └── traveling_salesman.py  # TravelingSalesmanSystem — QUBO-mapped TSP
│   ├── interactions/         # Coupling tensor generators (J_ij)
│   │   ├── base.py           # Interaction — abstract base
│   │   └── standard.py       # PeriodicNearestNeighbor, Decaying, CurieWeiss, Gaussian
│   ├── dynamics/             # Time-evolution rules (MC updates)
│   │   ├── base.py           # Dynamics — abstract base
│   │   ├── metropolis_hastings.py  # MetropolisHastings — standard MC for Ising/Spherical
│   │   ├── traveling_salesman.py   # TravelingSalesmanDynamics — 2-opt / swap moves
│   │   └── tracker.py        # Tracker — records observables during sweeps
│   └── measurements/         # Observable definitions
│       ├── base.py           # Measurement — abstract base
│       ├── scalars.py        # Energy, Magnetization, MagneticSusceptibility
│       ├── correlations.py   # OverlapMatrix (Q_ab for replicas)
│       └── gauge.py          # Plaquette, WilsonLoop (stubs for Wegner)
├── examples/                 # Runnable demonstration scripts
│   ├── ising.py              # 2D Ising temperature sweep + susceptibility plot
│   ├── wegner.py             # Wegner annealing animation (Z₂ gauge theory)
│   ├── tsp.py                # Traveling Salesman optimization via simulated annealing
│   └── images/               # Generated plots and animations
├── tests/                    # pytest test suite
│   ├── test_models.py
│   ├── test_dynamics.py
│   ├── test_interactions.py
│   ├── test_measurements.py
│   ├── test_tracker.py
│   └── cross_checking/       # Validation against analytical results
├── notebooks/                # Jupyter notebooks (exploratory, gitignored)
├── pyproject.toml            # Project metadata, dependencies, setuptools config
├── requirements-dev.txt      # Pinned dev dependencies
└── MILESTONES.md             # Research roadmap and next steps
```

---

## Architecture Overview

The engine follows a **four-module pipeline** pattern:

```
Interactions ──▶ Models ──▶ Dynamics ──▶ Measurements
  (J_ij)        (H, state)   (MC steps)    (observables)
```

### 1. `models/` — Physical Systems
- `BaseSpinSystem(tf.Module, ABC)` holds `spin_state` (a `tf.Variable`), lattice geometry, and defines `compute_energy()`.
- Constructor takes `lattice_dim`, `lattice_length`, `lattice_replicas`, and an optional `initial_spin_state`.
- `spin_state` shape: `(replicas, L, L, ...)` for standard models; `(replicas, L, L, D)` for gauge models (extra dim for link directions).
- Subclasses must implement `initialize_state()` and `compute_energy()`.

### 2. `interactions/` — Coupling Tensors
- `Interaction(ABC)` defines `.generate(D, L) -> np.ndarray` producing a tensor of shape `(L,)*D*2`.
- These are passed to model constructors (e.g., `IsingSystem(interaction_matrix=...)`) — **not used by WegnerSystem**, which computes plaquette products directly.

### 3. `dynamics/` — Monte Carlo Updates
- `Dynamics(ABC)` requires `.step()` and `.sweep()`.
- `MetropolisHastings` proposes spin flips, computes ΔE, and accepts/rejects via the Boltzmann criterion.
- `TravelingSalesmanDynamics` uses constraint-preserving moves (column swaps, segment reversals).
- `Tracker` is an observer passed to `.sweep()` that records `Measurement` values at configurable granularity using `tf.TensorArray`.

### 4. `measurements/` — Observables
- `Measurement(ABC)` defines `.compute(spin_state, system) -> tf.Tensor`.
- Uses `_resolve()` pattern to accept either an explicit state or fall back to `self.system.spin_state`.
- **Scalars**: `Energy`, `Magnetization`, `MagneticSusceptibility` (variance across replicas).
- **Correlations**: `OverlapMatrix` — computes Q_ab = (1/N) Σ_i s_i^a s_i^b.
- **Gauge** (stubs): `Plaquette`, `WilsonLoop` — raise `NotImplementedError`.

---

## Key Conventions

### Tensor Shapes
- All states are batched over replicas: first axis is always `replicas`.
- Spatial dimensions follow: `(replicas, L_1, ..., L_D)`.
- Gauge models add a final direction axis: `(replicas, L_1, ..., L_D, D)`.
- Interaction matrices are `(L,)*D*2` — a full `N×N` tensor reshaped as `(L, L, ..., L, L, ...)`.

### TensorFlow Patterns
- Simulation loops use `tf.while_loop` inside `@tf.function`-decorated `sweep()` methods.
- Energy tracking uses `tf.Variable` to avoid recomputation.
- `tf.TensorArray` is used in `Tracker` for graph-mode-compatible recording.
- Many `@tf.function` decorators are currently commented out during development (marked with `# @tf.function`).

### Type Checking
- The project uses `TYPE_CHECKING` guards for circular import avoidance.
- Some files use `# type: ignore` for TensorFlow/matplotlib typing issues.
- `cast()` from `typing` is used to satisfy type checkers around TF tensors.

### Testing
- Tests live in `tests/` and run via `pytest`.
- Run: `python -m pytest tests/ -v`
- Tests cover models, dynamics, interactions, measurements, and tracker.

### Development Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Running Examples
```bash
venv/bin/python examples/ising.py       # 2D Ising phase transition sweep
venv/bin/python examples/wegner.py      # Z₂ gauge theory annealing animation
venv/bin/python examples/tsp.py         # TSP optimization
```

> **AGENT ADDENDUM:** When executing scripts or running commands, AI agents MUST explicitly use the python interpreter from the virtual environment (e.g., `venv/bin/python` or `venv/bin/pytest`) instead of the global `python` command, as the global environment may lack required dependencies.

---

## Physics Context

### Current Scope
The framework currently implements:
- **Ising Model**: Discrete Z₂ spins on a periodic lattice. The 2D Ising model is the canonical example of a continuous phase transition with known critical temperature β_c ≈ 0.4407 (Onsager).
- **Spherical Model**: Continuous spins with optional global spherical constraint Σ s_i² = N.
- **Wegner Model (Z₂ Lattice Gauge Theory)**: Spins live on lattice *links*. Energy is the Wilson action — sum over plaquette products. This is a model of topological order with no local order parameter.
- **TSP Mapping**: Demonstrates the Ising–QUBO equivalence by encoding the Traveling Salesman Problem as a spin Hamiltonian.

### Research Direction
The project aims to explore two major frontiers:

1. **Spin Glass Phase Transitions**
   - Disordered systems with random couplings (frustration).
   - Edwards-Anderson and Sherrington-Kirkpatrick models.
   - Replica symmetry breaking, Parisi order parameter.
   - The overlap matrix `OverlapMatrix` measurement is already in place as infrastructure for this.

2. **Topological Phases of Matter**
   - The Wegner model is the simplest lattice gauge theory exhibiting topological order.
   - Wilson loops serve as non-local order parameters (area law vs. perimeter law).
   - Extensions toward Kitaev toric code and topological entanglement entropy.

---

## Important Notes for AI Agents

1. **Do not break the pipeline pattern.** New models should extend `BaseSpinSystem`, new dynamics should extend `Dynamics`, new measurements should extend `Measurement`.
2. **Respect the replica dimension.** All operations must handle the batch dimension (axis 0 = replicas).
3. **TensorFlow graph mode matters.** Code inside `sweep()` runs in `tf.function` context. Avoid Python-side effects, NumPy calls, or eager-only operations in these paths.
4. **The gauge measurements (`Plaquette`, `WilsonLoop`) are stubs.** They raise `NotImplementedError` and need proper implementation — this is a known priority.
5. **Interaction matrices are not used by WegnerSystem.** Gauge models define their own coupling structure via plaquettes. Don't try to pass interaction matrices to gauge models.
6. **Keep physics rigorous.** This is a research tool. Naming should follow established physics conventions (β for inverse temperature, J for couplings, H for Hamiltonian, etc.).
7. **Tests are essential.** Any new feature should include corresponding tests in `tests/`. Cross-check against analytical results when available.

---

## Simulation Convergence & Best Practices

When running scripts or executing phase transition sweeps, you MUST adhere to the following rules to ensure both computational efficiency and physical accuracy:

1. **Static Graph Management**: Instantiate the physical `System`, the `Dynamics` simulator, and the `Tracker` **ONCE outside** the temperature/beta loop. Re-instantiating them inside the loop destroys state persistence and triggers catastrophic `@tf.function` recompilations.
2. **Dynamic Sweep Scaling**: `sweep_length` represents individual spin flip proposals, not full lattice sweeps. To ensure equal equilibration, `sweep_length` must scale with lattice volume $N = L^D$. Always define `sweep_length = sweeps * N` (where `sweeps` $\ge 2000$).
3. **Simulated Annealing & State Persistence**: The lattice state must transition continuously between temperatures to remain in thermodynamic equilibrium.
   - **For Ferromagnets (Ising)**: Avoid Kibble-Zurek defect trapping by sweeping *cold-to-hot* (reverse $\beta$ loop: start at $\beta_{max}$ with `initial_magnetization=1.0`).
   - **For Spin Glasses (Edwards-Anderson)**: Traverse the rugged energy landscape via standard simulated annealing *hot-to-cold* (ascending $\beta$ loop: start at $\beta_{min}$ with `initial_magnetization=0.0`).
4. **Lower Critical Dimensions ($d_l$)**: Ensure the simulation geometry supports the expected phase transition. For example, the **Edwards-Anderson Spin Glass** only exhibits a finite-temperature phase transition ($T_c > 0$) in $3D$ or higher. In $2D$, the transition occurs strictly at $T=0$, meaning observables will only grow monotonically without a finite-$T$ peak.
