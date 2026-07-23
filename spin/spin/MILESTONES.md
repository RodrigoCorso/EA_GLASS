# MILESTONES.md — Spin System Research Roadmap

> Tracking improvements, next steps, and long-term goals toward exploring **Spin Glass Phase Transitions** and **Topological Phases of Matter**.

---

## Milestone 1 — Spin Glass Infrastructure ✅

**Goal**: Enable simulation and analysis of disordered spin systems with frustration.

### 1.1 Edwards-Anderson Model
- [x] Create `EdwardsAndersonSystem` in `models/edwards_anderson.py`
  - Ising spins on a *d*-dimensional lattice with **quenched random couplings** J_ij drawn from ±J or Gaussian distribution.
  - Reuse `BaseSpinSystem`; accept a pre-generated random interaction matrix from `GaussianInteraction` or a new `BinaryRandomInteraction(±J)`.
  - `compute_energy()` identical to `IsingSystem` but semantically distinct (no external field needed by default).
- [x] Add a `BinaryRandomInteraction` class to `interactions/standard.py`
  - `.generate(D, L)` returns J_ij ∈ {-J, +J} drawn uniformly, symmetric, with zero diagonal.
- [x] Write tests in `tests/test_edwards_anderson.py`
  - Verify energy computation matches hand-calculated values on a small 2×2 lattice.
  - Verify that the coupling matrix is symmetric with zero diagonal.

### 1.2 Sherrington-Kirkpatrick (SK) Model
- [x] Create `SherringtonKirkpatrickSystem` in `models/sk.py`
  - Fully connected Ising model with Gaussian random couplings J_ij ~ N(0, J/√N).
  - Can reuse `IsingSystem` with a `GaussianInteraction` whose std is scaled by 1/√N, or create a dedicated class for clarity.
  - Should support replica-wise computation for overlap analysis.
- [x] Add example script `examples/sk_glass.py`
  - Simulate the SK model across temperatures.
  - Plot energy and magnetization.

### 1.3 Overlap Distribution P(q)
- [x] Implement `OverlapDistribution` measurement in `measurements/correlations.py`
  - Uses the existing `OverlapMatrix` to extract off-diagonal elements q_ab.
  - Returns a 1D tensor of upper-triangular overlaps (replicas*(replicas-1)/2 values).
- [x] Implement `ParisiOverlapParameter` measurement
  - Computes ⟨q²⟩ - ⟨|q|⟩² as an indicator of replica symmetry breaking (RSB).
- [x] Add plotting utilities for P(q) visualization
  - KDE-based P(q) plots in `examples/sk_glass.py`, `examples/ea_glass.py`, and `examples/spin_glass_phase_diagram.py`.

### 1.4 Spin Glass Order Parameter & Phase Diagram
- [x] Implement `SpinGlassOrderParameter` (q_EA = [1/N Σ ⟨s_i⟩²]_avg)
  - Stateful measurement using `tf.Variable` accumulation over MC steps.
  - Includes `reset()` method for multi-temperature sweeps.
- [x] Create `examples/spin_glass_phase_diagram.py`
  - Temperature sweep showing paramagnetic → spin glass transition.
  - Plots q_EA vs T and P(q) distributions.
- [x] Add χ_SG = N⟨q²⟩ susceptibility analysis to `examples/sk_glass.py` and `examples/ea_glass.py`
  - Multi-dimension comparison (SK: D=1,2; EA: D=2,3).
  - Susceptibility peak detection to locate T_c numerically.
- [x] Create analytical documentation in `docs/spin_glass_analytics.md`
  - Derivation of T_c for SK and EA models, replica method, RSB, critical exponents.

**How to achieve**:
1. Start with the EA model since `GaussianInteraction` already exists — just wire it up.
2. The `OverlapMatrix` measurement is already implemented — extend it to compute full P(q).
3. The SK model is a special case of EA with all-to-all connectivity, so implementation can share infrastructure.
4. Use many replicas (`lattice_replicas ≥ 64`) to get statistically meaningful overlap distributions.

---

## Milestone 2 — Topological Phase Measurements ⬜

**Goal**: Complete the Wegner model tooling and implement non-local order parameters for topological phases.

### 2.1 Implement Plaquette Measurement
- [ ] Complete `Plaquette.compute()` in `measurements/gauge.py` (currently raises `NotImplementedError`)
  - Should compute the average plaquette value ⟨P⟩ = (1/N_p) Σ_p ∏_{l∈p} σ_l.
  - Leverage `WegnerSystem.compute_all_plaquettes()` which already returns per-plaquette values.
  - Return per-replica scalar.
- [ ] Test against known limits: ⟨P⟩ → 1 at β → ∞ (ordered), ⟨P⟩ → 0 at β → 0 (disordered).

### 2.2 Implement Wilson Loop Measurement
- [ ] Complete `WilsonLoop.compute()` in `measurements/gauge.py`
  - Compute the expectation value of rectangular R×T Wilson loops.
  - For a given loop, trace the product of link variables around the contour.
  - Support variable loop sizes.
- [ ] Implement **area law vs. perimeter law** analysis
  - In the confined phase: W(R,T) ~ exp(-σ·R·T) → area law.
  - In the deconfined phase: W(R,T) ~ exp(-μ·(R+T)) → perimeter law.
  - Provide utility to extract the string tension σ from Wilson loop data.

### 2.3 Wegner Model Phase Transition
- [ ] Create `examples/wegner_phase_transition.py`
  - Temperature sweep of the Wegner model in 2D and 3D.
  - Plot ⟨P⟩ vs. β to identify the confinement-deconfinement transition.
  - In 2D: no finite-temperature phase transition (Elitzur's theorem) — verify this.
  - In 3D: genuine phase transition exists — locate β_c.
- [ ] Implement `MetropolisHastings` support for gauge models
  - Currently `flip_spins` assumes site-based spins. Need a `flip_links` variant for gauge models that flips individual link variables.
  - The Wegner example currently uses the standard `MetropolisHastings` — verify correctness or create `GaugeDynamics`.

### 2.4 Topological Entanglement Entropy (Stretch Goal)
- [ ] Implement Kitaev-Preskill or Levin-Wen construction for extracting topological entanglement entropy S_topo from the ground state.
- [ ] This requires defining subsystem boundaries and computing Rényi entropies — significant new infrastructure.

**How to achieve**:
1. `Plaquette` is low-hanging fruit — `compute_all_plaquettes()` already does the hard work; just average.
2. `WilsonLoop` requires building rectangular loop products by chaining `tf.roll` and multiplication along the loop path — same technique used in `compute_all_plaquettes()` but for larger loops.
3. The 3D Wegner model just needs `lattice_dim=3` — the `WegnerSystem` already supports arbitrary dimensions.
4. Ensure `MetropolisHastings.flip_spins` works correctly with the `(replicas, L, ..., L, D)` tensor shape of gauge models.

---

## Milestone 3 — Dynamics & Algorithm Improvements ⬜

**Goal**: Improve simulation efficiency and physical realism.

### 3.1 Annealing Inside Sweep
- [ ] Implement temperature scheduling within `sweep()` to avoid TF graph retracing
  - Currently, each temperature requires a new `sweep()` call → new graph trace.
  - Pass a `beta_schedule` tensor (like the Wegner example does manually) as part of the standard `Dynamics` interface.
  - There is a `TODO` in `dynamics/base.py` about this.

### 3.2 Cluster Algorithms
- [ ] Implement **Wolff single-cluster algorithm** in `dynamics/wolff.py`
  - Dramatically reduces critical slowing down near phase transitions.
  - Essential for accurate measurements near β_c.
- [ ] Implement **Swendsen-Wang multi-cluster algorithm** as an alternative.
- [ ] These algorithms are specific to Ising/Potts-type models — enforce this via type checks.

### 3.3 Parallel Tempering (Replica Exchange)
- [ ] Implement `ReplicaExchange` dynamics
  - Run replicas at different temperatures simultaneously.
  - Periodically propose swaps of configurations between adjacent temperatures.
  - Critical for spin glasses where the energy landscape is rugged.
- [ ] This naturally leverages the existing replica infrastructure (`lattice_replicas`).

### 3.4 Enable `@tf.function` Compilation
- [ ] Audit and re-enable all commented-out `@tf.function` decorators.
  - `compute_energy()` in Ising/Spherical models.
  - `step()` in MetropolisHastings.
  - Ensure no Python-side effects break graph compilation.
- [ ] Benchmark performance improvement from graph compilation.

### 3.5 Quenched Disorder Averaging — Batched on GPU
- [ ] Don't loop disorder samples in Python (slow, re-traces nothing parallel) — batch them the same way `lattice_replicas` is already batched, by adding a `disorder_samples` (D) axis to the interaction tensor.
- [ ] Add a `disorder_samples` constructor param to `EdwardsAndersonSystem` and `SherringtonKirkpatrickSystem`.
  - Generate a stacked interaction tensor of shape `(D, N, N)` via D calls to `BinaryRandomInteraction`/`GaussianInteraction.generate()` with D distinct seeds, `np.stack`ed.
  - Keep `lattice_replicas` = `D × replicas_per_sample` (the existing flat batch axis) so `Dynamics`, `Tracker`, and disorder-agnostic measurements (`Energy`, `Magnetization`) need **zero changes** — same reason replica-parallelism was free.
- [ ] Rewrite `compute_energy()`/`compute_delta_energy()` in both models to use a **batched matmul/gather** instead of one shared matmul:
  - Reshape spins to `(D, R, N)`, `J` to `(D, N, N)`, contract with `tf.matmul(J, spin_transposed)` (broadcasts over the leading `D` batch dim) instead of `spin_flat @ J_flat`.
  - Same idea for the `tf.gather` of `J` rows in `compute_delta_energy` — gather per-disorder-slice, not from one shared `J_flat`.
- [ ] **Catch — must ship together, not as a follow-up**: `MagneticSusceptibility` and the whole Overlap family (`OverlapMatrix`, `OverlapDistribution`, `ParisiOverlapParameter`) reduce/compare *across the entire replica axis*. If `D` and `R` are silently flattened into that axis:
  - `MagneticSusceptibility` would conflate disorder-to-disorder fluctuation into what should be a purely thermal variance (biased high).
  - The Overlap family would compute overlaps between replicas from *different* J realizations — physically meaningless.
  - Add disorder-aware variants that reshape back to `(D, R, ...)` and reduce/compare only within each `R` block before touching the `D` axis.
- [ ] Combine per-disorder thermal error (5.4, jackknife over `R` within each disorder slice) with disorder-to-disorder variance (across `D`) in quadrature for the final error bar.
- [ ] Cost is now GPU memory/compute (`O(D·N²)` for the interaction tensor, `O(D·R·N²)` per energy eval), not wall-clock multiplication from a Python loop — same tradeoff replicas already made.

**How to achieve**:
1. Start with the annealing-inside-sweep refactor — it's a TODO already flagged in the code.
2. Wolff algorithm is the highest-impact addition for phase transition studies.
3. Parallel tempering is essential for spin glass research — without it, the system gets trapped in metastable states.
4. `@tf.function` audit should be done incrementally, testing each decorated function.

---

## Milestone 4 — New Models & Extensions ⬜

**Goal**: Expand the range of physical systems the engine can simulate.

### 4.1 Potts Model
- [ ] Create `PottsSystem` in `models/potts.py`
  - q-state generalization of Ising: spins take values in {0, 1, ..., q-1}.
  - Energy: H = -J Σ δ(s_i, s_j) (Kronecker delta interaction).
  - One-hot encoding for TF compatibility.
- [ ] Implement corresponding dynamics (single-spin Metropolis + Wolff cluster).

### 4.2 XY Model
- [ ] Create `XYSystem` in `models/xy.py`
  - Continuous planar spins: s_i = (cos θ_i, sin θ_i).
  - Energy: H = -J Σ cos(θ_i - θ_j).
  - Exhibits the Berezinskii-Kosterlitz-Thouless (BKT) transition in 2D.
- [ ] Implement vortex detection and counting as measurements.

### 4.3 Heisenberg Model
- [ ] Create `HeisenbergSystem` in `models/heisenberg.py`
  - 3-component unit vector spins on S².
  - Extends the existing `SphericalSystem` concept.
- [ ] Add over-relaxation dynamics for efficiency.

### 4.4 Kitaev Toric Code
- [ ] Create `ToricCodeSystem` in `models/toric_code.py`
  - Qubits on edges of a square lattice.
  - Hamiltonian: H = -Σ_v A_v - Σ_p B_p (vertex and plaquette operators).
  - Ground state degeneracy depends on topology.
- [ ] Implement anyonic excitation detection (e-particles and m-particles).

**How to achieve**:
1. Potts and XY are natural generalizations of the existing Ising/Spherical models.
2. The `BaseSpinSystem` abstraction should accommodate these without modification — just implement `initialize_state()` and `compute_energy()`.
3. The Toric Code builds on the gauge theory infrastructure from Wegner.

---

## Milestone 5 — Analysis & Visualization ⬜

**Goal**: Build reusable analysis tools for phase transition studies.

### 5.1 Finite-Size Scaling
- [ ] Implement utilities for running simulations at multiple lattice sizes.
- [ ] Binder cumulant: U_4 = 1 - ⟨m⁴⟩ / (3⟨m²⟩²) — crossing point gives β_c.
- [ ] Data collapse analysis to extract critical exponents (ν, β, γ).

### 5.2 Autocorrelation Analysis
- [ ] Implement integrated autocorrelation time τ_int measurement.
- [ ] Implement binning/blocking analysis: variance of the per-bin mean vs. bin size, used to confirm the estimate has plateaued (i.e. `granularity` and burn-in actually decorrelate samples).
- [ ] Use this to determine proper thermalization and decorrelation periods.
- [ ] Essential for error estimation in Monte Carlo — run as a diagnostic against existing cached `Tracker.history` data before trusting any error bars built on top (5.4).

### 5.3 Visualization Toolkit
- [ ] Standardize plotting utilities across examples.
- [ ] Animated lattice visualization for all model types (extend the Wegner animation pattern).
- [ ] Phase diagram plotting with error bars and finite-size scaling annotations (depends on 5.4).

### 5.4 Resampling-Based Error Propagation (Jackknife/Bootstrap)
- [ ] Implement generic `jackknife(samples, estimator_fn, axis=0)` and `bootstrap(samples, estimator_fn, n_resample, axis=0)` utilities.
  - Resample over the `replicas` axis — each replica is already an independent thermal chain under the same quenched J, so this is free (no extra simulation).
  - Needed because the observables we care about are nonlinear functions of primary moments (C_v = β²N(⟨E²⟩-⟨E⟩²), χ_SG = βN⟨q²⟩, Binder cumulant, Parisi parameter ⟨q²⟩-⟨|q|⟩²) — naive Gaussian error propagation through these is biased; jackknife/bootstrap handles it correctly with one implementation reused across all of them.
- [ ] Retrofit one example first (`examples/ea_observables.py`) to emit `(value, error)` pairs instead of bare floats, and switch `plt.plot` → `plt.errorbar`.
- [ ] Caveat: this only captures replica-sampling error, not within-chain autocorrelation (5.2) or disorder variance (5.5) — those are separate, complementary checks.

**How to achieve**:
1. Binder cumulant requires adding ⟨m²⟩ and ⟨m⁴⟩ measurements — straightforward extensions of the `Measurement` class.
2. Autocorrelation and binning can be computed post-hoc from `Tracker.history` data — pure NumPy, no `Measurement`/`Tracker` changes needed.
3. Consider a `spin_engine.analysis` subpackage for these tools (`autocorrelation.py`, `resampling.py`) — all post-hoc on `tracker.history[name].numpy()`, so the graph-mode restrictions on `sweep()` don't apply.
4. Suggested order: 5.2 (validate current granularity/burn-in) → 5.4 (cheap, reuses existing replica chains) → 3.5 (expensive, only once 5.4 is in place).

---

## Priority Order

| Priority | Milestone | Rationale |
|----------|-----------|-----------|
| 🔴 High | 2.1–2.2 Gauge measurements | Unblock topological phase research (stubs exist) |
| ✅ Done | 1.1–1.2 Spin glass models | Core research goal, infrastructure mostly exists |
| 🟡 Medium | 3.1–3.2 Annealing + Wolff | Performance bottleneck for accurate phase transition studies |
| ✅ Done | 1.3–1.4 Overlap & phase diagram | Requires spin glass models first |
| 🟡 Medium | 2.3 Wegner phase transition | Requires gauge measurements first |
| 🟢 Lower | 3.3–3.4 Parallel tempering + TF | Quality-of-life improvements |
| 🟢 Lower | 4.x New models | Expands scope but not blocking core goals |
| 🟢 Lower | 5.x Analysis toolkit | Can be built incrementally as needed |

---

## Status Legend

- ⬜ Not started
- 🟨 In progress
- ✅ Complete
