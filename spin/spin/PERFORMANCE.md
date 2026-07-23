# PERFORMANCE.md — Spin System Performance Analysis & Optimization Roadmap

> **Goal**: Document the key performance bottlenecks in the spin engine, propose concrete optimizations, and estimate their impact.
>
> This document is intended as a **companion to MILESTONES.md**, focusing exclusively on computational performance.

---

## Current Performance Landscape

### Hardware Context
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, compute capability 8.9)
- **Framework**: TensorFlow with `@tf.function` graph compilation and `tf.while_loop`

### Benchmark Results (Unoptimized vs Vectorized vs Incremental $\Delta E$ vs XLA)
A benchmark was run using `examples/benchmark_lite.py` tracking the progression of optimizations:

| Test Case | Unoptimized | Vectorized Indices | Incremental $\Delta E$ (O(1)) | XLA + Delta E | Speedup (vs Unoptimized) |
|---|---|---|---|---|---|
| **Ising 2D L=8 ($N=64$)** | 168 steps/s | 209 steps/s | 708 steps/s | **9,316 steps/s** | **55.5x** |
| **Ising 2D L=16 ($N=256$)** | 171 steps/s | 344 steps/s | 814 steps/s | **7,850 steps/s** | **45.9x** |
| **Ising 2D L=32 ($N=1024$)** | 178 steps/s | 350 steps/s | 866 steps/s | **7,164 steps/s** | **40.2x** |
| **EA 3D L=4 ($N=64$)** | 146 steps/s | 201 steps/s | 889 steps/s | **7,987 steps/s** | **54.7x** |
| **EA 3D L=6 ($N=216$)** | 161 steps/s | 498 steps/s | 908 steps/s | **6,711 steps/s** | **41.7x** |
| **EA 3D L=8 ($N=512$)** | 177 steps/s | 546 steps/s | 974 steps/s | **8,243 steps/s** | **46.6x** |

### Projected Runtimes for Example Scripts

| Example Script | Unoptimized (Python Loop) | Vectorized Index Gen | Incremental $\Delta E$ | **XLA + Delta E (Now)** |
|---|---|---|---|---|
| **`examples/ea_observables.py`** (L={4,6}, 5 betas) | ~8.0 hours | ~3.6 hours | ~1.4 hours (83 min) | **~10.7 min** |
| **`examples/ea_glass.py`** (L={4,8}, 25 betas) | ~115.6 hours | ~43.6 hours | ~20.8 hours (1245 min) | **~2.44 hours** (146 min) |
| **`examples/ising.py`** (L={8,16,32}, 25 betas) | ~41.8 hours | ~22.9 hours | ~8.8 hours (530 min) | **~57.5 min** |

**Key takeaway**: By removing the $O(N^2)$ energy recomputations, replacing them with local neighborhood updates ($O(D)$), and enabling full XLA JIT compilation over the dynamics loop, the cost per Monte Carlo step is completely decoupled from the system volume $N$. Simulated annealing of large 3D spin glasses can now complete in hours/minutes rather than taking days/weeks!

---

## Bottleneck Analysis

### ✅ P0 — Full Energy Recomputation per Step (Fixed)

**Where**: [`MetropolisHastings.flip_spins()`](src/spin_engine/dynamics/metropolis_hastings.py) → calls [`system.compute_energy(updated)`](src/spin_engine/models/ising.py)

**Problem**: Every single MC step proposes flipping **1 spin**, then recomputes the **entire system energy** via a full matrix multiply:

```python
h_local = tf.matmul(spin_state_flat, interaction_matrix_flat)  # (replicas, N) @ (N, N) → O(replicas × N²)
```

This is done for **every step** inside the `tf.while_loop`. For a sweep of `sweeps × N` steps, the total cost is:

```
Total FLOPs ≈ sweeps × N × replicas × N² = sweeps × replicas × N³
```

**The General Derivation (Model-Agnostic $\Delta E$)**:
Let the original spin vector be $\sigma$ and the new one (after updating a set $D$ of sites) be $\tilde\sigma$. Define the change:
$$\Delta\sigma_i := \tilde\sigma_i - \sigma_i.$$
Only indices $i\in D$ have $\Delta\sigma_i\neq0$; for $i\notin D$, $\Delta\sigma_i=0$.

Start from:
$$\mathcal H = -\tfrac12\sum_{i,j} \sigma_i J_{ij}\sigma_j, \qquad \tilde{\mathcal H} = -\tfrac12\sum_{i,j} \tilde\sigma_i J_{ij}\tilde\sigma_j,$$
so:
$$\Delta E = \tilde{\mathcal H}-\mathcal H = -\tfrac12\sum_{i,j}J_{ij}\big(\tilde\sigma_i\tilde\sigma_j-\sigma_i\sigma_j\big).$$

Using $\tilde\sigma_i=\sigma_i+\Delta\sigma_i$ and expanding:
$$\tilde\sigma_i\tilde\sigma_j-\sigma_i\sigma_j = \sigma_i\Delta\sigma_j+\sigma_j\Delta\sigma_i+\Delta\sigma_i\Delta\sigma_j.$$

Assuming $J_{ij}=J_{ji}$ (symmetric coupling matrix), we can combine the first two sums to get:
$$\Delta E = -\sum_{i,j} J_{ij}\sigma_i\Delta\sigma_j - \tfrac12\sum_{i,j}J_{ij}\Delta\sigma_i\Delta\sigma_j.$$

Because $\Delta\sigma_j=0$ for $j\notin D$, these sums reduce to:
$$\boxed{\displaystyle
\Delta E = -\sum_{j\in D} \Delta\sigma_j\, h_j \;-\; \tfrac12\sum_{i\in D}\sum_{j\in D} J_{ij}\,\Delta\sigma_i\,\Delta\sigma_j,
}$$
where $h_j$ is the *local field* (computed with the original spins $\sigma$):
$$h_j := \sum_{i} J_{ij}\sigma_i.$$

### Interpretation / Special Cases
* **Physical Interpretation**: The first term $-\sum_{j\in D} \Delta\sigma_j h_j$ is the interaction of each changed site with the rest of the system (original spins). The second term $-\tfrac12\sum_{i,j\in D} J_{ij}\Delta\sigma_i\Delta\sigma_j$ corrects for double counting and includes the mutual interaction among the disturbed sites themselves.
* **Single-site update** ($D=\{n\}$): $\Delta E = -\Delta\sigma_n h_n$. For a flip $\tilde\sigma_n = -\sigma_n$, we have $\Delta\sigma_n = -2\sigma_n$, giving $\Delta E = 2\sigma_n h_n$ (the standard Ising/EA update). Note that the second term vanishes as $J_{nn}=0$.
* **Two-site update** ($D=\{n,m\}$): $\Delta E = -(\Delta\sigma_n h_n + \Delta\sigma_m h_m) - J_{nm}\Delta\sigma_n\Delta\sigma_m$ (with $J_{ii}=0$).
* **Continuous spin models (Spherical System)**: $\Delta\sigma_j$ is continuous, but the same mathematical relation holds.

### Computational Note
To compute $\Delta E$ efficiently for $k = |D|$ disturbed sites:
1. Compute local fields $h_j=\sum_i J_{ij}\sigma_i$ for $j\in D$. Cost is $O(k \cdot N)$ or less if $J$ is sparse (nearest-neighbor is $O(k \cdot \text{max\_neighbors})$).
2. Compute the first term $-\sum_{j\in D}\Delta\sigma_j h_j$ (cost $O(k)$).
3. Compute the small dense sum $-\tfrac12\sum_{i,j\in D}J_{ij}\Delta\sigma_i\Delta\sigma_j$ (cost $O(k^2)$).

For $k \ll N$, this is extremely efficient. If $k$ is large, we can fallback to dense matrix operations directly:
$$\Delta E = -\sigma^\top J\,\Delta\sigma -\tfrac12 \Delta\sigma^\top J\,\Delta\sigma.$$

**Estimated speedup**:

| Model | Current | With $\Delta E$ | Speedup |
|---|---|---|---|
| Ising 2D L=32 (N=1024) | O(N²) = 1M ops/step | O(4) = 4 ops/step | **~250,000×** |
| EA 3D L=8 (N=512) | O(N²) = 262K ops/step | O(6) = 6 ops/step | **~44,000×** |
| SK N=128 | O(N²) = 16K ops/step | O(N) = 128 ops/step | **~128×** |

Even accounting for TF overhead and batching over replicas, this yields **orders of magnitude** improvement for nearest-neighbor models.

**Implementation approach**:
1. Add `compute_delta_energy(spin_state, updated_state, changed_indices)` as a unified method on `BaseSpinSystem`.
2. Compute $\Delta\sigma = \tilde\sigma - \sigma$.
3. Compute local fields $h_j$ for $j \in D$. The base class can use dense row gather.
4. Nearest-neighbor systems override local field lookup to utilize neighbor lists for $O(1)$ lookup.
5. `MetropolisHastings.step()` calls `compute_delta_energy()` instead of `compute_energy(proposed)`.
6. Update `current_energy` incrementally: `new_energy = current_energy + ΔE`.

**Files affected**:
- `src/spin_engine/models/base.py` — add abstract/default `compute_delta_energy()`
- `src/spin_engine/models/ising.py` — override local field extraction
    - `src/spin_engine/models/edwards_anderson.py` — override local field extraction
    - `src/spin_engine/dynamics/metropolis_hastings.py` — use `compute_delta_energy()` in `step()`

    **Caveat**: This fix covers Ising, EA, SK, and Spherical (anything with `self.interaction_matrix`). `WegnerSystem` has no interaction matrix and currently falls back to full recomputation on every step — see the correctness bugs below, which also block Wegner from being driven through `MetropolisHastings` at all for most replica counts.

    ---

    ### 🔴 P0 — O(N) Memory Copies for Single Spin Flips (`tf.where` overhead)

    **Where**: [`MetropolisHastings.step()`](src/spin_engine/dynamics/metropolis_hastings.py)

    **Problem**: Even with incremental $\Delta E$, proposing a flip required copying the entire lattice using `tf.tensor_scatter_nd_update` to create `updated`. Then, to conditionally accept the new state, `tf.where(accept, updated, original_state)` was called. Both of these operations are dense tensor updates, which means the GPU was doing an $O(N)$ memory copy just to flip a single spin! 

    For $L=12$ with 512 replicas, the lattice has $12^3 \times 512 = 884,736$ spins. Doing a full copy per step saturates the GPU's memory bandwidth entirely (VRAM bottleneck), leaving the math ALUs starving and drastically reducing GPU utilization for large systems.

    **Implementation approach**:
    1. Change `flip_spins()` to return only the specific indices and values of the proposed update, avoiding the creation of the full `updated` tensor since `compute_delta_energy()` relies strictly on indices anyway.
    2. Inside `step()`, evaluate the Metropolis `accept` criterion first.
    3. Use a single in-place (conceptually) `tf.tensor_scatter_nd_update` that modifies *only* the accepted indices, bypassing `tf.where` on the full lattice completely. 
    4. This drops the memory bandwidth requirement per step from $O(N)$ to $O(1)$.

    **Files affected**:
    - `src/spin_engine/dynamics/metropolis_hastings.py` — rewrote `step()` and `flip_spins()` to operate with scattered in-place updates.

    ---

### 🔴 P0 — Correctness Bugs Blocking Wegner Model Benchmarking

**Where**: [`MetropolisHastings.step()`](src/spin_engine/dynamics/metropolis_hastings.py#L87-L91) and [`MetropolisHastings.flip_spins()`](src/spin_engine/dynamics/metropolis_hastings.py#L22-L53)

**Discovered**: while extending the replica-scaling benchmark (`examples/benchmark_replicas.py`) to cover all five models instead of just Ising/EA, `WegnerSystem` driven through `MetropolisHastings` failed outright for most replica counts and was silently wrong for the rest.

**Bug 1 — wrong reshape rank for gauge models**:
```python
new_spin_state = tf.where(
    tf.reshape(accept, (-1,) + (1,) * self.system.lattice_dim),
    updated,
    self.system.spin_state
)
```
This assumes `spin_state.shape == (replicas,) + (L,) * lattice_dim`, which holds for Ising/EA/SK/Spherical. `WegnerSystem.spin_state` has an **extra trailing link-direction axis**: `(replicas, L, ..., L, lattice_dim)`. The reshape to `(-1, 1, 1)` (for `lattice_dim=2`) is one rank short, so `tf.where`'s broadcast either:
- **Raises `ValueError`** when `replicas != lattice_length` (e.g. R=32, L=8 → `Dimensions must be equal, but are 32 and 8`), or
- **Silently broadcasts wrong** when `replicas == lattice_length` — the per-replica accept/reject mask gets reinterpreted as a per-row mask along the lattice's first spatial axis instead, corrupting the physics with no error raised.

**Bug 2 — non-ergodic flip sampling for gauge models**:
```python
idx = tf.random.uniform(
    shape=(self.system.lattice_replicas, num_flips),
    maxval=tf.cast(self.system.number_spins, tf.int32),
    dtype=tf.int32
)
```
`number_spins = L ** lattice_dim` counts lattice **sites**, but a gauge model's flattened state has `L**lattice_dim * lattice_dim` entries (one per link direction per site). Sampling indices only up to `number_spins` restricts flips to roughly the first `1/lattice_dim` fraction of links — the rest can never be selected, breaking ergodicity. (`tests/test_delta_energy.py`'s `TestDeltaEnergyWegner` works around this correctly by sampling up to the true flattened size, which is how the bug was confirmed.)

**Compounding gap — no ΔE acceleration for Wegner**: `WegnerSystem` has no `interaction_matrix`, so `compute_delta_energy()` falls back to the base class's `compute_energy(updated) - compute_energy(spin_state)` — a **full plaquette recomputation on every single proposed flip**, independent of Bugs 1/2. The P0 ΔE optimization above does not apply to Wegner at all; every MC step costs a full O(N·D) plaquette sweep (paid twice — once per energy call), unlike Ising/EA/SK/Spherical's O(k·N) local update.

**Impact**: `WegnerSystem` cannot currently be driven through `MetropolisHastings.step()`/`.sweep()` for general replica counts. `examples/wegner.py` is unaffected only because it doesn't drive the model through `sim.sweep()` with a tracker in the standard pattern — don't assume it's safe to batch Wegner over replicas without fixing this first.

**Fix sketch**:
1. Reshape `accept` against the actual rank of `spin_state` (e.g. `tf.reshape(accept, (-1,) + (1,) * (len(self.system.spin_state.shape) - 1))`) instead of hardcoding `lattice_dim`.
2. Sample flip indices up to the true flattened size (`tf.size(spin_flat) // replicas`), not `number_spins`.
3. Give `WegnerSystem` its own `compute_delta_energy()` (recomputing only the plaquettes touching a flipped link) — full recompute is a separate, real performance gap once 1 and 2 are fixed.

**Files affected**:
- `src/spin_engine/dynamics/metropolis_hastings.py` — fix reshape rank and flip-index bound
- `src/spin_engine/models/wegner.py` — add a local `compute_delta_energy()` override

---

### 🔴 P0 — Dense Interaction Matrix Storage (Critical for large N)

**Where**: [`Interaction.generate()`](src/spin_engine/interactions/base.py) → produces `(L,)*D*2` dense tensor

**Problem**: The interaction matrix J is stored as a dense `(N, N)` tensor. For nearest-neighbor models, >99% of entries are zero:

| Model | N | Non-zeros | Density | Memory (float32) |
|---|---|---|---|---|
| EA 3D L=8 | 512 | 3,072 | 1.2% | 1 MB (dense) vs 12 KB (sparse) |
| EA 3D L=16 | 4,096 | 24,576 | 0.15% | 64 MB vs 96 KB |
| EA 3D L=32 | 32,768 | 196,608 | 0.02% | 4 GB vs 768 KB |
| SK N=128 | 128 | 16,384 | 100% | 64 KB (no gain) |

For L=32 in 3D, the dense matrix **does not fit in GPU memory** (4 GB for a single float32 matrix).

**Implementation approach**:
1. Add `generate_neighbor_list(D, L) -> (indices, weights)` to `Interaction` base class.
   - `neighbor_indices`: shape `(N, max_neighbors)` — padded indices of neighbors
   - `neighbor_weights`: shape `(N, max_neighbors)` — J values per neighbor
2. `PeriodicNearestNeighborInteraction` and `BinaryRandomInteraction` implement this natively.
3. Models store `self.neighbor_indices` and `self.neighbor_weights` instead of (or alongside) the full matrix.
4. `compute_delta_energy()` uses `tf.gather` on the neighbor list:
   ```python
   neighbor_spins = tf.gather(s_flat, self.neighbor_indices[i])  # (max_neighbors,)
   delta_E = -2 * s_i * tf.reduce_sum(self.neighbor_weights[i] * neighbor_spins)
   ```
5. Fully-connected models (SK, CurieWeiss) continue using the dense matrix — no change needed.

**Files affected**:
- `src/spin_engine/interactions/base.py` — add `generate_neighbor_list()` with default
- `src/spin_engine/interactions/standard.py` — implement for `PeriodicNearestNeighborInteraction`, `BinaryRandomInteraction`
- `src/spin_engine/models/ising.py` — optionally store neighbor list
- `src/spin_engine/models/edwards_anderson.py` — optionally store neighbor list

---

### 🟡 P1 — `tf.function` Retracing on New Lattice Sizes

**Where**: [`MetropolisHastings.sweep()`](src/spin_engine/dynamics/metropolis_hastings.py) — decorated with `@tf.function`

**Problem**: When changing lattice size (new L), TensorFlow must retrace the entire graph. This is visible in the terminal output as the `loop_optimizer.cc` warnings. The first β for each L incurs a ~30-60 second tracing overhead.

**Mitigation**: This is inherent to TF's graph compilation and can't be eliminated, but the impact can be reduced by:
1. **Batching all betas into a single `sweep()` call** using a `beta_schedule` tensor (as the Wegner example already does). This avoids retracing between betas.
2. **Warming up the graph** with a short dummy sweep before the real simulation.
3. The tracing cost becomes negligible once `compute_delta_energy` makes individual sweeps much faster.

**Note**: The `dynamics/base.py` already has a `TODO` comment about this:
```python
# TODO: Add annealing inside the sweep to avoid retracing.
```

---

### 🟡 P1 — Measurement Overhead Inside `tf.while_loop`

**Where**: [`Tracker.track()`](src/spin_engine/dynamics/tracker.py) → called every step, conditionally computes measurements

**Problem**: Every step inside the `tf.while_loop`, the tracker evaluates a `tf.cond` to check if measurements should be recorded. When measurements *are* recorded (every `granularity` steps), the following happens:
1. **`Energy.compute()`** calls `system.compute_energy()` — another full O(N²) matmul, separate from the one in `flip_spins`.
2. **`OverlapDistribution.compute()`** computes `tf.matmul(S, S^T)` — shape `(replicas, N) @ (N, replicas)` = O(replicas² × N).
3. **`OverlapDistribution`** then applies `tf.boolean_mask` with a mask it recreates every call.

**Optimizations**:
1. **Energy measurement**: Once `compute_delta_energy` is implemented, `Energy` measurement should read `dynamics.current_energy` directly instead of recomputing from scratch. This is free — the energy is already tracked.
2. **Overlap mask caching**: The boolean mask for extracting upper-triangular overlaps is the same every time. Cache it once at tracker initialization.
3. **Granularity**: Consider larger granularity values for expensive measurements (OverlapDistribution). The overlap is O(replicas² × N) which for 64 replicas and N=512 is ~2M FLOPs per recording.

---

### 🟡 P1 — Redundant System/Dynamics Re-instantiation in Scripts

**Where**: [`examples/sk_glass.py`](examples/sk_glass.py) and [`examples/spin_glass_phase_diagram.py`](examples/spin_glass_phase_diagram.py)

**Problem**: These scripts re-create `System`, `MetropolisHastings`, and `Tracker` **inside** the β loop:

```python
# sk_glass.py lines 51-71 — creates new system PER BETA
for beta in betas:
    system = SherringtonKirkpatrickSystem(...)  # NEW system
    sim = MetropolisHastings(system)            # NEW dynamics
    tracker = Tracker(...)                       # NEW tracker
    sim.sweep(...)
```

This causes:
1. **Graph retracing** on every β (new TF objects → new graph signatures)
2. **Loss of spin state** between temperatures (no annealing — starts from random state each time)
3. **Wasted memory** from accumulated TF graph artifacts

The EA and Ising scripts (`ea_glass.py`, `ea_observables.py`, `ising.py`) correctly instantiate outside the loop.

**Fix**: Move instantiation outside the β loop in `sk_glass.py` and `spin_glass_phase_diagram.py`, following the pattern already used in the EA/Ising scripts.

---

### 🟢 P2 — `flip_spins()` Uses Python Loop for Index Generation

**Where**: [`MetropolisHastings.flip_spins()`](src/spin_engine/dynamics/metropolis_hastings.py#L29-L33)

**Problem**: Spin flip indices were originally generated with a Python `for` loop over replicas, resulting in 64 separate `tf.random.shuffle` operations compiled into the graph.

**The Fix**: Vectorizing the index generation using `tf.random.uniform` (as currently checked out in the working tree):
```python
idx = tf.random.uniform(
    shape=(self.system.lattice_replicas, num_flips),
    maxval=tf.cast(self.system.number_spins, tf.int32),
    dtype=tf.int32
)
```

**Benchmark Proof**:
This vectorization was tested using `examples/benchmark_lite.py` (with `SWEEPS = 10`):
- **Graph Compilation / Tracing time**: For EA 3D L=8, the trace time dropped from **30.09s** to **11.56s** (nearly 3x faster tracing), proving that the massive graph from the unrolled Python loop was choking the compiler.
- **Execution Speed (Energy-only measurement)**: For EA 3D L=4, performance increased from **157 steps/s** to **961 steps/s** (a 6.1x speedup). This confirms that Python loop overhead was completely bottlenecking the execution of small systems.

---

### 🟢 P2 — Commented-out `@tf.function` Decorators

**Where**: Multiple files

Several `@tf.function` decorators are commented out during development:
- [`BaseSpinSystem.compute_energy()`](src/spin_engine/models/base.py#L45)
- [`IsingSystem.compute_energy()`](src/spin_engine/models/ising.py#L60)
- [`SphericalSystem.compute_energy()`](src/spin_engine/models/spherical.py#L78)
- [`WegnerSystem.compute_energy()`](src/spin_engine/models/wegner.py#L57)
- [`MetropolisHastings.step()`](src/spin_engine/dynamics/metropolis_hastings.py#L62)

These are already called from within the `@tf.function`-decorated `sweep()`, so they're traced as part of that graph. The commented decorators would help if these methods are called standalone (e.g., in measurements). Re-enabling them after the `compute_delta_energy` refactor would be straightforward.

---

### 🟢 P2 — `OverlapDistribution` Mask Recomputation

**Where**: [`OverlapDistribution.compute()`](src/spin_engine/measurements/correlations.py#L53-L56)

**Problem**: Every call creates the upper-triangular boolean mask from scratch:
```python
mask = tf.linalg.band_part(tf.ones((replicas, replicas), dtype=tf.bool), 0, -1)
mask = tf.linalg.set_diag(mask, tf.zeros(replicas, dtype=tf.bool))
```

Since `replicas` is constant, this mask should be computed once at initialization and reused. Inside `tf.function` this is likely folded by the compiler, but it's still unnecessary computation in eager mode and adds graph complexity.

---

### 🟣 P3 — Extreme Replica Scaling: The Memory Bandwidth Wall (Advanced)

> [!TIP]
> **Advanced Optimization**: If you push the replica axis to extreme limits ($>10^6$ replicas of small systems like $N=64$) for massive ensemble averaging, the system will hit a memory bandwidth wall. 

**Where**: Memory access patterns in `tf.while_loop` and `MetropolisHastings.sweep()`.

**Problem**: At $2^{21}$ replicas of $N=64$, the state requires ~5.4 GB of VRAM. While the GPU has enough space, the Monte Carlo sweep reads and writes these 5.4 GB to Global Memory (VRAM) for every step. The limited global memory bandwidth (1-3 TB/s) bottlenecks the `steps/s` throughput significantly, causing a drop in simulation speed.

**Implementation approach (The Custom Kernel Fix)**:
To regain the speed seen at smaller replica counts (which fit entirely in the ultra-fast L2 cache), the algorithm must be rewritten to utilize GPU **Shared Memory**:
1. Write a custom CUDA kernel (or use a highly fused framework like JAX with `vmap`/`lax.scan`).
2. Assign each replica to a single GPU Thread Block. 
3. The block loads its entire $N=64$ state from VRAM into the SM's ultra-fast Shared Memory *once*.
4. The block performs the entire Monte Carlo sweep (thousands of flip attempts) entirely in Shared Memory/Registers without ever touching Global Memory.
5. Write the final state back to Global Memory only when the sweep is complete.

This completely bypasses the Global Memory bottleneck and would allow millions of small replicas to simulate simultaneously at the absolute theoretical compute limit of the GPU.

---

## Optimization Priority & Implementation Order

| Priority | Item | Expected Impact | Complexity | Dependencies |
|---|---|---|---|---|
| 🔴 P0 | `compute_delta_energy()` on models | **100-10,000× per step** | Medium | None |
| 🔴 P0 | Fix Wegner+`MetropolisHastings` shape/ergodicity bugs | **Correctness** — blocks all Wegner batching/benchmarking | Low-Medium | None |
| 🔴 P0 | Sparse/neighbor-list J storage | **Memory: 100×, enables large L** | Medium | Pairs well with P0 |
| 🟡 P1 | Energy measurement reads cached value | **2× fewer matmuls per tracked step** | Low | Requires P0 |
| 🟡 P1 | Fix SK/phase_diagram re-instantiation | **~N× fewer retracings** | Low | None |
| 🟡 P1 | Annealing inside sweep (beta schedule) | **Eliminates inter-β retracing** | Medium | TODO in base.py |
| 🟢 P2 | Vectorize flip index generation | Minor | Low | None |
| 🟢 P2 | Cache overlap mask | Minor | Low | None |
| 🟢 P2 | Re-enable `@tf.function` decorators | Minor to moderate | Low | After P0 refactor |

---

## Projected Impact

### Benchmark Baseline (Unoptimized vs Vectorized Index)
- **Unoptimized**: `ea_glass.py` (L=4 & L=8) takes **~115.6 hours** (4.8 days)
- **Vectorized index**: `ea_glass.py` takes **~43.6 hours** (1.8 days)

### Projected After P0 ($\Delta E$ + neighbor list):
- L=4 ($N=64$): Each step goes from $O(64^2) = 4096$ operations to $O(6) = 6$ operations per flip.
- L=8 ($N=512$): Each step goes from $O(512^2) = 262,144$ operations to $O(6) = 6$ operations per flip.
- Even with TensorFlow graph execution overhead, we expect a conservative **50-200x speedup** on the execution loop.
- Projected: The `ea_glass.py` (L={4,8}) run should take **under 15 minutes** instead of 43.6 hours, making larger lattices (L=16, L=32) completely feasible to run in minutes/hours rather than weeks.

---

## Relationship to MILESTONES.md

These optimizations should be considered **Milestone 3.0** — a prerequisite before the existing Milestone 3 items (Wolff algorithm, Parallel Tempering). The current bottleneck makes even basic simulations impractical at research-relevant lattice sizes, blocking progress on Milestones 1-5.

| MILESTONES.md Item | Blocked By |
|---|---|
| MS2: Wegner Phase Transition (3D) | Large L requires fast sweeps; also blocked by the Wegner+`MetropolisHastings` correctness bugs above — fix before relying on replica batching |
| MS3.1: Annealing inside sweep | Natural extension of P0 refactor |
| MS3.2: Wolff cluster algorithm | Separate dynamics, but benefits from same ΔE infrastructure |
| MS3.3: Parallel Tempering | Requires fast single-temperature sweeps |
| MS5.1: Finite-Size Scaling | Requires L={8,16,32,64} to be feasible |
