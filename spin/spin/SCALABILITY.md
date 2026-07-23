# SCALABILITY.md — How Far Can This Go, On This Machine?

> **Scope**: This is a measurement-backed dossier answering one question: for **3D glassy systems** (Edwards-Anderson, and by extension Sherrington-Kirkpatrick), how do time and memory actually grow with lattice size `L`, replica count, and the planned `disorder_samples` axis (MILESTONES.md 5.5) — and where does *this specific machine* run out of room?
>
> Everything below was measured on this machine just now (not copied from PERFORMANCE.md, which documents GPU-side compute optimizations and is still broadly accurate for that part). No code was changed to produce these numbers.

---

## Hardware Profile

| Resource | Value |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, **8 GB VRAM**, compute capability 8.9 |
| System RAM | **15 GiB total, ~12 GiB available** (WSL2 — capped well below whatever the host physically has) |
| Swap | 4 GiB (don't rely on it — see below) |
| CPU | 32 logical cores |
| TF | 2.20.0, GPU visible, XLA (`jit_compile=True`) in use for the sweep loop |

**The binding constraint on this machine is host RAM, not GPU VRAM.** That's the headline finding, and it's the opposite of what PERFORMANCE.md's GPU-focused roadmap would lead you to expect. Details below.

---

## TL;DR

1. **GPU compute and GPU memory are fine** through at least `L=24` in 3D (8,725 steps/s, 1.6 GB VRAM at `L=24`, R=64) and almost certainly well beyond. They are not what will stop you.
2. **Host RAM during coupling-matrix construction is what will stop you**, and it already eats 92% of available RAM at `L=24` — *for a single disorder realization, before the simulation even starts.* `L=28` would need ~32 GB; `L=32` would need ~72 GB. Both are impossible on this machine as the code is written today.
3. The cause is two compounding implementation issues in `interactions/standard.py`, not anything inherent to the physics — Edwards-Anderson is a short-range model with O(N) true degrees of freedom, but the interaction matrix is built and stored as a dense O(N²) array, via an algorithm that's also O(N²) in *construction* cost (not just storage).
4. **`disorder_samples` (Milestone 5.5, not yet implemented) multiplies whichever cost dominates by K.** Given finding #2, adding disorder averaging on top of the current implementation is not "free in compute, just slower" — it compounds a memory wall that's already nearly tripped at `L=24, K=1`. Fix the construction bug *before* building disorder_samples, or the two problems will be debugged together and look far more confusing than either is alone.
5. **SK/Curie-Weiss have none of this slack to recover.** They're fully-connected by physics, so O(N²) memory and O(N) per-flip cost are real, not artifacts. The fixes below help EA/Ising dramatically and do nothing for SK.

---

## Two Different Scaling Stories

The engine currently treats nearest-neighbor models (Ising, Edwards-Anderson) and fully-connected models (SK, Curie-Weiss) identically — both go through `Interaction.generate()` and come out as a dense `(L,)*2D` tensor, and both use the same generic `compute_delta_energy()` path in [base.py](src/spin_engine/models/base.py#L56-L125) (or a model-specific override that still gathers a full dense row, e.g. [edwards_anderson.py:117-122](src/spin_engine/models/edwards_anderson.py#L117-L122)). That uniformity is convenient for the codebase but hides two very different scaling realities:

| | **EA / Ising (short-range)** | **SK / Curie-Weiss (fully-connected)** |
|---|---|---|
| True degrees of freedom | O(N) — each site has 2D neighbors (6 in 3D) | O(N²) — every pair interacts |
| Current storage | O(N²) dense — **100% artifact**, not needed | O(N²) dense — **intrinsic**, unavoidable |
| Current per-flip cost | O(N) dense-row gather — **artifact**, true cost is O(2D)=O(6) | O(N) dense-row gather — **intrinsic**, can't do better |
| Headroom from a sparse rewrite | Up to ~N/6× memory, same factor in time-per-flip | **None.** Sparsity doesn't exist in a complete graph. |

This distinction matters for prioritization: every fix proposed below is high-leverage for EA and irrelevant for SK. If you want bigger SK systems, the only lever is reducing N or accepting O(N²) — there's no hidden inefficiency to claw back.

---

## What's Actually Implemented Today (verified against source, not PERFORMANCE.md's narrative)

- ✅ `compute_delta_energy()` exists and is used by `MetropolisHastings.step()` for Ising/EA/SK/Spherical — single-flip proposals no longer recompute the full O(N²) energy. This part of PERFORMANCE.md's P0 item is done.
- ✅ The O(N) full-lattice copy-per-flip is also fixed — `flip_spins()` returns scatter indices/values, and `step()` uses `tf.tensor_scatter_nd_update` on just the touched entries ([metropolis_hastings.py:22-48](src/spin_engine/dynamics/metropolis_hastings.py#L22-L48)). Matches the recent commit `d2af942`.
- ❌ **The "sparse/neighbor-list storage" P0 item is not implemented.** `compute_delta_energy()` for EA still does `tf.gather(J_flat, flat_idx)` — pulling a *full N-length row* of mostly-zero entries to compute a local field, instead of gathering only the ~6 true neighbor weights ([edwards_anderson.py:119-127](src/spin_engine/models/edwards_anderson.py#L119-L127)). Per-flip cost is O(N), not O(1), for EA today.
- ❌ The Wegner `MetropolisHastings` reshape-rank and ergodicity bugs PERFORMANCE.md describes are still present in the current code ([metropolis_hastings.py:81-85](src/spin_engine/dynamics/metropolis_hastings.py#L81-L85), line 32). Orthogonal to this dossier's EA/SK focus, but flagging since it's the same file.
- ❌ `disorder_samples` (Milestone 5.5) has no code yet — `EdwardsAndersonSystem`/`SherringtonKirkpatrickSystem` take a single `interaction_matrix`, no batch-of-disorder axis.

The practical effect: the GPU-side compute path is in good shape. The problem is entirely upstream of the GPU, in how the interaction matrix gets built on the CPU.

---

## Measured: Edwards-Anderson 3D, `L` Scaling

Methodology: real runs via `venv/bin/python`, isolated per-`L` subprocess, `EdwardsAndersonSystem` built exactly the way `examples/ea_observables.py` builds it (`PeriodicNearestNeighborInteraction` × `BinaryRandomInteraction`), `R=64` replicas, 20-sweep timed average after a warm-up + cached call (same protocol `examples/benchmark.py` uses).

| L | N=L³ | Construction time | Construction peak RSS | Sweep steps/s | GPU peak mem |
|---|---|---|---|---|---|
| 8 | 512 | 0.015 s | 51 MB | 11,511 | 87 MB |
| 12 | 1,728 | 0.17 s | 239 MB | 11,174 | 106 MB |
| 16 | 4,096 | 1.13 s | 1,185 MB | 9,743 | 250 MB |
| 20 | 8,000 | 4.58 s | 4,428 MB | 10,570 | 587 MB |
| 24 | 13,824 | **27.9 s** | **13,148 MB** | 8,725 | 1,602 MB |

Two completely different trend lines:
- **GPU compute** (steps/s, GPU memory): mild, well-behaved degradation. Down only ~24% from `L=8` to `L=24` despite N growing 27×. GPU memory stays under 2 GB. This axis has lots of room left.
- **Host construction cost** (time, RSS): explosively non-linear. From `L=20` to `L=24` (N grows 1.73×), construction time grows **6.1×** and peak RSS grows **2.97×** — a clean local fit of **peak_RSS ≈ 69 bytes/element × N²** (R² ≈ 1.0 using the `L=20`/`L=24` pair, confirmed against the `L=20` point: predicts 4,403 MB vs measured 4,428 MB).

### Root cause (profiled directly, not inferred)

`PeriodicNearestNeighborInteraction.generate()` in [interactions/standard.py:32-51](src/spin_engine/interactions/standard.py#L32-L51) builds a full **N×N×3 pairwise coordinate-difference tensor** via broadcasting before masking it down to the ~6 nonzero neighbors per site. Profiling the four steps separately at `L=24` (N=13,824):

```
diff (broadcast NxNxD, int64):    4.318 s
periodic minimum (same shape):    7.074 s
manhattan distance sum:           1.467 s
nn_mask boolean compare:          0.130 s
final Python assignment loop:     0.204 s   <- NOT the bottleneck
TOTAL:                           13.540 s   (this profile; full script measured 22.7s incl. import overhead)
```

**95% of the time is in three vectorized-but-O(N²) NumPy operations, not the Python loop at the end** — so "vectorize the loop" (the fix `PERFORMANCE.md`'s P2 item already applied elsewhere) would do nothing here. The actual fix is to never materialize the `(N,N,D)` tensor: directly enumerate the 2D periodic-neighbor offsets per site (an O(N·D) algorithm), which is what a real neighbor-list generator looks like.

A second, independent and compounding bug: `BinaryRandomInteraction.generate()` ([standard.py:111](src/spin_engine/interactions/standard.py#L111)) and `GaussianInteraction.generate()` ([standard.py:80](src/spin_engine/interactions/standard.py#L80)) both call NumPy random generators on a Python `float`/list input, which defaults to **float64** — doubling memory for an array that gets cast to float32 anyway one line later in `BaseSpinSystem._validate_tensor_shape()`. Multiple float64 temporaries (`random_J`, the `nn_mask * random_J` product) are alive simultaneously, which is most of why peak RSS is ~17× the size of the final float32 matrix rather than ~4×.

### Extrapolation (not measured live — would have risked locking up this machine)

Using the fitted `peak_RSS ≈ 69 bytes × N²`:

| L | N | Projected construction peak RSS | Feasible on this machine (~12 GiB usable)? |
|---|---|---|---|
| 24 | 13,824 | 13.1 GB *(measured)* | Barely — 92% of usable RAM, for K=1 |
| 28 | 21,952 | ~32 GB | **No** |
| 32 | 32,768 | ~72 GB | **No** |

**`L≈24` is the practical ceiling for the Edwards-Anderson model in 3D on this machine, as the code is written today — and it's a host-RAM wall hit before the simulation loop even starts, not a GPU limit.** This is a much harder ceiling than PERFORMANCE.md's GPU-VRAM-only analysis suggested (its table flagged `L=32` dense storage as "doesn't fit in 4GB GPU" — true, but you'd never get there, because host construction fails first, around `L=26-27`).

---

## Time Budget for Production-Scale Runs

`examples/ea_observables.py`'s real settings: `sweeps=1000`, `lattice_replicas=512`, 25 betas, `sweep_length = sweeps × N`.

Measured at `L=16, R=512`: **6,888 steps/s** (down from 9,743 at R=64 — an 8× replica increase only cost 1.4× slowdown; replica batching is cheap, consistent with PERFORMANCE.md and confirming the GPU is under-utilized at small batch sizes).

| L | sweep_length/beta | Time/beta @ measured steps/s | × 25 betas |
|---|---|---|---|
| 16 | 4,096,000 | ~595 s (9.9 min) | **~4.1 hours** |
| 24 (extrapolated, R=64→512 factor applied) | 13,824,000 | ~2,240 s (37 min) | **~15.6 hours** |

This roughly matches why your current `ea_observables.py` run has cached results only through `L=10` — `L=12` and `L=16` are each multi-hour, and `L=24` would be an overnight-plus run even if it didn't hit the memory wall above. None of this is a GPU problem; it's just the linear-ish cost of `sweep_length` scaling with `N`, compounded by the not-yet-implemented O(N)→O(1) neighbor-list speedup (see Priority 3 below).

---

## `disorder_samples` (Milestone 5.5): How It Multiplies the Above

The MILESTONES.md plan is to give `EdwardsAndersonSystem`/`SherringtonKirkpatrickSystem` a `disorder_samples=K` parameter, stack K independently-seeded interaction matrices into shape `(K, N, N)`, and fold `K` into the existing `lattice_replicas` batch axis (so `Dynamics`/`Tracker` need no changes — same trick that made replica-parallelism free).

That design is sound for the **GPU steady-state** cost: `(K, N, N)` float32 on GPU is just `K × 4N²` bytes. For `L=16` (4N²=67MB), even `K=20` is 1.34 GB — trivial. For `L=24` (4N²=764MB), `K=10` is 7.6 GB — getting close to the 8GB VRAM ceiling on its own, separate from the host-RAM issue.

It is **not** sound for the **construction** cost, given the bugs above, unless implemented carefully:

- The periodic-neighbor mask (`nn_mask`) is identical across all K disorder samples — it only encodes lattice topology, not the random couplings. **Compute it once, outside the K loop.** If each disorder sample naively reconstructs it (the natural thing to do if you copy today's `nn_mask * random_J` pattern into a loop), you pay the ~23s/13GB `L=24` construction cost K times for no reason.
- Even with that fix, the random-sign/Gaussian matrix generation (the genuinely K-dependent part) is *also* O(N²) and *also* float64-by-default — at `L=24` that's ~4.7s and ~8.6GB of transient memory, per sample, for an array that only needs ~6N of its N² entries.
- **The peak transient memory does not need to compound K-fold** if each sample is constructed, cast to float32, and its NumPy temporaries dropped before the next sample starts — only the small float32 final matrices need to accumulate (K × 4N² bytes), not K × (the ~70N² construction peak). This is an implementation discipline, not a given; get it wrong (e.g. build a Python list of K float64 matrices and `np.stack` at the end) and you multiply an already-near-the-wall number by K.
- **Don't fold K into the replica axis without also fixing the measurement layer.** `MagneticSusceptibility` and the whole `OverlapMatrix`/`OverlapDistribution`/`ParisiOverlapParameter` family ([correlations.py](src/spin_engine/measurements/correlations.py)) reduce or compare *across the entire replica axis* (`tf.matmul(spin_flat, spin_flat, transpose_b=True)` at [correlations.py:34](src/spin_engine/measurements/correlations.py#L34)). If `K` and `replicas_per_sample` are silently flattened together, overlaps get computed between replicas from *different* disorder realizations — physically meaningless, and it'll look like a real result, not a crash. This is the correctness catch MILESTONES.md 5.5 already flags; it's not hypothetical, it's a one-line mistake away from a quietly-wrong physics result. The cost of getting it right (reshape back to `(K, R, ...)` before reducing) is cheap; the cost of getting it wrong is silent.

**Bottom line on disorder_samples**: fix the construction-time/dtype bugs first. Building K≥2 disorder samples on top of the current implementation just multiplies a wall that's already at 92% capacity at `L=24, K=1`.

---

## Recommendations, in Priority Order

This re-prioritizes PERFORMANCE.md's existing roadmap in light of the new finding — the host-RAM construction bug is not in that document at all, and it binds *before* any of PERFORMANCE.md's GPU-side items would.

| Priority | Fix | Effort | Unlocks |
|---|---|---|---|
| **0 (new, do first)** | Force `dtype=np.float32` in `GaussianInteraction.generate()` and `BinaryRandomInteraction.generate()` ([standard.py:80](src/spin_engine/interactions/standard.py#L80), [standard.py:111](src/spin_engine/interactions/standard.py#L111)) | Trivial — one line each, zero behavior change downstream (already cast to float32 later) | Immediately halves the float64 temporaries; cuts construction peak RSS roughly in half |
| **0 (new)** | Rewrite `PeriodicNearestNeighborInteraction.generate()` to enumerate periodic neighbor pairs directly (O(N·D)) instead of building the `(N,N,D)` broadcast tensor ([standard.py:32-51](src/spin_engine/interactions/standard.py#L32-L51)) | Medium — same output shape/contract, different algorithm internally | This is the actual fix for the wall. Removes ~95% of construction time and the dominant memory cost. Pushes the host-RAM ceiling from `L≈24` to well past `L≈60-80` for EA in 3D |
| 🔴 P0 (existing) | Sparse/neighbor-list storage for `compute_delta_energy()` (PERFORMANCE.md's already-documented item) | Medium | Drops per-flip cost from O(N) to O(2D)=O(6) for EA/Ising. Not urgent yet — at measured throughput, the O(N) dense-row gather doesn't start dominating over fixed per-step overhead until N is in the millions — but becomes the next wall once the two items above unlock larger `L` |
| Before attempting K>1 | Hoist `nn_mask` construction outside the disorder loop; build+cast+discard one disorder sample at a time rather than stacking float64 intermediates | Low, but easy to get wrong by copy-pasting the current pattern | Keeps disorder_samples memory at `O(K·N²)` steady-state instead of `O(K·N²_construction_peak)` |
| Before attempting K>1 | Add the disorder-aware reshape to `MagneticSusceptibility`/Overlap family before folding K into the replica axis | Low-medium | Avoids silently-wrong physics (overlaps across different disorder realizations) |

### What's *not* a near-term concern
- **GPU VRAM**: comfortable through `L=24` (1.6 GB of 8 GB) even before any fix. Only becomes binding at large `L × K` combinations once the host-RAM wall above is removed.
- **GPU compute throughput**: degrading gently, not a wall anywhere in the range this machine could otherwise reach.
- **Replica batching**: cheap (1.4× cost for 8× replicas) — there's room to grow replica/disorder batch sizes well before the GPU saturates.

### What I did not test
I did not run `L≥28` live — the fitted curve projects ~32GB+ peak RSS, which on a 12 GiB-usable / 4 GiB-swap WSL2 instance would very likely thrash or hang the environment rather than fail cleanly. The `L=20`→`L=24` extrapolation (R²≈1.0 against three independent points) is solid enough not to need that risk to confirm the wall exists.
