"""
Benchmark suite for Spin System simulations.

Covers two complementary axes:
  1. Lattice-size (N) scaling at fixed replica count, per model.
  2. Replica-count (R) scaling at fixed N≈64, per model — isolates how the
     batch dimension affects throughput and GPU memory independently of
     per-model physics. Also includes a standalone OverlapDistribution
     micro-benchmark, since its cost is O(R²) rather than O(R) and would
     otherwise be hidden inside the sweep loop's O(R) cost.

Methodology notes (see PERFORMANCE.md):
- Timing is GPU-synchronized (tf.test.experimental.sync_devices()) and
  averaged over several cached repeats, since single-shot timing is
  sensitive to GPU clock ramp-up and shared-GPU contention noise.
- A throwaway warm-up sweep runs once before any measured section, and
  each benchmarked config gets one extra untimed cached call before the
  timed repeats — otherwise whichever config runs first/largest reads up
  to ~30% slower for no reason related to its own performance (confirmed
  empirically: an Ising R=512 config measured 3,742 steps/s when run
  first-in-process vs 5,496 steps/s for the identical config measured
  alone in an already-warm process).
- Peak GPU memory is reported via tf.config.experimental.get_memory_info,
  reset per config via reset_memory_stats. Because all sections run in one
  process, TensorFlow's allocator can satisfy a later config's memory
  request from a block already pooled by an earlier one — absolute
  peak-memory comparisons ACROSS models/sections are not reliable, but
  trends WITHIN one model (e.g. across replica counts) are.
- WegnerSystem has known shape/ergodicity bugs when driven through
  MetropolisHastings (see PERFORMANCE.md, "Correctness Bugs Blocking
  Wegner") — sections involving it are wrapped in try/except and expected
  to fail until those are fixed.

Raw results are dumped to examples/output/benchmark_results.json
(gitignored) and plots are saved to examples/images/.
"""

import os
import re
import json
import time
import gc
import statistics
from typing import cast, Callable, List, Dict, Any, Optional

import numpy as np
import tensorflow as tf

_CUPTI_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "venv", "lib", "python3.12", "site-packages",
    "nvidia", "cuda_cupti", "lib",
)
_CUPTI_DIR = os.path.realpath(_CUPTI_DIR)
os.environ["LD_LIBRARY_PATH"] = (
    _CUPTI_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

_GPUS = tf.config.list_physical_devices('GPU')
for _gpu in _GPUS:
    tf.config.experimental.set_memory_growth(_gpu, True)

import matplotlib.pyplot as plt

from spin_engine.models import (
    IsingSystem, EdwardsAndersonSystem, SherringtonKirkpatrickSystem,
    SphericalSystem, WegnerSystem,
)
from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.interactions.standard import BinaryRandomInteraction
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization
from spin_engine.measurements.correlations import OverlapDistribution

OUTPUT_DIR = "examples/output"
IMAGES_DIR = "examples/images"


def _clean_label(label: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', label).strip('_')

def _sync() -> None:
    if _GPUS:
        tf.test.experimental.sync_devices()


def _gpu_peak_mb() -> Optional[float]:
    if not _GPUS:
        return None
    return tf.config.experimental.get_memory_info('GPU:0')['peak'] / 1e6


def _warm_up_gpu() -> None:
    """Run throwaway sweeps so GPU clocks reach steady state before timing."""
    L = 8
    J = PeriodicNearestNeighborInteraction().generate(2, L)
    system = IsingSystem(lattice_dim=2, lattice_length=L, lattice_replicas=64,
                          interaction_matrix=J, initial_magnetization=0.5)
    sim = MetropolisHastings(system)
    tracker = Tracker(measurements=[Energy(system)], granularity=200)
    for _ in range(3):
        sim.sweep(tracker=tracker, beta=tf.constant(1.0),
                  num_disturbances=cast(tf.Tensor, tf.constant(1)), sweep_length=3200)
    _sync()
    del system, sim, tracker
    gc.collect()


def benchmark_sweep(
    label: str, system: Any, measurements: List[Any], sweep_length: int,
    granularity: int = 100, beta: float = 1.0, repeats: int = 5,
    profile: bool = False,
) -> Dict[str, Any]:
    """Run a sweep, then re-run it `repeats` times (GPU-synced, cached graph)
    to report a stable mean +/- std throughput, plus peak GPU memory.
    """
    num_flips = cast(tf.Tensor, tf.constant(1))
    sim = MetropolisHastings(system)
    tracker = Tracker(measurements=measurements, granularity=granularity)

    if _GPUS:
        tf.config.experimental.reset_memory_stats('GPU:0')

    # First sweep: includes tf.function tracing / XLA compilation.
    t0 = time.perf_counter()
    sim.sweep(tracker=tracker, beta=tf.constant(beta, dtype=tf.float32),
              num_disturbances=num_flips, sweep_length=sweep_length)
    _sync()
    t_trace = time.perf_counter() - t0

    if profile:
        options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=2, device_tracer_level=1, python_tracer_level=0)
        run_logdir = os.path.join(OUTPUT_DIR, "profiler_logs", _clean_label(label))
        tf.profiler.experimental.start(run_logdir, options=options)

    # One untimed cached-graph call so the GPU isn't still ramping clocks
    # during the first of the timed repeats below.
    if profile:
        with tf.profiler.experimental.Trace("sweep", step_num=0, _r=1):
            sim.sweep(tracker=tracker, beta=tf.constant(beta, dtype=tf.float32),
                      num_disturbances=num_flips, sweep_length=sweep_length)
            _sync()
    else:
        sim.sweep(tracker=tracker, beta=tf.constant(beta, dtype=tf.float32),
                  num_disturbances=num_flips, sweep_length=sweep_length)
        _sync()

    times = []
    for step_idx in range(repeats):
        t1 = time.perf_counter()
        if profile:
            with tf.profiler.experimental.Trace("sweep", step_num=step_idx+1, _r=1):
                sim.sweep(tracker=tracker, beta=tf.constant(beta, dtype=tf.float32),
                          num_disturbances=num_flips, sweep_length=sweep_length)
                _sync()
        else:
            sim.sweep(tracker=tracker, beta=tf.constant(beta, dtype=tf.float32),
                      num_disturbances=num_flips, sweep_length=sweep_length)
            _sync()
        times.append(time.perf_counter() - t1)
        
    if profile:
        tf.profiler.experimental.stop()

    t_run = statistics.mean(times)
    t_std = statistics.stdev(times) if len(times) > 1 else 0.0
    
    steps_per_s_list = [sweep_length / t for t in times]
    steps_per_sec = statistics.mean(steps_per_s_list)
    steps_per_s_std = statistics.stdev(steps_per_s_list) if len(steps_per_s_list) > 1 else 0.0
    
    peak_mb = _gpu_peak_mb()

    mem_str = f"{peak_mb:7.1f}MB" if peak_mb is not None else "    n/a"
    print(f"  {label:<32s} | steps={sweep_length:>8d} | "
          f"trace={t_trace:>6.2f}s | cached={t_run*1000:>7.1f}ms(±{t_std*1000:5.1f}) | "
          f"{steps_per_sec:>8.0f} (±{steps_per_s_std:>5.0f}) steps/s | peak_mem={mem_str}")

    return {
        "label": label, "sweep_length": sweep_length, "t_trace": t_trace,
        "t_run": t_run, "t_std": t_std, "steps_per_sec": steps_per_sec,
        "steps_per_s_std": steps_per_s_std,
        "peak_mem_mb": peak_mb,
    }


def benchmark_replica_scaling(
    label: str, make_system: Callable[[int], Any], replicas_grid: List[int],
    sweeps: int = 50, granularity: int = 200, repeats: int = 5,
    profile: bool = False,
) -> List[Dict[str, Any]]:
    """Hold N fixed and vary `lattice_replicas`, isolating throughput/memory
    scaling along the batch dimension. Failures (e.g. Wegner's known
    MetropolisHastings bugs) are caught per-replica-count and recorded.
    """
    results = []
    print(f"\n{label}")
    print("-" * 96)
    for R in replicas_grid:
        if _GPUS:
            tf.config.experimental.reset_memory_stats('GPU:0')

        system = make_system(R)
        N = int(system.number_spins.numpy())
        sweep_length = sweeps * N
        sim = MetropolisHastings(system)
        tracker = Tracker(measurements=[Energy(system)], granularity=granularity)
        num_flips = cast(tf.Tensor, tf.constant(1))

        try:
            t0 = time.perf_counter()
            sim.sweep(tracker=tracker, beta=tf.constant(1.0), num_disturbances=num_flips, sweep_length=sweep_length)
            _sync()
            t_trace = time.perf_counter() - t0

            if profile:
                options = tf.profiler.experimental.ProfilerOptions(
                    host_tracer_level=2, device_tracer_level=1, python_tracer_level=0)
                # Append cleaned label and R_{R} to the log directory
                run_logdir = os.path.join(OUTPUT_DIR, "profiler_logs", _clean_label(label), f"R_{R}")
                tf.profiler.experimental.start(run_logdir, options=options)

            times = []
            for step_idx in range(repeats):
                t1 = time.perf_counter()
                if profile:
                    with tf.profiler.experimental.Trace("sweep", step_num=step_idx, _r=1):
                        sim.sweep(tracker=tracker, beta=tf.constant(1.0), num_disturbances=num_flips, sweep_length=sweep_length)
                        _sync()
                else:
                    sim.sweep(tracker=tracker, beta=tf.constant(1.0), num_disturbances=num_flips, sweep_length=sweep_length)
                    _sync()
                times.append(time.perf_counter() - t1)
                
            if profile:
                tf.profiler.experimental.stop()

        except Exception as exc:
            print(f"  R={R:>5d} | N={N:>4d} | FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
            results.append({"label": label, "replicas": R, "N": N, "error": str(exc).splitlines()[0]})
            del system, sim, tracker
            gc.collect()
            continue

        t_mean = statistics.mean(times)
        t_std = statistics.stdev(times) if len(times) > 1 else 0.0
        
        steps_per_s_list = [sweep_length / t for t in times]
        steps_per_s = statistics.mean(steps_per_s_list)
        steps_per_s_std = statistics.stdev(steps_per_s_list) if len(steps_per_s_list) > 1 else 0.0
        
        peak_mb = _gpu_peak_mb()

        mem_str = f"{peak_mb:8.1f} MB" if peak_mb is not None else "     n/a"
        print(f"  R={R:>5d} | N={N:>4d} | trace={t_trace:6.2f}s | "
              f"run={t_mean*1000:9.2f}ms (±{t_std*1000:5.2f}) | "
              f"{steps_per_s:>9.0f} (±{steps_per_s_std:>5.0f}) steps/s | peak_mem={mem_str}")

        results.append({
            "label": label, "replicas": R, "N": N,
            "t_trace": t_trace, "t_mean": t_mean, "t_std": t_std,
            "steps_per_s": steps_per_s, "steps_per_s_std": steps_per_s_std,
            "peak_mem_mb": peak_mb,
        })

        del system, sim, tracker
        gc.collect()

    return results


def benchmark_overlap_scaling(replicas_grid: List[int], n_spins: int = 64, repeats: int = 5, profile: bool = False) -> List[Dict[str, Any]]:
    """Isolated O(R²) cost of OverlapDistribution, decoupled from sweep cost."""
    print("\nOverlapDistribution micro-benchmark (replicas² scaling, isolated from sweep)")
    print("-" * 96)
    measurement = OverlapDistribution()
    results = []
    for R in replicas_grid:
        if _GPUS:
            tf.config.experimental.reset_memory_stats('GPU:0')
        state = tf.sign(tf.random.uniform((1, R, n_spins), minval=-1.0, maxval=1.0))

        out = measurement.compute(spin_state=state)
        _sync()
        
        if profile:
            options = tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=2, device_tracer_level=1, python_tracer_level=0)
            run_logdir = os.path.join(OUTPUT_DIR, "profiler_logs", "OverlapDistribution", f"R_{R}")
            tf.profiler.experimental.start(run_logdir, options=options)

        times = []
        for step_idx in range(repeats):
            t0 = time.perf_counter()
            if profile:
                with tf.profiler.experimental.Trace("overlap", step_num=step_idx, _r=1):
                    out = measurement.compute(spin_state=state)
                    _sync()
            else:
                out = measurement.compute(spin_state=state)
                _sync()
            times.append(time.perf_counter() - t0)
            
        if profile:
            tf.profiler.experimental.stop()
            
        t_mean = statistics.mean(times)
        peak_mb = _gpu_peak_mb()
        mem_str = f"{peak_mb:7.2f} MB" if peak_mb is not None else "    n/a"
        print(f"  R={R:>6d} | pairs={out.shape[0]:>9d} | time={t_mean*1000:9.3f}ms | peak_mem={mem_str}")
        results.append({"replicas": R, "t_mean": t_mean, "pairs": int(out.shape[0]), "peak_mem_mb": peak_mb})
    return results


def make_ising_1d(L: int, R: int):
    J = PeriodicNearestNeighborInteraction().generate(1, L)
    return IsingSystem(lattice_dim=1, lattice_length=L, lattice_replicas=R,
                        interaction_matrix=J, initial_magnetization=0.5)

def make_ising_2d(L: int, R: int):
    J = PeriodicNearestNeighborInteraction().generate(2, L)
    return IsingSystem(lattice_dim=2, lattice_length=L, lattice_replicas=R,
                        interaction_matrix=J, initial_magnetization=0.5)

def make_ising_3d(L: int, R: int):
    J = PeriodicNearestNeighborInteraction().generate(3, L)
    return IsingSystem(lattice_dim=3, lattice_length=L, lattice_replicas=R,
                        interaction_matrix=J, initial_magnetization=0.5)

def make_ising(R: int):
    return make_ising_2d(8, R)


def make_ea(R: int):
    L = 4
    nn = PeriodicNearestNeighborInteraction().generate(3, L)
    rnd = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L, quenched=1)
    J = nn * rnd
    return EdwardsAndersonSystem(lattice_length=L, lattice_dim=3, lattice_replicas=R,
                                  interaction_matrix=J, initial_magnetization=0.0)


def make_sk(R: int):
    return SherringtonKirkpatrickSystem(lattice_length=64, lattice_dim=1, lattice_replicas=R,
                                         J=1.0, seed=42, initial_magnetization=0.0)


def make_spherical(R: int):
    L = 8
    J = PeriodicNearestNeighborInteraction().generate(2, L)
    return SphericalSystem(lattice_length=L, lattice_dim=2, lattice_replicas=R,
                            interaction_matrix=J, initial_magnetization=0.0)


def make_wegner(R: int):
    return WegnerSystem(lattice_length=8, lattice_dim=2, lattice_replicas=R)


def _apply_plot_style() -> None:
    """Premium publication-quality style, matching examples/ising.py."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 15,
        'legend.fontsize': 9,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_lattice_scaling(lattice_results: Dict[str, List[Dict[str, Any]]]) -> None:
    _apply_plot_style()
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, len(lattice_results))]
    for color, (label, rows) in zip(colors, lattice_results.items()):
        Ns = [r["N"] for r in rows]
        rates = [r["steps_per_sec"] for r in rows]
        rates_err = [r.get("steps_per_s_std", 0) for r in rows]
        ax.errorbar(Ns, rates, yerr=rates_err, fmt='o-', color=color, markersize=5, linewidth=1.5, capsize=3, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Lattice size N")
    ax.set_ylabel("Steps / s (cached)")
    ax.set_title("Throughput vs. Lattice Size (replicas=64)", weight='bold')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "benchmark_lattice_scaling.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {path}")
    plt.close(fig)


def plot_replica_scaling(model_results: Dict[str, List[Dict[str, Any]]]) -> None:
    _apply_plot_style()
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(model_results))]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for color, (label, rows) in zip(colors, model_results.items()):
        ok_rows = [r for r in rows if "error" not in r]
        reps = [r["replicas"] for r in ok_rows]
        rates = [r["steps_per_s"] for r in ok_rows]
        if reps:
            rates_err = [r.get("steps_per_s_std", 0) for r in ok_rows]
            axs[0].errorbar(reps, rates, yerr=rates_err, fmt='o-', color=color, markersize=5, linewidth=1.5, capsize=3, label=label)

    axs[0].set_xscale("log", base=2)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Replicas")
    axs[0].set_ylabel("Steps / s")
    axs[0].set_title("Throughput vs. Replicas (fixed N≈64)", weight='bold')
    axs[0].legend()

    for color, (label, rows) in zip(colors, model_results.items()):
        ok_rows = [r for r in rows if "error" not in r]
        reps = [r["replicas"] for r in ok_rows]
        mems = [r["peak_mem_mb"] for r in ok_rows]
        if any(m is not None for m in mems):
            axs[1].plot(reps, mems, 'o-', color=color, markersize=5, linewidth=1.5, label=label)

    axs[1].set_xscale("log", base=2)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Replicas")
    axs[1].set_ylabel("Peak GPU memory (MB)")
    axs[1].set_title("Peak GPU Memory vs. Replicas (fixed N≈64)", weight='bold')
    axs[1].legend()

    plt.suptitle("Replica-Scaling Benchmark", y=0.98, weight='bold')
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "benchmark_replicas_scaling.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")
    plt.close(fig)


def plot_overlap_scaling(overlap_results: List[Dict[str, Any]]) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    reps = [r["replicas"] for r in overlap_results]
    times_ms = [r["t_mean"] * 1000 for r in overlap_results]
    ax.plot(reps, times_ms, 'o-', color='firebrick', markersize=6, linewidth=1.5, label="OverlapDistribution time")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Replicas")
    ax.set_ylabel("Time (ms)")
    ax.set_title("OverlapDistribution Cost vs. Replicas (O(R²))", weight='bold')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "benchmark_overlap_scaling.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")
    plt.close(fig)


def plot_cross_dimensional_scaling(cross_results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dims = ["1D", "2D", "3D"]
    
    for i, dim in enumerate(dims):
        ax = axes[i]
        dim_res = cross_results.get(dim, {})
        cmap = plt.get_cmap('viridis')
        colors = [cmap(j) for j in np.linspace(0.1, 0.9, max(1, len(dim_res)))]
        
        for color, (l_label, rows) in zip(colors, dim_res.items()):
            ok_rows = [r for r in rows if "error" not in r]
            reps = [r["replicas"] for r in ok_rows]
            rates = [r["steps_per_s"] for r in ok_rows]
            if reps:
                rates_err = [r.get("steps_per_s_std", 0) for r in ok_rows]
                ax.errorbar(reps, rates, yerr=rates_err, fmt='o-', color=color, markersize=5, linewidth=1.5, capsize=3, label=l_label)
        
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Replicas")
        if i == 0:
            ax.set_ylabel("Steps / s")
        ax.set_title(f"{dim} Ising", weight='bold')
        ax.legend()
        
    plt.suptitle("Cross-Dimensional Scaling: L vs Replicas", y=0.98, weight='bold')
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "benchmark_cross_dimensional.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")
    plt.close(fig)


def plot_stress_test(stress_results: List[Dict[str, Any]]) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ok_rows = [r for r in stress_results if "error" not in r]
    reps = [r["replicas"] for r in ok_rows]
    rates = [r["steps_per_s"] for r in ok_rows]
    
    if reps:
        rates_err = [r.get("steps_per_s_std", 0) for r in ok_rows]
        ax.errorbar(reps, rates, yerr=rates_err, fmt='o-', color='crimson', markersize=6, linewidth=1.5, capsize=3, label="Ising 2D (L=8)")
    
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Replicas")
    ax.set_ylabel("Steps / s")
    ax.set_title("Replica Limit Stress Test", weight='bold')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, "benchmark_stress_test.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("=" * 96)
    print("SPIN SYSTEM BENCHMARK")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {_GPUS[0].name if _GPUS else 'None (CPU only)'}")
    print("=" * 96)

    if _GPUS:
        print("\nWarming up GPU clocks before measurement...")
        _warm_up_gpu()

    replicas = 64
    granularity = 100
    SWEEPS = 10
    all_results: Dict[str, Any] = {}

    # ---- [1] Lattice-size scaling: 2D Ising ----
    print(f"\n[1] Lattice-Size Scaling: 2D Ising (replicas={replicas}, sweeps={SWEEPS})")
    print("-" * 96)
    ising_lattice = []
    for L in [8, 16, 32]:
        N = L * L
        sweep_length = SWEEPS * N
        interaction_matrix = PeriodicNearestNeighborInteraction().generate(2, L)
        system = IsingSystem(
            lattice_dim=2, lattice_length=L, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=1.0,
        )
        r = benchmark_sweep(f"Ising 2D L={L} N={N}", system,
                             [Energy(system), Magnetization(system)], sweep_length, granularity, profile=True)
        r.update(model="Ising 2D", L=L, N=N)
        ising_lattice.append(r)
    all_results["ising_lattice_scaling"] = ising_lattice

    # ---- [2] Lattice-size scaling: 3D EA ----
    print(f"\n[2] Lattice-Size Scaling: 3D Edwards-Anderson (replicas={replicas}, sweeps={SWEEPS})")
    print("-" * 96)
    ea_lattice = []
    for L in [4, 6, 8]:
        N = L ** 3
        sweep_length = SWEEPS * N
        nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
        random_J = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L, quenched=1)
        interaction_matrix = nn_mask * random_J
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_dim=3, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=0.0,
        )
        r = benchmark_sweep(f"EA 3D L={L} N={N}", system,
                             [Energy(system), OverlapDistribution(system)], sweep_length, granularity, profile=True)
        r.update(model="EA 3D", L=L, N=N)
        ea_lattice.append(r)
    all_results["ea_lattice_scaling"] = ea_lattice

    # ---- [3] Single-point reference: SK ----
    print(f"\n[3] Single-Point Reference: Sherrington-Kirkpatrick (replicas={replicas}, sweeps={SWEEPS})")
    print("-" * 96)
    L = N = 64
    sweep_length = SWEEPS * N
    system = SherringtonKirkpatrickSystem(
        lattice_length=L, lattice_dim=1, lattice_replicas=replicas,
        J=1.0, seed=42, initial_magnetization=0.0,
    )
    sk_result = benchmark_sweep(f"SK N={N} (fully connected)", system,
                                 [Energy(system), OverlapDistribution(system)], sweep_length, granularity, profile=True)
    sk_result.update(model="SK", N=N)
    all_results["sk_reference"] = sk_result

    # ---- [4] Single-point reference: Spherical ----
    print(f"\n[4] Single-Point Reference: Spherical 2D (replicas={replicas}, sweeps={SWEEPS})")
    print("-" * 96)
    L = 8
    N = L * L
    sweep_length = SWEEPS * N
    interaction_matrix = PeriodicNearestNeighborInteraction().generate(2, L)
    system = SphericalSystem(
        lattice_length=L, lattice_dim=2, lattice_replicas=replicas,
        interaction_matrix=interaction_matrix, initial_magnetization=0.0,
    )
    spherical_result = benchmark_sweep(f"Spherical 2D L={L} N={N}", system,
                                        [Energy(system)], sweep_length, granularity, profile=True)
    spherical_result.update(model="Spherical 2D", N=N)
    all_results["spherical_reference"] = spherical_result

    # ---- [5] Wegner / Z2 Gauge Theory ----
    print(f"\n[5] Wegner 2D / Z2 Gauge Theory (replicas={replicas}, sweeps={SWEEPS})")
    print("-" * 96)
    print("  NOTE: MetropolisHastings has known shape/ergodicity bugs for gauge")
    print("  models (see PERFORMANCE.md, 'Correctness Bugs Blocking Wegner').")
    print("  This section is expected to fail until those are fixed.")
    L = 8
    N = L * L
    sweep_length = SWEEPS * N
    try:
        system = WegnerSystem(lattice_length=L, lattice_dim=2, lattice_replicas=replicas)
        wegner_result = benchmark_sweep(f"Wegner 2D L={L} N={N}", system,
                                         [Energy(system)], sweep_length, granularity, profile=True)
        wegner_result.update(model="Wegner 2D", N=N)
        all_results["wegner_reference"] = wegner_result
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")
        all_results["wegner_reference"] = {"error": str(exc).splitlines()[0]}

    # ---- [6] Replica scaling across all models ----
    print(f"\n[6] Replica-Scaling: All Models (fixed N≈64)")
    print("-" * 96)
    replicas_grid = [1, 8, 32, 128, 512]
    model_results: Dict[str, List[Dict[str, Any]]] = {}
    model_results["Ising 2D (L=8, N=64)"] = benchmark_replica_scaling(
        "  -- Ising 2D (L=8, N=64) --", make_ising, replicas_grid, profile=True)
    model_results["EA 3D (L=4, N=64)"] = benchmark_replica_scaling(
        "  -- EA 3D (L=4, N=64) --", make_ea, replicas_grid, profile=True)
    model_results["SK (N=64, fully connected)"] = benchmark_replica_scaling(
        "  -- SK (N=64, fully connected) --", make_sk, replicas_grid, profile=True)
    model_results["Spherical 2D (L=8, N=64)"] = benchmark_replica_scaling(
        "  -- Spherical 2D (L=8, N=64) --", make_spherical, replicas_grid, profile=True)
    model_results["Wegner 2D (L=8, N=64 sites/128 links)"] = benchmark_replica_scaling(
        "  -- Wegner 2D (L=8, N=64 sites / 128 links) --", make_wegner, replicas_grid, profile=True)
    all_results["replica_scaling"] = model_results

    # ---- [7] OverlapDistribution micro-benchmark ----
    print(f"\n[7] OverlapDistribution Micro-Benchmark (O(R²) scaling)")
    print("-" * 96)
    overlap_results = benchmark_overlap_scaling([16, 64, 256, 1024, 4096, 16384], profile=True)
    all_results["overlap_scaling"] = overlap_results

    # ---- [8] Measurement overhead comparison ----
    print(f"\n[8] Measurement Overhead Comparison (EA 3D L=4, sweeps={SWEEPS})")
    print("-" * 96)
    L, N = 4, 64
    sweep_length = SWEEPS * N
    nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
    random_J = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L, quenched=1)
    interaction_matrix = nn_mask * random_J

    measurement_overhead = []
    for label, make_meas in [
        ("Energy only", lambda s: [Energy(s)]),
        ("Energy+Mag", lambda s: [Energy(s), Magnetization(s)]),
        ("Energy+Overlap", lambda s: [Energy(s), OverlapDistribution(s)]),
        ("All three", lambda s: [Energy(s), Magnetization(s), OverlapDistribution(s)]),
    ]:
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_dim=3, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=0.0,
        )
        r = benchmark_sweep(f"EA3D L=4 [{label}]", system, make_meas(system), sweep_length, granularity, profile=True)
        measurement_overhead.append(r)
    all_results["measurement_overhead"] = measurement_overhead

    # ---- [9] Granularity impact ----
    print(f"\n[9] Granularity Impact (EA 3D L=4, sweeps={SWEEPS})")
    print("-" * 96)
    granularity_impact = []
    for gran in [50, 100, 500, 1000]:
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_dim=3, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=0.0,
        )
        num_recordings = sweep_length // gran
        r = benchmark_sweep(f"EA3D L=4 gran={gran} ({num_recordings} recs)",
                             system, [Energy(system), OverlapDistribution(system)], sweep_length, gran, profile=True)
        granularity_impact.append(r)
    all_results["granularity_impact"] = granularity_impact

    # ---- [11] Replica Limit Stress Test ----
    print(f"\n[11] Replica Limit Stress Test (Ising 2D L=8, sweeps=10)")
    print("-" * 96)
    stress_grid = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
    stress_results = benchmark_replica_scaling(
        "  -- Ising 2D Stress Test (L=8, N=64) --",
        lambda R: make_ising_2d(8, R),
        stress_grid, sweeps=10, granularity=100, repeats=5, profile=True
    )
    all_results["stress_test"] = stress_results

    # ---- [12] Cross-Dimensional Scaling (L vs R) ----
    print(f"\n[12] Cross-Dimensional Scaling (L vs R)")
    print("-" * 96)
    cross_results = {}
    R_grid_cross = [1, 16, 256, 4096]
    
    print("  -- 1D Ising --")
    cross_results["1D"] = {}
    for L in [16, 64, 256, 1024]:
        cross_results["1D"][f"L={L}"] = benchmark_replica_scaling(
            f"  -- 1D L={L} --", lambda R: make_ising_1d(L, R), R_grid_cross, sweeps=10, granularity=100, repeats=2, profile=True
        )
        
    print("  -- 2D Ising --")
    cross_results["2D"] = {}
    for L in [4, 8, 16, 32]:
        cross_results["2D"][f"L={L}"] = benchmark_replica_scaling(
            f"  -- 2D L={L} --", lambda R: make_ising_2d(L, R), R_grid_cross, sweeps=10, granularity=100, repeats=2, profile=True
        )
        
    print("  -- 3D Ising --")
    cross_results["3D"] = {}
    for L in [2, 4, 6, 8]:
        cross_results["3D"][f"L={L}"] = benchmark_replica_scaling(
            f"  -- 3D L={L} --", lambda R: make_ising_3d(L, R), R_grid_cross, sweeps=10, granularity=100, repeats=2, profile=True
        )
    all_results["cross_dimensional"] = cross_results

    # ---- [10] Projected runtimes for example scripts ----
    print("\n" + "=" * 96)
    print("PROJECTED RUNTIMES FOR EXAMPLE SCRIPTS")
    print("=" * 96)

    rate_by = {}
    for r in ising_lattice + ea_lattice:
        rate_by[(r["model"], r["N"])] = r["steps_per_sec"]

    scripts = [
        {
            "name": "examples/ea_observables.py",
            "configs": [
                {"model": "EA 3D", "L": 4, "N": 64, "sweeps": 4000, "betas": 5},
                {"model": "EA 3D", "L": 6, "N": 216, "sweeps": 3000, "betas": 5},
            ]
        },
        {
            "name": "examples/ea_glass.py",
            "configs": [
                {"model": "EA 3D", "L": 4, "N": 64, "sweeps": 5000, "betas": 25},
                {"model": "EA 3D", "L": 8, "N": 512, "sweeps": 5000, "betas": 25},
            ]
        },
        {
            "name": "examples/ising.py",
            "configs": [
                {"model": "Ising 2D", "L": 8, "N": 64, "sweeps": 2344, "betas": 25},
                {"model": "Ising 2D", "L": 16, "N": 256, "sweeps": 1172, "betas": 25},
                {"model": "Ising 2D", "L": 32, "N": 1024, "sweeps": 586, "betas": 25},
            ]
        },
    ]

    projections = []
    for script in scripts:
        print(f"\n  --- {script['name']} ---")
        total = 0
        for cfg in script["configs"]:
            key = (cfg["model"], cfg["N"])
            sweep_length = cfg["sweeps"] * cfg["N"]
            if key in rate_by:
                rate = rate_by[key]
                t_per_beta = sweep_length / rate
                t_total = t_per_beta * cfg["betas"]
                total += t_total
                print(f"    L={cfg['L']:>2d} N={cfg['N']:>5d}: "
                      f"{cfg['betas']} betas × {sweep_length:>9d} steps "
                      f"@ {rate:.0f} steps/s ≈ {t_total/60:.1f} min")
            else:
                print(f"    L={cfg['L']:>2d} N={cfg['N']:>5d}: (no benchmark data)")
        if total > 0:
            print(f"    TOTAL ≈ {total/60:.1f} min ({total/3600:.2f} hours)")
        projections.append({"script": script["name"], "total_seconds": total})
    all_results["projections"] = projections

    # ---- Save raw results ----
    out_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results: {out_path}")

    # ---- Plots ----
    plot_lattice_scaling({"Ising 2D": ising_lattice, "EA 3D": ea_lattice})
    plot_replica_scaling(model_results)
    plot_overlap_scaling(overlap_results)
    if "stress_test" in all_results:
        plot_stress_test(all_results["stress_test"])
    if "cross_dimensional" in all_results:
        plot_cross_dimensional_scaling(all_results["cross_dimensional"])

    print("\n" + "=" * 96)
    print("DONE")
    print("=" * 96)


if __name__ == "__main__":
    main()
