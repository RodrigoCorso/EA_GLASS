"""
Sherrington-Kirkpatrick (SK) Spin Glass — Temperature Sweep with Overlap Analysis.

This script simulates the SK model (fully connected Ising spins with Gaussian
random couplings J_ij ~ N(0, J/√N)) for several effective dimensions (lattice_dim).
For each dimension it:
  1. Sweeps inverse temperature β around the mean-field critical value β_c = 1/J.
  2. Records magnetization evolution, overlap distribution P(q), and
     the spin-glass susceptibility χ_SG = N · ⟨q²⟩ (whose peak locates β_c).
  3. Plots four panels per dimension:
       (a) magnetization traces  (b) ⟨q²⟩ vs β
       (c) χ_SG vs β             (d) P(q) kernel-density estimates
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast

from spin_engine.models import SherringtonKirkpatrickSystem
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization
from spin_engine.measurements.correlations import OverlapDistribution


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sk_sweep(
    lattice_length: int,
    lattice_dim: int,
    betas: list[float],
    lattice_replicas: int = 64,
    J: float = 1.0,
    sweeps: int = 5000,
    granularity: int = 100,
    seed: int | None = 42,
):
    """Run SK model across *betas* and return per-β measurement history."""
    num_flips = cast(tf.Tensor, tf.constant(1))
    results: dict = {}

    N = lattice_length ** lattice_dim
    sweep_length = sweeps * N
    print(f"  SK: L={lattice_length}, D={lattice_dim}, N={N}, "
          f"replicas={lattice_replicas}, sweeps={sweeps} (total steps={sweep_length})")

    # Initialize ONCE outside the loop to enable Simulated Annealing (hot to cold)
    system = SherringtonKirkpatrickSystem(
        lattice_length=lattice_length,
        lattice_dim=lattice_dim,
        lattice_replicas=lattice_replicas,
        J=J,
        initial_magnetization=0.0, # Hot state (T=infinity)
        seed=seed,
    )
    sim = MetropolisHastings(system)

    tracker = Tracker(
        measurements=[
            Energy(system),
            Magnetization(system),
            OverlapDistribution(system),
        ],
        granularity=granularity,
    )

    for beta in betas:
        print(f"    β = {beta:.4f}")

        sim.sweep(
            tracker=tracker,
            beta=tf.constant(beta, dtype=tf.float32),
            num_disturbances=num_flips,
            sweep_length=sweep_length,
        )

        beta_data: dict = {}
        for name, var in tracker.history.items():
            beta_data[name] = var.numpy()
        results[beta] = beta_data

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _equilibrium_q2(results: dict, betas: list[float]):
    """Return arrays of ⟨q²⟩ averaged over the second half of each sweep."""
    q2_means, q2_stds = [], []
    for beta in betas:
        pq = results[beta]["OverlapDistribution"]
        half = len(pq) // 2
        q_flat = pq[half:]                # (steps, 1, n_pairs)
        q2_per_step = np.mean(q_flat ** 2, axis=2)
        q2_means.append(np.mean(q2_per_step))
        q2_stds.append(np.std(q2_per_step))
    return np.array(q2_means), np.array(q2_stds)


def plot_sk_results(
    all_results: dict[int, dict],
    dims: list[int],
    betas: list[float],
    N_per_dim: dict[int, int],
    J: float = 1.0,
):
    """Create a multi-row figure: one row per dimension."""
    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 4, figsize=(22, 5 * n_dims))
    if n_dims == 1:
        axes = axes[np.newaxis, :]

    temps = np.array([1.0 / b for b in betas])

    for row, D in enumerate(dims):
        results = all_results[D]
        N = N_per_dim[D]
        q2_means, q2_stds = _equilibrium_q2(results, betas)

        # (a) Magnetization traces (last β only — deeply frozen)
        ax_mag = axes[row, 0]
        last_beta = betas[-1]
        mag = results[last_beta]["Magnetization"]
        steps = np.arange(mag.shape[0]) * 100
        for r in range(min(10, mag.shape[2])):
            ax_mag.plot(steps, mag[:, 0, r], alpha=0.3, linewidth=0.8)
        ax_mag.plot(steps, np.mean(mag, axis=(1, 2)), "k--", lw=2, label="Mean")
        ax_mag.set_title(f"D={D}  Magnetization (β={last_beta})")
        ax_mag.set_xlabel("MC step")
        ax_mag.set_ylabel("m")
        ax_mag.legend(fontsize=8)
        ax_mag.grid(alpha=0.3)

        # (b) ⟨q²⟩ vs T
        ax_q2 = axes[row, 1]
        ax_q2.errorbar(temps, q2_means, yerr=q2_stds,
                       fmt="o-", color="crimson", capsize=4, markersize=5)
        ax_q2.axvline(x=J, ls="--", color="grey", label=r"$T_c^{\rm MF}=J$")
        ax_q2.set_xlabel(r"$T = 1/\beta$")
        ax_q2.set_ylabel(r"$\langle q^2 \rangle$")
        ax_q2.set_title(f"D={D}  Order Parameter")
        ax_q2.legend(fontsize=8)
        ax_q2.grid(alpha=0.3)

        # (c) Spin-glass susceptibility χ_SG = N·⟨q²⟩
        ax_chi = axes[row, 2]
        chi = N * q2_means
        chi_err = N * q2_stds
        ax_chi.errorbar(temps, chi, yerr=chi_err,
                        fmt="s-", color="teal", capsize=4, markersize=5)
        peak_idx = np.argmax(chi)
        ax_chi.axvline(x=temps[peak_idx], ls=":", color="red",
                       label=rf"peak $T \approx {temps[peak_idx]:.2f}$")
        ax_chi.axvline(x=J, ls="--", color="grey", label=r"$T_c^{\rm MF}=J$")
        ax_chi.set_xlabel(r"$T$")
        ax_chi.set_ylabel(r"$\chi_{\rm SG} = N\,\langle q^2\rangle$")
        ax_chi.set_title(f"D={D}  Susceptibility")
        ax_chi.legend(fontsize=8)
        ax_chi.grid(alpha=0.3)

        # (d) P(q) for selected temperatures
        ax_pq = axes[row, 3]
        selected = [b for b in [0.5, 0.8, 1.0, 1.5, 2.0] if b in results]
        palette = sns.color_palette("coolwarm", len(selected))
        for i, beta in enumerate(selected):
            pq = results[beta]["OverlapDistribution"]
            half = len(pq) // 2
            q_vals = pq[half:].flatten()
            sns.kdeplot(q_vals, ax=ax_pq, color=palette[i], fill=True,
                        alpha=0.2, label=f"T={1/beta:.2f}")
        ax_pq.set_xlabel(r"$q$")
        ax_pq.set_ylabel(r"$P(q)$")
        ax_pq.set_title(f"D={D}  Overlap Distribution")
        ax_pq.legend(fontsize=8)
        ax_pq.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("examples/images", exist_ok=True)
    path = "examples/images/sk_glass_overlap.png"
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    betas = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    J = 1.0

    # SK is a mean-field (d→∞) model, so "dimension" only changes N = L^D.
    # We run D=1 (N=128) and D=2 (N=10²=100) for comparison.
    configs = [
        {"lattice_dim": 1, "lattice_length": 128},
        {"lattice_dim": 2, "lattice_length": 10},
    ]

    dims = [c["lattice_dim"] for c in configs]
    N_per_dim = {c["lattice_dim"]: c["lattice_length"] ** c["lattice_dim"]
                 for c in configs}
    all_results: dict[int, dict] = {}

    for cfg in configs:
        D = cfg["lattice_dim"]
        L = cfg["lattice_length"]
        print(f"\n=== SK  D={D}  L={L}  N={L**D} ===")
        all_results[D] = run_sk_sweep(
            lattice_length=L,
            lattice_dim=D,
            betas=betas,
            lattice_replicas=64,
            J=J,
            sweeps=5000,
            granularity=100,
        )

    plot_sk_results(all_results, dims, betas, N_per_dim, J=J)


if __name__ == "__main__":
    main()
