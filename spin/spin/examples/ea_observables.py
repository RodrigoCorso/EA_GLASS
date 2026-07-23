"""
Edwards-Anderson (EA) Spin Glass — Observables and Parisi Distribution Plot.

This script simulates the 3D EA model across a range of temperatures/betas
for multiple lattice sizes, calculates the Specific Heat, Overlap, Overlap
Susceptibility, and the Parisi Overlap Distribution, and saves a 2x2 grid plot
using premium publication-quality style parameters.

It stores simulation results incrementally per lattice size L into JSON files in
examples/data/ to prevent data loss, enable resuming runs, and group datasets.

This version uses Population Annealing dynamics instead of Metropolis-Hastings.
Population Annealing anneals from beta_min to beta_max in a single sweep,
resampling replicas based on their Boltzmann weights at each temperature step.
"""

import os
import glob
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast, List, Dict, Any

from spin_engine.models import EdwardsAndersonSystem
from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.interactions.standard import BinaryRandomInteraction
from spin_engine.dynamics import PopulationAnnealing
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy
from spin_engine.measurements.correlations import OverlapDistribution


def generate_betas(
    num_betas: int = 25,
    critical_beta: float = 0.892,
    min_beta: float = 0.2,
    max_beta: float = 2.5,
) -> List[float]:
    """Generates a list of betas concentrated around the critical temperature (Tc ~ 1.124 => beta_c ~ 0.892)."""
    dense_range = 0.2
    num_dense = int(0.6 * num_betas)
    num_sparse = num_betas - num_dense

    betas_dense = np.linspace(
        critical_beta - dense_range,
        critical_beta + dense_range,
        num_dense
    )
    lower_tail = np.linspace(
        min_beta,
        critical_beta - dense_range - 0.05,
        num_sparse // 2
    )
    upper_tail = np.linspace(
        critical_beta + dense_range + 0.05,
        max_beta,
        num_sparse - len(lower_tail)
    )

    betas = np.concatenate([lower_tail, betas_dense, upper_tail])
    betas = np.sort(np.unique(betas))
    return betas.tolist()


def run_simulation(is_test: bool = False, force_run: bool = False):
    # Simulation Parameters
    if is_test:
        L_list = [4]
        lattice_replicas = 16
        betas = generate_betas(3, min_beta=0.1, max_beta=1.5)
        sweeps = 200
    else:
        L_list = [8,10,12,14]
        lattice_replicas = 20
        betas = generate_betas(30, min_beta=0.01, max_beta=1.3)
        sweeps = 600
        
    J = 1.0
    coupling_seed = 13
    num_flips = cast(tf.Tensor, tf.constant(1))
    
    data_dir = 'examples/data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Starting Edwards-Anderson Spin Glass Observables Simulation (3D)")
    print(f"Lattice sizes: {L_list}")
    print(f"Number of betas: {len(betas)}")
    print(f"Replicas: {lattice_replicas}")
    print(f"Sweeps per temperature: {sweeps}")
    
    for L in L_list:
        N = L ** 3
        sweep_length = sweeps * N
        num_couplings = int(np.sqrt(N))  # number of independent disorder (quenched) realizations to average over
        # Scale granularity to always record ~200 steps to prevent OOM errors
        granularity = max(1, sweep_length // 200)

        cache_file = os.path.join(data_dir, f"ea_observables_L{L}.json")
        
        # Caching/Resuming Check
        if os.path.exists(cache_file) and not force_run:
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verify that parameters match to prevent loading wrong configurations
                betas_match = len(cached_data.get('betas', [])) == len(betas) and np.allclose(cached_data.get('betas', []), betas)
                params_match = (
                    cached_data.get('lattice_replicas') == lattice_replicas and
                    cached_data.get('sweeps') == sweeps and
                    cached_data.get('granularity') == granularity and
                    cached_data.get('num_couplings') == num_couplings and   # add
                    betas_match
                )

                
                if params_match:
                    print(f"\n--- Found cached data for L={L} ({cache_file}). Skipping simulation. ---")
                    continue
                else:
                    print(f"\n--- Parameter mismatch in cache for L={L}. Re-running simulation. ---")
            except Exception as e:
                print(f"\n--- Error reading cache for L={L} ({e}). Re-running simulation. ---")
        
        # Build quenched coupling matrix (same for all betas)
        nn_mask = PeriodicNearestNeighborInteraction().generate(3, L).astype(np.int8)   # (L,...,L), no quenched axis — this class doesn't take one
        random_J = BinaryRandomInteraction(J=J, seed=coupling_seed).generate(3, L, quenched=num_couplings)  # (num_couplings, L,...,L)
        interaction_matrix = nn_mask * random_J   # broadcasts fine -> (num_couplings, L,...,L)

        # For Population Annealing, we don't need burn-in as the annealing process
        # naturally equilibrates the system. Each beta point gets one measurement.
        
        print(f"\n--- Running for Lattice Size L={L} (sweeps={sweeps}, num_betas={len(betas)}) ---")
        print(f"    Equilibration steps per temperature transition: {equilibration_steps}")
        

        # Build one persistent system/simulation/tracker per coupling
        # For Population Annealing, we need to create a beta schedule from min to max
        system = EdwardsAndersonSystem(
            lattice_length=L,
            lattice_dim=3,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix,
            quenched_variable_replicas=num_couplings,
            initial_magnetization=0.0,
        )
        simulation = PopulationAnnealing(system)
        
        # For PA, we need to determine equilibration steps per temperature step
        # Total sweeps distributed across (num_betas - 1) temperature transitions
        num_beta_steps = len(betas) - 1
        equilibration_steps = max(1, sweeps // num_beta_steps)
        
        tracker = Tracker(
            measurements=[Energy(system), OverlapDistribution(system)],
            granularity=granularity,
        )


        q2_list = []
        chi_sg_list = []
        cv_list = []
        binder_list = []
        parisi_hists = []   # one pooled histogram per beta, across all couplings
        q_bins = None

        # Population Annealing: single sweep through entire beta schedule
        # The sweep method expects beta_schedule as a tensor
        beta_schedule = tf.constant(betas, dtype=tf.float32)
        
        simulation.sweep(
            tracker=tracker,
            beta_schedule=beta_schedule,
            equilibration_steps=tf.constant(equilibration_steps, dtype=tf.int32),
            num_disturbances=num_flips
        )

        # After PA sweep, we have measurements at each beta in the schedule
        # Extract data for each beta point
        E_hist_full = tracker.history['Energy'].numpy()        # (beta_steps, Q, R)
        pq_hist_full = tracker.history['OverlapDistribution'].numpy()  # (beta_steps, Q, pairs)

        for beta_idx, beta in enumerate(betas):
            E_hist = E_hist_full[beta_idx:beta_idx+1, :, :]        # (1, Q, R) for this beta
            pq_hist = pq_hist_full[beta_idx:beta_idx+1, :, :]      # (1, Q, pairs) for this beta

            e_hist = E_hist / N
            e_avg_per_q = e_hist.mean(axis=(0, 2))          # (Q,) — thermal avg per coupling
            e2_avg_per_q = (e_hist**2).mean(axis=(0, 2))
            cv_per_q = (beta**2) * N * (e2_avg_per_q - e_avg_per_q**2)
            cv = float(np.mean(cv_per_q))                    # disorder avg

            pq_flat = pq_hist.reshape(pq_hist.shape[0], pq_hist.shape[1], -1)  # (1, Q, pairs)
            q2_per_q = np.mean(pq_flat**2, axis=(0, 2))
            q4_per_q = np.mean(pq_flat**4, axis=(0, 2))
            q2_avg = float(np.mean(q2_per_q))
            chi_sg = beta * N * q2_avg
            binder = 0.5 * (3.0 - np.mean(q4_per_q / q2_per_q**2)) if np.all(q2_per_q > 0) else 0.0

            # Parisi: pool samples across couplings for this beta
            pq_flat_all = pq_hist.flatten()
            hist, bin_edges = np.histogram(pq_flat_all, bins=100, range=(-1.05, 1.05), density=True)
            if q_bins is None:
                q_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
            parisi_hists.append(hist.tolist())

            q2_list.append(q2_avg)
            chi_sg_list.append(float(chi_sg))
            cv_list.append(cv)
            binder_list.append(float(binder))

            print(f"  beta={beta:.4f} | T={1/beta:.3f} | <q^2> (disorder avg)={q2_avg:.4f}")
    
        # Write results for this size L to its individual JSON file
        size_data = {
            'L': L,
            'lattice_replicas': lattice_replicas,
            'sweeps': sweeps,
            'granularity': granularity,
            'num_couplings': num_couplings,   # add
            'betas': betas,
            'q2': q2_list,
            'chi_sg': chi_sg_list,
            'cv': cv_list,
            'binder': binder_list,
            'parisi_hists': parisi_hists,
            'q_bins': q_bins.tolist() if q_bins is not None else []
        }
        
        with open(cache_file, 'w') as f:
            json.dump(size_data, f)
        print(f"Saved L={L} results to {cache_file}")

        del system, simulation, tracker, interaction_matrix
        import gc; gc.collect()
        tf.keras.backend.clear_session()  # if using a Keras-backed TF session/context
        
    print("\nSimulation phase complete. Compiling all available data for plotting...")
    
    # Compile results for all matching JSON files in examples/data/
    file_pattern = os.path.join(data_dir, "ea_observables_L*.json")
    all_files = glob.glob(file_pattern)
    
    plot_data: Dict[str, Any] = {
        'L_list': [],
        'betas': betas,
        'data': {},
        'parisi': {},
        'q_bins': []
    }
    
    for fp in all_files:
        try:
            with open(fp, 'r') as f:
                d = json.load(f)
            
            # Verify if this file's parameters match the current configuration
            betas_match = len(d.get('betas', [])) == len(betas) and np.allclose(d.get('betas', []), betas)
            L_val = d.get('L', 0)
            expected_granularity = max(1, (sweeps * (L_val**3)) // 200)
            # expected_num_couplings = num_couplings   # matches your `N // 10` formula
            params_match = (
                d.get('lattice_replicas') == lattice_replicas and
                d.get('sweeps') == sweeps and
                d.get('granularity') == expected_granularity and
                # d.get('num_couplings') == expected_num_couplings and   # add
                betas_match
            )
            
            if params_match:
                L_str = str(L_val)
                plot_data['L_list'].append(L_val)
                plot_data['data'][L_str] = {
                    'q2': d['q2'],
                    'chi_sg': d['chi_sg'],
                    'cv': d['cv'],
                    'binder': d.get('binder', [])
                }
                plot_data['parisi'][L_str] = d.get('parisi_hists', [])
                if not plot_data['q_bins'] and 'q_bins' in d:
                    plot_data['q_bins'] = d['q_bins']
        except Exception as e:
            print(f"Warning: Failed to load file {fp} for plotting: {e}")
            
    # Sort the lattice sizes for plotting
    plot_data['L_list'] = sorted(list(set(plot_data['L_list'])))
    
    if not plot_data['L_list']:
        print("Error: No valid datasets found matching current parameters.")
        return
        
    # Save the compiled results summary to JSON for convenience
    os.makedirs('examples/output', exist_ok=True)
    summary_file = 'examples/output/ea_observables_results.json'
    with open(summary_file, 'w') as f:
        json.dump(plot_data, f)
    print(f"Saved compiled summary JSON to {summary_file}")
    
    print(f"Plotting results for lattice sizes: {plot_data['L_list']}")
    plot_results(plot_data)


def plot_results(results_dict: Dict[str, Any]):
    L_list = results_dict['L_list']
    betas = np.array(results_dict['betas'])
    temps = 1.0 / betas
    data = results_dict['data']
    parisi = results_dict['parisi']
    q_bins = np.array(results_dict['q_bins'])
    
    Tc_exact = 0.892  # Expected critical temperature in 3D
    
    # Set premium publication-quality style parameters
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 15,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(L_list))]
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    
    for i, L in enumerate(L_list):
        L_str = str(L)
        color = colors[i]
        
        # 1. Overlap vs T
        axs[0].plot(betas, data[L_str]['q2'], 'o-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 2. Overlap Susceptibility vs T
        axs[1].plot(betas, data[L_str]['chi_sg'], 's-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 3. Binder Cumulant vs T
        axs[2].plot(betas, data[L_str]['binder'], 'd-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 4. Specific Heat vs T
        axs[3].plot(betas, data[L_str]['cv'], '^-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 5. Parisi Distribution at lowest temperature
        if len(parisi[L_str]) > 0:
            lowest_temp_hist = parisi[L_str][-1]  # Highest beta is last
            axs[4].plot(q_bins, lowest_temp_hist, '-', color=color, linewidth=2, label=f'L={L}')
            axs[4].fill_between(q_bins, 0, lowest_temp_hist, color=color, alpha=0.15)

         # --- NEW: Add explicit L label in the top-right corner of each subplot ---
        for ax_idx in range(5):  # Subplots 0 to 4 show all L sizes
            axs[ax_idx].text(0.95, 0.95, f'L = {L}', 
                            transform=axs[ax_idx].transAxes,
                            fontsize=10, 
                            fontweight='bold',
                            color=color,
                            verticalalignment='top', 
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))       


    # 6. Parisi Distribution Heatmap vs T (for largest L)
    largest_L = str(L_list[-1])
    if len(parisi[largest_L]) > 0:
        parisi_matrix = np.array(parisi[largest_L]).T  # shape (len(q_bins), len(betas))
        sort_idx = np.argsort(temps)
        sorted_temps = temps[sort_idx]
        sorted_matrix = parisi_matrix[:, sort_idx]
        
        T_mesh, Q_mesh = np.meshgrid(sorted_temps, q_bins)
        c = axs[5].pcolormesh(T_mesh, Q_mesh, sorted_matrix, cmap='magma', shading='auto')
        fig.colorbar(c, ax=axs[5], label=r'$P(q)$ Density')
        axs[5].axvline(Tc_exact, color='cyan', linestyle=':', linewidth=2, alpha=0.8, label=r'$T_c \approx 1.124$')
    
    # Add labels and format plots
    for idx in range(4):
        ax = axs[idx]
        ax.axvline(Tc_exact, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label=r'$T_c \approx 1.124$')
        ax.set_xlabel(r'$\beta$')
        ax.grid(True, alpha=0.3)
        
    axs[0].set_ylabel(r'$\langle q^2 \rangle$')
    axs[0].set_title('Spin Glass Order Parameter')
    
    axs[1].set_ylabel(r'$\chi_{\rm SG} = \beta N \langle q^2 \rangle$')
    axs[1].set_title('Overlap Susceptibility')
    
    axs[2].set_ylabel(r'$U = \frac{1}{2} \left(3 - \frac{\langle q^4 \rangle}{\langle q^2 \rangle^2}\right)$')
    axs[2].set_title('Binder Cumulant')
    
    axs[3].set_ylabel(r'$C_v$')
    axs[3].set_title('Specific Heat')
    
    axs[4].set_xlabel(r'$q$')
    axs[4].set_ylabel(r'$P(q)$')
    axs[4].set_title(f'Parisi Overlap Distribution at $T = {1.0/betas[-1]:.2f}$')
    axs[4].grid(True, alpha=0.3)
    axs[4].axvline(0, color='black', linewidth=0.8, alpha=0.5)
    
    axs[5].set_xlabel(r'$\beta$')
    axs[5].set_ylabel(r'$q$')
    axs[5].set_title(f'Evolution of P(q) vs T (L={largest_L})')
    
    # Single unified legend to avoid crowding
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(unique_labels), borderaxespad=0.)
    
    plt.suptitle('3D Edwards-Anderson Spin Glass Observables', y=1.02, weight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs('examples/images', exist_ok=True)
    plt.savefig('examples/images/ea_observables.png', dpi=300, bbox_inches='tight')
    print("Saved premium observables plot to examples/images/ea_observables.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate 3D Edwards-Anderson model.")
    parser.add_argument("--test", action="store_true", help="Run a quick test simulation.")
    parser.add_argument("--force", action="store_true", help="Force re-running simulations even if cache exists.")
    args = parser.parse_args()
    
    run_simulation(is_test=args.test, force_run=args.force)
