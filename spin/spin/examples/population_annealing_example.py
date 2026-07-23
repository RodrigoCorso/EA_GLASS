"""
Example script demonstrating Population Annealing dynamics for the 2D Ising model.

Population annealing is a sequential Monte Carlo method that:
1. Maintains a population of replicas at each temperature
2. Gradually anneals from high to low temperature (beta: 0 -> beta_max)
3. Resamples replicas based on their Boltzmann weights at each step
4. Provides an estimate of the partition function

This example shows how to use PopulationAnnealing as a drop-in replacement
for MetropolisHastings with a different sweep interface.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import cast, List, Dict, Any

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import PopulationAnnealing
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization


def generate_betas(num_betas: int = 25, critical_beta: float = 0.44068) -> List[float]:
    """Generates a list of betas concentrated around the critical temperature."""
    dense_range = 0.05
    num_dense = int(0.6 * num_betas)
    num_sparse = num_betas - num_dense

    betas_dense = np.linspace(critical_beta - dense_range,
                              critical_beta + dense_range,
                              num_dense)
    lower_tail = np.linspace(0.1, critical_beta - dense_range - 0.02, num_sparse // 2)
    upper_tail = np.linspace(critical_beta + dense_range + 0.02, 0.8, num_sparse - len(lower_tail))

    betas = np.concatenate([lower_tail, betas_dense, upper_tail])
    betas = np.sort(np.unique(betas))
    return betas.tolist()


def run_simulation():
    # Simulation Parameters
    L_list = [4, 6, 8, 10]  # Reduced for demonstration
    lattice_replicas = 128  # Larger population for better annealing
    betas = generate_betas(15)
    
    # For population annealing, we track at each beta step
    granularity = 1  # Track every beta step
    
    num_flips = cast(tf.Tensor, tf.constant(1))
    equilibration_steps = 10  # Metropolis steps between resampling
    
    # Storage for results
    results_dict: Dict[str, Any] = {'L_list': L_list, 'betas': betas, 'data': {}}
    
    print(f"Starting Ising Population Annealing Simulation.")
    print(f"Lattice sizes: {L_list}")
    print(f"Population size: {lattice_replicas}")
    print(f"Number of betas: {len(betas)}")
    print(f"Critical Beta (Onsager): 0.44068")

    for L in L_list:
        results_dict['data'][str(L)] = {
            'mag': [], 'chi': [], 'cv': [], 'binder': []
        }
        
        N = L * L
        interaction_matrix = PeriodicNearestNeighborInteraction().generate(2, L)
        
        print(f"\n--- Running for Lattice Size L={L} ---")
        
        # Initialize system with random state (better for PA than ordered state)
        ising_system = IsingSystem(
            lattice_dim=2,
            lattice_length=L,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=0.0  # Random initial state
        )
        
        # Use Population Annealing instead of MetropolisHastings
        simulation = PopulationAnnealing(ising_system)
        
        tracker = Tracker(measurements=[
            Energy(ising_system),
            Magnetization(ising_system)
        ], granularity=granularity)
        
        # Convert betas to tensor for the annealing schedule
        beta_schedule = tf.constant(betas, dtype=tf.float32)
        
        # Perform population annealing sweep
        # Note: PA sweeps through ALL betas in one call
        simulation.sweep(
            tracker=tracker,
            beta_schedule=beta_schedule,
            equilibration_steps=tf.constant(equilibration_steps, dtype=tf.int32),
            num_disturbances=num_flips
        )
        
        # Extract data from the full annealing run
        E_hist = tracker.history['Energy'].numpy()  # shape: (num_betas, replicas)
        M_hist = tracker.history['Magnetization'].numpy()
        
        # Normalize energy per spin
        e_hist = E_hist / N
        m_abs_hist = np.abs(M_hist)
        
        # Compute observables at each beta (using all replicas, no burn-in needed for PA)
        for beta_idx, beta in enumerate(betas):
            e_at_beta = e_hist[beta_idx, :]
            m_at_beta = M_hist[beta_idx, :]
            m_abs_at_beta = m_abs_hist[beta_idx, :]
            
            e_avg = np.mean(e_at_beta)
            e2_avg = np.mean(e_at_beta**2)
            
            m_abs_avg = np.mean(m_abs_at_beta)
            m2_avg = np.mean(m_at_beta**2)
            m4_avg = np.mean(m_at_beta**4)
            
            # Critical Observables
            cv = (beta**2) * N * (e2_avg - e_avg**2)
            chi = beta * N * (m2_avg - m_abs_avg**2)
            binder = 1.0 - (m4_avg / (3.0 * m2_avg**2))
            
            results_dict['data'][str(L)]['mag'].append(float(m_abs_avg))
            results_dict['data'][str(L)]['chi'].append(float(chi))
            results_dict['data'][str(L)]['cv'].append(float(cv))
            results_dict['data'][str(L)]['binder'].append(float(binder))
            
            if beta_idx % 5 == 0:
                print(f"  beta={beta:.4f} | |m|={m_abs_avg:.4f} | chi={chi:.4f} | cv={cv:.4f} | U4={binder:.4f}")
        
        # Get partition function estimate
        Z_estimate = simulation.get_partition_function_estimate().numpy()
        print(f"  Partition function estimate: {Z_estimate:.4e}")
            
    # Save results to JSON
    with open('examples/population_annealing_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print("\nSimulation complete. Plotting results...")
    plot_results(results_dict)


def plot_results(results_dict: Dict[str, Any]):
    L_list = results_dict['L_list']
    betas = np.array(results_dict['betas'])
    data = results_dict['data']
    beta_c_exact = 0.44068
    
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
    
    # Using a perceptually uniform colormap to represent size scale
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(L_list))]
    
    # ---------------------------------------------------------
    # 1. Main Observables Plot
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axs = axs.flatten()
    
    for idx, L in enumerate(L_list):
        L_str = str(L)
        color = colors[idx]
        axs[0].plot(betas, data[L_str]['mag'], 'o-', color=color, markersize=3, linewidth=1, label=f'L={L}')
        axs[1].plot(betas, data[L_str]['chi'], 'o-', color=color, markersize=3, linewidth=1, label=f'L={L}')
        axs[2].plot(betas, data[L_str]['cv'], 'o-', color=color, markersize=3, linewidth=1, label=f'L={L}')
        axs[3].plot(betas, data[L_str]['binder'], 'o-', color=color, markersize=3, linewidth=1, label=f'L={L}')
        
    for ax in axs:
        ax.axvline(beta_c_exact, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label=r'$\beta_c$ exact')
        ax.set_xlim(0.1, 0.8)
        
    axs[0].set_ylabel(r'$\langle |m| \rangle$')
    axs[0].set_title('Absolute Magnetization')
    
    axs[1].set_ylabel(r'$\chi$')
    axs[1].set_title('Magnetic Susceptibility')
    
    axs[2].set_ylabel(r'$C_v$')
    axs[2].set_title('Specific Heat')
    axs[2].set_xlabel(r'$\beta$')
    
    axs[3].set_ylabel(r'$U_4$')
    axs[3].set_title('Binder Cumulant')
    axs[3].set_xlabel(r'$\beta$')
    
    # Single unified legend
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='center right', bbox_to_anchor=(1.12, 0.5), borderaxespad=0.)
    
    plt.suptitle('2D Ising Model - Population Annealing', y=0.98, weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    
    plt.savefig('examples/ising_pa_observables.png', dpi=300, bbox_inches='tight')
    print("Saved observables plot to examples/ising_pa_observables.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 2. Finite-Size Scaling Plot
    # ---------------------------------------------------------
    fig_fss, axs_fss = plt.subplots(1, 3, figsize=(16, 5))
    
    L_array = np.array(L_list)
    chi_max = []
    cv_max = []
    m_tc = []
    
    # Find values for scaling
    for L in L_list:
        L_str = str(L)
        chi_max.append(np.max(data[L_str]['chi']))
        cv_max.append(np.max(data[L_str]['cv']))
        m_at_tc = np.interp(beta_c_exact, betas, data[L_str]['mag'])
        m_tc.append(m_at_tc)
        
    chi_max = np.array(chi_max)
    cv_max = np.array(cv_max)
    m_tc = np.array(m_tc)
    log_L = np.log(L_array)
    
    # a) Susceptibility peak scaling
    axs_fss[0].plot(L_array, chi_max, 'ko', markersize=6, label='Data')
    axs_fss[0].set_xscale('log')
    axs_fss[0].set_yscale('log')
    axs_fss[0].set_xlabel('L')
    axs_fss[0].set_ylabel(r'$\chi_{max}$')
    axs_fss[0].set_title(r'Susceptibility Scaling ($\gamma/\nu \approx 1.75$)')
    axs_fss[0].grid(True, alpha=0.3, which="both", ls="--")
    
    slope_chi, intercept_chi = np.polyfit(log_L, np.log(chi_max), 1)
    axs_fss[0].plot(L_array, np.exp(intercept_chi)*L_array**slope_chi, 'r--', 
                    label=f'Fit slope: {slope_chi:.3f}')
    axs_fss[0].legend()
 
    # b) Magnetization scaling at Tc
    axs_fss[1].plot(L_array, m_tc, 'ko', markersize=6, label='Data')
    axs_fss[1].set_xscale('log')
    axs_fss[1].set_yscale('log')
    axs_fss[1].set_xlabel('L')
    axs_fss[1].set_ylabel(r'$m(\beta_c)$')
    axs_fss[1].set_title(r'Magnetization Scaling at $T_c$ ($-\beta/\nu \approx -0.125$)')
    axs_fss[1].grid(True, alpha=0.3, which="both", ls="--")
    
    log_m = np.log(m_tc)
    slope_m, intercept_m = np.polyfit(log_L, log_m, 1)
    axs_fss[1].plot(L_array, np.exp(intercept_m)*L_array**slope_m, 'r--', 
                    label=f'Fit slope: {slope_m:.3f}')
    axs_fss[1].legend()
 
    # c) Specific heat scaling
    axs_fss[2].plot(log_L, cv_max, 'ko', markersize=6, label='Data')
    axs_fss[2].set_xlabel(r'$\ln(L)$')
    axs_fss[2].set_ylabel(r'$C_{v, max}$')
    axs_fss[2].set_title(r'Specific Heat Scaling ($\alpha = 0$)')
    axs_fss[2].grid(True, alpha=0.3)
    
    slope_cv, intercept_cv = np.polyfit(log_L, cv_max, 1)
    axs_fss[2].plot(log_L, slope_cv*log_L + intercept_cv, 'r--', 
                    label=f'Fit: {slope_cv:.3f} ln(L) + {intercept_cv:.3f}')
    axs_fss[2].legend()
    
    plt.suptitle('2D Ising Finite-Size Scaling fits (Population Annealing)', y=0.98, weight='bold')
    plt.tight_layout()
    plt.savefig('examples/ising_pa_fss.png', dpi=300)
    print("Saved finite-size scaling plot to examples/ising_pa_fss.png")
    plt.close()


if __name__ == "__main__":
    run_simulation()
