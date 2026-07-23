# Replica Scaling Discovery & Analysis

## Overview

We conducted a deep dive into the Spin System's scaling capabilities along the batch (`replicas`) dimension. The primary goal was to answer a critical research question: **Can we repurpose the replica axis to scale quenched variables or simulate independent parts of a single system (effectively composing replicas to achieve a larger effective lattice size $L$)?**

To answer this, we implemented two new suites of benchmarks:
1.  **Replica Limit Stress Test:** Pushing the system to extreme replica counts for a fixed, small lattice ($N=64$) to find the memory and throughput limits.
2.  **Cross-Dimensional Scaling ($L$ vs. $R$):** Mapping out the performance landscape across different dimensions ($d=1, 2, 3$) while simultaneously varying lattice size ($L$) and replica count ($R$).

## 1. Replica Limit Stress Test

We ran the 2D Ising model on an $L=8$ ($N=64$) lattice and exponentially increased the number of replicas up to **4.19 Million** ($2^{22}$).

![Stress Test](../examples/images/benchmark_stress_test.png)

### Key Findings:
- **Massive Capacity:** The engine successfully allocated and simulated $4,194,304$ independent replicas of $N=64$ spins.
- **Memory Scaling:** At ~4.19M replicas, the peak GPU memory footprint was approximately **5.4 GB**. This indicates a highly efficient memory layout. The memory scales strictly linearly with the number of replicas, as expected for independent states.
- **Throughput Decay:** While memory scaled perfectly, throughput (measured in `steps/s`) began to degrade exponentially after passing $~16,384$ replicas. 
    - At $R=16,384$, we saw $\sim 8,941$ steps/s.
    - At $R=4,194,304$, we saw $\sim 77$ steps/s.
    - *Reasoning:* For extremely large tensors, the operations become heavily **memory bandwidth bound**. Moving 5.4 GB of memory across the GPU bus for a single Monte Carlo sweep simply takes time, even if the compute is fully vectorized.

## 2. Cross-Dimensional Scaling

We tested 1D, 2D, and 3D Ising models for various $L$ sizes and replica counts ($R \in \{1, 16, 256, 4096\}$).

![Cross-Dimensional Scaling](../examples/images/benchmark_cross_dimensional.png)

### Key Findings:
- **Consistent Scaling Across Dimensions:** The replica scaling behavior is nearly identical regardless of the underlying lattice dimensionality. Whether it's 1D, 2D, or 3D, the performance drop-off begins around the same point (when the total tensor size begins to saturate the GPU's L2 cache and bandwidth limits).
- **Larger $L$ vs. Larger $R$:** 
    - Simulating one large lattice (e.g., 2D $L=32$, $N=1024$, $R=1$) runs at $\sim 11,500$ steps/s.
    - Simulating an equivalent number of spins via replicas (e.g., 2D $L=4$, $N=16$, $R=64$) runs slightly slower ($\sim 7,500$ steps/s).
    - *Conclusion:* The GPU is slightly more efficient at processing contiguous spatial dimensions (larger $L$) than larger batch dimensions (larger $R$), but both are well within the same order of magnitude.

## Conclusion: Composing Replicas for Larger Systems

> [!TIP]
> **Yes, you can absolutely use the replica axis to simulate massive arrays of independent systems or different realizations of quenched variables.**

**Guidelines for Repurposing the Replica Axis:**
1.  **Quenched Disorder (Spin Glasses):** This architecture is perfect for Spin Glasses (like EA or SK models). You can easily pack $10,000+$ replicas of a system to average over different random interaction matrices $J_{ij}$ simultaneously. 
2.  **Domain Decomposition (Composing larger $L$):** If you break a massive system into smaller independent sub-systems (e.g., using mean-field boundaries or neglecting boundary interactions), packing them into the replica axis is completely viable. The GPU will easily handle millions of such subsystems as long as they fit in VRAM.
3.  **Optimal Throughput:** To maximize raw `steps/s` throughput, aim to keep the total system size ($R \times N$) such that the VRAM footprint is under $\sim 500$ MB. Pushing into the multi-GB range will linearly slow down the simulation steps per second due to memory bandwidth limits.
4.  **Overcoming the Memory Wall (Advanced):** If you need to push past $10^6$ replicas, the bottleneck will be the GPU's Global Memory bandwidth. Because the individual subsystems are small ($N=64$), this can be completely solved in the future by writing a custom CUDA kernel (or using highly-fused JAX operations). A custom kernel would load a replica into the GPU's ultra-fast Shared Memory once, perform thousands of Monte Carlo sweeps entirely in-cache, and write back to Global Memory only when finished. This would allow millions of replicas to simulate at the theoretical maximum compute limit of the GPU.
