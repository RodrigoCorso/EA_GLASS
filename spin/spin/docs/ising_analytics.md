# 2D Ising Model — Analytical Framework

> A self-contained overview of the exact solution, critical exponents, and finite-size scaling for the 2D Ising model, serving as a baseline for `spin_engine` validations.

---

## 1 Hamiltonian

The **2D Ising Model** consists of discrete spins $s_i \in \{-1, +1\}$ on a square lattice of size $N = L \times L$. The standard Hamiltonian with nearest-neighbor interactions (without external magnetic field) is:

$$
H = -J \sum_{\langle i, j \rangle} s_i s_j
$$

where $J > 0$ favors ferromagnetic alignment.

---

## 2 The Onsager Solution

In 1944, Lars Onsager published the exact analytical solution for the 2D Ising model on a square lattice with zero external field. This was a landmark result in statistical mechanics, proving that continuous phase transitions can occur in models with local interactions.

### 2.1 Critical Temperature

The phase transition separates a high-temperature disordered (paramagnetic) phase from a low-temperature ordered (ferromagnetic) phase. The critical inverse temperature $\beta_c = 1 / T_c$ (setting $k_B = 1$, $J = 1$) is exactly:

$$
\sinh(2 \beta_c J) = 1 \implies \beta_c = \frac{1}{2} \ln(1 + \sqrt{2}) \approx 0.4406868
$$

which corresponds to $T_c \approx 2.269185$.

---

## 3 Critical Exponents

Near $T_c$, thermodynamic observables follow power-law scaling relations governed by universal **critical exponents**. For the 2D Ising universality class, these are known exactly:

| Exponent | Value | Physical Observable | Scaling Law near $T_c$ ($t = |T - T_c| / T_c \to 0$) |
|----------|-------|---------------------|-------------------------------------------------------|
| $\alpha$ | $0$ (log) | Specific Heat | $C_v \sim - \ln|t|$ |
| $\beta$  | $1/8$ | Magnetization | $m \sim (-t)^\beta$ for $T < T_c$, $0$ for $T > T_c$ |
| $\gamma$ | $7/4$ | Susceptibility | $\chi \sim |t|^{-\gamma}$ |
| $\delta$ | $15$  | Critical Isotherm | $m \sim H^{1/\delta}$ at $T = T_c$ |
| $\nu$    | $1$   | Correlation Length | $\xi \sim |t|^{-\nu}$ |
| $\eta$   | $1/4$ | Anomalous Dimension | Correlation function $G(r) \sim r^{-(d-2+\eta)}$ at $T_c$ |

These obey the scaling identities (e.g., Rushbrooke: $\alpha + 2\beta + \gamma = 2$, Hyperscaling: $d\nu = 2 - \alpha$).

---

## 4 Simulation Observables

To measure the properties in a Monte Carlo simulation (using `spin_engine` with $N$ spins), we compute averages over the equilibrated trajectory:

### 4.1 Magnetization
The absolute magnetization per spin:
$$
\langle |m| \rangle = \left\langle \left| \frac{1}{N} \sum_i s_i \right| \right\rangle
$$

### 4.2 Magnetic Susceptibility ($\chi$)
Using the fluctuation-dissipation theorem, susceptibility relates to magnetization fluctuations:
$$
\chi = \beta N \left( \langle m^2 \rangle - \langle |m| \rangle^2 \right)
$$

### 4.3 Specific Heat ($C_v$)
Similarly, specific heat relates to energy fluctuations ($e = E/N$):
$$
C_v = \beta^2 N \left( \langle e^2 \rangle - \langle e \rangle^2 \right)
$$

### 4.4 Binder Cumulant ($U_4$)
The Binder cumulant is a dimensionless ratio of moments of the magnetization:
$$
U_4 = 1 - \frac{\langle m^4 \rangle}{3 \langle m^2 \rangle^2}
$$
The value of $U_4$ is largely independent of system size exactly at $T_c$.

---

## 5 Finite-Size Scaling (FSS)

In a finite system of size $L$, divergences at $T_c$ are rounded off. The peak values depend on $L$, allowing us to extract critical exponents via Finite-Size Scaling.

### 5.1 Finding $T_c$ via Binder Cumulant
Because $U_4$ has scaling dimension 0, plotting $U_4(T)$ for different lattice sizes $L$ yields curves that **intersect exactly at $T_c$**. The slope at the intersection scales as $L^{1/\nu}$, providing a way to extract $\nu$.

### 5.2 Extracting $\gamma/\nu$ (Susceptibility)
The peak of the susceptibility $\chi_{max}$ scales with system size as:
$$
\chi_{max}(L) \sim L^{\gamma / \nu}
$$
In a log-log plot of $\chi_{max}$ vs $L$, the slope yields $\gamma/\nu = 1.75$.

### 5.3 Extracting $\beta/\nu$ (Magnetization)
At the critical temperature $T_c$, the magnetization scales as:
$$
\langle |m| \rangle (T_c, L) \sim L^{-\beta / \nu}
$$
A log-log plot of $\langle |m| \rangle(T_c)$ vs $L$ yields a slope of $-\beta/\nu = -0.125$.

### 5.4 Extracting $\alpha$ (Specific Heat)
Since $\alpha = 0$, the specific heat does not follow a power law but rather a logarithmic divergence:
$$
C_v^{max}(L) \sim A \ln L + B
$$
Plotting $C_v^{max}$ against $\ln L$ yields a straight line.

---

## 6 Monte Carlo Simulation & Optimization Analysis

To run large-scale benchmarks of the 2D Ising model up to $L = 64$ within reasonable compute times, several computational and physical optimizations were integrated into the `spin_engine` simulation workflow (`examples/ising.py`).

### 6.1 Computational & TensorFlow Graph Optimizations

Running Monte Carlo simulations within TensorFlow graph mode (`@tf.function`) introduces significant performance benefits but requires strict adherence to graph-compilation rules:

*   **Retracing Avoidance via Constant Wrapping**:
    In earlier iterations, the temperature parameter `beta` was passed as a raw Python float to the dynamics sweep function. Under `@tf.function`, passing varying Python values triggers a full recompilation of the computation graph on each step. Wrapping the parameter in a TensorFlow constant:
    ```python
    beta=tf.constant(beta, dtype=tf.float32)
    ```
    allows the compiled graph to accept `beta` as a dynamic input tensor, eliminating the compilation overhead. This resulted in an execution speedup of multiple orders of magnitude.
*   **Static Graph Variable Management**:
    Instantiating the `Tracker` object (which allocates internal `tf.Variable` instances for measurements) inside the temperature loop caused graph compilation failures. By instantiating the `Tracker` once per lattice size $L$ outside the temperature loop, we enforce a static graph layout and reuse the existing tensor buffers across all temperature steps.
*   **JSON Serialization Sanitization**:
    All values tracked and returned from the graph are TensorFlow or NumPy datatypes (e.g., `tf.float32`, `np.float32`). Trying to serialize them directly into JSON causes runtime errors. The data collector explicitly casts keys to strings (`str(L)`) and converts values to native Python types (`float(...)`) before exporting.

### 6.2 Physics & Phase Space Exploration Corrections

Simulating magnetic systems near critical points is challenging due to topological domain walls and critical slowing down:

*   **Transition from Quenching to Annealing (State Persistence)**:
    Re-initializing the spin configuration to a random state at each new $\beta$ represents a thermal *quench* ($T = \infty \to T$). Below the critical temperature $\beta > \beta_c$, the sudden quench traps the lattice in metastable configurations separated by domain walls (stripes of spins of opposite signs). The system lacks the thermodynamic energy to dissolve these domain boundaries using local Metropolis moves, causing the magnetization to remain low and the susceptibility $\chi$ to explode. By **reusing the spin state** from the previous temperature step, the lattice transitions continuously, allowing it to equilibrate into the true ferromagnetic ground state.
*   **Cold-to-Hot Sweeping (Reverse Annealing)**:
    Even with continuous annealing, cooling a large lattice ($L = 64$) from hot to cold can trigger the **Kibble-Zurek mechanism**, where local domains freeze out and form topological defect lines as the correlation length attempts to diverge near $T_c$. To guarantee topological defect-free convergence at low temperatures:
    1.  The simulation is swept in **reverse direction** (from cold to hot: $\beta_{max} \to \beta_{min}$).
    2.  The lattice is initialized in a fully ordered state:
        ```python
        initial_magnetization=1.0
        ```
    Starting from a perfect ferromagnetic configuration at $T=0$ ensures that the system starts in the global energy minimum. As we heat the system, it remains in thermal equilibrium and correctly displays the low-temperature drop in susceptibility $\chi \to 0$.
*   **Critical Slowing Down at Large L**:
    At the critical point $\beta_c$, the correlation length $\xi$ diverges, and the autocorrelation time $\tau$ of local single-spin Metropolis flips scales as:
    ```
    \tau \sim L^z \quad (with \ z \approx 2)
    ```
    Consequently, larger lattices ($L=64$) experience a dramatic slowing down in phase space exploration. While $30,000$ sweeps are more than sufficient to fully decorrelate and equilibrate $L=16$ and $L=32$ systems (resulting in accurate scaling fits), the peak of the susceptibility for $L=64$ is slightly damped and broadened under single-spin dynamics compared to the analytical infinite-limit scaling peak.
