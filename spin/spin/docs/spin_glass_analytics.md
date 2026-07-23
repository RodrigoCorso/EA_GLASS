# Spin Glass Models — Analytical Framework

> A self-contained derivation of the critical temperatures, order parameters,
> and susceptibilities used in the `spin_engine` simulations.

---

## 1  Notation

| Symbol | Meaning |
|--------|---------|
| $N$ | Total number of spins ($N = L^d$) |
| $d$ | Spatial dimension of the lattice |
| $\beta = 1/T$ | Inverse temperature (we set $k_B = 1$) |
| $J$ | Coupling scale |
| $J_{ij}$ | Quenched random coupling between sites $i$ and $j$ |
| $s_i \in \{-1, +1\}$ | Ising spin at site $i$ |
| $\langle \cdot \rangle$ | Thermal (Boltzmann) average at fixed disorder |
| $[\cdot]_J$ | Average over disorder realisations |

---

## 2  The Sherrington–Kirkpatrick (SK) Model

### 2.1  Hamiltonian

The SK model is a **fully connected** Ising system:

$$
H = -\sum_{i < j} J_{ij}\, s_i\, s_j
$$

with couplings drawn independently from a Gaussian distribution:

$$
J_{ij} \sim \mathcal{N}\!\left(0,\;\frac{J^2}{N}\right)
$$

The $1/N$ scaling ensures an extensive free energy in the thermodynamic limit.

### 2.2  Replica Method and the Free Energy

To compute the disorder-averaged free energy $[F]_J = -T\,[\ln Z]_J$, one uses
the **replica trick**:

$$
[\ln Z]_J = \lim_{n \to 0} \frac{[Z^n]_J - 1}{n}
$$

Introducing $n$ copies (replicas) $s_i^\alpha$ ($\alpha = 1, \ldots, n$) and
performing the Gaussian integral over disorder yields:

$$
[Z^n]_J = \operatorname{Tr}_{\{s^\alpha\}} \exp\!\left(
  \frac{\beta^2 J^2}{2N}\sum_{\alpha < \beta}\left(\sum_i s_i^\alpha s_i^\beta\right)^2
\right)
$$

This introduces the **overlap order parameter**:

$$
q_{\alpha\beta} = \frac{1}{N}\sum_{i=1}^{N} s_i^\alpha\, s_i^\beta
$$

### 2.3  Replica-Symmetric (RS) Solution

Under the **replica-symmetric** (RS) ansatz $q_{\alpha\beta} = q$ for all $\alpha \neq \beta$, the saddle-point equation gives:

$$
\boxed{q = \int_{-\infty}^{\infty} \frac{e^{-z^2/2}}{\sqrt{2\pi}}\,\tanh^2\!\left(\beta J\sqrt{q}\;z\right)\,dz}
$$

This is a self-consistency equation for $q$.

### 2.4  Critical Temperature

Near the transition $q \to 0$, we expand $\tanh^2(x) \approx x^2$ for small $x$:

$$
q \approx \beta^2 J^2 q \int \frac{e^{-z^2/2}}{\sqrt{2\pi}}\, z^2\, dz = \beta^2 J^2 q
$$

A non-trivial solution $q > 0$ first appears when:

$$
\boxed{\beta_c^2 J^2 = 1 \quad \Longrightarrow \quad T_c = J}
$$

For $J = 1$, this gives $\beta_c = 1$ (i.e.\ $T_c = 1$).

### 2.5  Spin-Glass Susceptibility

The **spin-glass susceptibility** is defined as:

$$
\chi_{\mathrm{SG}} = N\,\left[\langle q^2 \rangle\right]_J
$$

where $q$ is the overlap between two independent replicas. In the paramagnetic
phase ($T > T_c$), $\chi_{\mathrm{SG}}$ satisfies:

$$
\chi_{\mathrm{SG}} = \frac{1}{1 - \beta^2 J^2}
$$

which **diverges** at $T_c = J$. In a finite system, this divergence is rounded
into a peak whose position converges to $T_c$ as $N \to \infty$.

> [!IMPORTANT]
> In simulations, the peak of $\chi_{\mathrm{SG}} = N\langle q^2\rangle$ vs $T$
> is the primary method for locating $\beta_c$ numerically.

### 2.6  de Almeida–Thouless (AT) Instability

The RS solution is **unstable** below $T_c$. The de Almeida–Thouless condition
shows that the Hessian of the free energy in replica space has a negative
eigenvalue when:

$$
\beta^2 J^2 \int \frac{e^{-z^2/2}}{\sqrt{2\pi}}\,\operatorname{sech}^4\!\left(\beta J\sqrt{q}\;z\right) dz > 1
$$

Below this line (the **AT line**), the system enters a phase with **replica
symmetry breaking** (RSB), described by the Parisi solution.

### 2.7  Parisi Solution (RSB)

In the full Parisi picture, the overlap $q_{\alpha\beta}$ is no longer a
single number but a **function** $q(x)$ for $x \in [0, 1]$. The overlap
distribution $P(q)$ becomes non-trivial:

- **$T > T_c$** (paramagnet): $P(q) = \delta(q)$ — all replicas uncorrelated.
- **$T < T_c$** (spin glass): $P(q)$ develops structure; $\langle q^2 \rangle - \langle |q| \rangle^2 > 0$.

The **Parisi overlap parameter** we compute in `ParisiOverlapParameter` is:

$$
\Delta_P = \langle q^2 \rangle - \langle |q| \rangle^2
$$

which vanishes in the RS phase and becomes positive in the RSB phase.

### 2.8  Dimension Independence of $T_c$ in the SK Model

Because the SK model is **fully connected** (mean-field, effectively $d \to \infty$),
the critical temperature $T_c = J$ is **independent of the lattice dimension**.
Different values of `lattice_dim` in the simulation change only $N = L^d$
(the system size), not the physics. The SK model always has:

$$
T_c = J \quad \forall\; d
$$

Finite-size effects shift the apparent peak of $\chi_{\mathrm{SG}}$, but in the
thermodynamic limit the transition is universal at $T_c = J$.

---

## 3  The Edwards–Anderson (EA) Model

### 3.1  Hamiltonian

The EA model restricts interactions to **nearest neighbours** on a $d$-dimensional
hypercubic lattice with periodic boundary conditions:

$$
H = -\sum_{\langle i,j \rangle} J_{ij}\, s_i\, s_j
$$

The couplings $J_{ij}$ are quenched random variables. Two standard distributions:

| Distribution | $P(J_{ij})$ | Used in |
|---|---|---|
| Binary ($\pm J$) | $\frac{1}{2}\delta(J_{ij}-J) + \frac{1}{2}\delta(J_{ij}+J)$ | `BinaryRandomInteraction` |
| Gaussian | $\frac{1}{\sqrt{2\pi}J}\exp(-J_{ij}^2/2J^2)$ | `GaussianInteraction` |

### 3.2  Edwards–Anderson Order Parameter

The EA order parameter is defined as:

$$
q_{\mathrm{EA}} = \frac{1}{N}\sum_{i=1}^{N} \left[\langle s_i \rangle^2\right]_J
$$

In practice, the thermal average $\langle s_i \rangle$ is approximated by a
time average over Monte Carlo steps. In a paramagnetic phase, thermal fluctuations
ensure $\langle s_i \rangle \approx 0$, so $q_{\mathrm{EA}} = 0$. In the spin-glass
phase, spins freeze into random but definite orientations, so $q_{\mathrm{EA}} > 0$.

### 3.3  Overlap and Susceptibility

The overlap between two replicas $\alpha, \beta$ simulated with the **same**
disorder realisation is:

$$
q_{\alpha\beta} = \frac{1}{N}\sum_{i=1}^{N} s_i^\alpha\, s_i^\beta
$$

The spin-glass susceptibility is:

$$
\chi_{\mathrm{SG}} = N\,\left[\langle q^2 \rangle\right]_J
$$

As with the SK model, the **peak** of $\chi_{\mathrm{SG}}$ as a function of
temperature locates the phase transition.

### 3.4  Critical Temperature: Dependence on Dimension

The EA model's critical behaviour depends strongly on dimension $d$:

#### Lower Critical Dimension

The **lower critical dimension** for the Ising spin glass is:

$$
d_l \approx 2.5
$$

Below $d_l$, thermal fluctuations destroy any spin-glass ordering at any
finite temperature: $T_c = 0$.

#### $d = 2$: No Finite-Temperature Transition

Rigorous and numerical results establish:

$$
\boxed{T_c^{(d=2)} = 0}
$$

The 2D EA model has a **zero-temperature** phase transition only. The spin-glass
susceptibility $\chi_{\mathrm{SG}}$ diverges as $T \to 0$ but shows **no finite-$T$ peak**.

In simulations, this manifests as $\chi_{\mathrm{SG}}$ growing monotonically as
$T \to 0$ without any turnover.

#### $d = 3$: Finite-Temperature Transition

Extensive numerical work (e.g., Katzgraber & Young, 2006; Baity-Jesi et al., 2013)
gives for the binary ($\pm J$) distribution:

$$
\boxed{T_c^{(d=3)} \approx 1.1\,J}
$$

with critical exponents:

| Exponent | Value | Meaning |
|---|---|---|
| $\nu$ | $\approx 2.56$ | Correlation length: $\xi \sim |T - T_c|^{-\nu}$ |
| $\eta$ | $\approx -0.39$ | Anomalous dimension |
| $\beta_{\rm mag}$ | $\approx 0.77$ | Order parameter: $q_{\mathrm{EA}} \sim (T_c - T)^{\beta_{\rm mag}}$ |

For the **Gaussian** distribution, the critical temperature is slightly lower:

$$
T_c^{(d=3, \text{Gauss})} \approx 0.95\,J
$$

#### $d \geq 6$: Mean-Field Behaviour

For $d \geq d_u = 6$ (the **upper critical dimension**), the EA model recovers
mean-field (SK) critical behaviour:

$$
T_c \to 2dJ \cdot (\text{coordination-dependent factor})
$$

and the critical exponents take their mean-field values ($\nu = 1/2$, etc.).

### 3.5  Scaling of $\chi_{\mathrm{SG}}$ Near $T_c$ (d = 3)

Near the transition, the susceptibility scales as:

$$
\chi_{\mathrm{SG}} \sim |T - T_c|^{-\gamma} \quad \text{with} \quad \gamma = \nu(2 - \eta) \approx 6.1
$$

The large value of $\gamma$ means the peak in $\chi_{\mathrm{SG}}$ is **very sharp**,
making it an excellent probe of $T_c$ even on small lattices.

---

## 4  Connection to Simulation Observables

The quantities computed by the `spin_engine` measurement classes correspond
directly to the theoretical definitions:

| Measurement Class | Formula | Physics |
|---|---|---|
| `OverlapMatrix` | $Q_{\alpha\beta} = \frac{1}{N}\sum_i s_i^\alpha s_i^\beta$ | Replica overlap matrix |
| `OverlapDistribution` | Off-diagonal elements of $Q$ | Samples from $P(q)$ |
| `ParisiOverlapParameter` | $\langle q^2\rangle - \langle \lvert q\rvert\rangle^2$ | RSB indicator |
| `SpinGlassOrderParameter` | $\frac{1}{N}\sum_i \langle s_i\rangle_t^2$ | Edwards–Anderson $q_{\mathrm{EA}}$ |
| `MagneticSusceptibility` | $\mathrm{Var}_{\alpha}(m_\alpha)$ | Magnetic response (≈ 0 in SG) |

### 4.1  Extracting $T_c$ from Simulations

The primary method used in the example scripts:

1. **Compute** $\langle q^2 \rangle$ at each temperature by averaging
   $(q_{\alpha\beta})^2$ over all replica pairs and the equilibrated portion
   of the Monte Carlo trajectory.

2. **Form** $\chi_{\mathrm{SG}}(T) = N \cdot \langle q^2 \rangle(T)$.

3. **Locate** the peak of $\chi_{\mathrm{SG}}(T)$.

For a finite system, the peak is at $T_{\rm peak}(L) > T_c$. With finite-size
scaling:

$$
T_{\rm peak}(L) - T_c \sim L^{-1/\nu}
$$

one can extrapolate $T_c$ from simulations at multiple $L$ values.

### 4.2  Why Magnetization is Zero

In a spin glass, the disorder-averaged magnetization vanishes:

$$
[m]_J = \left[\frac{1}{N}\sum_i \langle s_i \rangle\right]_J = 0
$$

This is because the random couplings produce no preferred global orientation.
The system is **not** paramagnetic (spins are frozen), but the frozen pattern
is random. This is why magnetization is useless as an order parameter for spin
glasses, and one must use the **overlap** $q$ instead.

---

## 5  Summary of Critical Values

| Model | Dimension | $T_c / J$ | Method | Source |
|---|---|---|---|---|
| SK | any (mean-field) | **1.0** | $\beta^2 J^2 = 1$ | Sherrington & Kirkpatrick (1975) |
| EA ($\pm J$) | $d = 2$ | **0** | $d < d_l \approx 2.5$ | McMillan (1984), Bray & Moore (1984) |
| EA ($\pm J$) | $d = 3$ | **≈ 1.1** | MC + FSS | Katzgraber et al. (2006) |
| EA (Gauss) | $d = 3$ | **≈ 0.95** | MC + FSS | Katzgraber & Young (2004) |
| EA ($\pm J$) | $d \geq 6$ | mean-field | $d \geq d_u$ | — |

---

## References

1. Sherrington, D. & Kirkpatrick, S. (1975). *Solvable Model of a Spin-Glass*. Phys. Rev. Lett. **35**, 1792.
2. Edwards, S.F. & Anderson, P.W. (1975). *Theory of spin glasses*. J. Phys. F **5**, 965.
3. Parisi, G. (1979). *Infinite Number of Order Parameters for Spin-Glasses*. Phys. Rev. Lett. **43**, 1754.
4. de Almeida, J.R.L. & Thouless, D.J. (1978). *Stability of the Sherrington-Kirkpatrick solution*. J. Phys. A **11**, 983.
5. McMillan, W.L. (1984). *Domain-wall renormalization-group study of the two-dimensional random Ising model*. Phys. Rev. B **29**, 4026.
6. Katzgraber, H.G., Körner, M. & Young, A.P. (2006). *Universality in three-dimensional Ising spin glasses*. Phys. Rev. B **73**, 224432.
7. Baity-Jesi, M. et al. (Janus Collaboration) (2013). *Critical parameters of the three-dimensional Ising spin glass*. Phys. Rev. B **88**, 224416.
