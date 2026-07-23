# Population Annealing Implementation

## Overview

This implementation adds **Population Annealing (PA)** as an alternative sweep function to the existing Metropolis-Hastings dynamics in the spin simulation engine. Population annealing is a sequential Monte Carlo method that provides several advantages over standard single-replica Metropolis-Hastings:

### Key Features

1. **Population-based**: Maintains multiple replicas (a population) that evolve together
2. **Temperature annealing**: Gradually cools the system from high to low temperature
3. **Resampling**: At each temperature step, replicas are resampled based on their Boltzmann weights
4. **Partition function estimation**: Provides an estimate of the partition function ratio Z(β_final)/Z(β_initial)
5. **Better equilibration**: Particularly effective for systems with rough energy landscapes

## Files Added/Modified

### New Files

1. **`spin/src/spin_engine/dynamics/population_annealing.py`**
   - Main implementation of the `PopulationAnnealing` class
   - Inherits from the abstract `Dynamics` base class
   - Implements all required methods: `step()`, `sweep()`, and helper methods

2. **`spin/examples/population_annealing_example.py`**
   - Complete example demonstrating PA on the 2D Ising model
   - Shows how to use PA as a drop-in replacement for MetropolisHastings
   - Includes visualization and finite-size scaling analysis

3. **`spin/tests/test_population_annealing.py`**
   - Comprehensive test suite for the PA implementation
   - Tests initialization, weight computation, resampling, and full sweeps

### Modified Files

1. **`spin/src/spin_engine/dynamics/__init__.py`**
   - Added import for `PopulationAnnealing`

## API Usage

### Basic Usage

```python
from spin_engine.models import IsingSystem
from spin_engine.dynamics import PopulationAnnealing
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization
import tensorflow as tf

# Initialize system
system = IsingSystem(
    lattice_dim=2,
    lattice_length=8,
    lattice_replicas=128,  # Population size
    interaction_matrix=interaction_matrix
)

# Create PA dynamics
pa = PopulationAnnealing(system)

# Define temperature schedule (beta = 1/T)
beta_schedule = tf.constant([0.0, 0.1, 0.2, 0.3, 0.4], dtype=tf.float32)

# Setup tracker
tracker = Tracker(
    measurements=[Energy(system), Magnetization(system)],
    granularity=1  # Track at each beta step
)

# Run population annealing
pa.sweep(
    tracker=tracker,
    beta_schedule=beta_schedule,
    equilibration_steps=10,  # Metropolis steps between resampling
    num_disturbances=1  # Single spin flips
)

# Get results
energies = tracker.history['Energy'].numpy()
magnetizations = tracker.history['Magnetization'].numpy()

# Get partition function estimate
Z_ratio = pa.get_partition_function_estimate().numpy()
```

### Comparison with MetropolisHastings

**MetropolisHastings:**
```python
mh = MetropolisHastings(system)
mh.sweep(
    tracker=tracker,
    beta=0.44,  # Single temperature
    sweep_length=1000  # Number of steps
)
```

**PopulationAnnealing:**
```python
pa = PopulationAnnealing(system)
pa.sweep(
    tracker=tracker,
    beta_schedule=tf.constant([0.0, 0.1, ..., 0.44]),  # Full schedule
    equilibration_steps=10
)
```

## Algorithm Details

### Population Annealing Algorithm

1. **Initialize** a population of R replicas at β=0 (infinite temperature)

2. **For each temperature step** β_i → β_{i+1}:
   
   a. **Compute weights**: w_r = exp(-(β_{i+1} - β_i) × E_r) for each replica r
   
   b. **Update partition function**: Z ← Z × ⟨w⟩
   
   c. **Resample**: Draw R new replicas from the current population with probability ∝ w_r
   
   d. **Equilibrate**: Perform M Metropolis steps at β_{i+1} to restore equilibrium

3. **Track measurements** at each temperature step

### Key Parameters

- **population_size**: Number of replicas (larger = better statistics, more memory)
- **beta_schedule**: Array of inverse temperatures to anneal through
- **equilibration_steps**: Number of Metropolis steps between resampling (typically 5-20)
- **num_disturbances**: Number of spin flip attempts per Metropolis step

## Advantages Over Metropolis-Hastings

1. **Better sampling of rough landscapes**: The population can explore multiple energy basins simultaneously

2. **No burn-in required**: Each temperature starts from an equilibrated population

3. **Partition function estimation**: Unique capability among Monte Carlo methods

4. **Parallelizable**: All replicas can be updated independently during Metropolis steps

5. **Adaptive**: Can automatically adjust resampling rate based on weight fluctuations

## Disadvantages

1. **Higher memory usage**: Requires storing O(R) replicas instead of O(1)

2. **More parameters**: Need to choose population size and annealing schedule

3. **Different interface**: Sweep method takes a beta schedule instead of single beta

## Testing

Run the test suite:
```bash
cd /workspace/spin/spin
PYTHONPATH=/workspace/spin/spin/src pytest tests/test_population_annealing.py -v
```

## Example Execution

Run the example script:
```bash
cd /workspace/spin/spin
PYTHONPATH=/workspace/spin/spin/src python examples/population_annealing_example.py
```

This will:
1. Simulate the 2D Ising model for L = 4, 6, 8, 10
2. Compute thermodynamic observables (magnetization, susceptibility, specific heat, Binder cumulant)
3. Perform finite-size scaling analysis
4. Save results to JSON and generate publication-quality plots

## References

1. Hukushima, K., & Iba, Y. (2003). Population annealing and its application to a spin glass. *AIP Conference Proceedings*, 690(1), 200-206.

2. Machta, J. (2010). Population annealing with weighted averages: A Monte Carlo method for equilibrium statistical mechanics. *Physical Review E*, 82(2), 026705.

3. Barash, L., Weigel, M., Borovský, M., Janke, W., & Katzgraber, H. G. (2017). Markov chain versus population annealing simulations of the square-lattice Ising model. *Physical Review E*, 96(5), 053307.
