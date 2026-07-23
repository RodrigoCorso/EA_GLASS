# type: ignore
import matplotlib.animation as animation
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.base import Measurement
from spin_engine.models import WegnerSystem
from spin_engine.dynamics import MetropolisHastings

import tensorflow as tf

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_wegner_system(system):
    """
    Plots the spin state (links) and plaquettes of a Wegner system.

    Args:
        system: WegnerSystem instance.

    Returns:
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
    """
    spin_state = system.spin_state
    _, _, ny, nx, _ = spin_state.shape

    plaquettes = system.compute_all_plaquettes(spin_state)
    plaquettes_2d = plaquettes[0, 0, ..., 0].numpy()

    spin_vis = spin_state[0, 0].numpy()

    x_pts = []
    y_pts = []
    colors = []

    for i in range(ny):
        for j in range(nx):
            # y-oriented spin (index 0)
            x_pts.append(j)
            y_pts.append(i + 0.5)
            colors.append("red" if spin_vis[i, j, 0] > 0 else "blue")

            # x-oriented spin (index 1)
            x_pts.append(j + 0.5)
            y_pts.append(i)
            colors.append("red" if spin_vis[i, j, 1] > 0 else "blue")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pcolormesh(
        range(nx + 1),
        range(ny + 1),
        plaquettes_2d,
        cmap="coolwarm",
        alpha=0.3,
        edgecolors="gray",
        linewidth=0.5,
        vmin=-1,
        vmax=1,
    )

    # Plot links
    ax.scatter(x_pts, y_pts, c=colors, s=20/nx, zorder=10, marker="x")

    return fig, ax


# L = 10
# system = WegnerSystem(L, 1)
# dynamics = MetropolisHastings(system)


class SpinStateMeasurement(Measurement):
    def compute(self, spin_state=None, system=None):
        spin_state, _ = self._resolve(spin_state, system)
        return spin_state


@tf.function
def run_annealing(system, dynamics, tracker, beta_schedule, num_disturbances):
    """
    Runs an annealing schedule using the tracker.
    """
    schedule_length = tf.shape(beta_schedule)[0]
    tracking_arrays = tracker.init_run(schedule_length)

    # helper for loop body
    def body(i, tracking_arrays):
        beta = beta_schedule[i]
        dynamics.step(beta, num_disturbances)

        # Track (using i+1 because init_run allocates size+1 usually, or we track at i)
        # Tracker.track writes at floor(step/granularity).
        # If we want to track every step, granularity=1.

        tracking_arrays = tracker.track(i, system, tracking_arrays)
        return i + 1, tracking_arrays

    i0 = tf.constant(0, dtype=tf.int32)
    _, final_arrays = tf.while_loop(
        cond=lambda i, _: i < schedule_length,
        body=body,
        loop_vars=[i0, tracking_arrays]
    )

    tracker.finalize(final_arrays)


L = 16
system = WegnerSystem(L, 1)
dynamics = MetropolisHastings(system)

# Define measurement and tracker
spin_measurement = SpinStateMeasurement(system)
tracker = Tracker([spin_measurement], granularity=1)

# Annealing schedule: Low Beta (High Temp) -> High Beta (Low Temp)
# Beta = 1/T.
# T: 10.0 -> 0.1? => Beta: 0.1 -> 10.0
beta_schedule = tf.cast(np.logspace(
    np.log10(0.001), np.log10(5.0), num=1000), tf.float32)

print("Starting annealing...")
run_annealing(system, dynamics, tracker, beta_schedule, tf.constant(1))
print("Annealing finished.")

# Retrieve history
history = tracker.history
# (steps, 2*L*L) or similar depending on shape
spin_history = history['SpinStateMeasurement'].numpy()
# spin_state in Wegner is (1, L, L, 2) ?
# Let's check shape: (steps, 1, L, L, 2) likely.

print(f"History shape: {spin_history.shape}")

# Create Animation
fig, ax = plt.subplots(figsize=(6, 6))


def update(frame_idx):
    ax.clear()

    # Extract frame state
    # spin_history shape is likely [Steps, 1, L, L, 2]
    # We need to construct a temporary system or manually set state to plot
    # But plot_wegner_system takes a system.
    # We can temporarily assign the state to the system.

    current_state = spin_history[frame_idx]
    # plot_wegner_system logic expects system.spin_state to be a tensor
    # We can manually do plotting logic here to accept state directly
    # OR update system.spin_state (but that's a Variable, so use assign)

    # Updating variable is strictly side-effect, but for plotting it's fine.
    system.spin_state.assign(current_state)

    # We can reuse the plotting logic from plot_wegner_system code
    # But plot_wegner_system creates a FIGURE. using it inside update is hard if we want to reuse ax.
    # So let's refactor plot_wegner_system to take ax, OR just inline the logic.

    # Inline logic for speed/simplicity in this context:
    spin_state = system.spin_state
    _, _, ny, nx, _ = spin_state.shape
    plaquettes = system.compute_all_plaquettes(spin_state)
    plaquettes_2d = plaquettes[0, 0, ..., 0].numpy()
    spin_vis = spin_state[0, 0].numpy()

    x_pts = []
    y_pts = []
    colors = []

    for i in range(ny):
        for j in range(nx):
            # y-oriented spin (index 0)
            x_pts.append(j)
            y_pts.append(i + 0.5)
            colors.append("red" if spin_vis[i, j, 0] > 0 else "blue")

            # x-oriented spin (index 1)
            x_pts.append(j + 0.5)
            y_pts.append(i)
            colors.append("red" if spin_vis[i, j, 1] > 0 else "blue")

    ax.pcolormesh(
        range(nx + 1),
        range(ny + 1),
        plaquettes_2d,
        cmap="coolwarm",
        alpha=0.3,
        edgecolors="gray",
        linewidth=0.5,
        vmin=-1,
        vmax=1,
    )
    ax.scatter(x_pts, y_pts, c=colors, s=20/nx, zorder=10, marker="x")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")
    beta_val = beta_schedule[frame_idx]
    ax.set_title(f"Wegner Annealing\nBeta: {beta_val:.2f}")


ani = animation.FuncAnimation(
    fig, update, frames=len(beta_schedule), interval=50)

output_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "wegner_annealing.gif")

# Remove old files if they exist
old_files = ["plot.png", "plot_after_step.png"]
for f in old_files:
    p = os.path.join(output_dir, f)
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed {p}")

print(f"Saving animation to {output_path}...")
ani.save(output_path, writer='pillow', fps=10)
print("Done.")
plt.close(fig)
