# type: ignore
import numpy as np
import tensorflow as tf
import networkx as nx
import osmnx as ox
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spin_engine.models.traveling_salesman import TravelingSalesmanSystem
from spin_engine.measurements import Energy
from spin_engine.dynamics import MetropolisHastings, Tracker, TravelingSalesmanDynamics

# --- CONFIGURATION ---
# PLACE = "Union Square, San Francisco, California"
PLACE = "Londrina, Paraná, Brasil"
L = 10
NUM_SPINS = L**2
REPLICAS = 256
SEED = 42

# --- STEP 1: GEOSPATIAL DATA ---
print(f"Fetching map data for {PLACE}...")
G = ox.graph_from_address(PLACE, dist=1000, network_type="drive")
nodes_list = list(G.nodes)

np.random.seed(SEED)
all_active_nodes = np.random.choice(nodes_list, L, replace=False).tolist()

print(f"Calculating distance matrix for {L} nodes...")
W = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        if i == j:
            W[i, j] = 0
        else:
            try:
                W[i, j] = nx.shortest_path_length(
                    G, all_active_nodes[i], all_active_nodes[j], weight='length') / 1000.0
            except nx.NetworkXNoPath:
                W[i, j] = 100.0

# --- STEP 2: SYSTEM INITIALIZATION & ANNEALING ---
# Higher A_val ensures constraints (valid TSP route) are prioritized over distance
max_dist = np.max(W)
A_val = max_dist * 10.0
B_val = 1.0

system = TravelingSalesmanSystem(
    cost_matrix=W,
    lattice_replicas=REPLICAS,
    constraint_strength=A_val,
    distance_strength=B_val
)

# Improved annealing schedule: More steps help convergence
# betas = [0.001, 0.01, 0.1, 1.0]
# Slower, smoother schedule
betas = list(np.logspace(np.log10(0.001), np.log10(5.0), num=20))
sweep_len = 100
annealing_history = []

print("Starting simulated annealing...")
for b in betas:
    beta_val = tf.constant(b, dtype=tf.float32)
    simulation = TravelingSalesmanDynamics(system)
    # simulation = MetropolisHastings(system)
    tracker = Tracker([Energy(system)])
    simulation.sweep(tracker, beta=beta_val, sweep_length=sweep_len)
    annealing_history.append(tracker.history['Energy'].numpy())
    print(f"Finished sweep for beta={b:.2f}")

    # annealing_history has list of arrays of shape (steps, Q, R)
    # We want shape (total_steps, R)
    full_energy_history = np.concatenate(annealing_history, axis=0)[:, 0, :]


def create_integrated_dashboard_tsp(G, system, all_active_nodes, W, energy_history):
    print("Generating Interactive Dashboard (Robust Mode)...")

    # --- CONFIGURATION ---
    C_VALID = '#AB63FA'
    C_INVALID = '#EF553B'
    C_TELEPORT = '#FFA15A'

    PENALTY_THRESHOLD = 90.0

    replicas_spins = system.spin_state.value().numpy()[0]
    total_steps = energy_history.shape[0]

    # OPTIMIZATION: Downsample energy history for plotting (max ~1000 points)
    stride = max(1, total_steps // 1000)
    steps_x = np.arange(total_steps)[::stride]
    energy_history_plot = energy_history[::stride, :]  # Sliced view for plots

    final_energies = energy_history[-1, :]
    mean_energy = np.mean(energy_history_plot, axis=1)
    L = len(all_active_nodes)

    ref_node = all_active_nodes[0]
    center_lat, center_lon = G.nodes[ref_node]['y'], G.nodes[ref_node]['x']
    # Pre-calculate coordinates once
    node_lats = [G.nodes[n]['y'] for n in all_active_nodes]
    node_lons = [G.nodes[n]['x'] for n in all_active_nodes]

    replica_data = []
    range_L = range(L)  # Avoid recreating range object

    for r in range(len(replicas_spins)):
        spins = replicas_spins[r]
        x_mat = np.round((spins + 1) / 2).reshape(L, L)

        row_sums = np.sum(x_mat, axis=1)
        col_sums = np.sum(x_mat, axis=0)

        is_permutation = np.allclose(
            row_sums, 1.0) and np.allclose(col_sums, 1.0)

        idx_seq = [np.argmax(x_mat[:, t]) for t in range_L]

        try:
            ref_pos = idx_seq.index(0)
            # List slicing is faster than np.roll here
            idx_seq = idx_seq[ref_pos:] + idx_seq[:ref_pos]
        except ValueError:
            pass

        # OPTIMIZATION: Calculate distance using indices directly (W lookup)
        # Avoiding list.index() inside the loop
        dist_km = 0.0
        has_teleport = False

        # Walk the path
        prev = idx_seq[0]
        for i in range(1, L):
            curr = idx_seq[i]
            weight = W[prev, curr]
            if weight >= PENALTY_THRESHOLD:
                has_teleport = True
            dist_km += weight
            prev = curr

        # Close loop
        weight = W[prev, idx_seq[0]]
        if weight >= PENALTY_THRESHOLD:
            has_teleport = True
        dist_km += weight

        if not is_permutation:
            status = "INVALID (Matrix)"
            color = C_INVALID
            is_valid_bool = False
        elif has_teleport:
            status = "INVALID (No Path)"
            color = C_TELEPORT
            is_valid_bool = False
        else:
            status = "VALID"
            color = C_VALID
            is_valid_bool = True

        # Only construct full node list for storage
        route_nodes = [all_active_nodes[i] for i in idx_seq]
        route_nodes.append(route_nodes[0])

        replica_data.append({
            'id': r,
            'energy': final_energies[r],
            'valid': status,
            'color': color,
            'is_valid_bool': is_valid_bool,
            'route': route_nodes,
            'idx_seq': idx_seq,
            'distance_km': dist_km,
            'x_mat': x_mat
        })

    replica_data.sort(key=lambda x: x['energy'])

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        specs=[
            [{"type": "map", "rowspan": 2}, {"type": "xy"}],
            [None,                          {"type": "xy"}]
        ],
        vertical_spacing=0.08,
        subplot_titles=("Hamiltonian Cycle",
                        "Energy Evolution", "Spin State Matrix")
    )

    trace_idx = 0

    num_bg = min(50, energy_history.shape[1])
    for r in range(num_bg):
        # OPTIMIZATION: Use Scattergl for performance and downsampled data
        fig.add_trace(go.Scattergl(
            x=steps_x, y=energy_history_plot[:, r], mode='lines',
            line=dict(color='rgba(150,150,150,0.1)', width=1),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=2)
        trace_idx += 1

    fig.add_trace(go.Scattergl(
        x=steps_x, y=mean_energy, mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Mean Energy'
    ), row=1, col=2)
    trace_idx += 1

    best = replica_data[0]

    init_stats_text = (f"<b>Metrics</b><br>"
                       f"Total: {best['distance_km']:.2f} km<br>"
                       f"Status: {best['valid']}")

    idx_energy = trace_idx
    # OPTIMIZATION: Downsampled data
    fig.add_trace(go.Scattergl(
        x=steps_x, y=energy_history_plot[:, best['id']],
        mode='lines', line=dict(color=best['color'], width=3),
        name='Selected Replica', showlegend=False
    ), row=1, col=2)
    trace_idx += 1

    idx_heatmap = trace_idx
    fig.add_trace(go.Heatmap(
        z=best['x_mat'],
        colorscale=[[0, 'white'], [1, best['color']]],
        showscale=False, zmin=0, zmax=1, xgap=1, ygap=1,
        name="Permutation"
    ), row=2, col=2)
    trace_idx += 1

    idx_route = trace_idx
    fig.add_trace(go.Scattermap(
        lat=[], lon=[], mode='lines',
        name=f"Route ({best['valid']})",
        line=dict(width=4, color=best['color'])
    ), row=1, col=1)
    trace_idx += 1

    idx_nodes = trace_idx
    fig.add_trace(go.Scattermap(
        lat=node_lats, lon=node_lons,
        mode='markers+text',
        textposition="top center",
        marker=dict(size=12, color='#636EFA'),
        textfont=dict(family="Arial", size=14, color="black"),
        name="Visit Order"
    ), row=1, col=1)
    trace_idx += 1

    fig.add_annotation(
        text=init_stats_text,
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        showarrow=False,
        xanchor="left", yanchor="top", align="left",
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="black", borderwidth=1,
        font=dict(size=14)
    )
    metrics_idx = len(fig.layout.annotations) - 1

    slider_steps = []
    target_indices = [idx_energy, idx_heatmap, idx_route, idx_nodes]

    for rank, data in enumerate(replica_data[:100]):

        curr_color = data['color']
        curr_name = f"Route ({data['valid']})"

        stats_text = (f"<b>Metrics</b><br>"
                      f"Total: {data['distance_km']:.2f} km<br>"
                      f"Status: {data['valid']}")

        labels = [""] * L
        for step_i, node_id in enumerate(data['idx_seq']):
            labels[node_id] = str(step_i + 1)

        rlats, rlons = [], []
        for u, v in zip(data['route'][:-1], data['route'][1:]):
            try:
                path = nx.shortest_path(G, u, v, weight='length')
                rlats.extend([G.nodes[n]['y'] for n in path] + [None])
                rlons.extend([G.nodes[n]['x'] for n in path] + [None])
            except:
                rlats.extend([G.nodes[u]['y'], G.nodes[v]['y'], None])
                rlons.extend([G.nodes[u]['x'], G.nodes[v]['x'], None])

        step = {
            "method": "update",
            "args": [
                {
                    # OPTIMIZATION: Pass the downsampled energy array
                    "y": [energy_history_plot[:, data['id']], None, None, None],
                    "z": [None, data['x_mat'], None, None],
                    "lat": [None, None, rlats, node_lats],
                    "lon": [None, None, rlons, node_lons],
                    "text": [None, None, None, labels],
                    "line.color": [curr_color, None, curr_color, None],
                    "colorscale": [None, [[0, 'white'], [1, curr_color]], None, None],
                    "name": [None, None, curr_name, None]
                },
                {
                    f"annotations[{metrics_idx}].text": stats_text
                },
                target_indices
            ],
            "label": f"{rank+1}"
        }
        slider_steps.append(step)

    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20),
        map_style="carto-positron",
        map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=13),
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Rank: "},
            "pad": {"t": 50},
            "steps": slider_steps
        }],
        title_text=f"TSP Optimization: {PLACE}",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(scaleanchor='x2', scaleratio=1, constrain='domain',
                     showticklabels=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(title_text="Energy (Log)", type="log", row=1, col=2)

    return fig


fig = create_integrated_dashboard_tsp(
    G, system, all_active_nodes, W, full_energy_history)

fig.write_html(
    f"examples/output/tsp_dashboard_fullscreen_{SEED}_seed_{L}_stops_{REPLICAS}_replicas_{sweep_len}_steps.html",
    include_plotlyjs='cdn',
    full_html=True,
    config={'responsive': True}
)
print("Dashboard saved to examples/tsp_dashboard_fullscreen.html")
