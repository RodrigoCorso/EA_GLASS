"""
Tests for compute_delta_energy() across all spin system models.

Each test verifies that:
    old_energy + compute_delta_energy(...) ≈ compute_energy(new_state)

This cross-checks the incremental ΔE against full energy recomputation.
"""
import pytest
import tensorflow as tf
import numpy as np
from typing import cast, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem

from spin_engine.interactions.standard import (
    PeriodicNearestNeighborInteraction,
    BinaryRandomInteraction,
    GaussianInteraction,
)
from spin_engine.models.ising import IsingSystem
from spin_engine.models.edwards_anderson import EdwardsAndersonSystem
from spin_engine.models.sk import SherringtonKirkpatrickSystem
from spin_engine.models.spherical import SphericalSystem
from spin_engine.models.wegner import WegnerSystem
from spin_engine.dynamics import MetropolisHastings, Tracker
from spin_engine.measurements.base import Measurement


def _flip_and_get_indices(system, num_flips=1):
    """Helper: flip random spins and return (updated_state, indices)."""
    Q = system.quenched_replicas
    R = system.lattice_replicas
    spin_flat = tf.reshape(system.spin_state,
                           (Q, R, -1))

    # Generate unique indices for each replica using top_k to prevent duplicates
    num_spins_int = tf.cast(system.number_spins, tf.int32)
    random_vals = tf.random.uniform((Q, R, num_spins_int))
    _, idx = tf.math.top_k(random_vals, k=num_flips)
    idx = tf.cast(idx, tf.int32)

    q_idx = tf.repeat(tf.range(Q)[:, None, None], R, axis=1)
    q_idx = tf.repeat(q_idx, num_flips, axis=2)

    r_idx = tf.repeat(tf.range(R)[None, :, None], Q, axis=0)
    r_idx = tf.repeat(r_idx, num_flips, axis=2)

    scatter_indices = tf.stack([q_idx, r_idx, idx], axis=-1)
    scatter_indices = tf.reshape(scatter_indices, (-1, 3))

    updates = tf.reshape(
        -tf.gather_nd(spin_flat, scatter_indices), [-1])

    updated_flat = tf.tensor_scatter_nd_update(
        spin_flat, scatter_indices, updates)
    updated = tf.reshape(updated_flat, system.spin_state.shape)

    return updated, idx


class TestDeltaEnergyIsing:
    """Tests for IsingSystem.compute_delta_energy()."""

    def test_single_flip_no_field(self):
        """ΔE matches full recomputation for a single-site flip without field."""
        L, D, R = 4, 2, 8
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        system = IsingSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-4,
                                 message="Ising ΔE mismatch (no field)")

    def test_single_flip_with_field(self):
        """ΔE matches full recomputation with external field."""
        L, D, R = 4, 2, 4
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        h = tf.random.normal([L] * D)
        system = IsingSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, external_field=h, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-4,
                                 message="Ising ΔE mismatch (with field)")

    def test_multi_flip(self):
        """ΔE matches for multiple simultaneous flips."""
        L, D, R = 6, 2, 4
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        system = IsingSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=3)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-3,
                                 message="Ising ΔE mismatch (multi-flip)")

    def test_3d_lattice(self):
        """ΔE works correctly on a 3D lattice."""
        L, D, R = 4, 3, 4
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        system = IsingSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-3,
                                 message="Ising 3D ΔE mismatch")


class TestDeltaEnergyEA:
    """Tests for EdwardsAndersonSystem.compute_delta_energy()."""

    def test_single_flip_2d(self):
        """ΔE matches for EA model in 2D."""
        L, D, R = 4, 2, 8
        J = BinaryRandomInteraction(seed=42).generate(D, L, quenched=1)
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-4,
                                 message="EA 2D ΔE mismatch")

    def test_single_flip_3d(self):
        """ΔE matches for EA model in 3D."""
        L, D, R = 4, 3, 4
        J = BinaryRandomInteraction(seed=42).generate(D, L, quenched=1)
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-3,
                                 message="EA 3D ΔE mismatch")

    def test_gaussian_couplings(self):
        """ΔE works with Gaussian random couplings."""
        L, D, R = 4, 2, 4
        J = GaussianInteraction(seed=42).generate(D, L, quenched=1)
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-2,
                                 message="EA Gaussian ΔE mismatch")


class TestDeltaEnergySK:
    """Tests for SherringtonKirkpatrickSystem (inherits Ising ΔE)."""

    def test_single_flip(self):
        """SK inherits Ising ΔE — verify it works with fully connected J."""
        L, D, R = 8, 1, 4
        system = SherringtonKirkpatrickSystem(
            lattice_length=L, lattice_replicas=R,
            lattice_dim=D, seed=42
        )

        old_energy = system.compute_energy()
        updated, idx = _flip_and_get_indices(system, num_flips=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-3,
                                 message="SK ΔE mismatch")


class TestDeltaEnergySpherical:
    """Tests for SphericalSystem.compute_delta_energy()."""

    def _perturb_sites(self, system, num_perturb=1):
        """Helper: perturb random continuous spins and return (updated, indices)."""
        Q = system.quenched_replicas
        R = system.lattice_replicas
        spin_flat = tf.reshape(system.spin_state,
                               (Q, R, -1))

        idx = tf.random.uniform(
            shape=(Q, R, num_perturb),
            maxval=tf.cast(system.number_spins, tf.int32),
            dtype=tf.int32
        )

        # Apply a random perturbation at the selected sites
        q_idx = tf.repeat(tf.range(Q)[:, None, None], R, axis=1)
        q_idx = tf.repeat(q_idx, num_perturb, axis=2)

        r_idx = tf.repeat(tf.range(R)[None, :, None], Q, axis=0)
        r_idx = tf.repeat(r_idx, num_perturb, axis=2)

        scatter_indices = tf.stack([q_idx, r_idx, idx], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 3))

        perturbation = tf.random.normal(
            shape=(Q * R * num_perturb,))
        new_values = tf.gather_nd(spin_flat, scatter_indices) + perturbation

        updated_flat = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, new_values)
        updated = tf.reshape(updated_flat, system.spin_state.shape)

        return updated, idx

    def test_single_perturbation_no_field(self):
        """ΔE matches for continuous spin perturbation without field."""
        L, D, R = 4, 2, 4
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        system = SphericalSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = self._perturb_sites(system, num_perturb=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-2,
                                 message="Spherical ΔE mismatch (no field)")

    def test_single_perturbation_with_field(self):
        """ΔE matches with external field."""
        L, D, R = 4, 2, 4
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        h = tf.random.normal([L] * D)
        system = SphericalSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, external_field=h, lattice_dim=D
        )

        old_energy = system.compute_energy()
        updated, idx = self._perturb_sites(system, num_perturb=1)
        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-2,
                                 message="Spherical ΔE mismatch (with field)")


class TestDeltaEnergyWegner:
    """Tests for WegnerSystem (uses base class fallback)."""

    def test_fallback_to_full_recomputation(self):
        """Wegner has no interaction_matrix — falls back to full recompute."""
        L, D, R = 4, 2, 2
        system = WegnerSystem(
            lattice_length=L, lattice_replicas=R, lattice_dim=D
        )

        old_energy = system.compute_energy()

        Q = system.quenched_replicas
        R = system.lattice_replicas
        # Flip a random link (treat as flat)
        spin_flat = tf.reshape(system.spin_state, (Q, R, -1))
        total_elements = tf.shape(spin_flat)[2]
        idx = tf.random.uniform(
            shape=(Q, R, 1), maxval=total_elements, dtype=tf.int32)

        q_idx = tf.repeat(tf.range(Q)[:, None, None], R, axis=1)
        r_idx = tf.repeat(tf.range(R)[None, :, None], Q, axis=0)

        scatter_indices = tf.reshape(
            tf.stack([q_idx, r_idx, idx], axis=-1), (-1, 3))
        updates = -tf.gather_nd(spin_flat, scatter_indices)
        updated_flat = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, tf.reshape(updates, [-1]))
        updated = tf.reshape(updated_flat, system.spin_state.shape)

        new_energy = system.compute_energy(updated)

        delta_E = system.compute_delta_energy(system.spin_state, updated, idx)

        expected = new_energy - old_energy
        tf.debugging.assert_near(delta_E, expected, atol=1e-4,
                                 message="Wegner fallback ΔE mismatch")


class TestDeltaEnergyBaseClass:
    """Tests for the base class default compute_delta_energy()."""

    def test_base_default_matches_ising(self):
        """
        Verify the base class generic ΔE formula gives the same result
        as the Ising-specialized override.
        """
        L, D, R = 4, 2, 4
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        system = IsingSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )

        updated, idx = _flip_and_get_indices(system, num_flips=1)

        # Ising override
        delta_E_ising = system.compute_delta_energy(
            system.spin_state, updated, idx)

        # Base class default (call directly via super)
        from spin_engine.models.base import BaseSpinSystem
        delta_E_base = BaseSpinSystem.compute_delta_energy(
            system, system.spin_state, updated, idx)

        tf.debugging.assert_near(delta_E_ising, delta_E_base, atol=1e-3,
                                 message="Base vs Ising ΔE mismatch")


class TestDeltaEnergyEndToEnd:
    """End-to-end tests with MetropolisHastings dynamics."""

    def test_energy_tracking_stays_consistent(self):
        """
        After multiple MC steps, dynamics.current_energy should still match
        system.compute_energy(). Tests that accumulated ΔE doesn't drift.
        """
        L, D, R = 4, 2, 4
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        system = IsingSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )
        dynamics = MetropolisHastings(system)

        # Run many steps eagerly
        for _ in range(100):
            dynamics.step(beta=0.5, num_disturbances=tf.constant(1))

        # After 100 steps, tracked energy should match recomputed energy
        recomputed = system.compute_energy()
        tf.debugging.assert_near(
            dynamics.current_energy, recomputed, atol=1e-2,
            message="Energy drift after 100 MC steps"
        )

    def test_sweep_energy_consistency(self):
        """
        After a sweep with tracking, dynamics.current_energy matches recomputed.
        """
        L, D, R = 4, 2, 8
        J = PeriodicNearestNeighborInteraction().generate(D, L)
        system = IsingSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )
        dynamics = MetropolisHastings(system)

        class DummyMeasurement(Measurement):
            def compute(self, spin_state=None, system=None):
                return tf.constant(0.0)

        tracker = Tracker([DummyMeasurement()])

        N = L ** D
        dynamics.sweep(tracker, beta=0.4, sweep_length=10 * N,
                       num_disturbances=1)

        recomputed = system.compute_energy()
        tf.debugging.assert_near(
            dynamics.current_energy, recomputed, atol=1e-1,
            message="Energy drift after sweep"
        )

    def test_ea_sweep_energy_consistency(self):
        """EA model energy tracking through a sweep."""
        L, D, R = 4, 3, 4
        J = BinaryRandomInteraction(seed=42).generate(D, L, quenched=1)
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_replicas=R,
            interaction_matrix=J, lattice_dim=D
        )
        dynamics = MetropolisHastings(system)

        class DummyMeasurement(Measurement):
            def compute(self, spin_state=None, system=None):
                return tf.constant(0.0)

        tracker = Tracker([DummyMeasurement()])

        N = L ** D
        dynamics.sweep(tracker, beta=0.3, sweep_length=5 * N,
                       num_disturbances=1)

        recomputed = system.compute_energy()
        tf.debugging.assert_near(
            dynamics.current_energy, recomputed, atol=1e-1,
            message="EA energy drift after sweep"
        )
