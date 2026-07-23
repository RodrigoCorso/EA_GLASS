import numpy as np
from .base import Interaction
from typing import Optional


class DecayingInteraction(Interaction):
    def __init__(self, J0: float = 10, alpha: float = 1):
        self.J0 = J0
        self.alpha = alpha

    def generate(self, D: int, L: int) -> np.ndarray:
        coords = np.array(np.meshgrid(
            *[np.arange(L)]*D, indexing='ij')).reshape(D, -1).T

        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        J_flat = self.J0 * np.exp(-self.alpha * distances)
        np.fill_diagonal(J_flat, 0)

        tensor_shape = (L,)*D*2
        J_tensor = J_flat.reshape(tensor_shape)

        return J_tensor


class PeriodicNearestNeighborInteraction(Interaction):
    """
    Vectorized nearest-neighbor coupling tensor with periodic boundaries.
    J[i1,...,iD,j1,...,jD] = 1 if periodic Manhattan distance = 1, else 0
    """

    def generate(self, D: int, L: int) -> np.ndarray:
        coords = np.array(np.meshgrid(
            *[np.arange(L)]*D, indexing='ij')).reshape(D, -1).T
        N = coords.shape[0]

        diff = np.abs(coords[:, None, :] - coords[None, :, :])

        diff = np.minimum(diff, L - diff)

        manhattan_dist = diff.sum(axis=2)

        nn_mask = (manhattan_dist == 1)

        J_tensor = np.zeros((L,)*D*2, dtype=np.float32)

        idx_i, idx_j = np.nonzero(nn_mask)
        for i, j in zip(idx_i, idx_j):
            J_tensor[tuple(coords[i]) + tuple(coords[j])] = 1.0

        return J_tensor


class AntiPeriodicNearestNeighborInteraction(Interaction):
    """
    Vectorized nearest-neighbor coupling tensor with anti-periodic boundaries.
    J[i1,...,iD,j1,...,jD] = 1 if bulk nearest neighbor,
                            -1 if boundary nearest neighbor (wrapping around),
                             0 otherwise.
    """

    def generate(self, D: int, L: int) -> np.ndarray:
        coords = np.array(np.meshgrid(
            *[np.arange(L)]*D, indexing='ij')).reshape(D, -1).T
        N = coords.shape[0]

        diff_orig = np.abs(coords[:, None, :] - coords[None, :, :])
        diff = np.minimum(diff_orig, L - diff_orig)
        manhattan_dist = diff.sum(axis=2)
        nn_mask = (manhattan_dist == 1)

        # Identify boundary bonds (where absolute diff in any dimension is L - 1)
        boundary_mask = np.any(diff_orig == L - 1, axis=2) & nn_mask

        J_flat = np.zeros((N, N), dtype=np.float32)
        
        # Bulk bonds are 1
        J_flat[nn_mask] = 1.0
        
        # Boundary bonds are -1 (only meaningful for L > 2)
        if L > 2:
            J_flat[boundary_mask] = -1.0

        tensor_shape = (L,) * D * 2
        return J_flat.reshape(tensor_shape)



class CurieWeissInteraction(Interaction):
    def __init__(self, J0: float = 1.0):
        self.J0 = J0

    def generate(self, D: int, L: int) -> np.ndarray:
        N = L**D

        J_flat = (self.J0 / N) * (np.ones((N, N)) - np.eye(N))

        tensor_shape = (L,) * D * 2
        J_tensor = J_flat.reshape(tensor_shape)

        return J_tensor


class GaussianInteraction(Interaction):
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None, nearest_neighbor_only: bool = True):
        self.mean = mean
        self.std = std
        self.seed = seed
        self.nearest_neighbor_only = nearest_neighbor_only

    def generate(self, D: int, L: int, quenched: int = 1) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)

        N = L**D
        J_flat = np.random.normal(self.mean, self.std, size=(quenched, N, N)).astype(np.float32)

        J_flat = 0.5 * (J_flat + J_flat.transpose(0, 2, 1))

        J_flat[:, np.arange(N), np.arange(N)] = 0

        tensor_shape = (quenched,) + (L,) * D * 2
        J_tensor = J_flat.reshape(tensor_shape)

        if self.nearest_neighbor_only:
            nn_mask = PeriodicNearestNeighborInteraction().generate(D, L)
            J_tensor = J_tensor * nn_mask[np.newaxis, ...]

        return J_tensor


class BinaryRandomInteraction(Interaction):
    """
    Generates a purely binary random interaction matrix for spin glass models.
    Couplings are drawn uniformly from {-J, +J} with a strictly zero diagonal,
    forming a symmetric matrix.

    Args:
        J (float, optional): The coupling strength magnitude. Defaults to 1.0.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
    """

    def __init__(self, J: float = 1.0, seed: Optional[int] = None, nearest_neighbor_only: bool = True):
        self.J = J
        self.seed = seed
        self.nearest_neighbor_only = nearest_neighbor_only

    def generate(self, D: int, L: int, quenched: int = 1) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)

        N = L**D
        J_flat = np.random.choice([-self.J, self.J], size=(quenched, N, N)).astype(np.int8)

        upper = np.triu(J_flat, 1)
        J_flat = upper + upper.transpose(0, 2, 1)

        tensor_shape = (quenched,) + (L,) * D * 2
        J_tensor = J_flat.reshape(tensor_shape)

        if self.nearest_neighbor_only:
            nn_mask = PeriodicNearestNeighborInteraction().generate(D, L)
            J_tensor = J_tensor * nn_mask[np.newaxis, ...]

        return J_tensor

