import tensorflow as tf
from typing import Optional, Union, Callable
from .ising import IsingSystem
from ..interactions.standard import GaussianInteraction
import numpy as np


class SherringtonKirkpatrickSystem(IsingSystem):
    """
    Sherrington-Kirkpatrick (SK) Model for Spin Glass Systems.
    
    This model implements a fully connected Ising system with Gaussian random couplings.
    The couplings are drawn from a normal distribution with mean 0 and standard deviation J / sqrt(N),
    where N is the total number of spins.

    Inherits from IsingSystem and generates a fully connected tensor scaled 
    appropriately for the total number of spins N = L^D.

    Args:
        lattice_length (int): The size of each dimension in the lattice.
        lattice_replicas (int): The number of independent replicas to simulate in parallel.
        lattice_dim (int, optional): Number of spatial dimensions. Defaults to 1.
        J (float, optional): The coupling strength magnitude. Defaults to 1.0.
        external_field (Optional[Union[tf.Tensor, np.ndarray]], optional): External magnetic field. Defaults to None.
        initial_magnetization (float, optional): Sets the probability of spins initializing to +1. Defaults to 0.5.
        seed (Optional[int], optional): Random seed for the Gaussian couplings. Defaults to None.
        initial_spin_state (Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]], optional):
            Pre-defined spin states to initialize with. Defaults to None.
    """
    def __init__(
        self,
        lattice_length: int,
        lattice_replicas: int,
        lattice_dim: int = 1,
        J: float = 1.0,
        external_field: Optional[Union[tf.Tensor, np.ndarray]] = None,
        initial_magnetization: float = 0.5,
        seed: Optional[int] = None,
        initial_spin_state: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]] = None,
    ):
        self.lattice_length = lattice_length
        self.lattice_dim = lattice_dim
        self.J = J
        
        num_spins = lattice_length ** lattice_dim
        
        # Calculate standard deviation for SK model
        # GaussianInteraction averages J and J.T, which reduces variance by half.
        # To get a final std of J / sqrt(num_spins), we scale the input std by sqrt(2).
        std = (self.J / np.sqrt(num_spins)) * np.sqrt(2)
        
        # Generate the fully connected interaction matrix shaped for L and D
        interaction = GaussianInteraction(mean=0.0, std=std, seed=seed, nearest_neighbor_only=False)
        interaction_matrix = interaction.generate(D=lattice_dim, L=lattice_length, quenched=1)[0]

        # Call the parent IsingSystem
        super().__init__(
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix,
            external_field=external_field,
            initial_magnetization=initial_magnetization,
            lattice_dim=lattice_dim,
            initial_spin_state=initial_spin_state,
        )
