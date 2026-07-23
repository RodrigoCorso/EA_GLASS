import abc
import numpy as np


class Interaction(abc.ABC):
    """
    Abstract base class for all interaction types.
    """
    @abc.abstractmethod
    def generate(self, D: int, L: int, quenched: int) -> np.ndarray:
        """
        Generates the interaction matrix/tensor.

        Args:
           D: Dimension of the lattice
           L: Length of the lattice side
           quenched: Number of quenched variation of the interaction

        Returns:
            np.ndarray: Interaction tensor of shape (quenched,) + (L,)*D*2 or matrix (quenched, N, N) depending on usage,
                        but consistently (quenched,) + (L,)*D*2 for the provided examples.
        """
        pass
