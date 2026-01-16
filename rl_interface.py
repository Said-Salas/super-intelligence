import abc
import numpy as np
from typing import Tuple, Dict, Any, List

class GenesisEnvironment(abc.ABC):
    """
    Abstract Base Class for Genesis Protocol Environments.
    This enforces a standard API whether we are in a simple Python mock
    or connected to Unreal Engine 5 via sockets.
    """

    @property
    @abc.abstractmethod
    def action_space_size(self) -> int:
        """Number of continuous actions (motor outputs)."""
        pass

    @property
    @abc.abstractmethod
    def observation_space_size(self) -> int:
        """Number of continuous observations (sensory inputs)."""
        pass

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """
        Resets the environment to an initial state.
        Returns:
            observation (np.ndarray): The initial sensory input.
        """
        pass

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step in the environment.
        Args:
            action (np.ndarray): The motor commands [Batch, Actions] or [Actions].
        Returns:
            observation (np.ndarray): New sensory input.
            reward (float): The fitness/success signal (for PPO).
            done (bool): Whether the episode has ended (died/success).
            info (dict): Diagnostic info (e.g., 'internal_state').
        """
        pass

    @abc.abstractmethod
    def close(self):
        """Cleanup resources (sockets, threads, etc)."""
        pass

