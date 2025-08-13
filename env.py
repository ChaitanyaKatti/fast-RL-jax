from flax import struct
import jax.numpy as jnp
from typing import Tuple

@struct.dataclass
class EnvParams:
    """
    Base class for environment parameters.
    Contains common attributes that all environment parameters should have.
    """

@struct.dataclass
class EnvState:
    """
    Base class for environment state.
    Contains the entire state of the environment, including physics and any other relevant information.
    """

class Env:
    @classmethod
    def step(cls, key: jnp.ndarray, state:EnvState, action: jnp.ndarray, params:EnvParams) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        """
        Perform a step in the environment given the action for each agent.
        Args:
            key: jnp.ndarray of random keys for stochastic environments.
            state: Current state of the environment.
            action: jnp.array of shape (num_agents, action_dim) with action for each agent.
            params: Environment parameters.
        Returns:
            Tuple containing
                - obs: Observations after the step.
                - state: New state of the environment.
                - reward: Rewards for each agent.
                - terminated: Boolean array indicating if the episode has terminated for each agent.
                - info: Additional information.
        """
        raise NotImplementedError()

    @classmethod
    def reset(cls, key: jnp.ndarray, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
        """
        Reset the environment to an initial state.
        Args:
            key: jnp.ndarray of random keys for initialization.
            params: Environment parameters.
        Returns:
            Tuple containing
                - obs: Initial observations.
                - state: Initial state of the environment.
        """
        raise NotImplementedError()

    @staticmethod
    def action_space(params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the action space of the environment.
        Returns:
            Tuple of low and high bounds for the action space.
        """
        raise NotImplementedError()

    @staticmethod
    def observation_space(params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns the observation space of the environment.
        Returns:
            Tuple of low and high bounds for the observation space.
        """
        raise NotImplementedError()

    @staticmethod
    def render(state: EnvState, params: EnvParams) -> None:
        """
        Render the environment state.
        Args:
            state: Current state of the environment.
            params: Environment parameters.
        """
        raise NotImplementedError()

    @staticmethod
    def make_params(**kwargs) -> EnvParams:
        """
        Create environment parameters with default values.
        Args:
            **kwargs: Additional parameters to override defaults.
        Returns:
            An instance of EnvParams with the specified parameters.
        """
        return EnvParams(**kwargs)