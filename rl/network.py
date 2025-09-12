import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from typing import Tuple

from .distribution import (
    Distribution,
    BetaDistribution,
    MultivariateNormalDiag,
    TanhMultivariateNormalDiag,
    TruncatedMultivariateNormalDiag,
    CategoricalDistribution,
)

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Actor Network
        actor = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        actor = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor))

        # Normal Distribution
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        log_std = self.param("log_std", nn.initializers.constant(jnp.log(1.0)), (self.action_dim,)) # State independent log std
        pi = MultivariateNormalDiag(mean, jnp.exp(log_std))

        # # Tanh Normal Distrbution | Entropy is maximized when std=0.8744 and mean=0, so we use this values to initialize
        # log_std = self.param("log_std", nn.initializers.constant(jnp.log(0.8744)), (self.action_dim,)) # State independent log std
        # mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        # pi = TanhMultivariateNormalDiag(mean, jnp.exp(log_std))

        # # Truncated Normal distribution | Mean is not really a mean, but a location parameter
        # log_std = self.param("log_std", nn.initializers.constant(jnp.log(0.8744)), (self.action_dim,)) # State independent log std
        # loc = nn.tanh(nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor))
        # pi = TruncatedMultivariateNormalDiag(loc, jnp.exp(log_std))

        # # Beta Distribution | Reparaemeterization for alpha and beta parameters
        # mu = nn.sigmoid(nn.Dense(self.action_dim, kernel_init=orthogonal(1))(actor))
        # log_v = self.param("log_v", nn.initializers.constant(jnp.log(2.0)), (self.action_dim,))
        # alpha = jnp.clip(mu * jnp.exp(log_v), 1e-5, None)
        # beta  = jnp.clip((1 - mu) * jnp.exp(log_v), 1e-5, None)
        # pi = BetaDistribution(alpha, beta)

        # # Categorical Distribution | For discrete action spaces
        # logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        # pi = CategoricalDistribution(logits)


        # Critic Network
        critic = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        critic = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic))
        critic = nn.Dense(1, kernel_init=orthogonal(1))(critic)
        critic = jnp.squeeze(critic, axis=-1)

        return pi, critic

    def apply(self, variables, *args, rngs = None, method = None, mutable = False, capture_intermediates = False, **kwargs) -> Tuple[Distribution, jnp.ndarray]:
        return super().apply(variables, *args, rngs=rngs, method=method, mutable=mutable, capture_intermediates=capture_intermediates, **kwargs)

class CNNActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Shared Convolutional Network
        # base = nn.relu(nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding=1, kernel_init=orthogonal(jnp.sqrt(2)))(x))     # (B, 32, 32,  1) -> (B, 16, 16, 32)
        base = nn.relu(nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding=1, kernel_init=orthogonal(jnp.sqrt(2)))(x))  # (B, 16, 16, 32) -> (B,  8,  8, 32)
        base = nn.relu(nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding=1, kernel_init=orthogonal(jnp.sqrt(2)))(base))  # (B,  8,  8, 32) -> (B,  4,  4, 32)
        base = nn.relu(nn.Conv(features=32, kernel_size=(4, 4), strides=(1, 1), padding=0, kernel_init=orthogonal(jnp.sqrt(2)))(base))  # (B,  4,  4, 32) -> (B,  1,  1, 32)
        base = base.reshape((base.shape[0], -1))  # Flatten, (B, 1*1*32) = (B, 32)

        # Actor Network
        actor = nn.relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(base))
        actor = nn.relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.00001))(actor)
        pi = CategoricalDistribution(logits)

        # Critic Network
        critic = nn.relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(base))
        critic = nn.relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic))
        critic = nn.Dense(1, kernel_init=orthogonal(1))(critic)
        critic = jnp.squeeze(critic, axis=-1)

        return pi, critic

    def apply(self, variables, *args, rngs = None, method = None, mutable = False, capture_intermediates = False, **kwargs) -> Tuple[Distribution, jnp.ndarray]:
        return super().apply(variables, *args, rngs=rngs, method=method, mutable=mutable, capture_intermediates=capture_intermediates, **kwargs)
