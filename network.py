import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from distribution import Distribution, BetaDistribution, MultivariateNormalDiag, TanhMultivariateNormalDiag, TruncatedMultivariateNormalDiag
from typing import Tuple


class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Actor network
        actor = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        actor = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor))

        # Tanh Entropy is maximized when std=0.8744 and mean=0, so we use this values to initialize
        log_std = self.param("log_std", nn.initializers.constant(jnp.log(0.8744)), (self.action_dim,)) # State independent log std
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        # log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(jnp.log(0.8744)))(actor)  # State dependent log std
        pi = MultivariateNormalDiag(mean, jnp.exp(log_std))
        # pi = TanhMultivariateNormalDiag(mean, jnp.exp(log_std))
        # pi = TruncatedMultivariateNormalDiag(mean, jnp.exp(log_std)) # Truncated Normal distribution

        # # Reparaemeterization for Beta distribution
        # mu = nn.sigmoid(nn.Dense(self.action_dim, kernel_init=orthogonal(1))(actor))
        # log_v = self.param("log_v", nn.initializers.constant(jnp.log(2.0)), (self.action_dim,))
        # alpha = jnp.clip(mu * jnp.exp(log_v), 1e-5, None)
        # beta  = jnp.clip((1 - mu) * jnp.exp(log_v), 1e-5, None)
        # pi = BetaDistribution(alpha, beta)

        # Critic network
        critic = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        critic = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic))
        critic = nn.Dense(1, kernel_init=orthogonal(1))(critic)
        critic = jnp.squeeze(critic, axis=-1)

        return pi, critic

    def apply(self, variables, *args, rngs = None, method = None, mutable = False, capture_intermediates = False, **kwargs) -> Tuple[Distribution, jnp.ndarray]:
        return super().apply(variables, *args, rngs=rngs, method=method, mutable=mutable, capture_intermediates=capture_intermediates, **kwargs)

class PrivilegedActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, actor_obs: jnp.ndarray, critic_obs: jnp.ndarray):
        # Actor network
        actor = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor_obs))
        actor = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor))
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        log_std = self.param("log_std", nn.initializers.constant(jnp.log(0.8744)), (self.action_dim,))
        pi = TanhMultivariateNormalDiag(mean, jnp.exp(log_std))

        # Critic network
        critic = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic_obs))
        critic = nn.tanh(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic))
        critic = nn.Dense(1, kernel_init=orthogonal(1))(critic)
        critic = jnp.squeeze(critic, axis=-1)

        return pi, critic
