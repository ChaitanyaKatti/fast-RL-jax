import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal

class MultivariateNormalDiag:
    def __init__(self, loc, scale_diag):
        """
        loc: mean vector (..., k) where k is the dimensionality of the distribution
        scale_diag: std deviation vector (..., k), must be positive
        """
        self.loc = jnp.asarray(loc)                 # Shape (..., k)
        self.scale_diag = jnp.asarray(scale_diag)   # Shape (..., k)
        self.event_shape = self.loc.shape[-1:]      # Shape (k,)
        self.batch_shape = self.loc.shape[:-1]      # Shape (...)

    def sample(self, key, sample_shape=()):
        """
        key: PRNGKey
        sample_shape: tuple of leading sample dimensions
        returns: samples of shape sample_shape + batch_shape + event_shape
        """
        eps = jax.random.normal(key, shape=sample_shape + self.loc.shape)
        return self.loc + eps * self.scale_diag

    def log_prob(self, value):
        """
        value: tensor of shape [..., k]
        returns: log probability of each value with shape [...]
        """
        diff = (value - self.loc) / self.scale_diag
        log_det = -jnp.sum(jnp.log(self.scale_diag), axis=-1)
        norm_const = -0.5 * self.event_shape[0] * jnp.log(2 * jnp.pi)
        quadratic = -0.5 * jnp.sum(diff**2, axis=-1)
        return log_det + norm_const + quadratic

    def entropy(self):
        """
        returns: entropy for each batch element
        """
        k = self.event_shape[0]
        return 0.5 * k * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(jnp.log(self.scale_diag), axis=-1)

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Actor network
        actor = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        actor = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor))
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        pi = MultivariateNormalDiag(mean, jnp.exp(log_std))

        # Critic network
        critic = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(x))
        critic = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic))
        critic = nn.Dense(1, kernel_init=orthogonal(1))(critic)
        critic = jnp.squeeze(critic, axis=-1)

        return pi, critic

class PrivilegedActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, actor_obs: jnp.ndarray, critic_obs: jnp.ndarray):
        # Actor network
        actor = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor_obs))
        actor = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(actor))
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(actor)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.action_dim,))
        pi = MultivariateNormalDiag(mean, jnp.exp(log_std))

        # Critic network
        critic = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic_obs))
        critic = nn.leaky_relu(nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)))(critic))
        critic = nn.Dense(1, kernel_init=orthogonal(1))(critic)
        critic = jnp.squeeze(critic, axis=-1)

        return pi, critic
