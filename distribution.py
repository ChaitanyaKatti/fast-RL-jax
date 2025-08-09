import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

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
        k = self.event_shape[0]
        diff = (value - self.loc) / self.scale_diag
        return (
            -0.5 * k * jnp.log(2 * jnp.pi)
            - jnp.sum(jnp.log(self.scale_diag), axis=-1)
            -0.5 * jnp.sum(jnp.square(diff), axis=-1)
        )

    def entropy(self):
        """
        returns: entropy for each batch element
        """
        k = self.event_shape[0]
        return 0.5 * k * (jnp.log(2 * jnp.pi * jnp.e)) + jnp.sum(jnp.log(self.scale_diag), axis=-1)

class TanhMultivariateNormalDiag(MultivariateNormalDiag):
    def sample(self, key, sample_shape=()):
        """
        Sample from the distribution and apply tanh to the samples.
        Arguments:
            key: PRNGKey
            sample_shape: tuple of leading sample dimensions
        Returns: samples of shape sample_shape + batch_shape + event_shape
        """
        samples = super().sample(key, sample_shape)
        return jnp.tanh(samples)

    def log_prob(self, value):
        """
        Compute log probability of the tanh-transformed value.
        Arguments:
            value: tensor of shape [..., k] where k is the dimensionality of the distribution
        Returns: log probability of each value with shape [...]
        """
        # Apply inverse tanh transformation
        transformed_value = jnp.arctanh(value)
        return super().log_prob(transformed_value) - jnp.sum(jnp.log(1 - jnp.square(value)), axis=-1)

    def entropy(self):
        """
        Compute the entropy of the tanh-transformed distribution.
        Returns: entropy for each batch element
        """
        # No analytical entropy for tanh transformation, use Monte Carlo estimation
        # Optimal samples for fast MC entropy estimation
        samples32 = jnp.array([
            -2.2825078964233400, 2.2825078964233400,
            -1.9211180210113525, 1.9211180210113525,
            -1.3525558710098267, 1.3525558710098267,
            -1.0677014589309692, 1.0677014589309692,
            -0.8902221918106079, 0.8902221918106079,
            -0.8044639229774475, 0.8044639229774475,
            -0.7819824814796448, 0.7819824814796448,
            -0.7775918245315552, 0.7775918245315552,
            -0.7773172259330750, 0.7773172259330750,
            -0.7008596658706665, 0.7008596658706665,
            -0.6292149424552917, 0.6292149424552917,
            -0.2993766069412231, 0.2993766069412231,
            -0.1358855664730072, 0.1358855664730072,
            -0.0954640880227089, 0.0954640880227089,
            -0.0365058332681656, 0.0365058332681656,
            -0.0124982642009854, 0.0124982642009854
        ])

        # Entropy of the transformed distribution
        # Using the chain rule for entropy: H(Y) = H(X) + E[ln(1 - tanh(X)^2)]
        # Expectation is approximated by the mean over the optimal samples

        # Transform the samples
        z = self.loc[..., None] + self.scale_diag[..., None] * samples32
        # Compute the log of the absolute determinant of the Jacobian
        log_abs_det_jacobians = jnp.log1p(-jnp.square(jnp.tanh(z)) + 1e-6)                  # Shape (minibatch_size, action_dim, num_samples)
        # Average over the samples, sum over the action dimension
        log_abs_det_jacobians = jnp.sum(jnp.mean(log_abs_det_jacobians, axis=-1), axis=-1)  # Shape (minibatch_size,)
        # Return the total entropy
        return super().entropy() + log_abs_det_jacobians

class BetaDistribution:
    def __init__(self, alpha, beta):
        """
        alpha: tensor of shape (..., k) where k is the number of Beta variables
        beta: tensor of shape (..., k) where k is the number of Beta variables
        """
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.event_shape = self.alpha.shape[-1:]
        self.batch_shape = self.alpha.shape[:-1]
    
    def sample(self, key, sample_shape=()):
        """
        key: PRNGKey
        sample_shape: tuple of leading sample dimensions
        returns: samples of shape sample_shape + batch_shape + event_shape
        """
        return jax.random.beta(key, self.alpha, self.beta, shape=sample_shape + self.batch_shape + self.event_shape)
    
    def log_prob(self, value):
        """
        value: tensor of shape [..., k] where k is the number of Beta variables
        returns: log probability of each value with shape [...]
        """
        return (
            (self.alpha - 1) * jnp.log(value) +
            (self.beta - 1) * jnp.log(1 - value) -
            gammaln(self.alpha) -
            gammaln(self.beta) +
            gammaln(self.alpha + self.beta)
        )
    
    def entropy(self):
        """
        returns: entropy for each batch element
        """
        return (
            gammaln(self.alpha + self.beta) -
            gammaln(self.alpha) -
            gammaln(self.beta) +
            (self.alpha - 1) * jnp.log(self.alpha / (self.alpha + self.beta)) +
            (self.beta - 1) * jnp.log(self.beta / (self.alpha + self.beta))
        )
