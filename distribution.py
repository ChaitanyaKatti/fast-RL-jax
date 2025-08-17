import jax
import jax.numpy as jnp
from jax.scipy.special import betaln, digamma
from jax.scipy.stats import norm

class Distribution:
    def sample(self, key, sample_shape=()):
        """
        Sample from the distribution.

        Args:
            key (jnp.ndarray): PRNGKey
            sample_shape (tuple): tuple of leading sample dimensions

        Returns:
            samples of shape sample_shape + batch_shape + event_shape
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def log_prob(self, value):
        """
        Compute the log probability of the given value.

        Args:
            value (jnp.ndarray): tensor of shape [..., k] where k is the dimensionality of the distribution
        Returns:
            log probability of each value with shape [...]
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def entropy(self):
        """
        Returns:
            Entropy for each batch element
        """
        raise NotImplementedError("Subclasses should implement this method.")

class MultivariateNormalDiag(Distribution):
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
        eps = jax.random.normal(key, shape=sample_shape + self.loc.shape)
        return self.loc + eps * self.scale_diag

    def log_prob(self, value):
        k = self.event_shape[0]
        diff = (value - self.loc) / self.scale_diag
        return (
            -0.5 * k * jnp.log(2 * jnp.pi)
            - jnp.sum(jnp.log(self.scale_diag), axis=-1)
            -0.5 * jnp.sum(jnp.square(diff), axis=-1)
        )

    def entropy(self):
        k = self.event_shape[0]
        return 0.5 * (jnp.log(2 * jnp.pi * jnp.e)) + jnp.mean(jnp.log(self.scale_diag), axis=-1) # Mean entropy per dimension

class TanhMultivariateNormalDiag(MultivariateNormalDiag):
    def sample(self, key, sample_shape=()):
        samples = super().sample(key, sample_shape)
        return jnp.tanh(samples)

    def log_prob(self, value):
        value = jnp.clip(value, -0.999999, 0.999999) # Avoid log(0), arctanh(+/- 1) issues
        transformed_value = jnp.arctanh(value)
        return super().log_prob(transformed_value) - jnp.sum(jnp.log(1 - jnp.square(value) + 1e-6), axis=-1)

    def entropy(self):
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
        log_abs_det_jacobians = jnp.log1p(-jnp.square(jnp.tanh(z)) + 1e-6)      # Shape (minibatch_size, action_dim, num_samples)
        # Average over the samples, mean over the action dimension
        log_abs_det_jacobians = jnp.mean(log_abs_det_jacobians, axis=(-1, -2))  # Shape (minibatch_size,)
        # Return the total entropy
        return super().entropy() + log_abs_det_jacobians


class TruncatedMultivariateNormalDiag(MultivariateNormalDiag):
    """
    This distribution mimics the behaviour when resampling is used until the samples are within [-1, 1].
    p ~ normaldist(mean, std) # Stadard Gaussian
    q ~ normaldist(mean, std) / integral_{-1}^{1}{p(t)dt} # Normalized Truncated Gaussian, support=[-1,1]
    """
    def __init__(self, loc, scale_diag):
        super().__init__(loc, scale_diag)
        # Precompute the CDF values for the truncation points
        self.cdf_m1 = norm.cdf(-1, loc=self.loc, scale=self.scale_diag)
        self.cdf_p1 = norm.cdf(1, loc=self.loc, scale=self.scale_diag)
        self.cdf_m1_to_p1 = self.cdf_p1 - self.cdf_m1

    def sample(self, key, sample_shape=()):
        """
        Sample from the distribution uniformly in the range [-1, 1].
        This is done by sampling from the CDF of the truncated normal distribution.
        The CDF is computed as:
        CDF(x) = (CDF(x) - CDF(-1)) / (CDF(1) - CDF(-1))
        where CDF is the cumulative distribution function of the normal distribution.
        The samples are then transformed using the inverse CDF (quantile function) of the normal distribution.
        The samples are then clipped to [-1, 1] to ensure they are within the truncated range.
        """
        # Generate uniform random numbers
        u = jax.random.uniform(key, shape=sample_shape + self.loc.shape)
        cdf = self.cdf_m1 + u * (self.cdf_m1_to_p1)
        samples = norm.ppf(cdf, loc=self.loc, scale=self.scale_diag)
        return jnp.clip(samples, -1.0, 1.0)
    
    def log_prob(self, value):
        """
        Compute the log probability of the given value, considering the truncation.
        """
        return super().log_prob(value) - jnp.sum(jnp.log(self.cdf_m1_to_p1), axis=-1)

    def entropy(self):
        return jnp.mean(jnp.log(2*(1.0 - jnp.exp(-2 * self.scale_diag)))) # Very approximate entropy for truncated normal distribution
    
class BetaDistribution(Distribution):
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
        return jax.random.beta(
            key,
            self.alpha,
            self.beta,
            shape=sample_shape + self.batch_shape + self.event_shape,
        )

    def log_prob(self, value):
        return (
            (self.alpha - 1) * jnp.log(value)
            + (self.beta - 1) * jnp.log(1.0 - value)
            - betaln(self.alpha, self.beta)
        )

    def entropy(self):
        return (
            betaln(self.alpha, self.beta)
            - (self.alpha - 1) * digamma(self.alpha)
            - (self.beta - 1) * digamma(self.beta)
            + (self.alpha + self.beta - 2) * digamma(self.alpha + self.beta)
        )
