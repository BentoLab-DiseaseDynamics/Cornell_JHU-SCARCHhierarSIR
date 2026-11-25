"""
This script contains all functions related to the Bayesian numpyro model

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

import sys
import time
import jax
import optax
import xarray as xr
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
from numpyro.infer import init_to_median, init_to_value


#####################
## Finding the MAP ##
#####################

def find_map(model, model_kwargs, n_preoptim=1000, lr=1e-2, map_params=None):
    """
    Find the MAP estimate of a NumPyro model using Adam optimization.

    Parameters
    ----------
    model : callable
        NumPyro model.

    model_kwargs : dict
        Keyword arguments passed to `training_model`.

    n_preoptim : int
        Number of Adam optimization steps.

    lr : float
        Adam learning rate.

    Returns
    -------
    map_params : dict
        MAP estimate in the constrained parameter space used by the NumPyro model.
    """

    # --------------------------------------------------
    # Initialize model in unconstrained space
    # --------------------------------------------------

    rng_key = jax.random.PRNGKey(int(time.time()))

    if map_params:
        init_strategy=init_to_value(values=map_params)
    else:
        init_strategy=init_to_median()

    init_params_info, potential_fn, postprocess_fn, _ = (
        initialize_model(
            rng_key,
            model,
            model_kwargs=model_kwargs,
            init_strategy=init_strategy
        )
    )

    z = init_params_info.z

    # --------------------------------------------------
    # Initialise Adams optimizer
    # --------------------------------------------------

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(z)

    # --------------------------------------------------
    # Define ptimization step
    # --------------------------------------------------

    @jax.jit
    def step(z, opt_state):

        potential_energy, grads = jax.value_and_grad(potential_fn)(z)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            z,
        )

        z = optax.apply_updates(
            z,
            updates,
        )

        return z, opt_state, potential_energy

    # --------------------------------------------------
    # Optimize and print progress
    # --------------------------------------------------

    for i in range(n_preoptim+1):

        z, opt_state, pe = step(z, opt_state)

        if i % 100 == 0:
            print(i, float(pe))
            sys.stdout.flush()

    # --------------------------------------------------
    # Transform back to constrained space
    # --------------------------------------------------

    map_params = postprocess_fn(z)

    return map_params


##############################
## Tempered NB distribution ##
##############################


def compute_season_weights(data):
    """
    Compute weights so each season-state contributes equally.

    Parameters
    ----------
    data : jax.numpy.ndarray (n_seasons, n_states, n_observations)

    Returns
    -------
    weights : jax.numpy.ndarray, shape (n_seasons, n_states, 1)
    """

    # max over observations per season-state
    max_per_season_state = jnp.sqrt(jnp.mean(data, axis=2))
    inv_max = 1.0 / max_per_season_state
    # normalize to mean 1
    normalized = inv_max / jnp.mean(inv_max)
    # expand dims for broadcasting across observations
    return normalized[:, :, None]


class WeightedNB(dist.Distribution):
    """
    Weighted negative-binomial distribution with a mean parameterization.

    This distribution evaluates the log-probability using a negative
    binomial distribution and applies observation-specific weights to the
    resulting pointwise log-probabilities. The weights therefore temper the
    contribution of individual observations to the log-likelihood but do
    not affect posterior-predictive sampling.

    Parameters
    ----------
    mu : jax.numpy.ndarray
        Mean parameter of the negative binomial distribution. Expected shape
        is ``(n_seasons, n_states, n_observations)``.

    alpha : jax.numpy.ndarray
        Concentration (inverse-dispersion) parameter of the negative binomial
        distribution. Expected shape is ``(n_states,)``. The parameter is
        broadcast across seasons and observations.

    weights : jax.numpy.ndarray
        Observation-specific weights used to temper the log-probability.
        Expected shape is ``(n_seasons, n_states, n_observations)``.

    validate_args : bool, optional
        Whether to validate distribution arguments.
    """

    support = dist.constraints.nonnegative_integer

    def __init__(self, mu, alpha, weights, validate_args=None):
        self.mu = mu
        self.alpha = alpha
        self.weights = weights

        # Batch shape must match the shape of the data
        super().__init__(batch_shape=mu.shape, validate_args=validate_args,)


    def log_prob(self, value):

        # Align axes
        v_aligned = jnp.moveaxis(value, 1, -1)
        mu_aligned = jnp.moveaxis(self.mu, 1, -1)
        w_aligned = jnp.moveaxis(self.weights, 1, -1)

        # Record which observations have an unsafe mean
        unsafe_mu = mu_aligned <= 0

        # Evaluate the NB log-probability normally using the original mu
        pointwise_logp = dist.NegativeBinomial2(
            mean=mu_aligned,
            concentration=self.alpha,
        ).log_prob(v_aligned)

        # Apply weights
        weighted_logp_aligned = w_aligned * pointwise_logp

        # Redact ONLY observations for which the original mu was unsafe
        weighted_logp_aligned = jnp.where(
            unsafe_mu,
            0.0,
            weighted_logp_aligned,
        )

        # Restore original axis ordering
        return jnp.moveaxis(weighted_logp_aligned, -1, 1)

    def sample(self, key, sample_shape=()):
        """
        Generate samples from the underlying negative-binomial distribution.

        The observation weights are not applied when sampling. They are used
        only to temper the likelihood during inference. The state-specific
        concentration parameter ``alpha`` is broadcast across seasons and
        observations to match the shape of ``mu``.

        Parameters
        ----------
        key : jax.random.PRNGKey
            JAX random number generator key used for sampling.

        sample_shape : tuple, optional
            Shape of the leading sample dimensions to generate.

        Returns
        -------
        jax.numpy.ndarray
            Samples from the negative-binomial distribution with shape
            ``sample_shape + (n_seasons, n_states, n_observations)``.
        """

        # Broadcast alpha from (n_states,) to (n_seasons, n_states, n_observations) to match mu.
        nb_dist = dist.NegativeBinomial2(mean=self.mu, concentration=self.alpha[None, :, None])

        return nb_dist.sample(key, sample_shape=sample_shape)



###########################
## Geometric Random Walk ##
###########################


def add_geometric_random_walk(da, dim, sigma, rng_key):
    """Apply geometric random walk to DataArray along specified dimension.
    
    This function adds time-correlated variance to a forecast by applying a
    geometric random walk along the specified dimension.
    
    The geometric random walk is implemented as:
        y_t = y_0 * exp(sigma * sum_{i=1}^t epsilon_i)
    where epsilon_i ~ N(0, 1) are independent normal innovations.
    
    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray to apply the GRW to.
    
    dim : str
        The dimension name along which to apply the random walk.
    
    sigma : float
        The standard deviation of the GRW innovations. Controls the
        amount of variance added. Higher values result in more dispersion.
    
    rng_key : jax.random.PRNGKey
        JAX random key for generating innovations. Should be split from
        the main random key to ensure reproducibility.
    
    Returns
    -------
    xarray.DataArray
        DataArray with GRW applied along the specified dimension.
        Has the same shape and dimensions as the input.
    """

    other_dims = [d for d in da.dims if d != dim]
    other_shape = tuple(da.sizes[d] for d in other_dims)
    n_grw = da.sizes[dim]
    
    # Generate normal innovations: shape (*other_shape, n_grw)
    eps = jax.random.normal(rng_key, shape=(*other_shape, n_grw))
    
    # Cumulative sum along GRW dimension (last axis)
    cumulative_eps = jnp.cumsum(eps, axis=-1)
    
    # Compute multipliers: exp(sigma * cumulative_eps)
    multipliers = jnp.exp(sigma * cumulative_eps)
    
    # Create DataArray with matching coordinates
    multiplier_da = xr.DataArray(
        multipliers,
        dims=(*other_dims, dim),
        coords={d: da.coords[d] for d in other_dims}
    )
    return da * multiplier_da