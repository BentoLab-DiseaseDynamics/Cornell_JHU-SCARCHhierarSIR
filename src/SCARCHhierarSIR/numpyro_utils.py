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
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
from numpyro.infer import init_to_median, init_to_sample, init_to_mean


#####################
## Finding the MAP ##
#####################

def find_map(model, model_kwargs, n_preoptim=1000, lr=1e-2):
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

    init_params_info, potential_fn, postprocess_fn, _ = (
        initialize_model(
            rng_key,
            model,
            model_kwargs=model_kwargs,
            init_strategy=init_to_median()
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