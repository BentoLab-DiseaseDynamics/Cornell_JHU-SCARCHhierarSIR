"""
This script contains all functions related to pre-pymc optimization of the forward-simulating SIR ODE model

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

# general purpose python
import numpy as np
from scipy.special import logit

# jax
import jax
import optax
import jax.numpy as jnp

from SCARCHhierarSIR.jax_forward_sim_model import forward_sim_preopt_jax

#####################
## Preoptimisation ##
#####################


def preoptimize_parameters(
    *,
    args_static,
    data,
    init_params,
    n_iter=1000,
    lr=1e-2,
):
    """
    Deterministically pre-optimize rho, fI, fR and delta_beta.

    beta and gamma are fixed through args_static.

    Parameters
    ----------
    args_static : tuple
        (t0, t1_max, modifier_length, beta, gamma, population, ts)

    data : array
        Observed data with shape
        (n_seasons, n_states, n_observations)

    init_params : dict
        Initial constrained parameter values:
            rho
            fI
            fR
            delta_beta

        Shapes:
            rho         : (n_seasons, n_states)
            fI          : (n_seasons, n_states)
            fR          : (n_seasons, n_states)
            delta_beta  : (n_seasons, n_states, n_modifiers)

    Returns
    -------
    optimized_params : dict
        Optimized constrained parameters with the same shapes as init_params.
    """

    # --------------------------------------------------
    # Transform initial constrained parameters
    # to unconstrained space
    # --------------------------------------------------

    def inv_softplus(x):
        return x + jnp.log(-jnp.expm1(-x))

    params_raw = {
        "rho": inv_softplus(init_params["rho"]),
        "fI": inv_softplus(init_params["fI"]),
        "fR": jax.scipy.special.logit(init_params["fR"]),
        "delta_beta": jnp.arctanh(
            init_params["delta_beta"] / 0.25
        ),
    }

    # --------------------------------------------------
    # Transform unconstrained -> constrained
    # --------------------------------------------------

    def constrain(params_raw):

        return {
            "rho": jax.nn.softplus(params_raw["rho"]),
            "fI": jax.nn.softplus(params_raw["fI"]),
            "fR": jax.nn.sigmoid(params_raw["fR"]),
            "delta_beta": (0.25 * jnp.tanh(params_raw["delta_beta"])),
        }

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------

    def loss_fn(params_raw):

        params = constrain(params_raw)

        pred = forward_sim_preopt_jax(
            rho=params["rho"],
            fI=params["fI"],
            fR=params["fR"],
            delta_beta=params["delta_beta"],
            args_static=args_static,
        )

        return jnp.sum((data - pred) ** 2)

    # --------------------------------------------------
    # Initialize optimizer
    # --------------------------------------------------

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params_raw)

    # --------------------------------------------------
    # JIT optimization step
    # --------------------------------------------------

    @jax.jit
    def step(params_raw, opt_state):

        loss, grads = jax.value_and_grad(loss_fn)(params_raw)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params_raw,
        )

        params_raw = optax.apply_updates(
            params_raw,
            updates,
        )

        return params_raw, opt_state, loss

    # --------------------------------------------------
    # Optimization loop
    # --------------------------------------------------

    for i in range(n_iter):

        params_raw, opt_state, loss = step(
            params_raw,
            opt_state,
        )

        if i % 100 == 0:
            print(i, float(loss))

    # --------------------------------------------------
    # Return constrained parameters
    # --------------------------------------------------

    return constrain(params_raw)


##################################################
## Estimate initial pyMC training model effects ##
##################################################

def decompose_effects(array_2d, transform=None):
    """
    Decompose a (n_seasons, n_states) array into:
        global + state effects + season effects

    Parameters
    ----------
    array_2d : np.ndarray
        Shape (n_seasons, n_states)
    transform : callable, optional
        Applied before decomposition (e.g. log, logit)

    Returns
    -------
    dict with:
        global
        state_effects
        season_effects
        reconstructed (in transformed space)
        error_mean
        error_max
    """

    if transform is not None:
        x = transform(array_2d)
    else:
        x = array_2d

    # global mean
    global_mean = np.mean(x)

    # state effects (columns)
    state_effects = np.mean(x, axis=0) - global_mean

    # season effects (rows)
    season_effects = np.mean(x, axis=1) - global_mean

    # reconstruction
    reconstructed = (
        global_mean
        + state_effects[None, :]
        + season_effects[:, None]
    )

    error = np.abs(reconstructed - x)

    return {
        "global": global_mean,
        "state": state_effects,
        "season": season_effects,
        "reconstructed": reconstructed,
        "error_mean": error.mean(),
        "error_max": error.max(),
    }


def compute_initial_effects(args_diff_preoptim):
    """
    Convert deterministically pre-optimized parameters into
    initial values for the hierarchical NumPyro model.
    """

    # get values
    rho = np.asarray(args_diff_preoptim["rho"])
    fI = np.asarray(args_diff_preoptim["fI"])
    fR = np.asarray(args_diff_preoptim["fR"])
    delta_beta = np.asarray(args_diff_preoptim["delta_beta"])

    results = {}

    # decompose parameters
    results["log_rho"] = decompose_effects(rho, transform=np.log)
    results["log_fI"] = decompose_effects(fI, transform=np.log,)
    results["logit_fR"] = decompose_effects(fR, transform=logit)
    results["delta_beta_mu"] = np.mean(delta_beta, axis=1) # delta_beta: mean across seasons

    return results