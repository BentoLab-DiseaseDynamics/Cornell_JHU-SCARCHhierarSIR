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

#####################
## Preoptimisation ##
#####################

def preoptimize_parameters(
    *,
    jitted_sol_op,
    args_static,
    args_nodiff,
    data,
    init_params,          # dict with initial guess for beta, rho, fI, fR, delta_beta
    n_seasons,
    n_states,
    n_iter=1000,
    lr=1e-2,
):
    """
    Returns optimized *constrained* parameters (same shape as model expects).
    """

    # unconstrain parameters
    single = unconstrain(init_params)
    args_diff = jnp.broadcast_to(
        single,
        (n_seasons, n_states, single.shape[0])
    )

    # define loss function
    def loss_fn(args_diff):
        constrained = constrain(args_diff)
        pred = jitted_sol_op(constrained, args_nodiff, args_static)
        return jnp.sum((data - pred) ** 2)

    # initialise optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(args_diff)

    # jax jit the loss function
    @jax.jit
    def step(args_diff, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(args_diff)
        updates, opt_state = optimizer.update(grads, opt_state)
        args_diff = optax.apply_updates(args_diff, updates)
        return args_diff, opt_state, loss

    # optimization loop
    for i in range(n_iter):
        args_diff, opt_state, loss = step(args_diff, opt_state)

        if i % 100 == 0:
            print(i, float(loss))

    # Return constrained params
    return constrain(args_diff)



############################
## Transformation helpers ##
############################

def constrain(x):
    """Map unconstrained -> constrained"""

    beta = 0.45 + 0.01 * jax.nn.sigmoid(x[..., 0:1])   # beta (0.45-0.46)
    rho_fI = jax.nn.softplus(x[..., 1:3])              # rho, fI (positive)
    fR = jax.nn.sigmoid(x[..., 3:4])                   # fR (0-1)
    delta_beta = 0.25 * jnp.tanh(x[..., 4:])           # delta_beta (-0.25 to 0.25)

    return jnp.concatenate([beta, rho_fI, fR, delta_beta], axis=-1)


def unconstrain(params):
    """Map constrained -> unconstrained"""

    beta = params["beta"]
    rho = params["rho"]
    fI = params["fI"]
    fR = params["fR"]
    delta_beta = params["delta_beta"]

    # inverse scaled sigmoid
    beta_scaled = jnp.clip((beta - 0.45) / 0.01, 1e-6, 1 - 1e-6)
    beta_unconstrained = jnp.log(beta_scaled / (1 - beta_scaled))

    return jnp.concatenate([
        jnp.array([
            beta_unconstrained,
            jnp.log(jnp.exp(rho) - 1),
            jnp.log(jnp.exp(fI) - 1),
            jnp.log(fR / (1 - fR)),
        ]),
        jnp.arctanh(delta_beta / 0.25)
    ])



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
    Convert optimized parameter tensor into PyMC initial effects.
    """

    beta = np.array(args_diff_preoptim[:, :, 0])
    rho = np.array(args_diff_preoptim[:, :, 1])
    fI = np.array(args_diff_preoptim[:, :, 2])
    fR = np.array(args_diff_preoptim[:, :, 3])
    delta_beta = np.array(args_diff_preoptim[:, :, 4:])

    results = {}

    # rho (log scale)
    results["log_rho"] = decompose_effects(rho, transform=np.log)

    # fI (log scale)
    results["log_fI"] = decompose_effects(fI, transform=np.log)

    # fR (logit scale)
    results["logit_fR"] = decompose_effects(fR, transform=logit)

    # delta_beta (no decomposition, but mean)
    results["delta_beta_mu"] = np.transpose(np.mean(delta_beta, axis=0))

    return results
