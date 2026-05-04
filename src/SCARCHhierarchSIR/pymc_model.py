"""
This script contains all functions related to the Bayesian pymc model

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

import numpy as np
import pymc as pm
import pytensor.tensor as pt



##############################
## Tempered NB distribution ##
##############################

def compute_season_weights(data):
    """
    Compute weights so each season-state contributes equally.

    Parameters
    ----------
    data : ndarray (n_seasons, n_states, n_observations)

    Returns
    -------
    weights : np.ndarray, shape (n_seasons, n_states, 1)
    """
    # max over observations per season-state
    max_per_season_state = np.sqrt(data.mean(axis=2))
    inv_max = 1.0 / max_per_season_state
    # normalize to mean 1
    normalized = inv_max / inv_max.mean()
    # expand dims for broadcasting across observations
    return normalized[:, :, None]



def weighted_nb_logp(value, mu, alpha, weights):
    """
    Weighted Negative Binomial log-probability.

    Parameters
    ----------
    value : observed counts
        shape (n_seasons, n_states, observations)

    mu : predicted mean
        shape (n_seasons, n_states, observations)

    alpha : NB dispersion parameter
        shape (n_states,)

    weights : season weights
        shape (n_seasons, n_states, 1)
    """

    # move state axis to the end so alpha (n_states,) broadcasts correctly
    mu = mu.dimshuffle(0, 2, 1)
    value = value.dimshuffle(0, 2, 1)
    weights = weights.dimshuffle(0, 2, 1)

    return pt.sum(weights * pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=alpha), value))



def weighted_nb_random(*args, rng=None, size=None):
    """
    Random draws from Negative Binomial for posterior predictive.
    weights are ignored during random draws
    """
    # mu, alpha: tensors -> convert to numpy
    mu_ = np.array(args[0])
    alpha_ = 1/np.array(args[1])

    # remove pyMC broadcast axes
    alpha_ = alpha_.reshape(-1)

    # broadcast to mu
    alpha_ = alpha_[None, :, None]

    # size: PyMC passes shape of batch/draws
    return rng.negative_binomial(n=1/alpha_, p=1/(1 + mu_ * alpha_), size=size)



####################################
## AR(1)-GARCH(1,1) step function ##
####################################

def AR_GARCH_step(eta_t, prev_z, prev_sigma2, prev_eps, psi, omega, a_garch, b_garch, use_garch):

    # --- Compute variance ---
    sigma2 = pt.switch(
        use_garch,
        omega + a_garch * (prev_eps ** 2) + b_garch * prev_sigma2,  # GARCH (1,1)
        prev_sigma2                                                 # constant (= sigma2_0)
    )
    
    # --- AR(1) ---
    eps = eta_t * pt.sqrt(sigma2)
    z = psi * prev_z + eps

    return z, sigma2, eps

