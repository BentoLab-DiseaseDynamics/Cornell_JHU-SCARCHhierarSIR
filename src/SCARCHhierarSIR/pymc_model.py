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
import xarray as xr
import pytensor.tensor as pt

#################################################
## Extraction of last sample from previous run ##
#################################################

def trace_to_initvals(trace, free_rvs):
    """
    Convert the last draw of an ArviZ InferenceData object
    into a PyMC initvals list.

    Parameters
    ----------
    trace : arviz.InferenceData
        Loaded trace.
    free_rvs : list[str]
        Names of free random variables expected by the model.

    Returns
    -------
    list[dict]
        One dictionary per chain, suitable for pm.sample(initvals=...)
    """

    posterior = trace.posterior 
    initvals = []

    for chain in posterior.chain.values:

        chain_init = {}

        for rv_name in free_rvs:

            if rv_name not in list(posterior.data_vars):
                raise KeyError(
                    f"Variable '{rv_name}' not found in trace."
                )

            value = (
                posterior[rv_name]
                .sel(chain=chain)
                .isel(draw=-1)
                .values
            )

            # convert xarray -> numpy scalar/array
            chain_init[rv_name] = np.asarray(value)

        initvals.append(chain_init)

    return initvals

####################################
## Concatenation of trace objects ##
####################################

def concat_traces(trace1: xr.DataTree, trace2: xr.DataTree) -> xr.DataTree:
    """
    Concatenate two PyMC/ArviZ DataTree traces along the draw dimension.

    Parameters
    ----------
    trace1 : DataTree
        Existing trace.
    trace2 : DataTree
        New continuation trace.

    Returns
    -------
    DataTree
        Combined trace with draw coordinates re-indexed.
    """

    children = {}

    # iterate over top-level groups
    for group_name in trace1.children:

        ds1 = trace1[group_name].ds

        # if missing from second trace, keep original
        if group_name not in trace2.children:
            children[group_name] = xr.DataTree(ds1)
            continue

        ds2 = trace2[group_name].ds

        # concatenate groups containing draws
        if "draw" in ds1.dims:

            ds = xr.concat(
                [ds1, ds2],
                dim="draw",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            )

            # re-index draw coordinate
            ds = ds.assign_coords(
                draw=np.arange(ds.sizes["draw"])
            )

            children[group_name] = xr.DataTree(ds)

        else:
            # observed_data etc.
            children[group_name] = xr.DataTree(ds1)

    return xr.DataTree(children=children)

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

def AR_GARCH_step(eta_t, prev_z, prev_sigma2, prev_eps, psi, omega, a_garch, b_garch):

    # --- Compute variance ---
    sigma2 = omega + a_garch * (prev_eps ** 2) + b_garch * prev_sigma2  # GARCH (1,1)
    
    # --- AR(1) ---
    eps = eta_t * pt.sqrt(sigma2)
    z = psi * prev_z + eps

    return z, sigma2, eps

