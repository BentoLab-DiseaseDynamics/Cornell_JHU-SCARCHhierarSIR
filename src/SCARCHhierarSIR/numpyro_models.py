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

import jax
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from SCARCHhierarSIR.numpyro_utils import  WeightedNB
from SCARCHhierarSIR.jax_forward_sim_model import forward_sim_jax


####################
## Training model ##
####################

training_RV_dims = {

    # ascertainment
    "rho_state_raw": ["state"],
    "rho_season_raw": ["season"],
    "rho_state": ["state"],
    "rho_season": ["season"],
    "rho": ["season", "state"],

    # initial infected
    "fI_state_raw": ["state"],
    "fI_season_raw": ["season"],
    "fI_state": ["state"],
    "fI_season": ["season"],
    "fI": ["season", "state"],

    # initial recovered
    "fR_state_raw": ["state"],
    "fR_season_raw": ["season"],
    "fR_state": ["state"],
    "fR_season": ["season"],
    "fR": ["season", "state"],

    # modifier parameters
    "eta_raw": ["modifier_eta", "season", "state"],
    "delta_beta_spline_raw": ["spline_basis", "state"],
    "delta_beta_spline_coef": ["spline_basis", "state"],
    "delta_beta_state_mean": ["modifier", "state"],

    # modifier trajectories
    "z": ["modifier", "season", "state"],
    "delta_beta": ["modifier", "season", "state"],
    "sigma2": ["modifier", "season", "state"],
    "eps": ["modifier", "season", "state"],

    # simuation output
    "H": ["season", "state", "observation"],

    # observation model
    "alpha_inv": ["state"],
    "alpha": ["state"],
    "data": ["season", "state", "observation"]
}


def training_model(data, weights, adj, phi, omega, a_garch, b_garch, spline_basis, args_static, n_states, n_seasons, n_modifiers):
    
    # ============================================================
    # Ascertainment: rho
    # ============================================================

    # Global
    log_rho_global_mean = numpyro.sample("log_rho_global_mean", dist.Normal(jnp.log(0.01), 1/2))   # 0.01, 1/2
    numpyro.deterministic("rho_global_mean", jnp.exp(log_rho_global_mean))

    # State
    rho_state_sd = numpyro.sample("rho_state_sd", dist.HalfNormal(1/5))
    rho_state_raw = numpyro.sample("rho_state_raw", dist.Normal(0, 1).expand([n_states]))
    numpyro.deterministic("rho_state", jnp.exp(rho_state_sd * rho_state_raw))

    # Season
    rho_season_sd = numpyro.sample("rho_season_sd", dist.HalfNormal(1/5))
    rho_season_raw = numpyro.sample("rho_season_raw", dist.Normal(0, 1).expand([n_seasons]))
    numpyro.deterministic("rho_season", jnp.exp(rho_season_sd * rho_season_raw))

    # Additive hierarchy
    log_rho = (
        log_rho_global_mean
        + rho_state_sd * rho_state_raw[None, :]
        + rho_season_sd * rho_season_raw[:, None]
    )

    rho = numpyro.deterministic("rho", jnp.exp(log_rho))


    # ============================================================
    # Initial infected: fI
    # ============================================================

    # Global
    log_fI_global_mean = numpyro.sample("log_fI_global_mean", dist.Normal(jnp.log(0.0001), 1/2))    # 0.0001, 1/2
    numpyro.deterministic("fI_global_mean", jnp.exp(log_fI_global_mean))

    # State
    fI_state_sd = numpyro.sample("fI_state_sd", dist.HalfNormal(1/5))
    fI_state_raw = numpyro.sample("fI_state_raw", dist.Normal(0, 1).expand([n_states]))
    numpyro.deterministic("fI_state", jnp.exp(fI_state_sd * fI_state_raw))

    # Season
    fI_season_sd = numpyro.sample("fI_season_sd", dist.HalfNormal(1/5))
    fI_season_raw = numpyro.sample("fI_season_raw", dist.Normal(0, 1).expand([n_seasons]))
    numpyro.deterministic("fI_season", jnp.exp(fI_season_sd * fI_season_raw))

    # Additive hierarchy
    log_fI = (
        log_fI_global_mean
        + fI_state_sd * fI_state_raw[None, :]
        + fI_season_sd * fI_season_raw[:, None]
    )

    fI = numpyro.deterministic("fI", jnp.exp(log_fI))


    # ============================================================
    # Initial recovered: fR
    # ============================================================

    # Global
    logit_fR_global_mean = numpyro.sample("logit_fR_global_mean", dist.Normal(jax.scipy.special.logit(0.375), 0.22)) # 0.375, 0.22
    fR_global_mean = jax.nn.sigmoid(logit_fR_global_mean)
    numpyro.deterministic("fR_global_mean", fR_global_mean)

    # State
    fR_state_sd = numpyro.sample("fR_state_sd", dist.HalfNormal(1/10))
    fR_state_raw = numpyro.sample("fR_state_raw", dist.Normal(0, 1).expand([n_states]))
    numpyro.deterministic("fR_state", jnp.exp(fR_state_sd * fR_state_raw))

    # Season
    fR_season_sd = numpyro.sample("fR_season_sd", dist.HalfNormal(1/5))
    fR_season_raw = numpyro.sample("fR_season_raw", dist.Normal(0, 1).expand([n_seasons]))
    numpyro.deterministic("fR_season", jnp.exp(fR_season_sd * fR_season_raw))

    # Additive hierarchy
    logit_fR = (
        logit_fR_global_mean
        + fR_state_sd * fR_state_raw[None, :]
        + fR_season_sd * fR_season_raw[:, None]
    )

    fR = numpyro.deterministic("fR", jax.nn.sigmoid(logit_fR))


    # ============================================================
    # Spatial correlation
    # ============================================================

    psi_1 = 1e-5 + (1 - 1e-5) * numpyro.sample("psi_1", dist.Beta(3, 3))
    psi_2 = numpyro.sample("psi_2", dist.Beta(3, 3),)

    I = jnp.eye(n_states)
    W = jnp.asarray(adj)
    D = jnp.diag(jnp.sum(W, axis=1))

    Q_modifiers = ((1 - psi_1) * I + psi_1 * (D - W))
    L_Q_modifiers = jnp.linalg.cholesky(Q_modifiers)
    L_cov_modifiers = jnp.linalg.solve(L_Q_modifiers, I)

    Q_shocks = ((1 - psi_2) * I + psi_2 * (D - W))
    L_Q_shocks = jnp.linalg.cholesky(Q_shocks)
    L_cov_shocks = jnp.linalg.solve(L_Q_shocks, I)

    # spatially correlate shocks
    eta_raw = numpyro.sample("eta_raw", dist.Normal(0, 1).expand([n_modifiers - 1, n_seasons, n_states]))
    eta = jnp.einsum("ij,tsj->tsi", L_cov_shocks, eta_raw)

    # ============================================================
    #  seasonal average modifiers per state (spatially correlated)
    # ============================================================

    n_basis = spline_basis.shape[1]

    # Spatially correlated spline coefficients
    delta_beta_spline_raw = numpyro.sample("delta_beta_spline_raw", dist.Laplace(1/3).expand([n_basis, n_states]))
    delta_beta_spline_coef = jnp.einsum("ij,bj->bi", L_cov_modifiers, delta_beta_spline_raw)
    numpyro.deterministic("delta_beta_spline_coef",  delta_beta_spline_coef)

    # Evaluate spline on every modifier week
    delta_beta_state_mean = jnp.einsum("db,bs->ds", spline_basis, delta_beta_spline_coef)
    numpyro.deterministic("delta_beta_state_mean", delta_beta_state_mean)

    # ============================================================
    # AR(1) - GARCH(1,1)
    # ============================================================

    numpyro.deterministic("phi", phi)
    numpyro.deterministic("a_garch", a_garch)
    numpyro.deterministic("b_garch", b_garch)
    omega = numpyro.deterministic("omega", omega)

    # ============================================================
    # Forward simulation
    # ============================================================

    H_raw, z_raw, sigma2_raw, eps_raw = forward_sim_jax(
        eta,
        phi,
        omega,
        a_garch,
        b_garch,
        delta_beta_state_mean,
        rho,
        fI,
        fR,
        args_static,
    )

    H = numpyro.deterministic("H", jax.nn.softplus(H_raw)) # at sample 0 --> H raw can become very negative --> underflows softplus --> H becomes zero --> negative binomial becomes nan
    numpyro.deterministic("z", z_raw)
    numpyro.deterministic("sigma2", sigma2_raw)
    numpyro.deterministic("eps", eps_raw)
    numpyro.deterministic("delta_beta", z_raw + delta_beta_state_mean[:, None, :])


    # ============================================================
    # Observation model
    # ============================================================

    alpha_inv = numpyro.sample("alpha_inv", dist.HalfNormal(0.003/3).expand([n_states]))
    alpha = numpyro.deterministic("alpha", 1.0/alpha_inv)

    numpyro.sample("data", WeightedNB(mu=H, alpha=alpha, weights=weights), obs=data if data is not None else None)

    pass


#######################
## Forecasting model ##
#######################

forecasting_RV_dims = {

    # ascertainment
    "rho_season_raw": ["season"],
    "rho": ["season", "state"],

    # initial infected
    "fI_season_raw": ["season"],
    "fI": ["season", "state"],

    # initial recovered
    "fR_season_raw": ["season"],
    "fR": ["season", "state"],

    # modifier parameters
    "eta_raw": ["modifier_eta", "season", "state"],
    "delta_beta_raw": ["modifier", "state"],

    # modifier trajectories
    "z": ["modifier", "season", "state"],
    "delta_beta": ["modifier", "season", "state"],
    "sigma2": ["modifier", "season", "state"],
    "eps": ["modifier", "season", "state"],

    # simuation output
    "H": ["season", "state", "observation"],

    # observation model
    "obs": ["season", "state", "horizon_observation"],
    "pred": ["state", "horizon_forecast"]
}

def forecasting_model(data, weights, posterior_params, adj, sigma_grw, args_static, n_states, n_seasons, n_modifiers, n_observations):

    # ============================================================
    # SIR model parameters
    # ============================================================

    ## ascertainment: rho
    ### global (rho_global_mean)
    ### state (rho_state)
    ### season (rho_season_sd)
    rho_season_raw = numpyro.sample("rho_season_raw", dist.Normal(0, 1).expand([n_seasons]))
    rho = numpyro.deterministic("rho", jnp.exp(jnp.log(posterior_params['rho_global_mean']) + jnp.log(posterior_params['rho_state'])[None, :] + posterior_params['rho_season_sd'] * rho_season_raw[:, None]))

    ## initial infected: fI
    ### global (fI_global_mean)
    ### state (fI_state)
    ### season (fI_season_sd)
    fI_season_raw = numpyro.sample("fI_season_raw", dist.Normal(0, 1).expand([n_seasons]))
    fI = numpyro.deterministic("fI", jnp.exp(jnp.log(posterior_params['fI_global_mean']) + jnp.log(posterior_params['fI_state'])[None, :] + posterior_params['fI_season_sd'] * fI_season_raw[:, None]))

    ## initial recovered: fR
    ### global (fR_global_mean)
    ### state (fR_state)
    ### season (fR_season_sd)
    fR_season_raw = numpyro.sample("fR_season_raw", dist.Normal(0, 1).expand([n_seasons]))
    fR = numpyro.deterministic("fR", jax.nn.sigmoid(jax.scipy.special.logit(posterior_params['fR_global_mean']) + jnp.log(posterior_params['fR_state'])[None, :] + posterior_params['fR_season_sd'] * fR_season_raw[:, None]))


    # ============================================================
    # Spatial correlation
    # ============================================================

    psi_2 = posterior_params['psi_2']

    I = jnp.eye(n_states)
    W = jnp.asarray(adj)
    D = jnp.diag(jnp.sum(W, axis=1))

    Q_shocks = ((1 - psi_2) * I + psi_2 * (D - W))
    L_Q_shocks = jnp.linalg.cholesky(Q_shocks)
    L_cov_shocks = jnp.linalg.solve(L_Q_shocks, I)

    # spatially correlate shocks
    eta_raw = numpyro.sample("eta_raw", dist.Normal(0, 1).expand([n_modifiers - 1, n_seasons, n_states]))
    eta = jnp.einsum("ij,tsj->tsi", L_cov_shocks, eta_raw)

    # ============================================================
    #  seasonal average modifiers per state (spatially correlated)
    # ============================================================

    delta_beta_state_mean = posterior_params['delta_beta_state_mean']


    # ============================================================
    # AR(1) - GARCH(1,1)
    # ============================================================

    omega = posterior_params['omega']
    phi = posterior_params['phi']
    a_garch = posterior_params['a_garch']
    b_garch = posterior_params['b_garch']

    # ===========================================================
    # Forward simulation
    # ============================================================

    H_raw, z_raw, sigma2_raw, eps_raw = forward_sim_jax(
        eta,
        phi,
        omega,
        a_garch,
        b_garch,
        delta_beta_state_mean,
        rho,
        fI,
        fR,
        args_static,
    )

    H = numpyro.deterministic("H", jax.nn.softplus(H_raw))
    numpyro.deterministic("z", z_raw,)
    numpyro.deterministic("delta_beta", z_raw + delta_beta_state_mean[:, None, :])
    numpyro.deterministic("sigma2", sigma2_raw)
    numpyro.deterministic("eps", eps_raw)

    # ============================================================
    # Observation model
    # ============================================================

    alpha = 1.0 / posterior_params['alpha_inv'] 

    if data is None:
        numpyro.sample("obs", WeightedNB(mu=H[:,:,:n_observations], alpha=alpha, weights=weights), obs=None)
    else:
        numpyro.sample("obs", WeightedNB(mu=H[:,:,:n_observations], alpha=alpha, weights=weights), obs=data[:,:,:n_observations])

    # ======================================================================
    # Forecast model (model output + NB observation noise + GRW white noise)
    # ======================================================================

    if data is None:

        # Geometric random walk innovations
        n_forecast = H.shape[2] - n_observations
        grw_innov = numpyro.sample("grw_innov", dist.Normal(0.0, sigma_grw).expand([n_states, n_forecast]))

        # Multiplicative GRW adjustment
        grw_log_scale = jnp.cumsum(grw_innov, axis=1)
        H_future_rw = H[:, :, n_observations:] * jnp.exp(grw_log_scale[None, :, :])

        # Forecast observation model
        numpyro.sample("pred", dist.NegativeBinomial2(mean=H_future_rw, concentration=alpha[None, :, None]), obs=None)

    pass
