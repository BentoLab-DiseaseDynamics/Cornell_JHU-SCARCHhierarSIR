"""
This script trains the model on historical data.

Author: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

# standard python libraries
import os
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from datetime import datetime, timedelta
# pyMC / pytensor
import pymc as pm
import arviz
# jax and diffrax
import jax.numpy as jnp
# model package
from SCARCHhierarSIR.data import get_demography, get_adjacency_matrix, get_NHSN_HRD_data
from SCARCHhierarSIR.SIR_model import get_jax_jitted_model, make_sol_op
from SCARCHhierarSIR.pymc_model import trace_to_initvals, concat_traces
from SCARCHhierarSIR.preoptimization import preoptimize_parameters, compute_initial_effects

from SCARCHhierarSIR.jax_model import ForwardOp, forward_jitted, forward_vjp_jitted, forward_jax

import numpyro
numpyro.set_host_device_count(4)

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive


class WeightedNB(dist.Distribution):
    # Dictates that the elements in the array are independent observations
    support = dist.constraints.nonnegative_integer

    def __init__(self, mu, alpha, weights, validate_args=None):
        self.mu = mu
        self.alpha = alpha
        self.weights = weights
        
        # Batch shape must match the shape of the data: (n_seasons, n_states, n_observations)
        super().__init__(batch_shape=mu.shape, validate_args=validate_args)

    def log_prob(self, value):
        # 1. Align axes to broadcast against alpha (shape: n_states,)
        # Moves n_states (axis 1) to the trailing position (-1)
        v_aligned = jnp.moveaxis(value, 1, -1)
        mu_aligned = jnp.moveaxis(self.mu, 1, -1)
        w_aligned = jnp.moveaxis(self.weights, 1, -1)

        # 2. Compute pointwise log-probabilities
        # Resulting shape matches the aligned dimensions
        pointwise_logp = dist.NegativeBinomial2(
            mean=mu_aligned,
            concentration=self.alpha,
        ).log_prob(v_aligned)

        # 3. Apply weights pointwise
        weighted_logp_aligned = w_aligned * pointwise_logp

        # 4. Restore the original shape: (n_seasons, n_states, n_observations)
        # Moves the trailing axis (-1) back to its original position (1)
        return jnp.moveaxis(weighted_logp_aligned, -1, 1)

    def sample(self, key, sample_shape=()):
        nb_dist = dist.NegativeBinomial2(
            mean=self.mu,
            concentration=self.alpha[None, :, None],
        )

        return nb_dist.sample(
            key,
            sample_shape=sample_shape,
        )

def compute_season_weights(data):

    data = jnp.asarray(data)

    max_per_season_state = jnp.sqrt(
        jnp.mean(data, axis=2)
    )

    inv_max = 1.0 / max_per_season_state

    normalized = inv_max / jnp.mean(inv_max)

    return normalized[:, :, None]


# needed to use the 'spawn' multiprocessing context manager
def run_training():
        
    # all paths defined relative to this file
    abs_dir = os.path.dirname(__file__)

    # global parameters go here
    ## model-structural
    a_garch = 0.0
    b_garch = 0.0
    phi = 0.5
    gamma = 1/3.5
    n_modifiers = 32
    modifier_length = 7
    start_simulation = 0 # (October 1)
    modifier_ref_month = 10
    modifier_ref_day = 1
    clustering_name = 'all'
    ## temporal extent of training
    n_observations = 35             # run until start of May
    seasons = ['2023-2024', '2024-2025', '2025-2026']
    ## sampling effort
    n_chains = 4
    n_sample = 50
    n_burn = 50
    training_name = f'test'
    n_preoptim = 500
    ## use previous sampling
    cont_sampling = False # To continue sampling, the number of chains and the observed data must match!

    ## save model-structural parameters and training metadata
    output_folder = os.path.join(abs_dir, f'../../data/interim/calibration/hierarchical-training/{training_name}')
    os.makedirs(output_folder, exist_ok=True)
    params = {"a_garch": a_garch, "b_garch": b_garch, "phi": phi, "gamma": 1 / 3.5, "n_modifiers": n_modifiers, "modifier_length": modifier_length, "start_simulation": start_simulation,
              "modifier_ref_month": modifier_ref_month, "modifier_ref_day": modifier_ref_day, 'clustering_name': clustering_name,
               "observations": n_observations, 'seasons': seasons}
    with open(os.path.join(output_folder, "model_config.json"), "w") as f:
        json.dump(params, f, indent=4)

    # derived products
    ## convert to a list of start and enddates (datetime)
    n_seasons = len(seasons)
    start_calibrations = [datetime(int(season[0:4]),modifier_ref_month, modifier_ref_day) + timedelta(days=start_simulation) for season in seasons] # start calibration at simulation start
    modifier_reference_dates = [datetime(int(season[0:4]), modifier_ref_month, modifier_ref_day) for season in seasons]

    # Get the clusters
    # ~~~~~~~~~~~~~~~~

    clusters = pd.read_csv(os.path.join(abs_dir, "../../data/interim/geography/clusters.csv"))
    cluster_indices = sorted(clusters[clustering_name].unique())

    # Loop over the clusters
    # ~~~~~~~~~~~~~~~~~~~~~~

    hyperparameters = []
    for cluster_idx in cluster_indices:

        print(f'\nworking on cluster {cluster_idx}')
        print('~~~~~~~~~~~~~~~~~~~~\n')

        print(f'states in cluster: {clusters[clusters[clustering_name] == cluster_idx]['abbreviation_state'].values.tolist()}\n')

        cluster_output_folder = os.path.join(output_folder, f'cluster_{cluster_idx}')

        # Get US demographics
        # ~~~~~~~~~~~~~~~~~~~

        state_fips_index, demo = get_demography(clusters[clusters[clustering_name] == cluster_idx]['abbreviation_state'])
        n_states = len(demo)

        # Get state adjacency matrix
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        adj = get_adjacency_matrix(state_fips_index['abbreviation_state'])

        # Get US incidences
        # ~~~~~~~~~~~~~~~~~

        reference_date, data, dt, ts, n_observations = get_NHSN_HRD_data(start_calibrations, modifier_reference_dates, n_observations, forecast_horizon=None, state_fips=state_fips_index['fips_state'].values) # (n_season, n_variables, n_observations)
        data = data / 7 # divide weekly incidence by 7

        # Outlier detection
        # ~~~~~~~~~~~~~~~~~

        from pygam import LinearGAM, s
        for season in range(data.shape[0]):
            for i,state in enumerate(range(data.shape[1])):

                d = data[season,state,:]

                y = np.log1p(np.asarray(d))
                x = np.arange(len(d))

                gam = LinearGAM(s(0), lam=0.05, n_splines=int(np.round(len(d)/2))).fit(x[:, None], y)

                trend = gam.predict(x[:, None])
                confint = gam.confidence_intervals(x[:, None], width=0.9994)

                outliers = (y < confint[:, 0]) | (y > confint[:, 1])

                # if state_fips_index.iloc[i]['abbreviation_state'] == 'DC':
                #     fig,ax=plt.subplots(figsize=(8.7, 11.3/4))
                #     ax.set_title(state_fips_index.iloc[i]['abbreviation_state'])
                #     ax.scatter(dt[season], np.expm1(y), marker='o', color='black', s=10)
                #     ax.plot(dt[season], np.expm1(trend), color='green')
                #     ax.fill_between(dt[season], np.expm1(confint[:,0]), np.expm1(confint[:,1]), color='green', alpha=0.15)
                #     ax.scatter(dt[season][outliers], np.expm1(y[outliers]), marker='x', color='red')
                #     plt.show()
                #     plt.close()

                y[outliers] = trend[outliers]
                data[season, state, :] = np.expm1(y)

        # TODO: assert if there's nan in data

        # Define a jax-jitted diffrax differential equation model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        jitted_sol_op_multi, jitted_vjp_sol_op_multi = get_jax_jitted_model()

        # Define the Op and VJPOp classes for the ODE problem
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        args_static = (start_simulation, max(ts[:,-1]), modifier_length)
        sol_op = make_sol_op(args_static, jitted_sol_op_multi, jitted_vjp_sol_op_multi)

        # Pre-optimize the forward simulation model's parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print('pre-optimization\n')
        print('(iter, score)')

        # stack args_nodiff so two leading axes are seasons, states and the third axes gives the arguments for the season-state combination
        gamma_vec = jnp.full((n_seasons, n_states, 1), gamma)
        pop_mat = jnp.broadcast_to(jnp.asarray(demo)[None, :, None], (n_seasons, n_states, 1))
        ts_mat = jnp.broadcast_to(ts[:, None, :], (n_seasons, n_states, ts.shape[1]))
        args_nodiff = np.array(jnp.concatenate([gamma_vec, pop_mat, ts_mat], axis=2))     # shape: (n_seasons, n_states, )  --> convert to numpy otherwise error in pt.as_tensor_variable(args_nodiff) in make_node of pyMC model

        # pre-optimize the initial guesses
        args_diff_preoptim = preoptimize_parameters(
            jitted_sol_op=jitted_sol_op_multi,
            args_static=args_static,
            args_nodiff=args_nodiff,
            data=data,
            init_params=dict(
                beta=0.455,
                rho=0.0025,
                fI=1e-4,
                fR=0.25,
                delta_beta=jnp.zeros(n_modifiers),
            ),
            n_seasons=n_seasons,
            n_states=n_states,
            n_iter=n_preoptim,
        )

        # run simulation
        out = jitted_sol_op_multi(args_diff_preoptim, args_nodiff, args_static)

        # visualise the result
        for s in range(n_states):
            fig, ax = plt.subplots(nrows=1, figsize=(8.7, 11.3/4))
            for i in range(n_seasons):
                ax.plot(dt[i, :], 7*out[i, s, :], color='red', label='pred')
                ax.scatter(dt[i, :], 7*data[i, s, :], marker='o', color='black', label='obs')
            fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
            fig.tight_layout()
            os.makedirs(os.path.join(cluster_output_folder, 'initial-optim'), exist_ok=True)
            plt.savefig(os.path.join(cluster_output_folder,f'initial-optim/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
            plt.close(fig)

        # compute pyMC initial effect sizes
        init = compute_initial_effects(args_diff_preoptim)

        # make dictionary with initial sampler values
        initvals = n_chains * [{'alpha_inv': 0.05 * jnp.ones(n_states), 'delta_beta_raw': init["delta_beta_mu"] / 0.25,
                'log_rho_global_mean': init["log_rho"]["global"], 'rho_state_sd': 0.2, 'rho_state_raw': init["log_rho"]["state"] / 0.2, 'rho_season_sd': 0.2, 'rho_season_raw': init["log_rho"]["season"] / 0.2,
                'log_fI_global_mean': init["log_fI"]["global"], 'fI_state_sd': 0.2, 'fI_state_raw': init["log_fI"]["state"] / 0.2, 'fI_season_sd': 0.2, 'fI_season_raw': init["log_fI"]["season"] / 0.2,
                'logit_fR_global_mean': init["logit_fR"]["global"], 'fR_state_sd': 0.2, 'fR_state_raw': init["logit_fR"]["state"] / 0.2, 'fR_season_sd': 0.2, 'fR_season_raw': init["logit_fR"]["season"] / 0.2,
                'log_omega_global_mean': jnp.log(0.05/3), 'omega_global_mean_shrinkage': 0.05/3, 'psi_1': 0.5, 'psi_2': 0.5}]

        print('\nparameter hierarchy reconstruction\n')

        print("Mean log-rho:", init["log_rho"]["global"])
        print("Mean reconstruction error:", init["log_rho"]["error_mean"])
        print("Max reconstruction error:", init["log_rho"]["error_max"])

        print("Mean log-fI:", init["log_fI"]["global"])
        print("Mean reconstruction error:", init["log_fI"]["error_mean"])
        print("Max reconstruction error:", init["log_fI"]["error_max"])

        print("Mean logit-fR:", init["logit_fR"]["global"])
        print("Mean reconstruction error:", init["logit_fR"]["error_mean"])
        print("Max reconstruction error:", init["logit_fR"]["error_max"])

        # Build jax model
        # ~~~~~~~~~~~~~~~

        population = np.asarray(demo, dtype=np.float64)

        # Build tempored NB distribution
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        weights = compute_season_weights(data)

        # Build pyMC model
        # ~~~~~~~~~~~~~~~~

        print('\ncompiling pymc model')

        # construct coordinates
        coords = {
            "observation": np.arange(n_observations),
            "state": state_fips_index['abbreviation_state'].values,
            "season": seasons,
            "modifier": np.arange(n_modifiers),
            "modifier_eta": np.arange(n_modifiers-1)
        }

        dims = {
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

            # spatial/temporal modifiers
            "delta_beta_raw": ["modifier", "state"],
            "delta_beta_state_mean": ["modifier", "state"],

            "eta_raw": [
                "modifier_eta",
                "season",
                "state",
            ],

            "omega_state_raw": ["state"],
            "omega_season_raw": ["season"],

            "omega_state": ["state"],
            "omega_season": ["season"],
            "omega": ["season", "state"],

            "z": [
                "modifier",
                "season",
                "state",
            ],

            "delta_beta": [
                "modifier",
                "season",
                "state",
            ],

            "sigma2": [
                "modifier",
                "season",
                "state",
            ],

            "eps": [
                "modifier",
                "season",
                "state",
            ],

            "H": [
                "season",
                "state",
                "observation",
            ],

            "alpha_inv": ["state"],
            "alpha": ["state"],

            "data": ["season", "state", "observation"]
        }

        def numpyro_model(data, weights, adj, population, ts, gamma, beta, phi, init, args_static, n_states, n_seasons, n_modifiers):
            
            # ============================================================
            # Transmission coefficient
            # ============================================================

            beta = jnp.asarray(beta)

            # ============================================================
            # Ascertainment: rho
            # ============================================================

            # Global
            log_rho_global_mean = numpyro.sample(
                "log_rho_global_mean",
                dist.Normal(
                    init["log_rho"]["global"],
                    1 / 3,
                ),
            )

            numpyro.deterministic(
                "rho_global_mean",
                jnp.exp(log_rho_global_mean),
            )

            # State
            rho_state_sd = numpyro.sample(
                "rho_state_sd",
                dist.HalfNormal(1 / 5),
            )

            rho_state_raw = numpyro.sample(
                "rho_state_raw",
                dist.Normal(0, 1).expand([n_states]),
            )

            numpyro.deterministic(
                "rho_state",
                jnp.exp(rho_state_sd * rho_state_raw),
            )

            # Season
            rho_season_sd = numpyro.sample(
                "rho_season_sd",
                dist.HalfNormal(1 / 5),
            )

            rho_season_raw = numpyro.sample(
                "rho_season_raw",
                dist.Normal(0, 1).expand([n_seasons]),
            )

            numpyro.deterministic(
                "rho_season",
                jnp.exp(rho_season_sd * rho_season_raw),
            )

            log_rho = (
                log_rho_global_mean
                + rho_state_sd * rho_state_raw[None, :]
                + rho_season_sd * rho_season_raw[:, None]
            )

            rho = jnp.exp(log_rho)

            numpyro.deterministic("rho", rho)


            # ============================================================
            # Initial infected: fI
            # ============================================================

            log_fI_global_mean = numpyro.sample(
                "log_fI_global_mean",
                dist.Normal(
                    init["log_fI"]["global"],
                    1 / 3,
                ),
            )

            numpyro.deterministic(
                "fI_global_mean",
                jnp.exp(log_fI_global_mean),
            )

            fI_state_sd = numpyro.sample(
                "fI_state_sd",
                dist.HalfNormal(1 / 5),
            )

            fI_state_raw = numpyro.sample(
                "fI_state_raw",
                dist.Normal(0, 1).expand([n_states]),
            )

            numpyro.deterministic(
                "fI_state",
                jnp.exp(fI_state_sd * fI_state_raw),
            )

            fI_season_sd = numpyro.sample(
                "fI_season_sd",
                dist.HalfNormal(1 / 5),
            )

            fI_season_raw = numpyro.sample(
                "fI_season_raw",
                dist.Normal(0, 1).expand([n_seasons]),
            )

            numpyro.deterministic(
                "fI_season",
                jnp.exp(fI_season_sd * fI_season_raw),
            )

            log_fI = (
                log_fI_global_mean
                + fI_state_sd * fI_state_raw[None, :]
                + fI_season_sd * fI_season_raw[:, None]
            )

            fI = jnp.exp(log_fI)

            numpyro.deterministic("fI", fI)


            # ============================================================
            # Initial recovered: fR
            # ============================================================

            logit_fR_global_mean = numpyro.sample(
                "logit_fR_global_mean",
                dist.Normal(
                    jax.scipy.special.logit(0.4),
                    1.0,
                ),
            )

            fR_global_mean = jax.nn.sigmoid(
                logit_fR_global_mean
            )

            numpyro.deterministic(
                "fR_global_mean",
                fR_global_mean,
            )

            fR_state_sd = numpyro.sample(
                "fR_state_sd",
                dist.HalfNormal(1 / 5),
            )

            fR_state_raw = numpyro.sample(
                "fR_state_raw",
                dist.Normal(0, 1).expand([n_states]),
            )

            numpyro.deterministic(
                "fR_state",
                jnp.exp(fR_state_sd * fR_state_raw),
            )

            fR_season_sd = numpyro.sample(
                "fR_season_sd",
                dist.HalfNormal(1 / 5),
            )

            fR_season_raw = numpyro.sample(
                "fR_season_raw",
                dist.Normal(0, 1).expand([n_seasons]),
            )

            numpyro.deterministic(
                "fR_season",
                jnp.exp(fR_season_sd * fR_season_raw),
            )

            logit_fR = (
                logit_fR_global_mean
                + fR_state_sd * fR_state_raw[None, :]
                + fR_season_sd * fR_season_raw[:, None]
            )

            fR = jax.nn.sigmoid(logit_fR)

            numpyro.deterministic("fR", fR)


            # ============================================================
            # Spatial correlation
            # ============================================================

            psi_1 = 1e-5 + (
                1 - 1e-5
            ) * numpyro.sample(
                "psi_1",
                dist.Beta(3, 3),
            )

            psi_2 = numpyro.sample(
                "psi_2",
                dist.Beta(3, 3),
            )

            I = jnp.eye(n_states)
            W = jnp.asarray(adj)
            D = jnp.diag(jnp.sum(W, axis=1))

            Q_modifiers = (
                (1 - psi_1) * I
                + psi_1 * (D - W)
            )

            L_Q_modifiers = jnp.linalg.cholesky(
                Q_modifiers
            )

            L_cov_modifiers = jnp.linalg.solve(
                L_Q_modifiers,
                I,
            )

            Q_shocks = (
                (1 - psi_2) * I
                + psi_2 * (D - W)
            )

            L_Q_shocks = jnp.linalg.cholesky(
                Q_shocks
            )

            L_cov_shocks = jnp.linalg.solve(
                L_Q_shocks,
                I,
            )


            # ============================================================
            # delta_beta temporal/state component
            # ============================================================

            delta_beta_raw = numpyro.sample(
                "delta_beta_raw",
                dist.Normal(0, 1).expand(
                    [n_modifiers, n_states]
                ),
            )

            delta_beta_state_mean = (
                1 / 4
            ) * jnp.einsum(
                "ij,mj->mi",
                L_cov_modifiers,
                delta_beta_raw,
            )

            numpyro.deterministic(
                "delta_beta_state_mean",
                delta_beta_state_mean,
            )


            # ============================================================
            # AR(1)
            # ============================================================

            numpyro.deterministic(
                "phi",
                phi,
            )


            # ============================================================
            # GARCH / ARCH
            # ============================================================

            eta_raw = numpyro.sample(
                "eta_raw",
                dist.Normal(0, 1).expand(
                    [n_modifiers - 1, n_seasons, n_states]
                ),
            )


            # omega
            omega_global_mean_shrinkage = numpyro.sample(
                "omega_global_mean_shrinkage",
                dist.HalfNormal(0.05 / 3),
            )

            log_omega_global_mean = numpyro.sample(
                "log_omega_global_mean",
                dist.Normal(
                    jnp.log(omega_global_mean_shrinkage),
                    1 / 5,
                ),
            )

            omega_global_mean = jnp.exp(
                log_omega_global_mean
            )

            numpyro.deterministic(
                "omega_global_mean",
                omega_global_mean,
            )

            omega_state_sd = numpyro.sample(
                "omega_state_sd",
                dist.HalfNormal(1 / 5),
            )

            omega_state_raw = numpyro.sample(
                "omega_state_raw",
                dist.Normal(0, 1).expand([n_states]),
            )

            numpyro.deterministic(
                "omega_state",
                jnp.exp(
                    omega_state_sd * omega_state_raw
                ),
            )

            omega_season_sd = numpyro.sample(
                "omega_season_sd",
                dist.HalfNormal(1 / 5),
            )

            omega_season_raw = numpyro.sample(
                "omega_season_raw",
                dist.Normal(0, 1).expand([n_seasons]),
            )

            numpyro.deterministic(
                "omega_season",
                jnp.exp(
                    omega_season_sd * omega_season_raw
                ),
            )

            log_omega = (
                log_omega_global_mean
                + omega_state_sd * omega_state_raw[None, :]
                + omega_season_sd * omega_season_raw[:, None]
            )

            omega = jnp.exp(log_omega)

            numpyro.deterministic(
                "omega",
                omega,
            )


            # a_garch
            numpyro.deterministic(
                "a_garch",
                0.0,
            )


            # b_garch
            numpyro.deterministic(
                "b_garch",
                0.0,
            )


            # ============================================================
            # Forward simulation
            # ============================================================

            H_raw, z_raw, sigma2_raw, eps_raw = forward_jax(
                eta_raw,
                phi,
                omega,
                a_garch,
                b_garch,
                delta_beta_state_mean,
                L_cov_shocks,
                beta,
                rho,
                fI,
                fR,
                gamma,
                population,
                ts,
                args_static,
            )

            H = numpyro.deterministic(
                "H",
                jax.nn.softplus(7*H_raw),
            )

            numpyro.deterministic(
                "z",
                z_raw,
            )

            delta_beta = (
                z_raw
                + delta_beta_state_mean[:, None, :]
            )

            numpyro.deterministic(
                "delta_beta",
                delta_beta,
            )

            numpyro.deterministic(
                "sigma2",
                sigma2_raw,
            )

            numpyro.deterministic(
                "eps",
                eps_raw,
            )


            # ============================================================
            # NB dispersion
            # ============================================================

            alpha_inv = numpyro.sample(
                "alpha_inv",
                dist.HalfNormal(0.002 / 3).expand(
                    [n_states]
                ),
            )

            alpha = numpyro.deterministic(
                "alpha",
                1.0 / alpha_inv,
            )


            # ============================================================
            # Tempered NB likelihood
            # ============================================================

            numpyro.sample("data", WeightedNB(mu=H, alpha=alpha, weights=weights), obs=data if data is not None else None)
            
        # Sample pyMC model
        # ~~~~~~~~~~~~~~~~~

        print('\nstarting the sampler..\n')

        from numpyro.infer import init_to_value

        import time
        rng_key = jax.random.PRNGKey(int(time.time()))
        rng_key, rng_predict = jax.random.split(rng_key)

        kernel = NUTS(
            numpyro_model,
            step_size=0.0002,
            adapt_step_size=False,
            max_tree_depth=12,
            init_strategy = init_to_value(values=initvals[0]),
        )

        mcmc = MCMC(
            kernel,
            num_warmup=n_burn,
            num_samples=n_sample,
            num_chains=n_chains,
            chain_method="parallel",
            progress_bar=True,
        )

        mcmc.run(
            rng_key,
            data=jnp.asarray(7 * data),
            weights=jnp.asarray(weights),
            adj=jnp.asarray(adj),
            population=jnp.asarray(population),
            ts=jnp.asarray(ts),
            gamma=jnp.asarray(gamma),
            beta=jnp.full((n_seasons, n_states), 0.455),
            phi=phi,
            init=init,
            args_static=args_static,
            n_states=n_states,
            n_seasons=n_seasons,
            n_modifiers=n_modifiers,
        )

        print('\n..finished sampling\n')
        print('\nsaving traces\n')

        trace = arviz.from_numpyro(mcmc, coords=coords, dims=dims)

        trace.to_netcdf(os.path.join(cluster_output_folder, f"trace.nc"))

        inv_mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix # you will save this as a .json and then use it to restart runs

        print('\nmaking traceplots\n')

        # Generate traces
        variables2plot = [
                        'alpha_inv',                                                                            # overdispersion
                        'rho_global_mean', 'rho_state_sd', 'rho_state', 'rho_season_sd', 'rho_season', 'rho',   # rho
                        'fI_global_mean', 'fI_state_sd', 'fI_state', 'fI_season_sd', 'fI_season', 'fI',         # fI
                        'fR_global_mean', 'fR_state_sd', 'fR_state', 'fR_season_sd', 'fR_season', 'fR',         # fR
                        'psi_2', 'psi_1',                                                                       # spatial correlation strength
                        'phi',                                                                                  # AR 
                        'omega_global_mean', 'omega_state_sd', 'omega_state', 'omega_season_sd', 'omega_season', 'omega', # GARCH(1,0) parameters
                        'omega_global_mean_shrinkage',
                        'a_garch', 'b_garch',
                        ]

        # Save original traces
        os.makedirs(os.path.join(cluster_output_folder,'traces'), exist_ok=True)
        for var in variables2plot:
            arviz.plot_trace_dist(trace, var_names=[var], compact=True, combined=True, kind='kde') 
            plt.savefig(os.path.join(cluster_output_folder,f'traces/trace-{var}.pdf'))
            plt.close()

        print('\nmaking posterior predictive\n')

        # Make posterior predictive
        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        from numpyro.infer import Predictive

        predictive = Predictive(
            numpyro_model,
            posterior_samples=mcmc.get_samples(),
            )

        posterior_predictive = predictive(
            rng_predict,
            data=None,
            weights=jnp.asarray(weights),
            adj=jnp.asarray(adj),
            population=jnp.asarray(population),
            ts=jnp.asarray(ts),
            gamma=jnp.asarray(gamma),
            beta=jnp.full((n_seasons, n_states), 0.455),
            phi=phi,
            init=init,
            args_static=args_static,
            n_states=n_states,
            n_seasons=n_seasons,
            n_modifiers=n_modifiers,
        )

        posterior_predictive = arviz.from_numpyro(mcmc, posterior_predictive=posterior_predictive, coords=coords, dims=dims)

        # Save posterior predictive
        posterior_predictive.to_netcdf(os.path.join(cluster_output_folder,"posterior_predictive.nc"))

        print('\nmaking posterior predictive visualisations\n')

        # Visualisations
        # ~~~~~~~~~~~~~~

        # pairplots of alpha_inv and omega per U.S. state or territory
        os.makedirs(os.path.join(cluster_output_folder,'traces/pairplots'), exist_ok=True)
        x = trace.posterior['alpha_inv'].stack(sample=("chain", "draw"))
        y = trace.posterior['omega_state'].stack(sample=("chain", "draw"))
        states = x["state"].values
        for state in states:
            fig,ax=plt.subplots(figsize=(8.3/2, 11.7/4))
            ax.scatter(x.sel(state=state), y.sel(state=state), marker='o', color='black', alpha=0.05)
            # add regression
            if a_garch is None:
                res = linregress(x.sel(state=state), y.sel(state=state))
                xx = np.array([x.sel(state=state).min(), x.sel(state=state).max()])
                ax.plot(xx, res.intercept + res.slope * xx, color="red")
                text = (f"$R^2$ = {res.rvalue**2:.3f}")
                ax.text(0.05, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=5, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.set_xlabel(r'$1/\alpha_i$')
            ax.set_ylabel(r'$\omega_i$')
            ax.set_title(f'{state}')
            plt.tight_layout()
            plt.savefig(os.path.join(cluster_output_folder,f'traces/pairplots/pairplot-alpha_omega-{state}.pdf'))
            plt.close()


        # pairplot of a_garch, omega_global and phi
        x1 = trace.posterior['a_garch'].stack(sample=("chain", "draw"))
        x2 = trace.posterior['omega_global_mean'].stack(sample=("chain", "draw"))
        x3 = trace.posterior['phi'].stack(sample=("chain", "draw"))

        fig,ax=plt.subplots(figsize=(8.3, 11.7/2), nrows=2, ncols=2)

        ax[0,0].scatter(x1, x3, marker='o', color='black', alpha=0.05)
        if a_garch is None:
            res = linregress(x1, x3)
            xx = np.array([x1.min(), x1.max()])
            ax[0,0].plot(xx, res.intercept + res.slope * xx, color="red")
            text = (f"$R^2$ = {res.rvalue**2:.3f}")
            ax[0,0].text(0.05, 0.95, text, transform=ax[0,0].transAxes, ha="left", va="top", fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=1))
        ax[0,0].set_ylabel(r'$\phi$')

        ax[1,0].scatter(x1, x2, marker='o', color='black', alpha=0.05)
        if a_garch is None:
            res = linregress(x1, x2)
            xx = np.array([x1.min(), x1.max()])
            ax[1,0].plot(xx, res.intercept + res.slope * xx, color="red")
            text = (f"$R^2$ = {res.rvalue**2:.3f}")
            ax[1,0].text(0.05, 0.95, text, transform=ax[1,0].transAxes, ha="left", va="top", fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=1))
        ax[1,0].set_xlabel(r'$\alpha_{GARCH}$')
        ax[1,0].set_ylabel(r'$\omega_{global}$')

        ax[1,1].scatter(x3, x2, marker='o', color='black', alpha=0.05)
        if a_garch is None:
            res = linregress(x3, x2)
            xx = np.array([x3.min(), x3.max()])
            ax[1,1].plot(xx, res.intercept + res.slope * xx, color="red")
            text = (f"$R^2$ = {res.rvalue**2:.3f}")
            ax[1,1].text(0.05, 0.95, text, transform=ax[1,1].transAxes, ha="left", va="top", fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=1))
        ax[1,1].set_xlabel(r'$\phi$')

        fig.delaxes(ax[0,1])

        plt.tight_layout()
        plt.savefig(os.path.join(cluster_output_folder,f'traces/pairplots/pairplot-a_garch-omega_global_mean-phi.pdf'))
        plt.close()
        

        # forestplot of alpha_inv
        fig,ax=plt.subplots(figsize=(8.3/3*2, 11.7))
        samples = trace.posterior['alpha_inv'].stack(sample=("chain", "draw"))
        # compute median and 50% & 95% HDI
        median = samples.median(dim="sample").values
        hdi = arviz.hdi(samples, prob=0.95, dim="sample")
        lower_95 = hdi.sel(ci_bound="lower").values
        upper_95 = hdi.sel(ci_bound="upper").values
        hdi = arviz.hdi(samples, prob=0.50, dim="sample")
        lower_75 = hdi.sel(ci_bound="lower").values
        upper_75 = hdi.sel(ci_bound="upper").values
        # labels
        states = samples["state"].values
        # y positions
        y = np.arange(len(states))
        # horizontal intervals
        ax.hlines(y, lower_75, upper_75, linewidth=3, color='forestgreen')
        ax.hlines(y, lower_95, upper_95, linewidth=1, color='forestgreen')
        # median points
        ax.plot(median, y, "o", color='black', markerfacecolor='white', markersize=3)
        # formatting
        ax.set_yticks(y)
        ax.set_yticklabels(states)
        ax.invert_yaxis()
        ax.set_title(r"$1/\alpha_i$ by U.S. state or territory", fontsize=12)
        # cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_output_folder,f'traces/forestplot-alpha_inv.pdf'))
        plt.close()


        # visualise forest plots of state and season effect sizes
        labels_params = [r'$\rho$', r'$f_I$', r'$f_R$', r'$\omega$']
        state_params = ["rho_state", "fI_state", "fR_state", "omega_state"]
        season_params = ["rho_season", "fI_season", "fR_season", "omega_season"]
        global_params = ["rho_global_mean", "fI_global_mean", "fR_global_mean", "omega_global_mean"]
        params = ['rho', 'fI', 'fR', 'omega']
        effect_type = ['Multiplicative', 'Multiplicative', 'Odds-ratio', 'Multiplicative']

        for n, p_state, p_season, g, p, e in zip(labels_params, state_params, season_params, global_params, params, effect_type):

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8.3, 11.7),
                                    gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [1, 1]})
            
            # ---- Top row: global effect, spanning both columns ----
            ax_global = axes[0, 0]
            ax_global2 = axes[0, 1]
            
            # hide the second subplot for spacing
            ax_global2.axis('off')
            
            global_samples = trace.posterior[g].stack(sample=("chain", "draw")).values
            ax_global.hist(global_samples, bins=15, density=True, color='forestgreen', alpha=0.8)
            ax_global.axvline(np.median(global_samples), color='black', linestyle='--', label='Median')
            ax_global.set_title(f"Global {n}", fontsize=14)
            ax_global.spines['left'].set_visible(False)
            ax_global.spines['right'].set_visible(False)
            ax_global.spines['top'].set_visible(False)
            ax_global.set_yticks([])
            ax_global.xaxis.set_major_locator(plt.MaxNLocator(3)) 

            # ---- Bottom row: state and season forest plots ----
            ## state
            samples = trace.posterior[p_state].stack(sample=("chain", "draw"))
            # compute median and 50% & 95% HDI
            median = samples.median(dim="sample").values
            hdi = arviz.hdi(samples, prob=0.95, dim="sample")
            lower_95 = hdi.sel(ci_bound="lower").values
            upper_95 = hdi.sel(ci_bound="upper").values
            hdi = arviz.hdi(samples, prob=0.50, dim="sample")
            lower_75 = hdi.sel(ci_bound="lower").values
            upper_75 = hdi.sel(ci_bound="upper").values
            # labels
            states = samples["state"].values
            # y positions
            y = np.arange(len(states))
            # horizontal intervals
            axes[1, 0].hlines(y, lower_75, upper_75, linewidth=3, color='forestgreen')
            axes[1, 0].hlines(y, lower_95, upper_95, linewidth=1, color='forestgreen')
            # median points
            axes[1, 0].plot(median, y, "o", color='black', markerfacecolor='white', markersize=3)
            # reference line
            axes[1, 0].axvline(1, color="black", linestyle="--")
            # formatting
            axes[1, 0].set_yticks(y)
            axes[1, 0].set_yticklabels(states)
            axes[1, 0].invert_yaxis()
            axes[1, 0].set_title(f"{e} state effects", fontsize=12)
            axes[1, 0].set_xlabel("Effect size")
            # cleanup
            axes[1, 0].spines['top'].set_visible(False)
            axes[1, 0].spines['right'].set_visible(False)

            ## season
            samples = trace.posterior[p_season].stack(sample=("chain", "draw"))
            # compute median and HDI
            median = samples.median(dim="sample").values
            hdi = arviz.hdi(samples, prob=0.95, dim="sample")
            lower_95 = hdi.sel(ci_bound="lower").values
            upper_95 = hdi.sel(ci_bound="upper").values
            hdi = arviz.hdi(samples, prob=0.50, dim="sample")
            lower_75 = hdi.sel(ci_bound="lower").values
            upper_75 = hdi.sel(ci_bound="upper").values
            # labels
            states = samples["season"].values
            # y positions
            y = np.arange(len(states))
            # horizontal intervals
            axes[1, 1].hlines(y, lower_75, upper_75, linewidth=3, color='forestgreen')
            axes[1, 1].hlines(y, lower_95, upper_95, linewidth=1, color='forestgreen')
            # median points
            axes[1, 1].plot(median, y, "o", color='black', markerfacecolor='white', markersize=3)
            # reference line
            axes[1, 1].axvline(1, color="black", linestyle="--")
            # formatting
            axes[1, 1].set_yticks(y)
            axes[1, 1].set_yticklabels(states)
            axes[1, 1].invert_yaxis()
            axes[1, 1].set_title(f"{e} season effects", fontsize=12)
            axes[1, 1].set_xlabel("Effect size")
            # cleanup
            axes[1, 1].spines['top'].set_visible(False)
            axes[1, 1].spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(cluster_output_folder,f'traces/forestplot-{p}.pdf'))
            plt.close()


        # Visualise across-season modifier trend + within-season median per state
        os.makedirs(os.path.join(cluster_output_folder,'modifiers'), exist_ok=True)
        # make dates
        x = pd.date_range(start=datetime(2000,10,15), periods=n_modifiers, freq='W')
        for s in range(n_states):
            fig,ax=plt.subplots(figsize=(8.3, 11.7/5))
            # average trend
            ax.plot(x, 1+trace.posterior['delta_beta_state_mean'].median(dim=['chain', 'draw']).values[:,s], color='green')
            ax.fill_between(x,
                            1+trace.posterior['delta_beta_state_mean'].quantile(dim=['chain', 'draw'], q=0.025).values[:,s],
                            1+trace.posterior['delta_beta_state_mean'].quantile(dim=['chain', 'draw'], q=0.975).values[:,s],
                            color='green', alpha=0.15)
            # individual seasons
            for i in range(n_seasons):
                ax.plot(x, 1+trace.posterior['delta_beta'].median(dim=['chain', 'draw']).values[:,i,s], color='black', alpha=0.3, linewidth=0.5)
            ax.axhline(y=1, color='red', linewidth=0.5)
            # decorations
            fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
            ax.set_ylabel(r'$\Delta \beta_t$')
            ax.set_ylim([0.65, 1.35])
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            plt.savefig(os.path.join(cluster_output_folder,f'modifiers/modifiers_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
            plt.close()


        # Visualise goodness-of-fit, delta_beta, z, sigma2 and eps per state and per season
        for s in range(n_states):
            os.makedirs(os.path.join(cluster_output_folder,f'goodness-fit/{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}/'), exist_ok=True)
            for i, season in enumerate(seasons):
                
                fig,ax=plt.subplots(nrows=5, figsize=(8.3, 11.7), sharex=True)
                # observed versus modeled
                ax[0].plot(dt[i, :], posterior_predictive.posterior_predictive['data'].median(dim=['chain', 'draw']).values[i,s,:], linewidth=1, color='green')
                ax[0].fill_between(dt[i, :],
                                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.025).values[i,s,:],
                                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.975).values[i,s,:],
                                color='green', alpha=0.1)
                ax[0].fill_between(dt[i, :],
                                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.25).values[i,s,:],
                                posterior_predictive.posterior_predictive['data'].quantile(dim=['chain', 'draw'], q=0.75).values[i,s,:],
                                color='green', alpha=0.2)
                ax[0].scatter(dt[i, :], posterior_predictive.observed_data['data'].values[i,s,:], marker='o', color='black')

                # across-season delta_beta trend
                yr = dt[i, 0].astype(object).year
                modifier_dates = pd.date_range(start=datetime(yr, modifier_ref_month, modifier_ref_day), periods=n_modifiers, freq=timedelta(weeks=1))
                ax[1].plot(modifier_dates, trace.posterior['delta_beta_state_mean'].median(dim=['chain', 'draw']).values[:,s], color='green')
                ax[1].fill_between(modifier_dates,
                                trace.posterior['delta_beta_state_mean'].quantile(dim=['chain', 'draw'], q=0.025).values[:,s],
                                trace.posterior['delta_beta_state_mean'].quantile(dim=['chain', 'draw'], q=0.975).values[:,s],
                                color='green', alpha=0.15)
                
                # within-season delta_beta, z, sigma2, eps
                for j, par in enumerate(['delta_beta', 'z', 'sigma2', 'eps']):
                    ax[j+1].plot(modifier_dates, trace.posterior[par].median(dim=['chain', 'draw']).values[:,i,s], color='black', linewidth=0.5)
                    ax[j+1].fill_between(modifier_dates,
                            trace.posterior[par].quantile(dim=['chain', 'draw'], q=0.025).values[:,i,s],
                            trace.posterior[par].quantile(dim=['chain', 'draw'], q=0.975).values[:,i,s],
                            color='black', alpha=0.15)
                    ax[j+1].set_ylabel(par)
                ax[0].set_title(season)
                plt.savefig(os.path.join(cluster_output_folder,f'goodness-fit/{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}/{season}_goodness-fit.pdf'))
                plt.close()


        # Save hyperdistributions
        # ~~~~~~~~~~~~~~~~~~~~~~~

        # save the hyperdistributions
        med = trace.posterior.median(dim=("chain", "draw")) # take median across chains and draws
        df = pd.DataFrame(index=state_fips_index['abbreviation_state'].values)

        # scalar parameters (repeat per state)
        scalar_params = [
            "rho_global_mean",
            "rho_season_sd",
            "fI_global_mean",
            "fI_season_sd",
            "fR_global_mean",
            "fR_season_sd",
            "psi_1",    
            "psi_2",
            "phi",
            "omega_global_mean",
            "omega_season_sd",
            "a_garch",
            "b_garch"
        ]
        for p in scalar_params:
            df[p] = float(med[p].values)

        # state parameters
        state_params = [
            "alpha_inv",
            "rho_state",
            "fI_state",
            "fR_state",
            "omega_state",
        ]
        for p in state_params:
            df[p] = med[p].values


        # delta_beta_state_mean (modifier x state)
        delta = med["delta_beta_state_mean"].values
        n_modifiers = delta.shape[0]
        for i in range(n_modifiers):
            df[f"delta_beta_state_mean_{i}"] = delta[i, :]

        # save to csv
        df.index.name = "state"
        df.to_csv(os.path.join(cluster_output_folder,f"hyperparameters-{training_name}_cluster-{cluster_idx}.csv"))

        # append to output list
        hyperparameters.append(df)

        print(f'\ntraining of cluster {cluster_idx} complete!\n')

    print(f'\nmerging hyperparameters of all clusters\n')

    # concatenate all hyperparameters and save them
    output = pd.concat(hyperparameters, axis=0)
    output.to_csv(os.path.join(output_folder,f"hyperparameters-{training_name}.csv"))

    print(f'\ntraining complete!\n')

# runs the script
run_training()