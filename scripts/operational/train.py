"""
This script trains the model on historical data.

Author: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

# standard python libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
# pyMC / pytensor
import pymc as pm
import arviz
import pytensor
import pytensor.tensor as pt
#pytensor.config.cxx = '/usr/bin/clang++'
#pytensor.config.on_opt_error = "ignore"
# jax and diffrax
import jax.numpy as jnp
# model package
from SCARCHhierarchSIR.data import get_demography, get_adjacency_matrix, get_NHSN_HRD_data
from SCARCHhierarchSIR.SIR_model import get_jax_jitted_model, make_sol_op
from SCARCHhierarchSIR.pymc_model import AR_GARCH_step, compute_season_weights, weighted_nb_logp, weighted_nb_random
from SCARCHhierarchSIR.preoptimization import preoptimize_parameters, compute_initial_effects


# all paths defined relative to this file
abs_dir = os.path.dirname(__file__)

# global parameters go here
## model-structural
use_garch = True
gamma = 1/3.5
n_modifiers = 26
modifier_length = 7
start_simulation = -15 # (October 1)
## temporal extent of training
n_observations = 30
start_calibration_month = 10
seasons = ['2023-2024', '2024-2025', '2025-2026']
## sampling effort
n_chains = 1
n_sample = 2
n_burn = 0
training_name = 'exclude_None-wGARCH'
n_preoptim = 500

# derived products
## convert to a list of start and enddates (datetime)
n_seasons = len(seasons)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
modifier_reference_dates = [datetime(int(season[0:4]), 10, 15) for season in seasons]
## misc
assert n_sample > n_burn, 'number of burned samples cannot exceed total number of samples'


# Get the clusters
# ~~~~~~~~~~~~~~~~

clusters = pd.read_csv(os.path.join(abs_dir, "../../data/interim/geography/clusters.csv"))
cluster_indices = sorted(clusters['cluster_idx'].unique())

# Loop over the clusters
# ~~~~~~~~~~~~~~~~~~~~~~

hyperparameters = []
for cluster_idx in cluster_indices:

    print(f'\nworking on cluster {cluster_idx}')
    print('~~~~~~~~~~~~~~~~~~~~\n')

    print(f'states in cluster: {clusters[clusters['cluster_idx'] == cluster_idx]['abbreviation_state'].values.tolist()}\n')

    output_folder = os.path.join(abs_dir, f'../../data/interim/calibration/training/{training_name}/cluster_{cluster_idx}')

    # Get US demographics
    # ~~~~~~~~~~~~~~~~~~~

    state_fips_index, demo = get_demography(clusters[clusters['cluster_idx'] == cluster_idx]['abbreviation_state'])
    n_states = len(demo)

    # Get state adjacency matrix
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    adj = get_adjacency_matrix(state_fips_index['abbreviation_state'])

    # Get US incidences
    # ~~~~~~~~~~~~~~~~~

    reference_date, data, dt, ts, n_observations = get_NHSN_HRD_data(start_calibrations, modifier_reference_dates, n_observations, forecast_horizon=None, state_fips=state_fips_index['fips_state'].values) # (n_season, n_variables, n_observations)
    data = data / 7 # divide weekly incidence by 7

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
        os.makedirs(os.path.join(output_folder, 'initial-optim'), exist_ok=True)
        plt.savefig(os.path.join(output_folder,f'initial-optim/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
        plt.close(fig)

    # compute pyMC initial effect sizes
    init = compute_initial_effects(args_diff_preoptim)

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

    # Build tempored NB distribution
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    weights = compute_season_weights(data)

    # Build pyMC model
    # ~~~~~~~~~~~~~~~~

    print('\ncompiling pymc model')

    # construct coordinates
    coords = {
        "state": state_fips_index['abbreviation_state'].values,
        "season": seasons,
        "modifier": np.arange(n_modifiers)
    }

    # Build pyMC probablistic model
    with pm.Model(coords=coords) as model:

        # Hyperparameters '<parameter>_<level>_<type>' with level: {global, state, season} and type: {mean, sd, offset}

        ## transmission coefficient: beta (fixed)
        beta = pt.as_tensor_variable(0.455*np.ones(shape=(n_seasons,n_states)))

        ## ascertainment: rho
        ### global
        log_rho_global_mean = pm.Normal("log_rho_global_mean", mu=init["log_rho"]["global"], sigma=1/3)    
        rho_global_mean = pm.Deterministic("rho_global_mean", pt.exp(log_rho_global_mean))
        ### state
        rho_state_sd = pm.HalfNormal("rho_state_sd", sigma=1/5)      
        rho_state_raw = pm.Normal("rho_state_raw", 0, 1, dims="state")
        rho_state = pm.Deterministic("rho_state", pt.exp(rho_state_sd * rho_state_raw), dims="state")
        ### season
        rho_season_sd = pm.HalfNormal("rho_season_sd", sigma=1/5)
        rho_season_raw = pm.Normal("rho_season_raw", 0, 1, dims="season")
        rho_season = pm.Deterministic("rho_season", pt.exp(rho_season_sd * rho_season_raw), dims="season")
        log_rho = log_rho_global_mean + rho_state_sd * rho_state_raw[None, :] + rho_season_sd * rho_season_raw[:, None]
        rho = pm.Deterministic("rho", pt.exp(log_rho))

        ## initial infected: fI
        ### global
        log_fI_global_mean = pm.Normal("log_fI_global_mean", mu=init["log_fI"]["global"], sigma=1/3)    
        fI_global_mean = pm.Deterministic("fI_global_mean", pt.exp(log_fI_global_mean))
        ### state
        fI_state_sd = pm.HalfNormal("fI_state_sd", sigma=1/5)      
        fI_state_raw = pm.Normal("fI_state_raw", 0, 1, dims="state")
        fI_state = pm.Deterministic("fI_state", pt.exp(fI_state_sd * fI_state_raw), dims="state")
        ### season
        fI_season_sd = pm.HalfNormal("fI_season_sd", sigma=1/5)
        fI_season_raw = pm.Normal("fI_season_raw", 0, 1, dims="season")
        fI_season = pm.Deterministic("fI_season", pt.exp(fI_season_sd * fI_season_raw), dims="season")
        log_fI = log_fI_global_mean + fI_state_sd * fI_state_raw[None, :] + fI_season_sd * fI_season_raw[:, None]
        fI = pm.Deterministic("fI", pt.exp(log_fI))

        ## initial recovered: fR
        ### global
        logit_fR_global_mean = pm.Normal("logit_fR_global_mean", mu=pm.math.logit(0.4), sigma=1.0)
        fR_global_mean = pm.Deterministic("fR_global_mean", pm.math.sigmoid(logit_fR_global_mean))
        ### state
        fR_state_sd = pm.HalfNormal("fR_state_sd", sigma=1/5)
        fR_state_raw = pm.Normal("fR_state_raw", 0, 1, dims="state")
        fR_state = pm.Deterministic("fR_state", pt.exp(fR_state_sd * fR_state_raw), dims="state")
        ### season
        fR_season_sd = pm.HalfNormal("fR_season_sd", sigma=1/5)
        fR_season_raw = pm.Normal("fR_season_raw", 0, 1, dims="season")
        fR_season = pm.Deterministic("fR_season", pt.exp(fR_season_sd * fR_season_raw), dims="season")
        logit_fR = logit_fR_global_mean + fR_state_sd * fR_state_raw[None, :] + fR_season_sd * fR_season_raw[:, None]
        fR = pm.Deterministic("fR", pm.math.sigmoid(logit_fR))

        # ------- AR-GARCH modifiers -----------

        # Spatial correlation
        psi_1 = pm.Beta("psi_1", 3, 3)
        psi_2 = pm.Beta("psi_2", 3, 1)

        I = pt.eye(n_states)
        W = pt.as_tensor_variable(adj)
        D = pt.diag(pt.sum(W, axis=1))

        Q_modifiers = (1 - psi_1) * I + psi_1 * (D - W)
        L_Q_modifiers = pt.slinalg.cholesky(Q_modifiers)
        L_cov_modifiers = pt.slinalg.solve(L_Q_modifiers, I)
        Q_shocks = (1 - psi_2) * I + psi_2 * (D - W)
        L_Q_shocks = pt.slinalg.cholesky(Q_shocks)
        L_cov_shocks = pt.slinalg.solve(L_Q_shocks, I)
            
        # Hyperparameter for delta_beta_temporal
        delta_beta_raw = pm.Normal("delta_beta_raw", 0, 1, dims=("modifier","state"))
        delta_beta_state_mean = pm.Deterministic("delta_beta_state_mean", (1/4) * pt.einsum("ij,mj->mi", L_cov_modifiers, delta_beta_raw), dims=("modifier","state"))

        # --- AR(1) kernel ---
        # Initial position
        z_0 = pt.zeros([n_seasons, n_states])
        eps_0 = pt.zeros([n_seasons, n_states])
        # Total AR persistence
        ## global
        logit_phi_global_mean = pm.Normal("logit_phi_global_mean", mu=0, sigma=1)
        phi_global_mean = pm.Deterministic("phi_global_mean", pm.math.sigmoid(logit_phi_global_mean))
        ## state
        phi_state_sd = pm.HalfNormal("phi_state_sd", sigma=1/5)
        phi_state_raw = pm.Normal("phi_state_raw", 0, 1, dims="state")
        phi_state = pm.Deterministic("phi_state", pt.exp(phi_state_sd * phi_state_raw), dims="state")
        ### season
        phi_season_sd = pm.HalfNormal("phi_season_sd", sigma=1/5)
        phi_season_raw = pm.Normal("phi_season_raw", 0, 1, dims="season")
        phi_season = pm.Deterministic("phi_season", pt.exp(phi_season_sd * phi_season_raw), dims="season")
        phi = pm.Deterministic("phi", pm.math.sigmoid(logit_phi_global_mean + phi_state_sd * phi_state_raw[None, :] + phi_season_sd * phi_season_raw[:, None]))

        # sample iid standard normals as shocks
        eta_raw = pm.Normal("eta_raw", mu=0.0, sigma=1.0, dims=("modifier","season","state"))
        # correlate them across space using the precision matrix
        eta = pm.Deterministic("eta", pt.einsum("ij,tsj->tsi", L_cov_shocks, eta_raw))

        # --- GARCH(1,1) parameters ---                                                                             
        # Total noise persistence
        ## global
        logit_kappa_global_mean = pm.Normal("logit_kappa_global_mean", mu=0, sigma=1)
        kappa_global_mean = pm.Deterministic("kappa_global_mean", pm.math.sigmoid(logit_kappa_global_mean))
        ## state
        kappa_state_sd = pm.HalfNormal("kappa_state_sd", sigma=1/5)
        kappa_state_raw = pm.Normal("kappa_state_raw", 0, 1, dims="state")
        kappa_state = pm.Deterministic("kappa_state", pt.exp(kappa_state_sd * kappa_state_raw), dims="state")
        ### season
        kappa_season_sd = pm.HalfNormal("kappa_season_sd", sigma=1/5)
        kappa_season_raw = pm.Normal("kappa_season_raw", 0, 1, dims="season")
        kappa_season = pm.Deterministic("kappa_season", pt.exp(kappa_season_sd * kappa_season_raw), dims="season")
        kappa = pm.Deterministic("kappa", pm.math.sigmoid(logit_kappa_global_mean + kappa_state_sd * kappa_state_raw[None, :] + kappa_season_sd * kappa_season_raw[:, None]))
        ## Split between a and b                                                   
        nu = pm.Beta("nu", 5, 1)                                                                  
        a_garch = pm.Deterministic("a_garch", kappa * nu)                                                          
        b_garch = pm.Deterministic("b_garch", kappa * (1 - nu))
        # Steady state noise
        omega = pm.LogNormal("omega", mu=pt.log(0.01/3), sigma=1/5)    
        # Initial noise                         
        sigma2_0 = pm.Deterministic("sigma2_0", omega/(1-kappa), dims=("season","state"))

        # Run AR-GARCH scan over T steps
        z_seq, sigma2_seq, eps_seq = pytensor.scan(
            fn=AR_GARCH_step,
            sequences=[eta,],
            outputs_info=[z_0, sigma2_0, eps_0],
            non_sequences=[phi, omega, a_garch, b_garch, pt.as_tensor_variable(1 if use_garch else 0)],
            return_updates=False
        )

        # Register deterministic variables to inspect later
        delta_beta = pm.Deterministic("delta_beta", z_seq + delta_beta_state_mean[:, None, :])
        z = pm.Deterministic("z", z_seq)
        sigma2 = pm.Deterministic("sigma2", sigma2_seq)
        eps = pm.Deterministic("eps", eps_seq)

        # concatenate parameters along the last axis
        args_diff = pt.concatenate(
            [beta[:, :, None], rho[:, :, None], fI[:, :, None], fR[:, :, None], pt.transpose(delta_beta, (1, 2, 0))],
            axis=2
        )

        # Run forward simulation model
        ys = 7*sol_op(args_diff, args_nodiff)
        ys = pt.math.softplus(ys)

        # Compute likelihood
        alpha_inv = pm.LogNormal("alpha_inv", mu=pt.log(0.0025), sigma=1/5, dims="state")
        pm.CustomDist("data", ys, 1/alpha_inv, weights, logp=weighted_nb_logp, random=weighted_nb_random, observed=7*data)

    # Sample pyMC model
    # ~~~~~~~~~~~~~~~~~

    print('\nstarting the sampler..\n')

    with model:
        # set step size directly
        step = pm.NUTS(step_scale=0.005, target_accept=0.8, max_treedepth=10)   # for US: step_scale: 0.002 + max_treedepth 12, For NE+MA: step_scale: 0.005 + max_treedepth 10
        # run sampler without tuning
        trace = pm.sample(n_sample, tune=0, chains=n_chains, init='adapt_diag', cores=1, progressbar=True, step = step,
                            initvals=n_chains*[{'alpha_inv': 0.05 * pt.ones(n_states), 'delta_beta_raw': init["delta_beta_mu"] / 0.25,
                                    'log_rho_global_mean': init["log_rho"]["global"], 'rho_state_sd': 0.2, 'rho_state_raw': init["log_rho"]["state"] / 0.2, 'rho_season_sd': 0.2, 'rho_season_raw': init["log_rho"]["season"] / 0.2,
                                    'log_fI_global_mean': init["log_fI"]["global"], 'fI_state_sd': 0.2, 'fI_state_raw': init["log_fI"]["state"] / 0.2, 'fI_season_sd': 0.2, 'fI_season_raw': init["log_fI"]["season"] / 0.2,
                                    'logit_fR_global_mean': init["logit_fR"]["global"], 'fR_state_sd': 0.2, 'fR_state_raw': init["logit_fR"]["state"] / 0.2, 'fR_season_sd': 0.2, 'fR_season_raw': init["logit_fR"]["season"] / 0.2,
                                    'logit_phi_global_mean': 0.75, 'logit_kappa_global_mean': 0.75}])
    
    print('\n..finished sampling\n')
    print('\ngenerating diagnostics\n')

    # manual burn
    trace = trace.isel(draw=slice(n_burn, None))

    # Generate traces
    variables2plot = [
                    'alpha_inv',                                                                            # overdispersion
                    'rho_global_mean', 'rho_state_sd', 'rho_state', 'rho_season_sd', 'rho_season', 'rho',   # rho
                    'fI_global_mean', 'fI_state_sd', 'fI_state', 'fI_season_sd', 'fI_season', 'fI',         # fI
                    'fR_global_mean', 'fR_state_sd', 'fR_state', 'fR_season_sd', 'fR_season', 'fR',         # fR
                    'delta_beta_state_mean',                                                                # delta_beta_mu
                    'psi_2', 'psi_1',                                                                       # spatial correlation strength
                    'phi_global_mean', 'phi_state_sd', 'phi_season_sd', 'phi',                              # AR 
                    'kappa_global_mean', 'kappa_state_sd', 'kappa_season_sd', 'kappa', 'omega', 'nu',      # GARCH parameters
                    'a_garch', 'b_garch', 'sigma2_0',
                    ]

    # Save original traces
    os.makedirs(os.path.join(output_folder,'traces'), exist_ok=True)
    for var in variables2plot:
        arviz.plot_trace(trace, var_names=[var]) 
        plt.savefig(os.path.join(output_folder,f'traces/trace-{var}.pdf'))
        plt.close()

    # Build pair plots
    arviz.plot_pair(trace, var_names=["kappa", "nu", "omega", "phi"], divergences=True)
    plt.savefig(os.path.join(output_folder,'traces/pairplot-ARGARCH.pdf'))
    plt.close()


    # Make posterior predictive
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # Predict
    with model:
        posterior_predictive = pm.sample_posterior_predictive(trace)

    # Save traces and posterior predictive
    arviz.to_netcdf(trace, os.path.join(output_folder,"trace.nc"))
    arviz.to_netcdf(posterior_predictive, os.path.join(output_folder,"posterior_predictive.nc"))

    # Visualisations
    # ~~~~~~~~~~~~~~

    # Visualise across-season modifier trend + within-season median per state
    os.makedirs(os.path.join(output_folder,'modifiers'), exist_ok=True)
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
        plt.savefig(os.path.join(output_folder,f'modifiers/modifiers_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
        plt.close()


    # Visualise goodness-of-fit, delta_beta, z, sigma2 and eps per state and per season
    for s in range(n_states):
        os.makedirs(os.path.join(output_folder,f'goodness-fit/{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}/'), exist_ok=True)
        for i, season in enumerate(seasons):
            
            fig,ax=plt.subplots(nrows=5, figsize=(8.3, 11.7))
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
            ax[1].plot(range(n_modifiers), trace.posterior['delta_beta_state_mean'].median(dim=['chain', 'draw']).values[:,s], color='green')
            ax[1].fill_between(range(n_modifiers),
                            trace.posterior['delta_beta_state_mean'].quantile(dim=['chain', 'draw'], q=0.025).values[:,s],
                            trace.posterior['delta_beta_state_mean'].quantile(dim=['chain', 'draw'], q=0.975).values[:,s],
                            color='green', alpha=0.15)
            
            # within-season delta_beta, z, sigma2, eps
            for j, par in enumerate(['delta_beta', 'z', 'sigma2', 'eps']):
                ax[j+1].plot(range(n_modifiers), trace.posterior[par].median(dim=['chain', 'draw']).values[:,i,s], color='black', linewidth=0.5)
                ax[j+1].fill_between(range(n_modifiers),
                        trace.posterior[par].quantile(dim=['chain', 'draw'], q=0.025).values[:,i,s],
                        trace.posterior[par].quantile(dim=['chain', 'draw'], q=0.975).values[:,i,s],
                        color='black', alpha=0.15)
                ax[j+1].set_ylabel(par)
            ax[0].set_title(season)
            plt.savefig(os.path.join(output_folder,f'goodness-fit/{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}/{season}_goodness-fit.pdf'))
            plt.close()


    # visualise forest plots of state and season effect sizes
    labels_params = [r'$\rho$', r'$f_I$', r'$f_R$', r'$\phi$', r'$\kappa$']
    state_params = ["rho_state", "fI_state", "fR_state", "phi_state", "kappa_state"]
    season_params = ["rho_season", "fI_season", "fR_season", "phi_season", "kappa_season"]
    global_params = ["rho_global_mean", "fI_global_mean", "fR_global_mean", "phi_global_mean", "kappa_global_mean"]
    params = ['rho', 'fI', 'fR', 'phi', 'kappa']
    effect_type = ['Multiplicative', 'Multiplicative', 'Odds-ratio', 'Odds-ratio', 'Odds-ratio']

    for n, p_state, p_season, g, p, e in zip(labels_params, state_params, season_params, global_params, params, effect_type):
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.7, 8.3),
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
        arviz.plot_forest(trace, var_names=[p_state], combined=True, hdi_prob=0.95, ax=axes[1, 0], colors='forestgreen')
        axes[1, 0].axvline(1, color="black", linestyle="--")
        axes[1, 0].set_title(f"{e} state effects", fontsize=12)
        ## season
        arviz.plot_forest(trace, var_names=[p_season], combined=True, hdi_prob=0.95, ax=axes[1, 1], colors='forestgreen')
        axes[1, 1].axvline(1, color="black", linestyle="--")
        axes[1, 1].set_title(f"{e} season effects", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,f'traces/forestplot-{p}.pdf'))
        plt.close()


    # Save hyperdistributions
    # ~~~~~~~~~~~~~~~~~~~~~~~

    # save the hyperdistributions
    med = trace.posterior.median(dim=("chain", "draw")) # take median across chains and draws
    df = pd.DataFrame(index=model.coords["state"])

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
        "phi_global_mean",
        "phi_season_sd",
        "kappa_global_mean",
        "kappa_season_sd",
        "omega",
        "nu",
    ]
    for p in scalar_params:
        df[p] = float(med[p].values)

    # state parameters
    state_params = [
        "alpha_inv",
        "rho_state",
        "fI_state",
        "fR_state",
        "phi_state",
        "kappa_state",
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
    df.to_csv(os.path.join(output_folder,f"hyperparameters-{training_name}_cluster-{cluster_idx}.csv"))

    # append to output list
    hyperparameters.append(df)

    print(f'\ntraining of cluster {cluster_idx} complete!\n')


# concatenate all hyperparameters and save them
output = pd.concat(hyperparameters, axis=0)
output.to_csv(os.path.join(output_folder,'..',f"hyperparameters-{training_name}.csv"))

print(f'\ntraining complete!')