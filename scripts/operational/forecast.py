"""
This script makes a forecast for unseen data.

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
import xarray as xr
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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
from SCARCHhierarSIR.data import get_demography, get_adjacency_matrix, get_NHSN_HRD_data, simout_to_hubverse_admissions, simout_to_hubverse_peak_admissions, simout_to_hubverse_peak_timing
from SCARCHhierarSIR.SIR_model import get_jax_jitted_model, make_sol_op
from SCARCHhierarSIR.pymc_model import AR_GARCH_step, compute_season_weights, weighted_nb_logp, weighted_nb_random
from SCARCHhierarSIR.preoptimization import preoptimize_parameters

# needed to use the 'spawn' multiprocessing context manager
def run_forecast():

    # all paths defined relative to this file
    abs_dir = os.path.dirname(__file__)

    # global parameters go here
    ## training metadata
    training_name = 'exclude_None'
    training_folder = os.path.join(abs_dir, f'../../data/interim/calibration/hierarchical-training/{training_name}')
    ## forecasting settings
    challenge_start_reference_date = datetime(2025, 10, 18) # must be a saturday
    challenge_end_reference_date = datetime(2026, 5, 30)    # must be the last saturday of may
    seasons = ['2025-2026',]        # script only works with one season
    n_observations = 4              # use all data available in the forecast season
    forecast_horizon = 20           # forecast 4 weeks ahead
    n_preoptim = 500
    n_sample = 75
    n_tune = 25
    n_chains = 4
    sigma_grw = 0.01

    ## load the model-structural parameters and training metadata
    with open(os.path.join(training_folder, "model_config.json"), "r") as f:
        params = json.load(f)

    b_garch = params["b_garch"]
    gamma = params["gamma"]
    n_modifiers = params["n_modifiers"]
    modifier_length = params["modifier_length"]
    start_simulation = params["start_simulation"]
    modifier_ref_month = params["modifier_ref_month"]
    modifier_ref_day = params["modifier_ref_day"]
    clustering_name = params["clustering_name"]

    # derived products
    ## convert to a list of start and enddates (datetime)
    n_seasons = len(seasons)
    start_calibrations = [datetime(int(season[0:4]), modifier_ref_month, modifier_ref_day) + timedelta(days=start_simulation) for season in seasons]    # calibrations started at same time as simulation
    modifier_reference_dates = [datetime(int(season[0:4]), modifier_ref_month, modifier_ref_day) for season in seasons]
    model_name = 'SCARCHhierarSIR'

    # Get the clusters
    # ~~~~~~~~~~~~~~~~

    clusters = pd.read_csv(os.path.join(abs_dir, "../../data/interim/geography/clusters.csv"))
    cluster_indices = sorted(clusters[clustering_name].unique())

    # Loop over the clusters
    # ~~~~~~~~~~~~~~~~~~~~~~

    forecasts = []
    for cluster_idx in cluster_indices:

        print(f'\nworking on cluster {cluster_idx}')
        print('~~~~~~~~~~~~~~~~~~~~\n')

        print(f'states in cluster: {clusters[clusters[clustering_name] == cluster_idx]['abbreviation_state'].values.tolist()}\n')


        # Get US demographics
        # ~~~~~~~~~~~~~~~~~~~

        state_fips_index, demo = get_demography(clusters[clusters[clustering_name] == cluster_idx]['abbreviation_state'])
        n_states = len(demo)

        # Get state adjacency matrix
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        adj = get_adjacency_matrix(state_fips_index['abbreviation_state'])

        # Get US incidence data
        # ~~~~~~~~~~~~~~~~~~~~~

        # get data
        reference_date, data, dt, ts, n_observations = get_NHSN_HRD_data(start_calibrations, modifier_reference_dates, n_observations,
                                                                        type = 'preliminary',
                                                                        forecast_horizon=forecast_horizon,
                                                                        state_fips=state_fips_index['fips_state'].values) # (n_season, n_variables, n_observations)
        data = data / 7                                     # divide weekly incidence by 7
        reference_date = dt[-1][-1] + timedelta(weeks=1) - timedelta(weeks=forecast_horizon)    # compute true reference date based on data instead of filename

        # output folder name
        output_folder = os.path.join(abs_dir, f'../../data/interim/calibration/forecast/{training_name}/reference_date-{reference_date.strftime('%Y-%m-%d')}/cluster_{cluster_idx}/')

        # Get the hyperparameters
        # ~~~~~~~~~~~~~~~~~~~~~~~

        # get
        hyperpars = pd.read_csv(os.path.join(training_folder, f'hyperparameters-{training_name}.csv'))

        # slice states
        hyperpars = hyperpars[hyperpars['state'].isin(state_fips_index['abbreviation_state'])]

        # unpack
        ## (global) scalar
        rho_global_mean         = hyperpars['rho_global_mean'].unique()[0]
        rho_season_sd           = hyperpars['rho_season_sd'].unique()[0]
        fI_global_mean          = hyperpars['fI_global_mean'].unique()[0]
        fI_season_sd            = hyperpars['fI_season_sd'].unique()[0]
        fR_global_mean          = hyperpars['fR_global_mean'].unique()[0]
        fR_season_sd            = hyperpars['fR_season_sd'].unique()[0]
        psi_2                   = hyperpars['psi_2'].unique()[0]
        phi_global_mean         = hyperpars['phi_global_mean'].unique()[0]
        phi_season_sd           = hyperpars['phi_season_sd'].unique()[0]
        omega_global_mean       = hyperpars['omega_global_mean'].unique()[0]
        omega_season_sd         = hyperpars['omega_season_sd'].unique()[0]
        a_garch                 = hyperpars['a_garch'].unique()[0]
        b_garch                 = hyperpars['b_garch'].unique()[0]
        ## (state) vectors
        alpha_inv               = hyperpars['alpha_inv'].values
        rho_state               = hyperpars['rho_state'].values
        fI_state                = hyperpars['fI_state'].values
        fR_state                = hyperpars['fR_state'].values
        phi_state               = hyperpars['phi_state'].values
        omega_state             = hyperpars['omega_state'].values

        ## hypermodifiers
        modifier_cols = [c for c in hyperpars.columns if c.startswith("delta_beta_state_mean_")]
        delta_beta_state_mean = np.transpose(hyperpars[modifier_cols].to_numpy())

        # Define a jax-jitted diffrax differential equation model
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        jitted_sol_op_multi, jitted_vjp_sol_op_multi = get_jax_jitted_model()

        # Pre-optimize the forward simulation model's parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print('pre-optimization\n')
        print('(iter, score)')

        # static arguments
        args_static = (start_simulation, float(max(ts[:,:n_observations][:,-1])), modifier_length)

        # stack args_nodiff so two leading axes are seasons, states and the third axes gives the arguments for the season-state combination
        gamma_vec = jnp.full((n_seasons, n_states, 1), gamma)
        pop_mat = jnp.broadcast_to(jnp.asarray(demo)[None, :, None], (n_seasons, n_states, 1))
        ts_mat = jnp.broadcast_to(ts[:, None, :n_observations], (n_seasons, n_states, ts[:,:n_observations].shape[1]))
        args_nodiff = np.array(jnp.concatenate([gamma_vec, pop_mat, ts_mat], axis=2))     # shape: (n_seasons, n_states, )  --> convert to numpy otherwise error in pt.as_tensor_variable(args_nodiff) in make_node of pyMC model

        # pre-optimize the initial guesses
        args_diff_preoptim = preoptimize_parameters(
            jitted_sol_op=jitted_sol_op_multi,
            args_static=args_static,
            args_nodiff=args_nodiff,
            data=data[:,:,:n_observations],
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

        # inspect result
        for s in range(n_states):
            fig, ax = plt.subplots(nrows=1, figsize=(8.7, 11.3/4))
            for i in range(n_seasons):
                ax.plot(dt[i, :n_observations], 7*out[i, s, :], color='red', label='pred')
                ax.scatter(dt[i, :n_observations], 7*data[i, s, :n_observations], marker='o', color='black', label='obs')
            fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
            fig.tight_layout()
            os.makedirs(os.path.join(output_folder, 'initial-optim'), exist_ok=True)
            plt.savefig(os.path.join(output_folder,f'initial-optim/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
            plt.close(fig)

        # Define the Op and VJPOp classes for the ODE problem
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # reshape args_static and args_nodiff to simulate SIR model until data end + forecast horizon
        args_static = (start_simulation, max(ts[:,-1]), modifier_length)
        ts_mat = jnp.broadcast_to(ts[:, None, :], (n_seasons, n_states, ts.shape[1]))
        args_nodiff = np.array(jnp.concatenate([gamma_vec, pop_mat, ts_mat], axis=2))

        # generate the pyMC probablistic node
        sol_op = make_sol_op(args_static, jitted_sol_op_multi, jitted_vjp_sol_op_multi)

        # Build tempored NB distribution
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        weights = compute_season_weights(data[:,:,:n_observations])

        # Build pyMC model
        # ~~~~~~~~~~~~~~~~

        print('\ncompiling pymc model')

        # construct coordinates
        coords = {
            "state": state_fips_index['abbreviation_state'].values,
            "season": seasons,
            "modifier": np.arange(n_modifiers),
            "horizon_forecast": np.arange(forecast_horizon),
            "horizon_observation": [-i for i in range(1, n_observations + 1)]
        }

        # Build pyMC probablistic model
        with pm.Model(coords=coords) as model:

            # Hyperparameters '<parameter>_<level>_<type>' with level: {global, state, season} and type: {mean, sd, offset}

            ## transmission coefficient: beta (fixed)
            beta = pt.as_tensor_variable(0.455*np.ones(shape=(n_seasons,n_states)))

            ## ascertainment: rho
            ### global (rho_global_mean)
            ### state (rho_state)
            ### season (rho_season_sd)
            rho_season_raw = pm.Normal("rho_season_raw", 0, 1, dims="season")
            rho = pm.Deterministic("rho", pt.exp(pt.log(rho_global_mean) + pt.log(rho_state)[None, :] + rho_season_sd * rho_season_raw[:, None]))

            ## initial infected: fI
            ### global (fI_global_mean)
            ### state (fI_state)
            ### season (fI_season_sd)
            fI_season_raw = pm.Normal("fI_season_raw", 0, 1, dims="season")
            fI = pm.Deterministic("fI", pt.exp(pt.log(fI_global_mean) + pt.log(fI_state)[None, :] + fI_season_sd * fI_season_raw[:, None]))

            ## initial recovered: fR
            ### global (fR_global_mean)
            ### state (fR_state)
            ### season (fR_season_sd)
            fR_season_raw = pm.Normal("fR_season_raw", 0, 1, dims="season")
            fR = pm.Deterministic("fR", pm.math.sigmoid(pm.math.logit(fR_global_mean) + pt.log(fR_state)[None, :] + fR_season_sd * fR_season_raw[:, None]))

            # ------- AR-GARCH modifiers -----------

            # Spatial correlation ('psi_2' hyperparameter)
            I = pt.eye(n_states)
            W = pt.as_tensor_variable(adj)
            D = pt.diag(pt.sum(W, axis=1))
            Q_shocks = (1 - psi_2) * I + psi_2 * (D - W)
            L_Q_shocks = pt.linalg.cholesky(Q_shocks)
            L_cov_shocks = pt.linalg.solve(L_Q_shocks, pt.eye(n_states))

            # Hyperparameter for delta_beta_temporal (delta_beta_state_mean hyperparameter, shape: n_modifiers x n_states)

            # --- AR(1) kernel (season axis removed) ---
            # Initial position
            z_0 = pt.zeros([n_states,])
            eps_0 = pt.zeros([n_states,])
            # Total AR persistence
            ### global (phi_global_mean)
            ### state (phi_state)
            ### season (phi_season_sd)
            phi_season_raw = pm.Normal("phi_season_raw", 0, 1, dims="season")
            phi = pm.Deterministic("phi", pt.squeeze(pm.math.sigmoid(pm.math.logit(phi_global_mean) + pt.log(phi_state)[None, :] + phi_season_sd * phi_season_raw[:, None])[0,:]))
            # sample iid standard normals as shocks
            eta_raw = pm.Normal("eta_raw", mu=0.0, sigma=1.0, shape=(n_modifiers-1, n_states))
            # correlate them across space using the precision matrix 
            eta = pm.Deterministic("eta", pt.einsum("ij,mj->mi", L_cov_shocks, eta_raw))    # shape: (modifier x state)

            # --- GARCH(1,1) parameters ---                                                                             
            # Baseline noise
            ### global (omega_global_mean)
            ### state (omega_state)
            ### season (omega_season_sd)
            omega_season_raw = pm.Normal("omega_season_raw", 0, 1, dims="season")
            omega = pm.Deterministic("omega", pt.exp(pt.log(omega_global_mean) + pt.log(omega_state)[None, :] + omega_season_sd * omega_season_raw[:, None])[0,:])

            # Initial state + a_garch, b_garch                                                                                                         
            a_garch = pm.Deterministic("a_garch", pt.as_tensor_variable(a_garch))                                                        
            b_garch = pm.Deterministic("b_garch", pt.as_tensor_variable(b_garch))                  
            sigma2_0 = pm.Deterministic("sigma2_0", omega, dims="state")

            # Run AR-GARCH scan over T steps
            z_seq, sigma2_seq, eps_seq = pytensor.scan(
                fn=AR_GARCH_step,
                sequences=[eta,],
                outputs_info=[z_0, sigma2_0, eps_0],
                non_sequences=[phi, omega, a_garch, b_garch],
                return_updates=False
            )

            # Register deterministic variables to inspect later
            z = pm.Deterministic("z", pt.concatenate([z_0[None, ...], z_seq], axis=0))  # prepend initial condition
            sigma2 = pm.Deterministic("sigma2", pt.concatenate([sigma2_0[None, ...], sigma2_seq], axis=0))
            eps = pm.Deterministic("eps", pt.concatenate([eps_0[None, ...], eps_seq], axis=0))
            delta_beta = pm.Deterministic("delta_beta", z + delta_beta_state_mean)

            # concatenate parameters along last axis (n_seasons, n_states, n_parameters)
            args_diff = pt.concatenate(
                [beta[:, :, None], rho[:, :, None], fI[:, :, None], fR[:, :, None], pt.transpose(delta_beta, (1, 0))[None, :, :]],
                axis=2
            )

            # Run forward simulation model
            ys = 7*sol_op(args_diff, args_nodiff)
            ys = pt.math.softplus(ys)

            # Compute likelihood (alpha_inv hyperparameter)
            pm.CustomDist("obs", ys[:,:,:n_observations], 1/alpha_inv, weights, logp=weighted_nb_logp, random=weighted_nb_random,
                          observed=7*data[:,:,:n_observations], dims=("season", "state", "horizon_observation"))

        # Sample pyMC model
        # ~~~~~~~~~~~~~~~~~

        print('\nstarting the sampler..\n')

        with model:
            # run sampler without tuning
            trace = pm.sample(n_sample, tune=n_tune, chains=n_chains, init='adapt_diag', cores=n_chains, mp_ctx=mp.get_context("spawn"), progressbar=True)

        print('\n..finished sampling\n')

        # Generate traces
        variables2plot = ['rho', 'fI', 'fR', 'phi', 'omega', 'sigma2_0']

        # Save original traces
        os.makedirs(os.path.join(output_folder, 'traces'), exist_ok=True)
        for var in variables2plot:
            arviz.plot_trace_dist(trace, var_names=[var], compact=True, combined=True, kind='kde') 
            plt.savefig(os.path.join(output_folder, f'traces/trace-{var}.pdf'))
            plt.close()

        # Make posterior predictive
        # ~~~~~~~~~~~~~~~~~~~~~~~~~

        print('\ngenerating posterior predictive\n')

        with model:

            # add a geometric random walk per state to simulation output
            grw_innov = pm.Normal("grw_innov", mu=0, sigma=sigma_grw, dims=("state", "horizon_forecast"))         # tune by LOOCV on WIS (currently set to NC stationary GRW baseline model optimal)
            ys_future_rw = ys[:, :, n_observations:] * pt.exp(pt.cumsum(grw_innov, axis=1)[None, :, :])

            # add sampling noise
            pred = pm.NegativeBinomial("pred", mu=ys_future_rw, alpha=1/alpha_inv[None, :, None], dims=("season", "state", "horizon_forecast"))

            # sample posterior predictive
            posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["obs", "pred"])


        # Save traces and posterior predictive
        trace.to_netcdf(os.path.join(output_folder, "trace.nc"))
        posterior_predictive.to_netcdf(os.path.join(output_folder, "posterior_predictive.nc"))

        # Visualise goodness-of-fit
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        
        print('\ngenerating diagnostic plots\n')

        # Visualise
        dates_obs = dt[0,:n_observations]
        dates_pred = dt[0,n_observations:]
        for s in range(n_states):
            fig,ax=plt.subplots()
            ## training
            ax.plot(dates_obs, posterior_predictive.posterior_predictive['obs'].median(dim=['chain', 'draw']).values[0,s,:], linewidth=1, color='black')
            ax.fill_between(dates_obs,
                            posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                            posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.975).values[0,s,:],
                            color='black', alpha=0.1)
            ax.fill_between(dates_obs,
                            posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                            posterior_predictive.posterior_predictive['obs'].quantile(dim=['chain', 'draw'], q=0.75).values[0,s,:],
                            color='black', alpha=0.1)    
            ax.scatter(dates_obs, posterior_predictive.observed_data['obs'].values[0,s,:], marker='o', color='black')
            ## forecast
            ax.plot(dates_pred, posterior_predictive.posterior_predictive['pred'].median(dim=['chain', 'draw']).values[0,s,:], linewidth=1, color='red')
            ax.fill_between(dates_pred,
                            posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                            posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.975).values[0,s,:],
                            color='red', alpha=0.1)
            ax.fill_between(dates_pred,
                            posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.25).values[0,s,:],
                            posterior_predictive.posterior_predictive['pred'].quantile(dim=['chain', 'draw'], q=0.75).values[0,s,:],
                            color='red', alpha=0.1)    
            fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
            fig.tight_layout()
            os.makedirs(os.path.join(output_folder, 'goodness-fit'), exist_ok=True)
            plt.savefig(os.path.join(output_folder,f'goodness-fit/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
            plt.close(fig)

        # TODO: add the modifier trajectories

        # Send simulation output to Hubverse format
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        print('\nconverting simulation output to Hubverse format\n')

        # remove 'seasons' dimension and flatten the 'chain' and 'draw' dimensions into 'draw'
        ## [forecast]
        pred = posterior_predictive.posterior_predictive['pred']
        pred = pred.sel(season=seasons).squeeze("season", drop=True)
        pred = (pred.stack(sample=("chain", "draw")).reset_index("sample", drop=True).rename({"sample": "draw"}))
        pred = pred.assign_coords(draw=np.arange(pred.sizes["draw"]))
        pred = pred.rename({"horizon_forecast": "horizon"})
        ## [observed]
        obs = posterior_predictive.posterior_predictive['obs']
        obs = obs.sel(season=seasons).squeeze("season", drop=True)
        obs = (obs.stack(sample=("chain", "draw")).reset_index("sample", drop=True).rename({"sample": "draw"}))
        obs = obs.assign_coords(draw=np.arange(obs.sizes["draw"]))
        obs = obs.rename({"horizon_observation": "horizon"})
        ## [merge]
        mrg = xr.merge([obs, pred], join='outer')
        mrg["merged"] = mrg["obs"].fillna(mrg["pred"])

        # estimate the peak admissions and convert to hubverse format
        hv_out_peak_admissions = simout_to_hubverse_peak_admissions(mrg["merged"],
                                                                        reference_date,
                                                                        dict(zip(state_fips_index["abbreviation_state"],
                                                                        state_fips_index["fips_state"])),
                                                                        quantiles=True)
        
        # estimate the peak timing and convert to hubverse format
        hv_out_peak_timing = simout_to_hubverse_peak_timing(mrg["merged"],
                                                                reference_date,
                                                                challenge_start_reference_date, 
                                                                challenge_end_reference_date,
                                                                dict(zip(state_fips_index["abbreviation_state"], state_fips_index["fips_state"])),
                                                                quantiles=True)
        
        # convert the admissions to hubverse format
        hv_out_admissions = simout_to_hubverse_admissions(pred,
                                                            reference_date,
                                                            dict(zip(state_fips_index["abbreviation_state"],
                                                            state_fips_index["fips_state"])),
                                                            quantiles=True)
        hv_out_admissions = hv_out_admissions[hv_out_admissions['horizon'] <= 3] # limit admissions to 4-week aheads

        # merge all metrics together
        hv_out = pd.concat([hv_out_admissions, hv_out_peak_timing, hv_out_peak_admissions], axis=0, ignore_index=True)
        hv_out = hv_out.fillna('NA')

        # save result
        hv_out.to_csv(os.path.join(output_folder, reference_date.strftime('%Y-%m-%d')+'-JHU_Cornell'+'-'+'SCARCHhierarSIR.csv'), index=False)

        # append to output list
        forecasts.append(hv_out)

        print(f'\nforecasts of cluster {cluster_idx} complete!\n')

    print(f'\nmerging forecasts of all clusters\n')

    # concatenate all forecasts and save them
    output = pd.concat(forecasts, axis=0)
    output.to_csv(os.path.join(output_folder,'../..',reference_date.strftime('%Y-%m-%d')+'-JHU_Cornell'+'-'+f'{model_name}.csv'), index=False)

    print(f'\nforecasting complete!\n')

# runs the script
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    run_forecast()