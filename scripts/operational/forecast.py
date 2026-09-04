"""
This script makes a single forecast for unseen data.

Author: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

n_chains = 8

# Suppress the specific UserWarning from JAX regarding int64 truncation
import warnings
warnings.filterwarnings(
    "ignore", 
    message="Explicitly requested dtype int64 requested in astype is not available"
)

# standard python libraries
import os
import time
import json
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# jax and diffrax
import jax
import jax.numpy as jnp
from numpyro.infer import NUTS, MCMC, Predictive, init_to_value
import numpyro
numpyro.set_host_device_count(n_chains)
import arviz # must be after numpyro.infer
# model package
from SCARCHhierarSIR.data import get_demography, get_adjacency_matrix, get_NHSN_HRD_data, simout_to_hubverse_admissions, simout_to_hubverse_peak_admissions, simout_to_hubverse_peak_timing
from SCARCHhierarSIR.numpyro_utils import compute_season_weights, find_map


# wrapper 
def main():

    # all paths defined relative to this file
    abs_dir = os.path.dirname(__file__)

    # global parameters go here
    ## training metadata
    training_name = 'exclude_None-a_garch_0.0-phi_0.5-omega_0.005'
    training_folder = os.path.join(abs_dir, f'../../data/interim/calibration/hierarchical-training/{training_name}')
    ## forecasting settings
    challenge_start_reference_date = datetime(2026, 10, 10) # must be a saturday
    challenge_end_reference_date = datetime(2027, 5, 29)    # must be the last saturday of may
    season = '2025-2026'            
    n_observations = 4            # use all data available in the forecast season
    forecast_horizon = 20           # forecast sufficiently ahead to capture peaks
    n_preoptim = 1000
    n_sample = 10
    n_tune = 10
    sigma_grw = 0.001

    ## load the model-structural parameters and training metadata
    with open(os.path.join(training_folder, "model_config.json"), "r") as f:
        params = json.load(f)

    beta = params["beta"]
    gamma = params["gamma"]
    n_basis = params["n_basis"]
    n_modifiers = params["n_modifiers"]
    modifier_length = params["modifier_length"]
    start_simulation = params["start_simulation"]
    modifier_ref_month = params["modifier_ref_month"]
    modifier_ref_day = params["modifier_ref_day"]

    # derived products
    ## convert to a list of start and enddates (datetime)
    n_seasons = 1
    start_calibrations = [datetime(int(season[0:4]), modifier_ref_month, modifier_ref_day) + timedelta(days=start_simulation)]    # calibrations started at same time as simulation
    modifier_reference_dates = [datetime(int(season[0:4]), modifier_ref_month, modifier_ref_day)]
    model_name = 'SCARCHhierarSIR'


    # Get US demographics
    # ~~~~~~~~~~~~~~~~~~~

    state_fips_index, demo = get_demography()
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
    reference_date = dt[-1][-1] + timedelta(weeks=1) - timedelta(weeks=forecast_horizon)    # compute true reference date based on data instead of filename

    # output folder name
    output_folder = os.path.join(abs_dir, f'../../data/interim/calibration/forecast/{training_name}/reference_date-{reference_date.strftime('%Y-%m-%d')}/')


    # Get the hyperparameters
    # ~~~~~~~~~~~~~~~~~~~~~~~

    # get
    hyperpars = pd.read_csv(os.path.join(training_folder, f'hyperparameters-{training_name}.csv'))

    # slice states
    hyperpars = hyperpars[hyperpars['state'].isin(state_fips_index['abbreviation_state'])]

    # unpack
    posterior_params = {
        'rho_global_mean':          hyperpars['rho_global_mean'].unique()[0],
        'rho_season_sd':            hyperpars['rho_season_sd'].unique()[0],
        'fI_global_mean':           hyperpars['fI_global_mean'].unique()[0],
        'fI_season_sd':             hyperpars['fI_season_sd'].unique()[0],
        'fR_global_mean':           hyperpars['fR_global_mean'].unique()[0],
        'fR_season_sd':             hyperpars['fR_season_sd'].unique()[0],
        'psi_2':                    hyperpars['psi_2'].unique()[0],
        'phi':                      hyperpars['phi'].unique()[0],
        'omega':                    hyperpars['omega'].unique()[0],
        'a_garch':                  hyperpars['a_garch'].unique()[0],
        'b_garch':                  hyperpars['b_garch'].unique()[0],
        'alpha_inv':                hyperpars['alpha_inv'].values,
        'rho_state':                hyperpars['rho_state'].values,
        'fI_state':                 hyperpars['fI_state'].values,
        'fR_state':                 hyperpars['fR_state'].values,
        'delta_beta_state_mean':    np.transpose(hyperpars[[c for c in hyperpars.columns if c.startswith("delta_beta_state_mean_")]].to_numpy())
    }


    # Build numpyro model
    # ~~~~~~~~~~~~~~~~~~~

    print('\ncompiling numpyro model\n')

    weights = compute_season_weights(data[:,:,:n_observations])

    args_static = (start_simulation, float(max(ts[:,-1])), modifier_length, jnp.full((n_seasons, n_states), beta), gamma, jnp.asarray(demo), ts)

    # load forecasting model and its RV dimensions
    from SCARCHhierarSIR.numpyro_models import forecasting_model, forecasting_RV_dims

    # construct coordinates
    coords = {
        "state": state_fips_index['abbreviation_state'].values,
        "season": [season,],
        "modifier": np.arange(n_modifiers),
        "modifier_eta": np.arange(n_modifiers-1),
        "observation": np.arange(len(np.squeeze(ts))),
        "horizon_forecast": np.arange(forecast_horizon),
        "horizon_observation": [-i for i in range(1, n_observations + 1)]
    }

    # construct its arguments
    model_kwargs = dict(
        data=jnp.asarray(data),
        weights=jnp.asarray(weights),
        posterior_params=posterior_params,
        adj=jnp.asarray(adj),
        sigma_grw=sigma_grw,
        args_static=args_static,
        n_states=n_states,
        n_seasons=n_seasons,
        n_modifiers=n_modifiers,
        n_observations=n_observations,
    )


    # Find the model's MAP
    # ~~~~~~~~~~~~~~~~~~~~

    print('pre-optimizing MAP\n')
    print('(iter, score)')

    # run optimisation
    map_params = find_map(forecasting_model, model_kwargs, n_preoptim)

    # visualise the result
    out = map_params['H'][:,:,:n_observations]
    for s in range(n_states):
        fig, ax = plt.subplots(nrows=1, figsize=(8.7, 11.3/4))
        for i in range(n_seasons):
            ax.plot(dt[i, :n_observations], out[i, s, :], color='red', label='pred')
            ax.scatter(dt[i, :n_observations], data[i, s, :n_observations], marker='o', color='black', label='obs')
        fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
        fig.tight_layout()
        os.makedirs(os.path.join(output_folder, 'initial-optim'), exist_ok=True)
        plt.savefig(os.path.join(output_folder,f'initial-optim/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
        plt.close(fig)


    # Sample numpyro model
    # ~~~~~~~~~~~~~~~~~~~~

    start_dt = datetime.now()
    start_time = time.time()

    print(f"\nstarting the NUTS sampler at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")

    rng_key = jax.random.PRNGKey(int(time.time()))
    rng_key, rng_predict = jax.random.split(rng_key)

    kernel = NUTS(
        forecasting_model,
        step_size=0.0002,
        adapt_step_size=True,
        max_tree_depth=12,
        target_accept_prob=0.98,
        dense_mass=True,
        init_strategy = init_to_value(values=map_params),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=n_tune,
        num_samples=n_sample,
        num_chains=n_chains,
        chain_method="parallel",
        progress_bar=False,
    )

    mcmc.run(
        rng_key,
        **model_kwargs,
        extra_fields=["potential_energy", "adapt_state.step_size"]
    )

    # Chain collection avoids weird sequencing of printouts
    _ = mcmc.get_samples()

    # Record the end timestamp and compute elapsed time
    end_dt = datetime.now()
    elapsed_seconds = time.time() - start_time
    elapsed_formatted = str(timedelta(seconds=int(elapsed_seconds)))

    print(f"..and finished sampling at: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"total elapsed time: {elapsed_formatted}\n")

    print('\nsaving traces\n')

    # convert to arviz
    trace = arviz.from_numpyro(mcmc, coords=coords, dims=forecasting_RV_dims)

    # save traces to a netcdf
    trace.to_netcdf(os.path.join(output_folder, "trace.nc"))

    # Generate traceplots
    variables2plot = ['rho', 'fI', 'fR']

    # Save original traces
    os.makedirs(os.path.join(output_folder, 'traces'), exist_ok=True)
    for var in variables2plot:
        arviz.plot_trace_dist(trace, var_names=[var], compact=True, combined=True, kind='kde') 
        plt.savefig(os.path.join(output_folder, f'traces/trace-{var}.pdf'))
        plt.close()

    # plot the step sizes
    fig,ax = plt.subplots(figsize=(8.3, 11.7/4))
    for chain in trace.sample_stats.step_size.coords['chain'].values:
        ax.plot(trace.sample_stats.step_size.sel(chain=chain), label=f"chain {chain}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder,f'traces/step_sizes.pdf'))
    plt.close()


    # Make posterior predictive
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    print('\ngenerating posterior predictive\n')

    predictive = Predictive(
        forecasting_model,
        posterior_samples=mcmc.get_samples(),
        return_sites = ["obs", "pred"]
    )

    model_kwargs.update({'data': None})

    posterior_predictive = predictive(rng_predict, **model_kwargs)

    # convert to arviz
    posterior_predictive = arviz.from_numpyro(mcmc, posterior_predictive=posterior_predictive) #, coords=coords, dims=forecasting_RV_dims)

    # Rename obs and pred dimensions
    pp = posterior_predictive["posterior_predictive"]
    obs = pp["obs"].rename({"obs_dim_0": "season", "obs_dim_1": "state", "obs_dim_2": "horizon_observation"})
    pred = pp["pred"].rename({"pred_dim_0": "season", "pred_dim_1": "state", "pred_dim_2": "horizon_forecast"})

    # Assign coordinates directly to the variables
    obs = obs.assign_coords({"season": coords["season"], "state": coords["state"], "horizon_observation": coords["horizon_observation"]})
    pred = pred.assign_coords({"season": coords["season"], "state": coords["state"], "horizon_forecast": coords["horizon_forecast"]})

    # Replace variables
    pp["obs"] = obs
    pp["pred"] = pred
    posterior_predictive["posterior_predictive"] = pp

    # do observed data too
    pp = posterior_predictive['observed_data']
    obs = pp["obs"].rename({"obs_dim_0": "season", "obs_dim_1": "state", "obs_dim_2": "horizon_observation"})
    obs = obs.assign_coords({"season": coords["season"], "state": coords["state"], "horizon_observation": coords["horizon_observation"]})
    pp["obs"] = obs
    posterior_predictive["observed_data"] = pp

    # save posterior predictive
    posterior_predictive.to_netcdf(os.path.join(output_folder, "posterior_predictive.nc"))


    # Visualise goodness-of-fit
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    # sum over all states to get USA totals
    data = xr.concat([posterior_predictive.observed_data['obs'], posterior_predictive.observed_data['obs'].sum(dim="state").assign_coords(state="USA").expand_dims("state")], dim="state")
    obs = xr.concat([posterior_predictive.posterior_predictive['obs'], posterior_predictive.posterior_predictive['obs'].sum(dim="state").assign_coords(state="USA").expand_dims("state")], dim="state")
    pred = xr.concat([posterior_predictive.posterior_predictive['pred'], posterior_predictive.posterior_predictive['pred'].sum(dim="state").assign_coords(state="USA").expand_dims("state")], dim="state")

    # expand fips index
    state_fips_index["fips_state"] = state_fips_index["fips_state"].map("{:02d}".format)
    new_row = pd.DataFrame([{"abbreviation_state": "USA", "name_state": "united states", "fips_state": "USA"}])
    state_fips_index = pd.concat([state_fips_index, new_row], ignore_index=True)


    print('\ngenerating diagnostic plots\n')

    # Visualise
    dates_obs = dt[0,:n_observations]
    dates_pred = dt[0,n_observations:]
    for s in range(len(state_fips_index)):
        fig,ax=plt.subplots()
        ## training
        ax.plot(dates_obs, obs.median(dim=['chain', 'draw']).values[0,s,:], linewidth=1, color='black')
        ax.fill_between(dates_obs,
                        obs.quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                        obs.quantile(dim=['chain', 'draw'], q=0.975).values[0,s,:],
                        color='black', alpha=0.1)
        ax.fill_between(dates_obs,
                        obs.quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                        obs.quantile(dim=['chain', 'draw'], q=0.75).values[0,s,:],
                        color='black', alpha=0.1)    
        ax.scatter(dates_obs, data.values[0,s,:], marker='o', color='black')
        ## forecast
        ax.plot(dates_pred, pred.median(dim=['chain', 'draw']).values[0,s,:], linewidth=1, color='red')
        ax.fill_between(dates_pred,
                        pred.quantile(dim=['chain', 'draw'], q=0.025).values[0,s,:],
                        pred.quantile(dim=['chain', 'draw'], q=0.975).values[0,s,:],
                        color='red', alpha=0.1)
        ax.fill_between(dates_pred,
                        pred.quantile(dim=['chain', 'draw'], q=0.25).values[0,s,:],
                        pred.quantile(dim=['chain', 'draw'], q=0.75).values[0,s,:],
                        color='red', alpha=0.1)    
        fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
        fig.tight_layout()
        os.makedirs(os.path.join(output_folder, 'goodness-fit'), exist_ok=True)
        plt.savefig(os.path.join(output_folder,f'goodness-fit/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
        plt.close(fig)


    # Send simulation output to Hubverse format
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print('\nconverting simulation output to Hubverse format\n')

    # remove 'seasons' dimension and flatten the 'chain' and 'draw' dimensions into 'draw'
    ## [forecast]
    pred = pred.squeeze("season", drop=True)
    pred = (pred.stack(sample=("chain", "draw")).reset_index("sample", drop=True).rename({"sample": "draw"}))
    pred = pred.assign_coords(draw=np.arange(pred.sizes["draw"]))
    pred = pred.rename({"horizon_forecast": "horizon"})
    ## [observed]
    obs = obs.squeeze("season", drop=True)
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
    hv_out.to_csv(os.path.join(output_folder, reference_date.strftime('%Y-%m-%d')+'-Cornell_JHU'+'-'+f'{model_name}.csv'), index=False)

    print(f'\nforecasting complete!\n')

# execute script
if __name__ == "__main__":

    main()
