"""
This script trains the model on historical data.

Author: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

n_chains = 12

# standard python libraries
import os
import json
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from datetime import datetime, timedelta
# jax and numpyro
import jax
import jax.numpy as jnp
import numpyro
numpyro.set_host_device_count(n_chains)
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive, init_to_value
import arviz
# model package
from SCARCHhierarSIR.data import get_demography, get_adjacency_matrix, get_NHSN_HRD_data, impute_outliers
from SCARCHhierarSIR.preoptimization import preoptimize_parameters, compute_initial_effects
from SCARCHhierarSIR.jax_forward_sim_model import forward_sim_preopt_jax
from SCARCHhierarSIR.numpyro_utils import compute_season_weights

# all paths defined relative to this file
abs_dir = os.path.dirname(__file__)

# global parameters go here
## model-structural
a_garch = 0.0
b_garch = 0.0
phi = 0.5
beta = 0.455
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
n_sample = 25
n_burn = 25
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

    data = impute_outliers(data)

    # Pre-optimize the forward simulation model's parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print('pre-optimization\n')
    print('(iter, score)')

    args_static = (start_simulation, max(ts[:,-1]), modifier_length, jnp.full((n_seasons, n_states), beta), gamma, jnp.asarray(demo), ts)

    # initial guess
    init_params = {
        "rho": jnp.full((n_seasons, n_states), 0.0025),
        "fI": jnp.full((n_seasons, n_states), 1e-4),
        "fR": jnp.full((n_seasons, n_states), 0.25),
        "delta_beta": jnp.zeros((n_modifiers, n_seasons, n_states)),
    }

    # pre-optimise it
    args_diff_preoptim = preoptimize_parameters(
        args_static=args_static,
        data=data,
        init_params=init_params,
        n_iter=n_preoptim,
    )

    # run simulation
    out = forward_sim_preopt_jax(
            rho=args_diff_preoptim["rho"],
            fI=args_diff_preoptim["fI"],
            fR=args_diff_preoptim["fR"],
            delta_beta=args_diff_preoptim["delta_beta"],
            args_static=args_static,
        )

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

    # compute the initial effect sizes
    init = compute_initial_effects(args_diff_preoptim)

    # make dictionary with initial sampler values
    initvals = n_chains * [{'alpha_inv': 0.05 * jnp.ones(n_states), 'delta_beta_raw': init["delta_beta_mu"] / 0.25,
            'log_rho_global_mean': init["log_rho"]["global"], 'rho_state_sd': 0.2, 'rho_state_raw': init["log_rho"]["state"] / 0.2, 'rho_season_sd': 0.2, 'rho_season_raw': init["log_rho"]["season"] / 0.2,
            'log_fI_global_mean': init["log_fI"]["global"], 'fI_state_sd': 0.2, 'fI_state_raw': init["log_fI"]["state"] / 0.2, 'fI_season_sd': 0.2, 'fI_season_raw': init["log_fI"]["season"] / 0.2,
            'logit_fR_global_mean': init["logit_fR"]["global"], 'fR_state_sd': 0.2, 'fR_state_raw': init["logit_fR"]["state"] / 0.2, 'fR_season_sd': 0.2, 'fR_season_raw': init["logit_fR"]["season"] / 0.2,
            'log_omega_global_mean': jnp.log(0.05/3), 'omega_global_mean_shrinkage': 0.05/3, 'psi_1': 0.5, 'psi_2': 0.5}]

    print('\nparameter hierarchy reconstruction\n')

    print(f"Mean log-rho: {init['log_rho']['global']:.2f}")
    print(f"Mean reconstruction error: {init['log_rho']['error_mean']:.2f} ({abs(init['log_rho']['error_mean'] / init['log_rho']['global'] * 100):.1f} %)")
    print(f"Max reconstruction error: {init['log_rho']['error_max']:.2f} ({abs(init['log_rho']['error_max'] / init['log_rho']['global'] * 100):.1f} %)")

    print(f"Mean log-fI: {init['log_fI']['global']:.2f}")
    print(f"Mean reconstruction error: {init['log_fI']['error_mean']:.2f} ({abs(init['log_fI']['error_mean'] / init['log_fI']['global'] * 100):.1f} %)")
    print(f"Max reconstruction error: {init['log_fI']['error_max']:.2f} ({abs(init['log_fI']['error_max'] / init['log_fI']['global'] * 100):.1f} %)")

    print(f"Mean logit-fR: {init['logit_fR']['global']:.2f}")
    print(f"Mean reconstruction error: {init['logit_fR']['error_mean']:.2f} ({abs(init['logit_fR']['error_mean'] / init['logit_fR']['global'] * 100):.1f} %)")
    print(f"Max reconstruction error: {init['logit_fR']['error_max']:.2f} ({abs(init['logit_fR']['error_max'] / init['logit_fR']['global'] * 100):.1f} %)")


    # Build numpyro model
    # ~~~~~~~~~~~~~~~~~~~

    print('\ncompiling numpyro model')

    weights = compute_season_weights(jnp.asarray(data))

    # load training model and its RV dimensions
    from SCARCHhierarSIR.numpyro_models import training_model, training_RV_dims

    # construct its coordinates
    coords = {
        "observation": np.arange(n_observations),
        "state": state_fips_index['abbreviation_state'].values,
        "season": seasons,
        "modifier": np.arange(n_modifiers),
        "modifier_eta": np.arange(n_modifiers-1)
    }


    # Sample numpyro model
    # ~~~~~~~~~~~~~~~~~~~~

    print('\nstarting the NUTS sampler..\n')
    
    rng_key = jax.random.PRNGKey(int(time.time()))
    rng_key, rng_predict = jax.random.split(rng_key)

    kernel = NUTS(
        training_model,
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
        phi=phi,
        a_garch=a_garch,
        b_garch=b_garch,
        init=init,
        args_static=args_static,
        n_states=n_states,
        n_seasons=n_seasons,
        n_modifiers=n_modifiers,
    )


    print('\n..finished sampling\n')
    print('\nsaving traces\n')

    # convert to arviz
    trace = arviz.from_numpyro(mcmc, coords=coords, dims=training_RV_dims)

    # save traces to a netcdf
    trace.to_netcdf(os.path.join(cluster_output_folder, f"trace.nc"))

    # TODO: save the inverse mass matrix
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


    # Sample posterior predictive
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print('\nsampling posterior predictive\n')

    predictive = Predictive(
        training_model,
        posterior_samples=mcmc.get_samples(),
        )

    posterior_predictive = predictive(
        rng_predict,
        data=None,
        weights=jnp.asarray(weights),
        adj=jnp.asarray(adj),
        phi=phi,
        a_garch=a_garch,
        b_garch=b_garch,
        init=init,
        args_static=args_static,
        n_states=n_states,
        n_seasons=n_seasons,
        n_modifiers=n_modifiers,
    )

    # convert to arviz
    posterior_predictive = arviz.from_numpyro(mcmc, posterior_predictive=posterior_predictive, coords=coords, dims=training_RV_dims)

    # save posterior predictive
    posterior_predictive.to_netcdf(os.path.join(cluster_output_folder,"posterior_predictive.nc"))


    # Visualisations
    # ~~~~~~~~~~~~~~

    print('\nmaking posterior predictive visualisations\n')

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
