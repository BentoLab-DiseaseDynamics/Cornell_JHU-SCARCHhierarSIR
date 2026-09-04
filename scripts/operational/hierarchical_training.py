"""
This script trains the model on historical data.

Author: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

nuts_progress_bar = False
n_chains = 4

# Suppress the specific UserWarning from JAX regarding int64 truncation
import warnings
warnings.filterwarnings(
    "ignore", 
    message="Explicitly requested dtype int64 requested in astype is not available"
)

# standard python libraries
import os
import json
import time
import numpy as np
import pandas as pd
from patsy import dmatrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
# jax and numpyro
import jax
import jax.numpy as jnp
import numpyro
numpyro.set_host_device_count(n_chains)
from numpyro.infer import NUTS, MCMC, Predictive, init_to_value
import arviz

# model package
from SCARCHhierarSIR.data import get_demography, get_adjacency_matrix, get_NHSN_HRD_data, impute_outliers
from SCARCHhierarSIR.numpyro_utils import compute_season_weights, find_map


# wrapper
def main():

    # all paths defined relative to this file
    abs_dir = os.path.dirname(__file__)

    # global parameters go here
    ## model-structural
    a_garch = 0.0
    b_garch = 0.0
    omega = 0.005
    phi = 0.50
    beta = 0.455
    gamma = 1/3.5
    n_basis = 20
    n_modifiers = 36
    modifier_length = 7
    start_simulation = 0 # (Sept 1)
    modifier_ref_month = 9
    modifier_ref_day = 1
    ## temporal extent of training
    n_observations = 36             # run until start of June
    seasons = ['2023-2024', '2024-2025', '2025-2026']
    ## sampling effort
    n_sample = 250
    n_burn = 150
    target_accept = 0.85
    n_preoptim = 5000
    training_name = f'exclude_None-a_garch_{a_garch}-phi_{phi}-omega_{omega}-targetaccept_{target_accept}'
    ## use previous sampling
    find_new_map = False

    ## save model-structural parameters and training metadata
    output_folder = os.path.join(abs_dir, f'../../data/interim/calibration/hierarchical-training/{training_name}')
    os.makedirs(output_folder, exist_ok=True)
    params = {"a_garch": a_garch, "b_garch": b_garch, "omega": omega, "phi": phi, "beta": 0.455, "gamma": 1 / 3.5, "n_modifiers": n_modifiers, "n_basis": n_basis, "modifier_length": modifier_length, "start_simulation": start_simulation,
                "modifier_ref_month": modifier_ref_month, "modifier_ref_day": modifier_ref_day, "observations": n_observations, 'seasons': seasons}
    with open(os.path.join(output_folder, "model_config.json"), "w") as f:
        json.dump(params, f, indent=4)


    print('\nretrieving data')

    ## get a previously obtained map estimate
    if not find_new_map:
        with jnp.load(os.path.join(output_folder, "../map_estimate.npz")) as archive:
            map_params = {key: jnp.asarray(archive[key]) for key in archive.files}
    else:
        map_params = None

    # derived products
    ## convert to a list of start and enddates (datetime)
    n_seasons = len(seasons)
    start_calibrations = [datetime(int(season[0:4]),modifier_ref_month, modifier_ref_day) + timedelta(days=start_simulation) for season in seasons] # start calibration at simulation start
    modifier_reference_dates = [datetime(int(season[0:4]), modifier_ref_month, modifier_ref_day) for season in seasons]

    # Get US demographics
    # ~~~~~~~~~~~~~~~~~~~

    state_fips_index, demo = get_demography()
    n_states = len(demo)

    # Get state adjacency matrix
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    adj = get_adjacency_matrix(state_fips_index['abbreviation_state'].values)

    # Get US incidences
    # ~~~~~~~~~~~~~~~~~

    reference_date, data, dt, ts, n_observations = get_NHSN_HRD_data(start_calibrations, modifier_reference_dates, n_observations, forecast_horizon=None, state_fips=state_fips_index['fips_state'].values) # (n_season, n_variables, n_observations)

    # Outlier detection
    # ~~~~~~~~~~~~~~~~~

    data = impute_outliers(data)

    # Build numpyro model
    # ~~~~~~~~~~~~~~~~~~~

    print('\ncompiling numpyro model\n')

    spline_basis = jnp.asarray(dmatrix(f"bs(x, df={n_basis-1}, degree=3, include_intercept=False)", {"x": np.arange(n_modifiers)}, return_type="dataframe").to_numpy())

    args_static = (start_simulation, max(ts[:,-1]), modifier_length, jnp.full((n_seasons, n_states), beta), gamma, jnp.asarray(demo), ts)

    weights = compute_season_weights(jnp.asarray(data))

    # load training model and its RV dimensions
    from SCARCHhierarSIR.numpyro_models import training_model, training_RV_dims

    # construct its coordinates
    coords = {
        "observation": np.arange(n_observations),
        "state": state_fips_index['abbreviation_state'].values,
        "season": seasons,
        "modifier": np.arange(n_modifiers),
        "modifier_eta": np.arange(n_modifiers-1),
        "spline_basis": np.arange(n_basis)
    }

    # construct its arguments
    model_kwargs = dict(
        data=jnp.asarray(data),
        weights=jnp.asarray(weights),
        adj=jnp.asarray(adj),
        phi=phi,
        omega=omega,
        a_garch=a_garch,
        b_garch=b_garch,
        spline_basis=spline_basis,
        args_static=args_static,
        n_states=n_states,
        n_seasons=n_seasons,
        n_modifiers=n_modifiers,
    )


    # Find the model's MAP
    # ~~~~~~~~~~~~~~~~~~~~

    print('pre-optimizing MAP\n')
    print('(iter, score)')

    # run optimisation
    map_params = find_map(training_model, model_kwargs, n_preoptim, map_params=map_params)

    # Save dictionary to a file
    if find_new_map:
        jnp.savez(os.path.join(output_folder, "../map_estimate.npz"), **map_params)

    # https://pmc.ncbi.nlm.nih.gov/articles/PMC11128746/    --> 30% average sterilizing antibodies
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC4169819/     --> pandemic: R = 1.8, seasonal R = 1.28 --> \approx 30% immunity

    # visualise the result
    out = map_params['H']
    for s in range(n_states):
        fig, ax = plt.subplots(nrows=1, figsize=(8.7, 11.3/4))
        for i in range(n_seasons):
            ax.plot(dt[i, :], out[i, s, :], color='red', label='pred')
            ax.scatter(dt[i, :], data[i, s, :], marker='o', color='black', label='obs')
        fig.suptitle(f'{state_fips_index.iloc[s]['abbreviation_state']}')
        fig.tight_layout()

        os.makedirs(os.path.join(output_folder,f'initial-optim'), exist_ok=True)
        plt.savefig(os.path.join(output_folder,f'initial-optim/state_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
        plt.close(fig)

    # Sample numpyro model
    # ~~~~~~~~~~~~~~~~~~~~

    start_dt = datetime.now()
    start_time = time.time()

    print(f"\nstarting the NUTS sampler at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ..")

    rng_key = jax.random.PRNGKey(int(time.time()))
    rng_key, rng_predict = jax.random.split(rng_key)

    kernel = NUTS(
        training_model,
        step_size=0.0002,
        adapt_step_size=True,
        max_tree_depth=12,
        target_accept_prob=target_accept,
        dense_mass=True,
        init_strategy = init_to_value(values=map_params),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=n_burn,
        num_samples=n_sample,
        num_chains=n_chains,
        chain_method="parallel",
        progress_bar=nuts_progress_bar,
    )

    mcmc.run(
        rng_key,
        **model_kwargs,
        extra_fields=["potential_energy", "adapt_state.step_size"]
    )

    # Chain collection prevents jax asynchronous dispatch from weirdly sequencing printouts
    time.sleep(1)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), mcmc.get_samples())

    # Record the end timestamp and compute elapsed time
    end_dt = datetime.now()
    elapsed_seconds = time.time() - start_time
    elapsed_formatted = str(timedelta(seconds=int(elapsed_seconds)))

    print(f"..and finished sampling at: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"total elapsed time: {elapsed_formatted}\n")

    print('\nsaving traces\n')

    # convert to arviz
    trace = arviz.from_numpyro(mcmc, coords=coords, dims=training_RV_dims)

    # save traces to a netcdf
    trace.to_netcdf(os.path.join(output_folder, f"trace.nc"))

    # TODO: save the inverse mass matrix
    inv_mass_matrix = mcmc.last_state.adapt_state.inverse_mass_matrix # you will save this as a .json and then use it to restart runs

    # plot the step sizes
    fig,ax = plt.subplots(figsize=(8.3, 11.7/4))
    for chain in trace.sample_stats.step_size.coords['chain'].values:
        ax.plot(trace.sample_stats.step_size.sel(chain=chain), label=f"chain {chain}")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.join(output_folder,'traces'), exist_ok=True)
    plt.savefig(os.path.join(output_folder,f'traces/step_sizes.pdf'))
    plt.close()

    print('\nmaking traceplots\n')

    # Generate traces
    variables2plot = [
                    'alpha_inv',                                                                            # overdispersion
                    'rho_global_mean', 'rho_state_sd', 'rho_state', 'rho_season_sd', 'rho_season', 'rho',   # rho
                    'fI_global_mean', 'fI_state_sd', 'fI_state', 'fI_season_sd', 'fI_season', 'fI',         # fI
                    'fR_global_mean', 'fR_state_sd', 'fR_state', 'fR_season_sd', 'fR_season', 'fR',         # fR
                    'psi_2', 'psi_1',                                                                       # spatial correlation strength
                    'phi', 'omega',                                                                         # AR - GARCH
                    'a_garch', 'b_garch',
                    ]

    # Save original traces
    for var in variables2plot:
        arviz.plot_trace_dist(trace, var_names=[var], compact=True, combined=True, kind='kde') 
        plt.savefig(os.path.join(output_folder,f'traces/trace-{var}.pdf'))
        plt.close()


    # Sample posterior predictive
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print('\nsampling posterior predictive\n')

    predictive = Predictive(
        training_model,
        posterior_samples=mcmc.get_samples(),
        )

    model_kwargs.update({'data': None})

    posterior_predictive = predictive(rng_predict, **model_kwargs)

    # convert to arviz
    posterior_predictive = arviz.from_numpyro(mcmc, posterior_predictive=posterior_predictive, coords=coords, dims=training_RV_dims)

    # save posterior predictive
    posterior_predictive.to_netcdf(os.path.join(output_folder,"posterior_predictive.nc"))


    # Visualisations
    # ~~~~~~~~~~~~~~

    print('\nmaking posterior predictive visualisations\n')

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
    plt.savefig(os.path.join(output_folder,f'traces/forestplot-alpha_inv.pdf'))
    plt.close()


    # visualise forest plots of state and season effect sizes
    labels_params = [r'$\rho$', r'$f_I$', r'$f_R$']
    state_params = ["rho_state", "fI_state", "fR_state"]
    season_params = ["rho_season", "fI_season", "fR_season"]
    global_params = ["rho_global_mean", "fI_global_mean", "fR_global_mean"]
    params = ['rho', 'fI', 'fR']
    effect_type = ['Multiplicative', 'Multiplicative', 'Odds-ratio']

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
        plt.savefig(os.path.join(output_folder,f'traces/forestplot-{p}.pdf'))
        plt.close()


    # Visualise across-season modifier trend + within-season median per state
    os.makedirs(os.path.join(output_folder,'modifiers'), exist_ok=True)
    # make dates
    x = pd.date_range(start=datetime(2000,modifier_ref_month,modifier_ref_day), periods=n_modifiers, freq='W')
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
        ax.set_ylim([0.5, 1.5])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.savefig(os.path.join(output_folder,f'modifiers/modifiers_{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}.pdf'))
        plt.close()


    # Visualise goodness-of-fit, delta_beta, z, sigma2 and eps per state and per season
    for s in range(n_states):
        os.makedirs(os.path.join(output_folder,f'goodness-fit/{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}/'), exist_ok=True)
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
            plt.savefig(os.path.join(output_folder,f'goodness-fit/{state_fips_index.iloc[s]['fips_state']}_{state_fips_index.iloc[s]['abbreviation_state']}/{season}_goodness-fit.pdf'))
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
        "omega",
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
    ]
    for p in state_params:
        df[p] = med[p].values


    # delta_beta_state_mean (modifier x state)
    delta = med["delta_beta_state_mean"].values
    n_modifiers = delta.shape[0]
    for i in range(n_modifiers):
        df[f"delta_beta_state_mean_{i}"] = delta[i, :]

    # save hyperparameters
    df.index.name = "state"
    df.to_csv(os.path.join(output_folder,f"hyperparameters-{training_name}.csv"))

    print(f'\ntraining complete!\n')

# execute script
if __name__ == "__main__":

    main()