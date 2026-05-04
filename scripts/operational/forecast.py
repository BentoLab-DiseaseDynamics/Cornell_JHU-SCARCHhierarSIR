"""
This script makes a forecast for unseen data.

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
from SCARCHhierarchSIR.data import get_demography, get_adjacency_matrix, get_NHSN_HRD_data, simout_to_hubverse
from SCARCHhierarchSIR.SIR_model import get_jax_jitted_model, make_sol_op
from SCARCHhierarchSIR.pymc_model import AR_GARCH_step, compute_season_weights, weighted_nb_logp, weighted_nb_random
from SCARCHhierarchSIR.preoptimization import preoptimize_parameters

# all paths defined relative to this file
abs_dir = os.path.dirname(__file__)

# global parameters go here
## model-structural
use_garch = True
gamma = 1/3.5
n_modifiers = 26
modifier_length = 7
start_simulation = -15
## geographical extent of training
regions = ['New England', 'Middle Atlantic']
## training metadata
start_calibration_month = 10
training_name = 'exclude_None-wGARCH'
training_folder = os.path.join(abs_dir, f'../../data/interim/calibration/training/{training_name}')
## forecasting settings
seasons = ['2025-2026',]        # script only works with one season
n_observations = 25             # use all data available in the forecast season
forecast_horizon = 4            # forecast 4 weeks ahead
n_preoptim = 500
n_sample = 10
n_tune = 10
n_chains = 2
sigma_grw = 0.375

# derived products
## convert to a list of start and enddates (datetime)
n_seasons = len(seasons)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
modifier_reference_dates = [datetime(int(season[0:4]), 10, 15) for season in seasons]

# Get US demographics
# ~~~~~~~~~~~~~~~~~~~

state_fips_index, demo = get_demography(regions)
n_states = len(demo)

# Get state adjacency matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

adj = get_adjacency_matrix(state_fips_index['abbreviation_state'])

# Get US incidence data
# ~~~~~~~~~~~~~~~~~~~~~

# get data
reference_date, data, dt, ts, n_observations = get_NHSN_HRD_data(start_calibrations, modifier_reference_dates, n_observations,
                                                                 type = 'hypothetical',
                                                                 forecast_horizon=forecast_horizon,
                                                                 state_fips=state_fips_index['fips_state'].values) # (n_season, n_variables, n_observations)
data = data / 7                                     # divide weekly incidence by 7
reference_date = dt[-1][-1] + timedelta(weeks=1) - timedelta(weeks=forecast_horizon)    # compute true reference date based on data instead of filename

# output folder name
output_folder = os.path.join(abs_dir, f'../../data/interim/calibration/forecasting/{training_name}/reference_date-{reference_date.strftime('%Y-%m-%d')}')

# Get the hyperparameters
# ~~~~~~~~~~~~~~~~~~~~~~~

# get
hyperpars = pd.read_csv(os.path.join(training_folder, f'hyperparameters-{training_name}.csv'))

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
kappa_global_mean       = hyperpars['kappa_global_mean'].unique()[0]
kappa_season_sd         = hyperpars['kappa_season_sd'].unique()[0]
omega                   = hyperpars['omega'].unique()[0]
nu                      = hyperpars['nu'].unique()[0]
## (state) vectors
alpha_inv               = hyperpars['alpha_inv'].values
rho_state               = hyperpars['rho_state'].values
fI_state                = hyperpars['fI_state'].values
fR_state                = hyperpars['fR_state'].values
phi_state               = hyperpars['phi_state'].values
kappa_state             = hyperpars['kappa_state'].values

## hypermodifiers
modifier_cols = [c for c in hyperpars.columns if c.startswith("delta_beta_state_mean_")]
delta_beta_state_mean = np.transpose(hyperpars[modifier_cols].to_numpy())

# Define a jax-jitted diffrax differential equation model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

jitted_sol_op_multi, jitted_vjp_sol_op_multi = get_jax_jitted_model()

# Pre-optimize the forward simulation model's parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

# construct coordinates
coords = {
    "state": state_fips_index['abbreviation_state'].values,
    "season": seasons,
    "modifier": np.arange(n_modifiers),
    "horizon": np.arange(forecast_horizon)
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
    L_Q_shocks = pt.slinalg.cholesky(Q_shocks)
    L_cov_shocks = pt.slinalg.solve(L_Q_shocks, pt.eye(n_states))

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
    eta_raw = pm.Normal("eta_raw", mu=0.0, sigma=1.0, dims=("modifier","state"))
    # correlate them across space using the precision matrix 
    eta = pm.Deterministic("eta", pt.einsum("ij,mj->mi", L_cov_shocks, eta_raw))    # shape: (modifier x state)

    # --- GARCH(1,1) parameters ---                                                                             
    # Total noise persistence
    ### global (kappa_global_mean)
    ### state (kappa_state)
    ### season (kappa_season_sd)
    kappa_season_raw = pm.Normal("kappa_season_raw", 0, 1, dims="season")
    kappa = pm.Deterministic("kappa", pt.squeeze(pm.math.sigmoid(pm.math.logit(kappa_global_mean) + pt.log(kappa_state)[None, :] + kappa_season_sd * kappa_season_raw[:, None])[0,:]))
     
    # Split between a and b (nu hyperparameter)                                                                                                             
    a_garch = pm.Deterministic("a_garch", kappa * nu)                                                        
    b_garch = pm.Deterministic("b_garch", kappa * (1 - nu))                         
    sigma2_0 = pm.Deterministic("sigma2_0", omega/(1-kappa), dims="state")

    # Run AR-GARCH scan over T steps
    z_seq, sigma2_seq, eps_seq = pytensor.scan(
        fn=AR_GARCH_step,
        sequences=[eta,],
        outputs_info=[z_0, sigma2_0, eps_0],
        non_sequences=[phi, omega, a_garch, b_garch, pt.as_tensor_variable(1 if use_garch else 0)],
        return_updates=False
    )

    # Register deterministic variables to inspect later
    delta_beta = pm.Deterministic("delta_beta", delta_beta_state_mean + z_seq)
    z = pm.Deterministic("z", z_seq)
    sigma2 = pm.Deterministic("sigma2", sigma2_seq)
    eps = pm.Deterministic("eps", eps_seq)

    # concatenate parameters along last axis (n_seasons, n_states, n_parameters)
    args_diff = pt.concatenate(
        [beta[:, :, None], rho[:, :, None], fI[:, :, None], fR[:, :, None], pt.transpose(delta_beta, (1, 0))[None, :, :]],
        axis=2
    )

    # Run forward simulation model
    ys = 7*sol_op(args_diff, args_nodiff)
    ys = pt.math.softplus(ys)

    # Compute likelihood (alpha_inv hyperparameter)
    pm.CustomDist("obs", ys[:,:,:n_observations], 1/alpha_inv, weights, logp=weighted_nb_logp, random=weighted_nb_random, observed=7*data[:,:,:n_observations])

# Sample pyMC model
# ~~~~~~~~~~~~~~~~~

with model:
    # run sampler without tuning
    trace = pm.sample(n_sample, tune=n_tune, chains=n_chains, init='adapt_diag', cores=1, progressbar=True)

print(f"Step size post-tuning: {trace.sample_stats.step_size_bar.values}")

# Generate traces
variables2plot = ['rho', 'fI', 'fR', 'phi', 'kappa', 'sigma2_0']

# Save original traces
os.makedirs(os.path.join(output_folder, 'traces'), exist_ok=True)
for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(os.path.join(output_folder, f'traces/trace-{var}.pdf'))
    plt.close()

# Make posterior predictive
# ~~~~~~~~~~~~~~~~~~~~~~~~~

with model:

    # add a geometric random walk per state to simulation output
    grw_innov = pm.Normal("grw_innov", mu=0, sigma=sigma_grw, dims=("state", "horizon"))         # tune by LOOCV on WIS (currently set to NC stationary GRW baseline model optimal)
    ys_future_rw = ys[:, :, n_observations:] * pt.exp(pt.cumsum(grw_innov, axis=1)[None, :, :])

    # add sampling noise
    pred = pm.NegativeBinomial("pred", mu=ys_future_rw, alpha=1/alpha_inv[None, :, None], dims=("season", "state", "horizon"))

    # sample posterior predictive
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["obs", "pred"])


# Save traces and posterior predictive
arviz.to_netcdf(trace, os.path.join(output_folder, "trace.nc"))
arviz.to_netcdf(posterior_predictive, os.path.join(output_folder, "posterior_predictive.nc"))

# Visualise goodness-of-fit
# ~~~~~~~~~~~~~~~~~~~~~~~~~

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

# Send simulation output to Hubverse format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# remove 'seasons' dimension and flatten the 'chain' and 'draw' dimensions into 'draw'
simout = posterior_predictive.posterior_predictive['pred']
simout = simout.sel(season=seasons).squeeze("season", drop=True)
simout = (simout.stack(sample=("chain", "draw")).reset_index("sample", drop=True).rename({"sample": "draw"}))
simout = simout.assign_coords(draw=np.arange(simout.sizes["draw"]))

# convert to hubverse format
hv_out = simout_to_hubverse(simout,
                            reference_date, 
                            dict(zip(state_fips_index["abbreviation_state"], state_fips_index["fips_state"])),
                            target='wk inc flu hosp',
                            quantiles=False)

# save result
hv_out.to_csv(os.path.join(output_folder, reference_date.strftime('%Y-%m-%d')+'-JHU_Cornell'+'-'+'SCARCHhierarchSIR.csv'), index=False)

