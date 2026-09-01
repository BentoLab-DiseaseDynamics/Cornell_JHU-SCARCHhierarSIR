import jax
import numpy as np
import jax.numpy as jnp
from jax.scipy.signal import convolve
import diffrax

######################
## SIR vector field ##
######################

def SIR_vector_field(t, y, args):

    # unpack states
    S, I = y

    # unpack parameters
    beta, daily_ts, delta_beta_daily, gamma = args
    
    # interpolate modifier at current ODE time
    delta_beta = 1.0 + jnp.interp(t, xp=daily_ts, fp=delta_beta_daily)

    # force of infection
    FOI = delta_beta * beta * I

    # SIR equations
    dS = -S * FOI
    dI = S * FOI - gamma * I

    return jnp.array([dS, dI])


#############################################
## Write the single state/season simulator ##
#############################################

def simulate_one_jax(beta, rho, fI, fR, delta_beta_daily, gamma, population, t0, t1, ts):
    """
    Simulate one season/state SIR system.

    Parameters
    ----------
    beta : scalar
    rho : scalar
    fI : scalar
    fR : scalar

    delta_beta_daily : array
        Shape (n_time,)

    gamma : scalar

    population : scalar

    t0, t1 : scalar/static
        Simulation bounds.

    ts : array
        Observation times.

    Returns
    -------
    ys : array
        Shape (n_observation, 4)
    """

    # Daily time points corresponding to delta_beta_daily
    daily_ts = jnp.arange(t0, t1)

    # Initial condition
    y0 = jnp.array([1.0 - fI - fR, fI])

    # Diffrax term
    term = diffrax.ODETerm(SIR_vector_field)

    # Solve
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Heun(),
        t0=t0,
        t1=t1,
        dt0=7,
        y0=y0,
        args=(beta, daily_ts, delta_beta_daily, gamma),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.DirectAdjoint(),
    )

    # Compute hospital admissions in week
    hosp_diff = population * rho * (sol.ys[:-1, 0] - sol.ys[1:, 0])
    hosp_first = (5.0 * hosp_diff[0] - hosp_diff[1] - hosp_diff[2]) / 3.0   # linear interpolation of first 3 weeks
    weekly_hosp = jnp.concatenate([hosp_first[None], hosp_diff])

    return weekly_hosp


##########################################################################################
## Write the wrapper batching the single season/state simulator over states and seasons ##
##########################################################################################

def simulate_all_jax(beta, rho, fI, fR, delta_beta_daily, gamma, population, t0, t1, ts):

    """
    Batched SIR simulation over all seasons and states.

    Returns
    -------
    ys : array
        Shape:
        (season, state, observation, 4)
    """

    # State-level vmap
    simulate_states = jax.vmap(
        simulate_one_jax,
        in_axes=(
            0,      # beta
            0,      # rho
            0,      # fI
            0,      # fR
            0,      # delta_beta_daily
            None,   # gamma
            0,      # population
            None,   # t0
            None,   # t1
            None,   # ts
        ),
    )

    # Season-level vmap
    simulate_seasons = jax.vmap(
        simulate_states,
        in_axes=(
            0,      # beta
            0,      # rho
            0,      # fI
            0,      # fR
            0,      # delta_beta_daily
            None,   # gamma
            None,   # population
            None,   # t0
            None,   # t1
            0,   # ts
        ),
    )

    return simulate_seasons(beta, rho, fI, fR, delta_beta_daily, gamma, population, t0, t1, ts)


###########################
## Modifier construction ##
###########################

def make_delta_beta_daily_batched(delta_beta, duration, t0, t1, sigma=1):
    """
    Convert block-level modifiers into smoothed daily modifiers.

    Parameters
    ----------
    delta_beta : jax.Array
        Shape (n_modifiers, n_seasons, n_states)

    duration : int
        Number of days represented by each modifier.

    t0, t1 : int
        Simulation bounds.

    sigma : float
        Gaussian kernel standard deviation.

    Returns
    -------
    daily : jax.Array
        Shape (n_seasons, n_states, t1 - t0)
    """

    n_modifiers = delta_beta.shape[0]

    # Simulation times
    ts = jnp.arange(t0, t1)
    total_len = n_modifiers * duration

    # Which modifier block corresponds to each time?
    block_ids = jnp.floor_divide(ts, duration).astype(jnp.int32)

    # Valid modifier support
    valid = ((ts >= 0) & (ts < total_len))

    # Prevent invalid indexing before applying the mask
    safe_block_ids = jnp.clip(block_ids, 0, n_modifiers - 1)

    # Expand blocks to daily values
    expanded = delta_beta[safe_block_ids]

    # Zero-pad outside support
    expanded = jnp.where(valid[:, None, None], expanded, 0.0)

    # Put time on final axis
    expanded = jnp.transpose(expanded, (1, 2, 0))

    n_seasons = expanded.shape[0]
    n_states = expanded.shape[1]
    n_times = expanded.shape[2]

    # Build Gaussian smoothing kernel
    x = jnp.linspace(-7.0, 7.0, num=15)
    kern = jnp.exp(-0.5 * (x / sigma) ** 2)
    kern = kern / kern.sum()

    # Convolve each season/state trajectory independently over time
    expanded_flat = expanded.reshape(n_seasons * n_states, n_times)

    def smooth(x):
        return convolve(x, kern, mode="same")

    expanded_flat = jax.vmap(smooth)(expanded_flat)

    # Restore dimensions
    daily = expanded_flat.reshape(n_seasons, n_states, n_times,)

    return daily


###################
## AR-GARCH scan ##
###################

def ar_garch_scan(eta, phi, omega, a_garch, b_garch):
    """
    Parameters
    ----------
    eta : jax.Array
        Shape (T-1, season, state)

    phi : scalar
        AR(1) coefficient.

    omega : scalar
        Baseline noise (= sigma if a/b garch = 0)

    a_garch : scalar
        ARCH coefficient.

    b_garch : scalar
        GARCH coefficient.

    Returns
    -------
    z : jax.Array
        Shape (T, season, state)

    sigma2 : jax.Array
        Shape (T, season, state)

    eps : jax.Array
        Shape (T, season, state)
    """

    # Construct initial statees
    z0 = jnp.zeros([eta.shape[1], eta.shape[2]])
    eps0 = jnp.zeros([eta.shape[1], eta.shape[2]])
    sigma20 = omega * jnp.ones_like(eps0)

    # Run the recursion
    def step(carry, eta_t):

        prev_z, prev_sigma2, prev_eps = carry

        sigma2 = omega + a_garch * prev_eps**2 + b_garch * prev_sigma2 # GARCH(1,1)

        eps = eta_t * jnp.sqrt(sigma2) # innovation

        z = phi * prev_z + eps # AR(1)

        return (z,sigma2,eps,), (z,sigma2,eps,)

    _, (z_seq, sigma2_seq, eps_seq) = jax.lax.scan(step,(z0, sigma20, eps0),eta)

    # Concatenate the initial states
    z = jnp.concatenate([z0[None, ...], z_seq], axis=0)
    sigma2 = jnp.concatenate([sigma20[None, ...], sigma2_seq], axis=0)
    eps = jnp.concatenate([eps0[None, ...], eps_seq], axis=0)

    return z, sigma2, eps


########################################################
## Combine all parts into one jax-function and jit it ##
########################################################

def forward_sim_jax(eta, phi, omega, a_garch, b_garch, delta_beta_state_mean, rho, fI, fR, args_static):
    """
    Complete JAX forward simulation model.

    Returns
    -------
    H : jax.Array
        Shape (season, state, observation)

    z : jax.Array
        Shape (modifier, season, state)

    sigma2 : jax.Array
        Shape (modifier, season, state)

    eps : jax.Array
        Shape (modifier, season, state)
    """
    
    # Unpack static arguments
    t0, t1, modifier_length, beta, gamma, population, ts = args_static

    # 1. AR-GARCH recursion
    z, sigma2, eps = ar_garch_scan(eta=eta, phi=phi, omega=omega, a_garch=a_garch, b_garch=b_garch) # shape: (modifier, season, state)

    # 2. Construct modifier
    delta_beta = z + delta_beta_state_mean[:, None, :]

    # 3. Convert modifier to daily values
    delta_beta_daily = make_delta_beta_daily_batched(delta_beta=delta_beta, duration=modifier_length, t0=t0, t1=t1)

    # 4. Run batched ODE
    H = simulate_all_jax(
        beta=beta,
        rho=rho,
        fI=fI,
        fR=fR,
        delta_beta_daily=delta_beta_daily,
        gamma=gamma,
        population=population,
        t0=t0,
        t1=t1,
        ts=ts,
    )

    # shape: (season, state, observation)

    return H, z, sigma2, eps