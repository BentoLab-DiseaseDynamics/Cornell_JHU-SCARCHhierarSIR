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
    S, I, R, H = y

    # unpack parameters
    beta, daily_ts, delta_beta_daily, gamma, rho = args
    
    # prevent negative state values due to rounding errors
    S = jax.nn.softplus(S)
    I = jax.nn.softplus(I)
    R = jax.nn.softplus(R)
    H = jax.nn.softplus(H)

    # total population
    N = S + I + R

    # interpolate modifier at current ODE time
    delta_beta = 1.0 + jnp.interp(
        t,
        xp=daily_ts,
        fp=delta_beta_daily,
    )

    # force of infection
    FOI = delta_beta * beta * I / N

    # SIR equations
    dS = -S * FOI
    dI = S * FOI - gamma * I
    dR = gamma * I

    # observation process
    dH = rho * S * FOI - H

    return jnp.array([dS, dI, dR, dH])


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
    y0 = population * jnp.array([1.0 - fI - fR, fI, fR, 0.0])

    # Diffrax term
    term = diffrax.ODETerm(SIR_vector_field)

    # Solve
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        args=(beta, daily_ts, delta_beta_daily, gamma, rho),
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4),
        adjoint=diffrax.DirectAdjoint(),
    )

    return sol.ys


##########################################################################################
## Write the wrapper batching the single season/state simulator over states and seasons ##
##########################################################################################

def simulate_all_jax(beta, rho, fI, fR, delta_beta_daily, gamma, population, t0, t1, ts,):

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

def make_delta_beta_daily_batched(delta_beta, duration, t0, t1, sigma=2.5):
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

    # ---------------------------------------
    # Expand blocks to daily values
    # ---------------------------------------

    # delta_beta:
    #   (modifier, season, state)
    #
    # indexed by block_ids:
    #   (time, season, state)

    expanded = delta_beta[safe_block_ids]

    # Zero-pad outside support
    expanded = jnp.where(valid[:, None, None], expanded, 0.0)

    # Put time on final axis
    expanded = jnp.transpose(expanded, (1, 2, 0))

    n_seasons = expanded.shape[0]
    n_states = expanded.shape[1]
    n_times = expanded.shape[2]

    # ---------------------------------------
    # Gaussian smoothing kernel
    # ---------------------------------------

    x = jnp.linspace(-7.0, 7.0, num=15)
    kern = jnp.exp(-0.5 * (x / sigma) ** 2)
    kern = kern / kern.sum()

    # ---------------------------------------
    # Convolve each season/state trajectory
    # independently over time
    # ---------------------------------------

    expanded_flat = expanded.reshape(n_seasons * n_states, n_times)

    def smooth(x):
        return convolve(x, kern, mode="same")

    expanded_flat = jax.vmap(smooth)(expanded_flat)

    # ---------------------------------------
    # Restore dimensions
    # ---------------------------------------

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

    omega : jax.Array
        Shape (season, state)

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
    z0 = jnp.zeros_like(omega)
    eps0 = jnp.zeros_like(omega)
    sigma20 = omega

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

def forward_jax(eta_raw, phi, omega, a_garch, b_garch, delta_beta_state_mean, L_cov_shocks, beta, rho, fI, fR, gamma, population, ts, args_static):
    """
    Complete JAX forward model.

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

    t0, t1_max, modifier_length = args_static

    # 1. Spatially correlate shocks
    eta = jnp.einsum("ij,tsj->tsi", L_cov_shocks, eta_raw)

    # 2. AR-GARCH recursion
    z, sigma2, eps = ar_garch_scan(eta=eta, phi=phi, omega=omega, a_garch=a_garch, b_garch=b_garch) # shape: (modifier, season, state)

    # 3. Construct modifier
    delta_beta = z + delta_beta_state_mean[:, None, :] # shape: (modifier, season, state)

    # 4. Convert modifiers to daily values
    delta_beta_daily = make_delta_beta_daily_batched(delta_beta=delta_beta, duration=modifier_length, t0=t0, t1=t1_max) # shape: (season, state, time)

    # 5. Run batched ODE
    ys = simulate_all_jax(
        beta=beta,
        rho=rho,
        fI=fI,
        fR=fR,
        delta_beta_daily=delta_beta_daily,
        gamma=gamma,
        population=population,
        t0=t0,
        t1=t1_max,
        ts=ts,
    )

    # shape: (season, state, observation, 4)

    # 6. Extract H
    H = ys[..., 3] # shape: (season, state, observation)

    return H, z, sigma2, eps

forward_jitted = jax.jit(forward_jax, static_argnums=-1)


#################################
## Define the jax VJP function ##
#################################


def forward_vjp_jax(
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
    gH,
    args_static,
):
    """
    VJP of forward_jax with respect to the forward inputs.

    Only H contributes to the likelihood, so the cotangents
    for z, sigma2, and eps are set to zero inside JAX.
    """

    def forward_fn(
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
    ):
        return forward_jitted(
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

    # Evaluate forward function and construct VJP
    outputs, pullback = jax.vjp(
        forward_fn,
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
    )

    H, z, sigma2, eps = outputs

    # Only H contributes to the likelihood.
    grads = pullback(
        (
            gH,
            jnp.zeros_like(z),
            jnp.zeros_like(sigma2),
            jnp.zeros_like(eps),
        )
    )

    return grads


forward_vjp_jitted = jax.jit(
    forward_vjp_jax,
    static_argnums=-1,
)


#######################
## Write the pyMC Op ##
#######################

# pytensor and pymc
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify





class ForwardOp(Op):

    def __init__(self, args_static, forward_jitted, forward_vjp_jitted):
        self.args_static = args_static
        self.forward_jitted = forward_jitted
        self.forward_vjp_op = ForwardVJPOp(
            args_static,
            forward_vjp_jitted,
        )

    def make_node(
        self,
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
    ):

        inputs = [
            pt.as_tensor_variable(eta_raw),
            pt.as_tensor_variable(phi),
            pt.as_tensor_variable(omega),
            pt.as_tensor_variable(a_garch),
            pt.as_tensor_variable(b_garch),
            pt.as_tensor_variable(delta_beta_state_mean),
            pt.as_tensor_variable(L_cov_shocks),
            pt.as_tensor_variable(beta),
            pt.as_tensor_variable(rho),
            pt.as_tensor_variable(fI),
            pt.as_tensor_variable(fR),
            pt.as_tensor_variable(gamma),
            pt.as_tensor_variable(population),
            pt.as_tensor_variable(ts),
        ]

        outputs = [
            pt.tensor3(),  # H
            pt.tensor3(),  # z
            pt.tensor3(),  # sigma2
            pt.tensor3(),  # eps
        ]

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):

        (
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
        ) = inputs

        H, z, sigma2, eps = self.forward_jitted(
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
            self.args_static,
        )

        outputs[0][0] = np.asarray(H, dtype=np.float64)
        outputs[1][0] = np.asarray(z, dtype=np.float64)
        outputs[2][0] = np.asarray(sigma2, dtype=np.float64)
        outputs[3][0] = np.asarray(eps, dtype=np.float64)

    def pullback(self, inputs, outputs, output_grads):

        (
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
        ) = inputs

        (
            gH,
            gz,
            gsigma2,
            geps,
        ) = output_grads

        # --------------------------------------------------
        # Only H contributes to the likelihood.
        #
        # z, sigma2, and eps are exposed as deterministic
        # outputs for the posterior, but currently have no
        # downstream likelihood contribution.
        #
        # PyTensor may therefore give us DisconnectedType
        # objects for gz, gsigma2, and geps. We deliberately
        # do not pass those to the JAX VJP.
        # --------------------------------------------------

        grads = self.forward_vjp_op(
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
            gH,
        )

        return list(grads)







class ForwardVJPOp(Op):

    def __init__(self, args_static, forward_vjp_jitted):
        self.args_static = args_static
        self.forward_vjp_jitted = forward_vjp_jitted

    def make_node(
        self,
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
        gH,
    ):

        inputs = [
            pt.as_tensor_variable(eta_raw),
            pt.as_tensor_variable(phi),
            pt.as_tensor_variable(omega),
            pt.as_tensor_variable(a_garch),
            pt.as_tensor_variable(b_garch),
            pt.as_tensor_variable(delta_beta_state_mean),
            pt.as_tensor_variable(L_cov_shocks),
            pt.as_tensor_variable(beta),
            pt.as_tensor_variable(rho),
            pt.as_tensor_variable(fI),
            pt.as_tensor_variable(fR),
            pt.as_tensor_variable(gamma),
            pt.as_tensor_variable(population),
            pt.as_tensor_variable(ts),
            pt.as_tensor_variable(gH),
        ]

        # Gradient outputs have exactly the same
        # TensorType as their corresponding inputs.
        outputs = [
            inputs[0].type(),   # eta_raw
            inputs[1].type(),   # phi
            inputs[2].type(),   # omega
            inputs[3].type(),   # a_garch
            inputs[4].type(),   # b_garch
            inputs[5].type(),   # delta_beta_state_mean
            inputs[6].type(),   # L_cov_shocks
            inputs[7].type(),   # beta
            inputs[8].type(),   # rho
            inputs[9].type(),   # fI
            inputs[10].type(),  # fR
            inputs[11].type(),  # gamma
            inputs[12].type(),  # population
            inputs[13].type(),  # ts
        ]

        return Apply(
            self,
            inputs,
            outputs,
        )

    def perform(self, node, inputs, outputs):

        (
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
            gH,
        ) = inputs

        grads = self.forward_vjp_jitted(
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
            gH,
            self.args_static,
        )
        for i,(output, grad) in enumerate(zip(outputs, grads)):
            output[0] = np.asarray(grad)






#############################
## Register nodes with JAX ##
#############################

@jax_funcify.register(ForwardVJPOp)
def forward_vjp_op_jax_funcify(op, **kwargs):

    return lambda eta_raw,phi,omega,a_garch,b_garch,delta_beta_state_mean,L_cov_shocks,beta,rho,fI,fR,gamma,population,ts,gH: op.forward_vjp_jitted(
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
        gH,
        op.args_static,
    )


@jax_funcify.register(ForwardOp)
def forward_op_jax_funcify(op, **kwargs):

    return lambda eta_raw, phi, omega, a_garch, b_garch, delta_beta_state_mean, L_cov_shocks, beta, rho, fI, fR, gamma, population, ts: op.forward_jitted(
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
        op.args_static,
    )
