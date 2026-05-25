"""
This script contains all functions related to the forward-simulating SIR ODE model

Authors: T.W. Alleman
Affiliation: Bento Lab, Cornell CVM
Copyright (c) 2026 T.W. Alleman

Licensed under CC BY-NC-SA 4.0
"""

##################
## Dependencies ##
##################

# general purpose python
import numpy as np

# jax and diffrax
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
import diffrax

# pytensor and pymc
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

##############################################
## Foward-simulating SIR model with diffrax ##
##############################################

# define ODE rhs
def SIR_vector_field(t, y, args):
    # unpack states and parameters
    S, I, R, H = y
    beta, delta_beta_daily, gamma, rho = args
    # prevent negative state values due to rounding errors
    S = jax.nn.softplus(S)
    I = jax.nn.softplus(I)
    R = jax.nn.softplus(R)
    H = jax.nn.softplus(H)
    # compute total population
    N = S + I + R
    # get modifier
    delta_beta = 1 + jnp.interp(t, xp=delta_beta_daily[0,:], fp=delta_beta_daily[1,:])
    # compute state derivatives
    FOI = delta_beta * beta * I / N
    dS = - S * FOI
    dI = S * FOI - gamma * I
    dR = gamma * I
    # observation
    dH = rho * S * FOI - H
    return jnp.array([dS, dI, dR, dH])

# define delta_beta[t] modifier function
def make_delta_beta_daily(delta_beta, duration, t0, t1, sigma=2.5):
    """
    Parameters
    ----------
    delta_beta : array (K,)
        Modifier values for each block.
    duration : int
        Number of days each entry is repeated.
    t0 : int
        Start day of the simulation (can be negative).
    t1 : int
        End day of the simulation (may exceed total expanded length).
    sigma: float
        Standard deviation of the Gausian filter.

    Returns
    -------
    vec : 2D array, shape: 2 x (t1 - t0)
        First row: Timesteps.
        Second row: Delta_beta(t) series with zero-padding outside support.
    """

    # Total support length = len(delta_beta) * duration
    total_len = delta_beta.shape[0]* duration

    # Indices of simulation range
    ts = jnp.arange(t0, t1)

    # Compute block boundaries
    # block i covers [i*duration , (i+1)*duration - 1]
    block_ids = jnp.floor_divide(ts, duration).astype(jnp.int32)

    # Mask: valid only if in range
    valid = (ts >= 0) & (ts < total_len)

    # Gather values, using mod-safe indexing (will be masked out anyway)
    expanded = jnp.where(valid, delta_beta[block_ids], 0.0)

    # Smooth with a guassian filter
    x = jnp.linspace(-7, 7, num=15)
    kern = jnp.exp(-0.5 * (x/sigma)**2)
    kern = kern / kern.sum()
    expanded = convolve(expanded, kern, mode="same")

    return jnp.stack([ts, expanded])

# build diffrax model wrapper
def stop_gradients(x):
    return jax.tree.map(jax.lax.stop_gradient, x)

def sol_op_jax(args_diff, args_nodiff, args_static):
    # unpack differentiable parameters
    beta = args_diff[0]
    rho = args_diff[1]
    fI = args_diff[2]
    fR = args_diff[3]
    delta_beta = args_diff[4:]
    # unpack non-differentiable parameters and block their gradients
    args_nodiff = stop_gradients(args_nodiff)
    gamma = args_nodiff[0]
    population = args_nodiff[1]
    ts = args_nodiff[2:]
    # unpack static arguments
    t0, t1_max, modifier_length = args_static
    # evaluate modifiers
    delta_beta_daily = make_delta_beta_daily(delta_beta, modifier_length, t0, t1_max)
    # wrap ODE rhs
    term = diffrax.ODETerm(SIR_vector_field)
    # solve ODE
    sol = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        t0=t0,
        t1=t1_max,
        dt0=0.1,
        y0=population * jnp.array([1-fI-fR, fI, fR, 0]),
        args = (beta, delta_beta_daily, gamma, rho),
        saveat=diffrax.SaveAt(ts=list(ts)),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4)
    )
    return sol.ys[:,-1] # return observed state only


##########################################
## JIT-compile forward simulating model ##
##########################################

# Define jax jitted foward simulation functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sol_op_single(args_diff, args_nodiff, args_static):
    """Wrapper for sol_op_jax to allow vmap."""
    return sol_op_jax(args_diff, args_nodiff, args_static)

def sol_op_multi(args_diff, args_nodiff, args_static):

    # vmap forward simulating model across the states
    state_vmapped = jax.vmap(
        sol_op_jax,
        in_axes=(0,0,None),
        out_axes=0
    )

    # vmap forward simulating model the vmapped states
    sol_op_multi = jax.vmap(
        state_vmapped,
        in_axes=(0,0,None),
        out_axes=0
    )
    return sol_op_multi(args_diff, args_nodiff, args_static)


# Define jax jitted VJP (gradient computation) functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def single_vjp(ad, g, an, args_static):
    _, pullback = jax.vjp(
        lambda th: sol_op_jax(th, an, args_static),
        ad
    )
    return pullback(g)[0]

def vjp_sol_op_multi(args_diff, gz, args_nodiff, args_static):

    # vmap gradients of forward simulating model across the states
    state_vjp = jax.vmap(
        single_vjp,
        in_axes=(0,0,0,None)
    )
    # vmap gradients of forward simulating model the vmapped states
    season_vjp = jax.vmap(
        state_vjp,
        in_axes=(0,0,0,None)
    )

    return season_vjp(args_diff, gz, args_nodiff, args_static)


# Wrap both functions and jax jit them
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_jax_jitted_model():
    """
    Write a docstring
    """
    return jax.jit(sol_op_multi, static_argnums=2), jax.jit(vjp_sol_op_multi, static_argnums=3)


###########################################################
## Register JIT-compiled jax ODE model for use with pyMC ##
###########################################################

# =========================
# Core Op implementations
# =========================

class VJPSolOp(Op):
    def __init__(self, args_static, jitted_vjp_op):
        self.args_static = args_static
        self.jitted_vjp_op = jitted_vjp_op

    def make_node(self, args_diff, gz, args_nodiff):
        return Apply(self, [
            pt.as_tensor_variable(args_diff),
            pt.as_tensor_variable(gz),
            pt.as_tensor_variable(args_nodiff)
        ], [pt.tensor3()])

    def perform(self, node, inputs, outputs):
        args_diff, gz, args_nodiff = inputs
        grad = self.jitted_vjp_op(args_diff, gz, args_nodiff, self.args_static)
        outputs[0][0] = np.asarray(grad, dtype=np.float64)


class SolOp(Op):
    def __init__(self, args_static, jitted_sol_op, jitted_vjp_op):
        self.args_static = args_static
        self.jitted_sol_op = jitted_sol_op
        self.vjp_sol_op = VJPSolOp(args_static, jitted_vjp_op)

    def make_node(self, args_diff, args_nodiff):
        return Apply(self, [
            pt.as_tensor_variable(args_diff),
            pt.as_tensor_variable(args_nodiff)
        ], [pt.tensor3()])

    def perform(self, node, inputs, outputs):
        args_diff, args_nodiff = inputs
        ys = self.jitted_sol_op(args_diff, args_nodiff, self.args_static)
        outputs[0][0] = np.asarray(ys, dtype=np.float64)

    def pullback(self, inputs, outputs, output_grads):
        args_diff, args_nodiff = inputs
        (gz,) = output_grads

        grad_wrt_args_diff = self.vjp_sol_op(args_diff, gz, args_nodiff)
        grad_wrt_args_nodiff = pt.zeros_like(args_nodiff)

        return [grad_wrt_args_diff, grad_wrt_args_nodiff]


# =========================
# JAX registration
# =========================

@jax_funcify.register(SolOp)
def sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff, args_nodiff: op.jitted_sol_op(
        args_diff, args_nodiff, op.args_static
    )


@jax_funcify.register(VJPSolOp)
def vjp_sol_op_jax_funcify(op, **kwargs):
    return lambda args_diff, gz, args_nodiff: op.jitted_vjp_op(
        args_diff, gz, args_nodiff, op.args_static
    )


# =========================
# Factory function
# =========================

def make_sol_op(args_static, jitted_sol_op, jitted_vjp_op):
    return SolOp(args_static, jitted_sol_op, jitted_vjp_op)