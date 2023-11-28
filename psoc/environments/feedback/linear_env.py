from functools import partial

import jax
from jax import numpy as jnp

import distrax
from flax import linen as nn

from psoc.abstract import StochasticDynamics
from psoc.abstract import Network
from psoc.abstract import FeedbackPolicy
from psoc.abstract import FeedbackLoop

from psoc.utils import Tanh

jax.config.update("jax_enable_x64", True)


@partial(jnp.vectorize, signature='(k),(h)->(k)')
def ode(
    x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:

    A = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 0.0]
        ]
    )
    B = jnp.array(
        [
            [0.0],
            [1.0]
        ]
    )
    return A @ x + B @ u


@partial(jnp.vectorize, signature='(k),()->()')
def reward(state, eta):
    goal = jnp.array([0.0, 0.0, 0.0])
    weights = jnp.array([1e2, 1e0, 1e-1])
    cost = jnp.dot(state - goal, weights * (state - goal))
    return - 0.5 * eta * cost


prior = distrax.MultivariateNormalDiag(
    loc=jnp.array([1.0, 2.0, 0.0]),
    scale_diag=jnp.ones((3,)) * 1e-4
)

dynamics = StochasticDynamics(
    dim=2,
    ode=ode,
    step=0.1,
    log_std=jnp.log(1e-2 * jnp.ones((2,)))
)


@partial(jnp.vectorize, signature='(k)->(h)')
def identity(x):
    return x


network = Network(
    dim=1,
    layer_size=[],
    transform=identity,
    activation=nn.relu,
    init_log_std=jnp.log(1.0 * jnp.ones((1,))),
)

bijector = distrax.Chain([
    distrax.ScalarAffine(0.0, 2.5),
    Tanh()
])


def create_env(params, eta):
    policy = FeedbackPolicy(
        network, bijector, params
    )

    closedloop = FeedbackLoop(
        dynamics, policy
    )

    anon_rwrd = lambda z: reward(z, eta)
    return prior, closedloop, anon_rwrd