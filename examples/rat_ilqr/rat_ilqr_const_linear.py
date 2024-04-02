import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from matplotlib import pyplot as plt

from psoc.abstract import StochasticDynamics
from psoc.environments.feedback import const_linear_env as linear
from rat_ilqr.algorithms import RATILQRState, rat_ilqr


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


dynamics = StochasticDynamics(
    dim=2, ode=linear.ode, step=0.1, stddev=1e-2 * jnp.ones((2,))
)


def rollout(rng, state, make_ileqg_problem):
    _, stochastic_transition, _, _ = make_ileqg_problem()
    x_init = state.x_nominal[0]
    nb_steps = state.l.shape[0]

    def step(x_k, vals):
        sub_key, k = vals
        u_k = state.control(x_k, k)
        next_x = stochastic_transition(sub_key, x_k, u_k)
        return next_x, (next_x, u_k)

    sub_keys = jr.split(rng, nb_steps)
    _, (trajectory, controls) = jax.lax.scan(
        step, x_init, (sub_keys, jnp.arange(nb_steps))
    )
    trajectory = jnp.insert(trajectory, 0, x_init, 0)
    return trajectory, controls


def make_ileqg_problem():

    cost = lambda x, u: -linear.reward(jnp.concatenate([x, u], -1), 1.0)
    terminal_cost = lambda x: 0.0
    return dynamics.mean, dynamics.sample, cost, terminal_cost


def create_state(action_dim, nb_steps, init_tempering, sigma, make_ileqg_problem, d):
    transition, _, _, _ = make_ileqg_problem()

    def step(x_k, u_k):
        next_x = transition(x_k, u_k)
        return next_x, next_x

    u = jnp.zeros((nb_steps, action_dim))
    x_init = jnp.array([1.0, 2.0])
    _, x_nominal = jax.lax.scan(step, x_init, u)
    x_nominal = jnp.insert(x_nominal, 0, x_init, 0)
    W = jnp.tile(dynamics.stddev * jnp.eye(2)[None, :], (nb_steps, 1, 1))

    return RATILQRState.create(
        x_nominal=x_nominal,
        u=u,
        W=W,
        init_tempering=init_tempering,
        sigma=sigma,
        d=d,
        nb_samples=100,
        nb_elite_samples=50,
    )


def main():
    key = jr.PRNGKey(1)

    nb_steps = 50
    nb_iter, nb_ileqg_iter, nb_cem_iter = 1, 1, 10
    d = 1e-9
    init_tempering, init_std = 1e-1, 0.1
    state = create_state(1, nb_steps, init_tempering, init_std, make_ileqg_problem, d)

    train_key, inference_key = jr.split(key)
    state, _ = rat_ilqr(
        train_key, state, make_ileqg_problem, nb_iter, nb_ileqg_iter, nb_cem_iter
    )
    trajectory, controls = rollout(inference_key, state, make_ileqg_problem)

    plt.figure()
    plt.plot(trajectory, label=["pos", "vel"])
    plt.plot(jnp.arange(1, nb_steps + 1), controls, label="u")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
