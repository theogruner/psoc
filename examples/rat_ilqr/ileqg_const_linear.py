import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from matplotlib import pyplot as plt

from psoc.abstract import StochasticDynamics
from psoc.environments.feedback import const_linear_env as linear
from rat_ilqr.algorithms import ILEQGState, ileqg


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


dynamics = StochasticDynamics(
    dim=2, ode=linear.ode, step=0.1, stddev=1e-1 * jnp.ones((2,))
)


def rollout(rng, state: ILEQGState, make_ileqg_problem):
    _, stochastic_transition, cost_fn, terminal_cost_fn = make_ileqg_problem()
    x_init = state.x_nominal[0]
    nb_steps = state.l.shape[0]

    def step(x_k, vals):
        sub_key, k = vals
        u_k = state.control(x_k, k)
        # jax.debug.print(
        #     "Gains L_k: {k}, {L_k}", k=k, L_k=state.L[k] @ (x_k - state.x_nominal[k])
        # )
        # jax.debug.print("Nominal l_k: {k}, {l_k}", k=k, l_k=state.l[k])
        # jax.debug.print("Control u_k: {k}, {u_k}", k=k, u_k=u_k)

        next_x = stochastic_transition(sub_key, x_k, u_k)
        cost_k = cost_fn(x_k, u_k)
        return next_x, (next_x, u_k, cost_k)

    sub_keys = jr.split(rng, nb_steps)
    x_terminal, (trajectory, controls, costs) = jax.lax.scan(
        step, x_init, (sub_keys, jnp.arange(nb_steps))
    )
    trajectory = jnp.insert(trajectory, 0, x_init, 0)
    terminal_cost = terminal_cost_fn(x_terminal)
    costs = jnp.append(costs, terminal_cost)
    return trajectory, controls, costs.sum()


def make_ileqg_problem():
    cost = lambda x, u: -linear.reward(jnp.concatenate([x, u], -1), 1.0)
    terminal_cost = lambda x: 0.0
    return dynamics.mean, dynamics.sample, cost, terminal_cost


def create_state(action_dim, nb_steps, tempering, make_ileqg_problem, mu=1e-6):
    transition, _, _, _ = make_ileqg_problem()

    def step(x_k, u_k):
        next_x = transition(x_k, u_k)
        return next_x, next_x

    u = jnp.zeros((nb_steps, action_dim))
    x_init = jnp.array([1.0, 2.0])
    _, x_nominal = jax.lax.scan(step, x_init, u)
    x_nominal = jnp.insert(x_nominal, 0, x_init, 0)
    W = jnp.tile((dynamics.stddev**2) * jnp.eye(2)[None, :], (nb_steps, 1, 1))

    return ILEQGState.create(x_nominal, u, tempering, W, mu)


def main():
    key = jr.PRNGKey(0)

    nb_steps = 50
    nb_iter = 1
    tempering = 1e-8  # 5e-2  # optimal 6e-2
    mu = 1e-12
    state = create_state(1, nb_steps, tempering, make_ileqg_problem, mu)
    print(f"ILEQG for nu={tempering}")
    state, _ = ileqg(state, make_ileqg_problem, nb_iter)

    trajectory, controls, cost = rollout(key, state, make_ileqg_problem)
    print(f"Accumulated cost: {cost}")

    plt.figure()
    plt.plot(trajectory, label=["pos", "vel"])
    plt.plot(jnp.arange(1, nb_steps + 1), controls, label="u")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
