import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental.host_callback import id_tap
from flax import struct

from ._cem import CEMState, cem
from ._ileqg import ILEQGState, ileqg


@struct.dataclass
class RATILQRState:
    ileqg_state: ILEQGState
    cem_state: CEMState

    d: float

    @classmethod
    def create(
        cls,
        x_nominal,
        u,
        W,
        init_tempering,
        sigma,
        d,
        max_reward=jnp.inf,
        regularizer=1e-9,
        nb_samples=100,
        nb_elite_samples=None,
    ):
        ileqg_state = ILEQGState.create(x_nominal, u, init_tempering, W, regularizer)
        cem_state = CEMState.create(
            init_tempering, sigma, nb_samples, nb_elite_samples, max_reward
        )
        return cls(ileqg_state=ileqg_state, cem_state=cem_state, d=d)

    def control(self, x, k):
        return self.ileqg_state.control(x, k)

    @property
    def x_nominal(self):
        return self.ileqg_state.x_nominal

    @property
    def l(self):
        return self.ileqg_state.l


def rat_ilqr(
    rng: jax.Array,
    state: RATILQRState,
    make_ileqg_problem,
    nb_iter,
    nb_ileqg_iter,
    nb_cem_iter,
    verbose=True,
):
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, cost-to-go: {z[1]:.4f}, temperature: {z[2]: .4f}", end="\n"
    )

    @jax.jit
    def body(state: RATILQRState, x):
        def make_cem_problem():

            def reward_fn(param):
                ileqg_state = state.ileqg_state
                ileqg_state = ileqg_state.replace(tempering=param)
                _, ctg = ileqg(
                    ileqg_state, make_ileqg_problem, nb_ileqg_iter, verbose=False
                )
                reward = ctg + state.d / param
                return reward

            return reward_fn

        rng = x
        cem_state, _ = cem(rng, state.cem_state, make_cem_problem, nb_cem_iter)
        ileqg_state = state.ileqg_state.replace(tempering=cem_state.mu)
        ileqg_state, cost_to_go = ileqg(
            ileqg_state, make_ileqg_problem, nb_ileqg_iter, verbose=False
        )
        state = state.replace(cem_state=cem_state, ileqg_state=ileqg_state)

        return state, (cost_to_go, cem_state.mu)

    sub_keys = jr.split(rng, nb_iter)
    cost_to_go = jnp.array(jnp.inf)
    for i in range(nb_iter):
        state, (cost_to_go, mu) = body(state, sub_keys.at[i].get())

        if verbose:
            id_tap(print_func, (i, cost_to_go, mu))

    # state, _ = jax.lax.scan(body, state, None, length=nb_iter)
    return state, cost_to_go
