from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from jax.experimental.host_callback import id_tap


@struct.dataclass
class ILEQGState:
    # Linear controler
    x_nominal: jax.Array
    L: jax.Array
    l: jax.Array
    cost_to_go: float

    # hyperparameter
    W: jax.Array  # system noise
    tempering: float
    mu: float

    @classmethod
    def create(cls, x_nominal, u, tempering, W, mu=1e-6):
        nb_steps, state_dim = x_nominal.shape
        nb_steps -= 1
        action_dim = u.shape[-1]
        return cls(
            x_nominal=x_nominal,
            L=jnp.empty((nb_steps, action_dim, state_dim)),
            l=u,
            tempering=tempering,
            mu=mu,
            W=W,
            cost_to_go=jnp.inf,
        )

    def control(self, x, k):
        return self.L[k] @ (x - self.x_nominal[k]) + self.l[k]


def ileqg(state: ILEQGState, make_ileqg_problem, nb_iter, verbose=True):
    """Iterative linear exponential quadratic Gaussian for
    time-discrete finite-horizon MDPs.
    """
    print_func = lambda z, *_: print(
        f"\riter: {z[0]}, cost-to-go: {z[1]:.4f}", end="\n"
    )

    @jax.jit
    def body(carry, x):
        return _ileqg(carry, make_ileqg_problem)

    cost_to_go = jnp.array(jnp.inf)
    for i in range(nb_iter):
        state, cost_to_go = body(state, None)

        if verbose:
            id_tap(print_func, (i, cost_to_go))

    # state, _ = jax.lax.scan(body, state, None, length=nb_iter)
    return state, cost_to_go


def _ileqg(state: ILEQGState, make_ileqg_problem):

    # 1. Local approximation
    q, q_vec, Q, r, R, P, A, B = _local_approximation(
        state.x_nominal, state.l, make_ileqg_problem
    )

    # 2. Backward pass
    L, dl = _backward_pass(
        q, q_vec, Q, r, R, P, A, B, state.W, state.tempering, state.mu
    )

    # 3. Line search
    x_nominal, l, cost_to_go = _line_search(
        L,
        state.l,
        dl,
        state.W,
        state.x_nominal,
        state.cost_to_go,
        make_ileqg_problem,
        state.tempering,
    )

    state = state.replace(x_nominal=x_nominal, L=L, l=l, cost_to_go=cost_to_go)

    return state, cost_to_go


def _local_approximation(x, u, make_ileqg_problem):
    dynamics, _, cost, terminal_cost = make_ileqg_problem()

    def local_linearization(x_k, u_k):
        A_k = jax.jacfwd(dynamics, argnums=0)(x_k, u_k)
        B_k = jax.jacfwd(dynamics, argnums=1)(x_k, u_k)
        q_k = cost(x_k, u_k)
        q_vec_k = jax.grad(cost, argnums=0)(x_k, u_k)
        Q_k = jax.hessian(cost, argnums=0)(x_k, u_k)
        r_k = jax.grad(cost, argnums=1)(x_k, u_k)
        R_k = jax.hessian(cost, argnums=1)(x_k, u_k)
        P_k = jax.jacfwd(jax.grad(cost, argnums=1), argnums=0)(x_k, u_k)
        return A_k, B_k, q_k, q_vec_k, Q_k, r_k, R_k, P_k

    A, B, q, q_vec, Q, r, R, P = jax.vmap(local_linearization)(x[:-1], u)

    # Calculate terminal reward
    terminal_state = x[-1]

    q_terminal = terminal_cost(terminal_state)
    q = jnp.append(q, q_terminal)

    q_vec_terminal = jax.grad(terminal_cost)(terminal_state)
    q_vec = jnp.append(q_vec, q_vec_terminal[None, :], 0)

    Q_terminal = jax.hessian(terminal_cost)(terminal_state)
    Q = jnp.append(Q, Q_terminal[None, :], 0)

    return q, q_vec, Q, r, R, P, A, B


def _backward_pass(q, q_vec, Q, r, R, P, A, B, W, tempering: float, mu: float):

    state_dim = W.shape[-1]

    def _update_gains(H_k, G_k, g_k):
        # jax.debug.print(
        #     "H positive definite: {h_pos}",
        #     h_pos=jnp.all(jnp.linalg.eigvals(H_k + mu * jnp.eye(H_k.shape[-1])) > 0),
        # )
        inverse_gain = jnp.linalg.inv(H_k + mu * jnp.eye(H_k.shape[-1]))
        L_k = -inverse_gain @ G_k
        dl_k = -inverse_gain @ g_k
        return L_k, dl_k

    def body(carry, x):
        s_prev, s_vec_prev, S_prev = carry
        A_k, B_k, q_k, q_vec_k, Q_k, r_k, R_k, P_k, W_k = x
        M_k = jnp.linalg.inv(W_k) - tempering * S_prev
        M_k_inv = jnp.linalg.inv(M_k)
        D_k = jnp.eye(state_dim) + tempering * S_prev @ M_k_inv
        g_k = r_k + B_k.transpose() @ (D_k @ s_vec_prev)
        G_k = P_k + B_k.transpose() @ (D_k @ (S_prev @ A_k))
        H_k = R_k + B_k.transpose() @ (D_k @ (S_prev @ B_k))

        L_k, dl_k = _update_gains(H_k, G_k, g_k)

        s_k = (
            q_k
            + s_prev
            + 0.5 * dl_k.transpose() @ (H_k @ dl_k)
            + dl_k.transpose() @ g_k
        )
        residual = jax.lax.cond(
            tempering == 0.0,
            lambda: 0.5 * jnp.trace(W_k @ S_prev),
            lambda: -jax.lax.div(0.5, tempering) * _logdet(W_k @ M_k)
            + 0.5 * tempering * s_vec_prev.transpose() @ (M_k_inv @ s_vec),
        )
        s_k += residual
        jax.debug.print("{wk}", wk=-jax.lax.div(0.5, tempering) * _logdet(W_k @ M_k))

        s_k_vec = (
            q_vec_k
            + A_k.transpose() @ (D_k @ s_vec_prev)
            + L_k.transpose() @ (H_k @ dl_k)
            + L_k.transpose() @ g_k
            + G_k.transpose() @ dl_k
        )
        S_k = (
            Q_k
            + A_k.transpose() @ (D_k @ (S_prev @ A_k))
            + L_k.transpose() @ (H_k @ L_k)
            + L_k.transpose() @ G_k
            + G_k.transpose() @ L_k
        )

        return (s_k, s_k_vec, S_k), (L_k, dl_k)

    s = q[-1]
    s_vec = q_vec[-1]
    S = Q[-1]
    _, (L, dl) = jax.lax.scan(
        body,
        (s, s_vec, S),
        (
            A[::-1],
            B[::-1],
            q[:-1][::-1],
            q_vec[:-1][::-1],
            Q[:-1][::-1],
            r[::-1],
            R[::-1],
            P[::-1],
            W[::-1],
        ),
    )

    return L[::-1], dl[::-1]


def _eval_cost_to_go(q, q_vec, Q, r, R, P, A, B, W, L, dl, tempering: float):
    s = q[-1]
    s_vec = q_vec[-1]
    S = Q[-1]
    state_dim = W.shape[-1]

    def body(carry, x):
        s_prev, s_vec_prev, S_prev = carry
        A_k, B_k, q_k, q_vec_k, Q_k, r_k, R_k, P_k, W_k, L_k, dl_k = x
        M_k = jnp.linalg.inv(W_k) - tempering * S_prev
        M_k_inv = jnp.linalg.inv(M_k)
        D_k = jnp.eye(state_dim) + tempering * S_prev @ M_k_inv
        g_k = r_k + B_k.transpose() @ (D_k @ s_vec_prev)
        G_k = P_k + B_k.transpose() @ (D_k @ (S_prev @ A_k))
        H_k = R_k + B_k.transpose() @ (D_k @ (S_prev @ B_k))

        s_k = (
            q_k
            + s_prev
            + 0.5 * dl_k.transpose() @ (H_k @ dl_k)
            + dl_k.transpose() @ g_k
        )
        # jax.debug.print("{sk}", sk=q_k + s_prev)
        residual = jax.lax.cond(
            tempering == 0.0,
            lambda: 0.5 * jnp.trace(W_k @ S_prev),
            lambda: -jax.lax.div(0.5, tempering) * _logdet(W_k @ M_k)
            + 0.5 * tempering * s_vec_prev.transpose() @ (M_k_inv @ s_vec),
        )
        s_k += residual
        s_k_vec = (
            q_vec_k
            + A_k.transpose() @ (D_k @ s_vec_prev)
            + L_k.transpose() @ (H_k @ dl_k)
            + L_k.transpose() @ g_k
            + G_k.transpose() @ dl_k
        )
        S_k = (
            Q_k
            + A_k.transpose() @ (D_k @ (S_prev @ A_k))
            + L_k.transpose() @ (H_k @ L_k)
            + L_k.transpose() @ G_k
            + G_k.transpose() @ L_k
        )

        return (s_k, s_k_vec, S_k), (s_k, s_k_vec, S_k, g_k, G_k, H_k, L_k, dl_k)

    (s_0, _, _), _ = jax.lax.scan(
        body,
        (s, s_vec, S),
        (
            A[::-1],
            B[::-1],
            q[:-1][::-1],
            q_vec[:-1][::-1],
            Q[:-1][::-1],
            r[::-1],
            R[::-1],
            P[::-1],
            W[::-1],
            L[::-1],
            dl[::-1],
        ),
    )
    return s_0


def _line_search(
    L,
    l,
    dl,
    W,
    x_nominal,
    reference_cost,
    make_ileqg_problem,
    tempering,
    lam=0.5,
    max_steps=100,
):
    dynamics, _, _, _ = make_ileqg_problem()
    epsilon = 1.0

    def _candidate_trajectory(eps) -> Tuple[jax.Array, jax.Array]:

        def _step(x_k, val):
            L_k, l_k, dl_k, nominal_x_k = val
            u_k = L_k @ (x_k - nominal_x_k) + l_k + eps * dl_k
            next_x = dynamics(x_k, u_k)
            return next_x, (next_x, u_k)

        _, (trajectory, controls) = jax.lax.scan(
            _step, x_nominal[0], (L, l, dl, x_nominal[:-1])
        )
        trajectory = jnp.insert(trajectory, 0, x_nominal[0], 0)
        return trajectory, controls

    def cond(val):
        cost_to_go, _, _, epsilon, i = val
        # jax.debug.print("should terminate: {diff}", diff=cost_to_go < reference_cost)
        return jnp.logical_or(
            jnp.logical_and((i < max_steps), cost_to_go > reference_cost), i == 0
        )

    def body(val):
        _, _, _, epsilon, i = val
        epsilon *= lam
        x_candidate, u_candidate = _candidate_trajectory(epsilon)
        q, q_vec, Q, r, R, P, A, B = _local_approximation(
            x_candidate, u_candidate, make_ileqg_problem
        )
        cost_to_go = _eval_cost_to_go(
            q, q_vec, Q, r, R, P, A, B, W, L, jnp.zeros_like(u_candidate), tempering
        )

        return cost_to_go, x_candidate, u_candidate, epsilon, i + 1

    cost_to_go, x_nominal, l, epsilon, i = jax.lax.while_loop(
        cond,
        body,
        (jnp.inf, x_nominal, l, epsilon / lam, 0),
    )
    return x_nominal, l, cost_to_go


def _logdet(m):
    sign, logabsdet = jnp.linalg.slogdet(m)
    return sign * logabsdet
