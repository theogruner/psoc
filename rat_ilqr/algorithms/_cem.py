import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct


@struct.dataclass
class CEMState:
    mu: float
    sigma: float

    # hyperparameters
    nb_samples: int = struct.field(pytree_node=False)
    nb_elite_samples: int = struct.field(pytree_node=False)
    max_cost: float = struct.field(pytree_node=False)

    @classmethod
    def create(cls, mu, sigma, nb_samples, nb_elite_samples=None, max_cost=jnp.inf):
        if nb_elite_samples is None:
            nb_elite_samples = nb_samples

        return cls(mu, sigma, nb_samples, nb_elite_samples, max_cost)


def cem(key, state, make_cem_problem, nb_iter: int):
    def body(state, val):
        key = val
        state, cost = _cem(key, state, make_cem_problem)
        return state, cost

    for i in range(nb_iter):
        sub_key, key = jr.split(key)
        state, _ = body(state, sub_key)

    return state, None


def _cem(key, state, make_cem_problem):
    # sample_key = jr.split(key, 1)
    samples, costs, _ = _sample_valid_parameters(
        key,
        state.mu,
        state.sigma,
        make_cem_problem,
        state.nb_samples,
        state.max_cost,
        state.nb_elite_samples,
    )

    elite_samples = _select_elite(samples, costs, state.nb_elite_samples)

    mu, sigma = _fit_gaussian(elite_samples)

    state = state.replace(mu=mu, sigma=sigma)

    return state, None


def _sample_valid_parameters(
    key,
    mu,
    sigma,
    make_cem_problem,
    nb_samples,
    max_cost,
    nb_elite_samples,
    max_tries=1000,
):
    cost_fn = make_cem_problem()

    init_samples = jnp.zeros((nb_samples,))
    init_costs = jnp.inf * jnp.ones((nb_samples,))
    init_acceptance_mask = jnp.zeros(nb_samples, dtype=jnp.bool_)

    def sampler(rng):
        return mu + jax.random.normal(rng) * sigma

    def cond(vals):
        _, i, acceptance_mask, _, _ = vals
        return jnp.logical_and(
            jnp.sum(acceptance_mask) < nb_elite_samples, i != max_tries
        )

    def resampling(vals):
        key, i, acceptance_mask, samples, costs = vals
        sub_key, key = jr.split(key)

        sub_keys = jr.split(sub_key, nb_samples)
        proposals = jax.vmap(sampler)(sub_keys)

        proposal_costs = jax.vmap(cost_fn)(proposals)

        acc_params = jnp.logical_and(proposal_costs < max_cost, proposals > 0.0)
        new_params_mask = jnp.logical_and(acc_params, jnp.logical_not(acceptance_mask))
        acceptance_mask = jnp.logical_or(acceptance_mask, new_params_mask)

        samples = jnp.where(new_params_mask, proposals, samples)
        costs = jnp.where(new_params_mask, proposal_costs, costs)

        return key, i + 1, acceptance_mask, samples, costs

    _, tries, acceptance_mask, samples, costs = jax.lax.while_loop(
        cond, resampling, (key, 0, init_acceptance_mask, init_samples, init_costs)
    )
    return samples, costs, acceptance_mask


def _select_elite(samples, costs, nb_elite_samples):
    # Sort samples by cost and return nb_elite_samples
    idx = jnp.argsort(costs)
    samples = samples.at[idx].get()
    return samples[:nb_elite_samples]


def _fit_gaussian(samples):
    mu = jnp.mean(samples)
    sigma = jnp.std(samples)
    return mu, sigma
