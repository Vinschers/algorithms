from typing import Callable, Any, Union, List
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["f", "max_iter"])
def _newton_raphson_jax(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x0: Any,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
) -> Union[Any, List[Any]]:

    x0_arr = jnp.array(x0, dtype=jnp.float32)
    original_shape = x0_arr.shape
    x = jnp.atleast_1d(x0_arr)

    def step(state):
        x, i, _, loss_history, converged = state

        f_val = f(x)
        f_val = jnp.atleast_1d(f_val)

        J = jax.jacrev(lambda arg: jnp.atleast_1d(f(arg)))(x)

        delta_x = jnp.linalg.solve(J, f_val)

        new_x = x - delta_x

        norm = jnp.linalg.norm(f_val)
        new_loss_history = loss_history.at[i].set(norm)

        converged = norm < tol

        return new_x, i + 1, norm, new_loss_history, converged

    def condition(state):
        _, i, norm, _, converged = state
        return (i < max_iter) & (~converged) & jnp.isfinite(norm)

    initial_loss_history = jnp.full((max_iter,), jnp.nan)
    init_state = (x, 0, jnp.finfo(jnp.float32).max, initial_loss_history, False)

    final_state = jax.lax.while_loop(condition, step, init_state)
    x, _, _, loss, converged = final_state
    x = x.reshape(original_shape)

    return x, loss, converged


def newton_raphson(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x0: Any,
    max_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-8,
    return_loss: bool = False,
    return_convergence: bool = False,
) -> Union[Any, List[Any]]:
    x_jax, loss_jax, conv_jax = _newton_raphson_jax(
        f, x0, max_iter, tol, epsilon
    )

    x = x_jax

    if jnp.ndim(x0) == 0:
        x = x.item()

    output = [x]

    if return_loss:
        valid_history = loss_jax[~jnp.isnan(loss_jax)]
        output.append(valid_history.tolist())

    if return_convergence:
        output.append(bool(conv_jax))

    if len(output) == 1:
        return output[0]

    return output
