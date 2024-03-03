
from jax import numpy as np, random, jit, value_and_grad
from jax.example_libraries import optimizers
from typing import Tuple, Any, Callable


def __jit_loss_fn(
        statistic_fn: Callable[[np.array], np.array]
) -> Callable[[np.array, np.array], np.array]:
    # @jit
    def compute_loss_fn(
            synthetic_dataset: np.array, target_statistics: np.array
    ) -> np.array:
        return np.linalg.norm(statistic_fn(synthetic_dataset) - target_statistics
                              )

    return compute_loss_fn

def __get_update_function(
            D_prime,
            learning_rate: float,
            optimizer: Callable[..., optimizers.Optimizer],
            loss_fn: Callable[[np.array, np.array], np.array],) -> Tuple[Callable[[np.array, np.array], np.array], np.array]:
        opt_init, opt_update, get_params = optimizer(learning_rate)
        opt_state = opt_init(D_prime)

    #    @jit
        def update(synthetic_dataset, target_statistics, state):
            """Compute the gradient and update the parameters"""
            value, grads = value_and_grad(loss_fn)(synthetic_dataset, target_statistics)
            state = opt_update(0, grads, state)
            return get_params(state), state, value

        return update, opt_state
