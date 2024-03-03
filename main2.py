from jax import random
from jax.example_libraries import optimizers

from privacy_budget_tracking import zCDPTracker
from relaxed_adaptive_projection import RAPConfiguration, RAP
from relaxed_adaptive_projection.constants import Norm, ProjectionInterval
from utils_data import data_sources, ohe_to_categorical
import numpy as np
import statistickway
import helpers,helpers2

# generate random k-way marginals
dataset = data_sources["adult"](False, "", False, 1000, 2)
D = np.asarray(dataset.get_dataset())
kway_attrs = dataset.randomKway(64, 3)
kway_compact_queries, _ = dataset.get_queries(kway_attrs)

k_way_queries, total_queries = dataset.get_queries(kway_attrs, N=-1)
k_way_queries = np.asarray(k_way_queries)
all_statistic_fn = statistickway.preserve_statistic(kway_compact_queries)
shape = (1000, 588)
key = random.PRNGKey(123)
D_prime = random.uniform(key=key, shape=shape)

#why is D_prime transformed onto [-1,1] while D is in {0,1}?
D_prime = 2 * (D_prime - 0.5)
initial_marginals = all_statistic_fn(D_prime)

# evaluate worse performing
tracker = zCDPTracker(
    0.15, 0, 1, 5, 1000
)
true_statistics = all_statistic_fn(D)
query_errs = np.abs(
    initial_marginals - true_statistics
)
sanitized_queries = np.array([])
rnm_selected_indices = tracker.select_noisy_q(
    query_errs, sanitized_queries, 5, tracker.budget_per_epoch() / 2
)
# apply noise on selected k-ways
rand_noise = zCDPTracker.generate_random_noise(key, rnm_selected_indices.shape)
target_statistics = true_statistics[rnm_selected_indices] + (0.004 * rand_noise)
stat_wo_noise = true_statistics[rnm_selected_indices]

# frank wolfe
sanitized_queries = np.asarray(
    np.append(sanitized_queries, rnm_selected_indices), dtype=np.int32
)

curr_queries = k_way_queries[sanitized_queries]
current_stat = initial_marginals[sanitized_queries]

fw_path = []
for T in range(5):  # condition

    max_dot = -1 * sanitized_queries.shape[0]
    direction = -1 * np.ones(sanitized_queries.shape)
    # find direction of movement
    for convex_vertex in helpers.get_enumeration(sanitized_queries.shape[0]):
        if np.dot(target_statistics - current_stat, convex_vertex - current_stat) > max_dot:
            max_dot = np.dot(target_statistics - current_stat, convex_vertex - current_stat)
            direction = convex_vertex
    fw_path.append(direction)
    # find optimal point in that direction
    optimal_point = helpers.closest_point_on_line(current_stat, direction - current_stat, target_statistics)



    # gradient descent towards optimal point to synthetic update data points------------------------------------------------>
    curr_statistic_fn = statistickway.preserve_subset_statistic(
        np.asarray(curr_queries)
    )
    loss_fn = helpers2.__jit_loss_fn(curr_statistic_fn)
    previous_loss = np.inf

    optimizer_learning_rate = (
        0.001
    )  # * 2 ** (-epoch)
    update, opt_state = helpers2.__get_update_function(
        D_prime, optimizer_learning_rate, optimizers.adam, loss_fn
    )

    for iteration in range(1000):

        # find fw vertex

        D_prime, opt_state, loss = update(
            D_prime, optimal_point, opt_state
        )
        if loss >= previous_loss - 0.001:
            break
        previous_loss = loss

    current_stat = synthetic_statistics = curr_statistic_fn(D_prime)
    statistics_l1 = np.mean(np.absolute(target_statistics - synthetic_statistics))

    statistics_max = np.amax(np.absolute(target_statistics - synthetic_statistics))

    all_synth_statistics = all_statistic_fn(D_prime)

    l2_error = np.linalg.norm(true_statistics - all_synth_statistics, ord=2)
    print(l2_error,fw_path)


