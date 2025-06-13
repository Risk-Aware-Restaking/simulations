from scipy import optimize
import numpy as np
from pyvis.network import Network


def reward_calc(
    v: int,
    s: int,
    sigma: np.ndarray,
    r: np.ndarray,
    deg: np.ndarray,
    w: np.ndarray,
    validators: range | None = None,
    split: bool = False,
    split_allocation: np.ndarray = np.zeros((0, 0)),
) -> np.ndarray:

    assert len(sigma) == v, "Stake must have length equal to number of validators."
    assert len(r) == s, "Reward must have length equal to number of services."
    assert len(deg) == s, "Degree must have length equal to number of services."
    if split:
        num_splits = len(np.unique(deg))
        assert split_allocation.shape == (
            v,
            num_splits,
        ), "Split allocations must be equal to (v, num_splits) or not defined"
    else:
        num_splits = 1

    assert np.shape(w) == (
        v * num_splits,
        s,
    ), "Weights must be a (v*num_splits) x s matrix."

    if validators is None:
        validators = range(v * num_splits)
    rewards = np.zeros(v * num_splits)
    row_sum = np.sum(w, axis=1)
    col_sum = np.sum(w, axis=0)
    for service in range(s):
        if col_sum[service] == 0:
            continue
        for validator in validators:
            if split:
                stake = split_allocation[
                    validator // num_splits, validator % num_splits
                ]
            else:
                stake = sigma[validator // num_splits]
            if stake == 0 or row_sum[validator] / stake > deg[service]:
                continue
            rewards[validator] += w[validator, service] / col_sum[service] * r[service]
    return rewards


def stake_ratio(v: int, s: int, r: np.ndarray, w: np.ndarray) -> np.ndarray:
    assert len(r) == s, "Reward must have length equal to number of services."

    stake_ratios = np.repeat(0.0, s)
    col_sum = np.sum(w, axis=0)
    for service in range(s):
        stake_ratios[service] = col_sum[service] / r[service]
    return stake_ratios


# sigma is stake per validator, r is reward per service
def original_w(v: int, s: int, sigma: np.ndarray, r: np.ndarray, d: int) -> np.ndarray:
    assert len(sigma) == v, "Stake must have length equal to number of validators."
    assert len(r) == s, "Reward must have length equal to number of services."
    assert d > 0, "d must be greater than 0."

    w = np.array([[sigma[i]] * s for i in range(v)])
    r_sum = np.sum(r)
    for service in range(s):
        for validator in range(v):
            w[validator, service] = d * (r[service] / r_sum) * sigma[validator]
    return w


def visualize(
    v: int,
    s: int,
    sigma: np.ndarray,
    r: np.ndarray,
    deg: np.ndarray,
    w: np.ndarray,
    filename: str = "mygraph.html",
):
    net = Network()
    net.add_nodes(
        ["v" + str(i) for i in range(v)],
        label=["v" + str(i) + " stake: " + str(sigma[i]) for i in range(v)],
        x=[0] * v,
        y=[100 * i for i in range(v)],
    )
    net.add_nodes(
        ["s" + str(i) for i in range(s)],
        label=[
            "s" + str(i) + " r: " + str(r[i]) + " deg: " + str(deg[i]) for i in range(s)
        ],
        x=[200] * s,
        y=[100 * i for i in range(s)],
    )
    for i in range(v):
        for j in range(s):
            if w[i][j] > 0:
                net.add_edge("v" + str(i), "s" + str(j), weight=5, title=str(w[i][j]))
    net.toggle_physics(False)
    net.show(filename, notebook=False)


# v = number of validators
# s = number of services
# sigma = stake per validator
# theta = threshold per service
# pi = prize per service
# r = reward per service
# deg = restaking degree limit per service
# n = number of iterations
# epsilon = convergence threshold
# init = initial allocation (optional)
# split = whether we use splitting or not
def simulate(
    v: int,
    s: int,
    sigma: np.ndarray,
    theta: np.ndarray,
    pi: np.ndarray,
    r: np.ndarray,
    deg: np.ndarray,
    n: int = 1000,
    epsilon: float = 1e-3,
    init: np.ndarray | None = None,
    split: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    assert len(sigma) == v, "Stake must have length equal to number of validators."
    assert len(r) == s, "Reward must have length equal to number of services."
    assert len(deg) == s, "Degree must have length equal to number of services."
    assert len(theta) == s, "Threshold must have length equal to number of services."
    assert len(pi) == s, "Prize must have length equal to number of services."

    print(f"Simulating with {v} validators, {s} contracts.")
    print(
        f"Stake: {sigma}, Thresholds: {theta}, Prizes: {pi}, Rewards: {r}, Degrees: {deg}"
    )

    if split:
        print(
            "Splitting is enabled. Each validator can split their stake across services according to the number of unique degrees."
        )
        num_splits = len(np.unique(deg))
    else:
        num_splits = 1

    # allocations
    if init is None:
        w = np.array(
            [
                [sigma[i // num_splits] / num_splits / s] * s
                for i in range(v * num_splits)
            ]
        )
    else:
        w = init

    assert w.shape == (
        v * num_splits,
        s,
    ), "Weights must be a (v*num_splits) x s matrix."

    equilibrium_count = 0

    all_split_allocs = np.array(
        [[sigma[i // num_splits] / num_splits] * num_splits for i in range(v)]
    )
    assert all_split_allocs.shape == (v, num_splits)

    for i in range(n):
        if i % 100 == 0:
            print(f"Iteration {i+1}/{n}")
        # validator i % v makes an allocation based on the current state
        curr_v = i % v
        curr_stake = sigma[curr_v]

        # l = number of splits
        # x[0] to x[l-1] are allocations for each split
        # x[l] to x[l+s-1] - stake of validator curr_v split 0 for all services
        # for each split x[0] <= split[0]
        # x[0] -> split 0 service 0 x[1] + split 0 service 1 ...
        # x[2] -> split 1 service 0 x[3] split 1 service 1
        def utility(x):
            w_copy = np.copy(w)
            if split:
                assert (
                    len(x) == num_splits + num_splits * s
                ), "x should have length equal to number of splits + number of splits * services."

                split_allocation = np.array(x[0:num_splits])
                temp_split_allocs = np.copy(all_split_allocs)
                temp_split_allocs[curr_v] = split_allocation

                validator_splits = np.array(x[num_splits:]).reshape(num_splits, s)
                if not np.isclose(np.sum(split_allocation), curr_stake, rtol=1e-1):
                    return 1e9
                validator_row_sums = np.sum(validator_splits, axis=1)
                assert len(validator_row_sums) == num_splits
                for i in range(num_splits):
                    if np.any(validator_splits[i] > split_allocation[i]):
                        return 1e9
                    restake = validator_row_sums[i] / split_allocation[i]
                    if restake > deg[i]:
                        return 1e9
                w_copy[curr_v * num_splits : curr_v * num_splits + num_splits, :] = (
                    validator_splits
                )

                curr_split_vals = range(
                    curr_v * num_splits, curr_v * num_splits + num_splits
                )

                rewards = reward_calc(
                    v,
                    s,
                    sigma,
                    r,
                    deg,
                    w_copy,
                    validators=curr_split_vals,
                    split=split,
                    split_allocation=temp_split_allocs,
                )

                for j in range(len(rewards)):
                    if j not in curr_split_vals:
                        assert (
                            rewards[j] == 0
                        ), "Rewards for non-active validators should be 0."

                return -1 * np.sum(rewards)
            else:
                assert len(x) == s, "x should have length equal to number of services."

                w_copy[curr_v] = np.array(x)
                return (
                    -1
                    * reward_calc(
                        v,
                        s,
                        sigma,
                        r,
                        deg,
                        w_copy,
                        validators=range(curr_v, curr_v + 1),
                    )[curr_v]
                )

        if split:
            bounds = [(0, curr_stake)] * (num_splits + num_splits * s)
        else:
            bounds = [(0, curr_stake)] * s

        results = optimize.dual_annealing(
            utility,
            bounds,
            x0=np.concatenate(
                (
                    all_split_allocs[curr_v],
                    w[
                        curr_v * num_splits : curr_v * num_splits + num_splits, :
                    ].flatten(),
                )
            ),
        )

        if not split:
            res = np.array(results.x)
            assert (
                len(res) == s
            ), "Results should have length equal to number of services."

            deltas = res - w[curr_v]
            w[curr_v] = res
        else:
            assert (
                len(results.x) == num_splits + num_splits * s
            ), "Results should have length equal to number of splits + number of splits * services."
            res = np.array(results.x[num_splits:]).reshape(num_splits, s)
            deltas = res - w[curr_v * num_splits : curr_v * num_splits + num_splits, :]
            w[curr_v * num_splits : curr_v * num_splits + num_splits, :] = res

            if not np.isclose(np.sum(results.x[:num_splits]), curr_stake, rtol=1e-1):
                print(w)
                print(results)
                raise AssertionError(
                    f"Warning: Split allocation for validator {curr_v} does not sum to stake."
                )
            all_split_allocs[curr_v] = np.array(results.x[:num_splits])

        if np.all(np.abs(deltas) < epsilon):
            equilibrium_count += 1
        else:
            equilibrium_count = 0

        if equilibrium_count >= v:
            print(f"Equilibrium reached after {i+1} iterations.")
            break

    return w, all_split_allocs


if __name__ == "__main__":
    print("Starting simulations...")

    # validators
    v = 2

    # contracts
    s = 3

    # stake per validator
    sigma = np.array([11, 11])

    # threshold per service
    theta = np.array([0.25, 0.5, 0.6])

    # prize per service
    pi = np.array([2, 1, 3])

    # reward per service
    r = np.array([3, 1, 3])

    # degree per service
    deg = np.array([1, 1.5, 1.5])

    # whether we use splitting or not
    split = True

    # maximize reward while minimizing stake
    w, split_alloc = simulate(v, s, sigma, theta, pi, r, deg, n=200, split=split)
    print("split allocs", split_alloc)
    print(w)
    print(
        "reward",
        reward_calc(
            v,
            s,
            sigma,
            r,
            deg,
            w,
            split=split,
            split_allocation=split_alloc,
        ),
    )
    print("stake ratio", stake_ratio(v, s, r, w))
    # with open("w.txt", "w") as f:
    #     print(w, file=f)

    # # original should be equal to above if degree is constant
    # w_2 = original_w(v, s, sigma, r, deg[0])
    # print(w_2)
    # print(reward_calc(v, s, sigma, r, deg, w_2))
    # print(stake_ratio(v, s, r, w_2))

    # w_3 = w
    # w_3[0] = w_2[0]
    # print(w_3)
    # print(reward_calc(v, s, sigma, r, deg, w_3))
    # visualize(v, s, sigma, r, deg, w_2)
