from scipy import optimize
import numpy as np
from pyvis.network import Network


def reward_calc(v, s, sigma, r, deg, w, validators=None, split=False):
    if split:
        num_splits = len(set(deg))
    else:
        num_splits = 1
    if validators is None:
        validators = range(v * num_splits)
    rewards = [0] * (v * num_splits)
    for service in range(s):
        for validator in validators:
            if sum(w[validator]) / sigma[validator // num_splits] > deg[service]:
                continue
            rewards[validator] += (
                w[validator][service] / sum(row[service] for row in w) * r[service]
            )
    return rewards


def stake_ratio(v, s, r, w):
    stake_ratios = [0] * s
    for service in range(s):
        stake_ratios[service] = sum(w[i][service] for i in range(v)) / r[service]
    return stake_ratios


# sigma is stake per validator, r is reward per service
def original_w(v, s, sigma, r, d):
    w = [[sigma[i]] * s for i in range(v)]
    r_sum = sum(r)
    for service in range(s):
        for validator in range(v):
            w[validator][service] = d * (r[service] / r_sum) * sigma[validator]
    return w


def visualize(v, s, sigma, r, deg, w, filename="mygraph.html"):
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


# n = number of iterations
def simulate(
    v, s, sigma, theta, pi, r, deg, n=1000, epsilon=1e-3, init=None, split=False
):
    print(f"Simulating with {v} validators, {s} contracts.")
    print(
        f"Stake: {sigma}, Thresholds: {theta}, Prizes: {pi}, Rewards: {r}, Degrees: {deg}"
    )

    if split:
        print(
            "Splitting is enabled. Each validator can split their stake across services."
        )
        num_splits = len(set(deg))
    else:
        num_splits = 1

    # allocations
    if init is None:
        w = [[sigma[i // num_splits] / s] * s for i in range(v * num_splits)]
    else:
        w = init

    equilibrium_count = 0

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
            if split:
                if sum(x[0:num_splits]) > curr_stake:
                    return 0
                for i in range(num_splits, len(x)):
                    if x[i] > x[(i - num_splits) // s]:
                        return 0

                for j in range(num_splits):
                    w[curr_v * num_splits + j] = x[
                        num_splits + j * s : num_splits + j * s + s
                    ]
                return -1 * sum(
                    reward_calc(
                        v,
                        s,
                        sigma,
                        r,
                        deg,
                        w,
                        validators=range(
                            curr_v * num_splits, curr_v * num_splits + num_splits
                        ),
                        split=split,
                    )
                )
            else:
                w[curr_v] = x
                return (
                    -1
                    * reward_calc(v, s, sigma, r, deg, w, range(curr_v, curr_v + 1))[
                        curr_v
                    ]
                )

        bounds = [(0, curr_stake)] * (num_splits + num_splits * s)

        results = optimize.dual_annealing(utility, bounds)

        if not split:
            if all(abs(w[curr_v][j] - results.x[j]) < epsilon for j in range(s)):
                equilibrium_count += 1
            else:
                equilibrium_count = 0

            if equilibrium_count >= v:
                print(f"Equilibrium reached after {i+1} iterations.")
                break
            w[curr_v] = results.x
        else:
            deltas = []
            for j in range(num_splits):
                delta = (
                    results.x[num_splits + j * s : num_splits + j * s + s]
                    - w[curr_v * num_splits + j]
                )

                w[curr_v * num_splits + j] += delta

                deltas.append(sum(delta))

            if all([abs(delta) < epsilon for delta in deltas]):
                equilibrium_count += 1
            else:
                equilibrium_count = 0
            if equilibrium_count >= v:
                print(f"Equilibrium reached after {i+1} iterations.")
                break

    return w


if __name__ == "__main__":
    print("Starting simulations...")

    # validators
    v = 3

    # contracts
    s = 3

    # stake per validator
    sigma = [10, 10, 10]

    # threshold per service
    theta = [0.25, 0.5, 0.6]

    # prize per service
    pi = [2, 1, 3]

    # reward per service
    r = [10, 10, 10]

    # degree per service
    deg = [2, 3, 3]

    # whether we use splitting or not
    split = True

    # maximize reward while minimizing stake
    w = simulate(v, s, sigma, theta, pi, r, deg, n=1000, split=split)
    print(w)
    print(reward_calc(v, s, sigma, r, deg, w, split=split))
    print(stake_ratio(v, s, r, w))
    # with open("w.txt", "w") as f:
    #     print(w, file=f)

    # # original should be equal to above if degree is constant
    # w_2 = original_w(v, s, sigma, r, deg[0])
    # print(w_2)
    # print(reward_calc(v, s, sigma, r, deg, w_2))

    # w_3 = w
    # w_3[0] = w_2[0]
    # print(w_3)
    # print(reward_calc(v, s, sigma, r, deg, w_3))
    # visualize(v, s, sigma, r, deg, w_2)

    # [array([5.67617361, 4.30650524, 5.67206526, 5.6652727 , 5.67019663, 4.29012822, 4.29811622, 4.30307471]),
    # array([4.29012822, 4.29167424, 4.30307471]),
    # array([5.36159713, 4.63826211, 5.36158598, 5.36135594, 5.36151405, 4.63808794, 4.63819042, 4.6382621 ]),
    # array([5.85325838, 5.85080777, 5.84936051]),
    # array([5.36158598, 5.36135594, 5.36151405]),
    # array([4.63808794, 4.63819042, 4.63220156])]
