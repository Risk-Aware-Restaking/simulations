from scipy import optimize
import numpy as np
from pyvis.network import Network

def reward_calc(v, s, sigma, r, deg, w):
    rewards = [0] * v
    for service in range(s):
        for validator in range(v):
            if sum(w[validator])/sigma[validator] > deg[service]:
                continue
            rewards[validator] += w[validator][service]/sum(w[i][service] for i in range(v)) * r[service]
    return rewards

# sigma is stake per validator, r is reward per service
def original_w(v, s, sigma, r, d):
    w = [[sigma[i]] * s for i in range(v)]
    r_sum = sum(r)
    for service in range(s):
        for validator in range(v):
            w[validator][service] = d*(r[service]/r_sum)*sigma[validator]
    return w

def visualize(v, s, sigma, r, deg, w):
    net = Network()
    net.add_node(["validator " + str(i) for i in range(v)], label=[])
    net.add_node(["service " + str(i) for i in range(s)], label=[])

    pass

# n = number of iterations
def simulate(v, s, sigma, theta, pi, r, deg, n=1000, epsilon=1e-3):
    print(f"Simulating with {v} validators, {s} contracts.")
    print(f"Stake: {sigma}, Thresholds: {theta}, Prizes: {pi}, Rewards: {r}, Degrees: {deg}")

    # allocations
    w = [[sigma[i] / s] * s for i in range(v)]

    equilibrium_count = 0

    for i in range(n):
        if i % 100 == 0:
            print(f"Iteration {i+1}/{n}")
        # validator i % v makes an allocation based on the current state
        curr_v = i % v
        curr_stake = sigma[curr_v]
    
        # utility function - x[0], x[1], ... are stake of validator curr_v per service
        def utility(x):
            w[curr_v] = x
            return -1 * reward_calc(v, s, sigma, r, deg, w)[curr_v]

        bounds = [(0, curr_stake)] * v

        results = optimize.dual_annealing(utility, bounds)
        if all(abs(w[curr_v][j] - results.x[j]) < epsilon for j in range(s)):
            equilibrium_count += 1
        else:
            equilibrium_count = 0

        if equilibrium_count >= v:
            print(f"Equilibrium reached after {i+1} iterations.")
            break

        w[curr_v] = results.x
    
    return w


if __name__  == "__main__":
    print("Starting simulations...")

    # validators
    v = 3

    # contracts
    s = 3

    # stake per validator
    sigma = [4, 5, 2]
    
    # threshold per service
    theta = [0.25, 0.5, 0.6]

    # prize per service
    pi =[2,1,3]
    
    # reward per service
    r = [3, 3, 3]

    # degree per service
    deg = [2, 2, 2]

    # maximize reward while minimizing stake
    w = simulate(v, s, sigma, theta, pi, r, deg, 2000)
    print(w)
    print(reward_calc(v, s, sigma, r, deg, w))
    # original should be equal to above if degree is constant
    w_2 = original_w(v, s, sigma, r, deg[0])
    print(w_2)
    print(reward_calc(v, s, sigma, r, deg, w_2))

    w_3 = w
    w_3[0] = w_2[0]
    print(w_3)
    print(reward_calc(v, s, sigma, r, deg, w_3))
