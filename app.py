from flask import Flask, request, send_from_directory, jsonify
import numpy as np
import os
from simulations import simulate, visualize, reward_calc, get_validator_reward
from algorithm import calculate_equilibrium_algorithm, get_results_str
import flask_caching

app = Flask(__name__)

config = {
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": ".cache",
    "CACHE_THRESHOLD": 10000,
    "CACHE_DEFAULT_TIMEOUT": 0,
}
app.config.from_mapping(config)
cache = flask_caching.Cache(app)


@app.route("/")
def index():
    return send_from_directory(".", "interface.html")


@app.route("/mygraph.html")
def graph():
    return send_from_directory(".", "mygraph.html")


@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    data = request.get_json()
    v = int(data["v"])
    s = int(data["s"])
    sigma = np.array(data["sigma"], dtype=float)
    theta = np.zeros(s)
    pi = np.zeros(s)
    r = np.array(data["r"], dtype=float)
    deg = np.array(data["deg"], dtype=float)
    n = int(data.get("n", 200))
    epsilon = float(data.get("epsilon", 1e-3))
    split = bool(data.get("split", False))
    optimizer_type = data.get("optimizer_type", "differential_evolution")
    print(optimizer_type)

    # Run simulation
    w, split_alloc = simulate_cache(
        v,
        s,
        sigma,
        theta,
        pi,
        r,
        deg,
        n=n,
        epsilon=epsilon,
        split=split,
        optimizer_type=optimizer_type,
    )
    # print(split_alloc)
    # print(w)
    reward = reward_calc(
        v, s, sigma, r, deg, w, split=split, split_allocation=split_alloc
    )
    validator_rewards = get_validator_reward(reward, v, len(np.unique(deg)))
    # Visualize
    visualize(
        v,
        s,
        sigma,
        r,
        deg,
        w,
        split=split,
        split_allocation=split_alloc,
        filename="mygraph.html",
    )

    # run algorithm
    algo_splits, algo_allocations = calculate_equilibrium_algorithm(deg, r, sigma)
    algo_results = get_results_str(deg, r, sigma, algo_splits, algo_allocations)

    return jsonify(
        {
            "status": "ok",
            "split_allocation": split_alloc.tolist(),
            "w": w.tolist(),
            "split_restaking_deg": (w.sum(axis=1) / split_alloc.reshape(-1)).tolist(),
            "reward": reward.tolist(),
            "validator_rewards": validator_rewards.tolist(),
            "algo_results": algo_results,
        }
    )


@cache.memoize()
def simulate_cache(
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
    split_init: np.ndarray | None = None,
    split: bool = False,
    optimizer_type: str = "differential_evolution",
):
    return simulate(
        v,
        s,
        sigma,
        theta,
        pi,
        r,
        deg,
        n=n,
        epsilon=epsilon,
        init=init,
        split_init=split_init,
        split=split,
        optimizer_type=optimizer_type,
    )


if __name__ == "__main__":
    app.run(debug=True)
