from flask import Flask, request, send_from_directory, jsonify
import numpy as np
import os
from simulations import simulate, visualize

app = Flask(__name__)


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
    theta = np.array(data["theta"], dtype=float)
    pi = np.array(data["pi"], dtype=float)
    r = np.array(data["r"], dtype=float)
    deg = np.array(data["deg"], dtype=float)
    n = int(data.get("n", 200))
    epsilon = float(data.get("epsilon", 1e-3))
    split = bool(data.get("split", False))

    # Run simulation
    w, split_alloc = simulate(
        v, s, sigma, theta, pi, r, deg, n=n, epsilon=epsilon, split=split
    )
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
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
