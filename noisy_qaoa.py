"""
QAOA MaxCut on a 3-regular graph: ideal vs. two implmentation with noisy 2-qubit gates: using a noise model with aer and an explicit dephasing quantum channel
QAOA convention: C = (1/2) sum_(u,v) (I - Z_u Z_v), making each e^{-i gamma C_uv} is CX(u,v) RZ(-gamma,v) CX(u,v) up to a global phase.
"""

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, phase_damping_error

import sys

def maxcut_scores(graph):
    """cost[x] = number of cut edges in computational basis state |x>."""
    n = graph.number_of_nodes()
    edges = list(graph.edges())
    return np.array(
        [
            sum(((x >> u) & 1) != ((x >> v) & 1) for u, v in edges)
            for x in range(1 << n)
        ],
        dtype=float,
    )


def two_qubit_dephasing(lam):
    """independent one-qubit phase damping on both participating qubits."""
    if not 0 <= lam <= 1:
        raise ValueError("lam must lie in [0, 1]")
    one = phase_damping_error(lam)
    return one.tensor(one)


def qaoa_circuit(graph, gammas, betas, after_cx=None):
    """construct QAOA from H, CX, RZ and RX; optionally insert noise after CX."""
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    n = graph.number_of_nodes()
    if set(graph.nodes()) != set(range(n)):
        raise ValueError("Graph nodes must be labeled 0, 1, ..., n-1")

    circuit = QuantumCircuit(n)
    circuit.h(range(n))  # |+>^n

    for gamma, beta in zip(gammas, betas):
        # cost layer: e^{-i gamma C_uv}, apart from a global phase.
        for u, v in sorted(tuple(sorted(edge)) for edge in graph.edges()):
            circuit.cx(u, v)
            if after_cx is not None:
                circuit.append(after_cx, [u, v])

            circuit.rz(-float(gamma), v)

            circuit.cx(u, v)
            if after_cx is not None:
                circuit.append(after_cx, [u, v])

        # mixer: e^{-i beta sum_i X_i}
        for q in range(n):
            circuit.rx(2 * float(beta), q)

    return circuit


def density_matrix(circuit, simulator):
    """return exact final density matrix from an aer density-matrix simulation."""
    run_circuit = circuit.copy()
    run_circuit.save_density_matrix(label="rho")
    saved = simulator.run(run_circuit).result().data(0)["rho"]
    return np.asarray(saved if isinstance(saved, np.ndarray) else saved.data)


def expected_cut(rho, cut_scores):
    """tr(C rho) because C is diagonal in the computational basis."""
    probabilities = np.real(np.diag(rho))
    return float(np.dot(cut_scores, probabilities))


def main():
    n = 8  # n must be even for 3-regular graphs
    depth = 1
    lam = 0.08
    graph = nx.random_regular_graph(d=3, n=n, seed=7)
    scores = maxcut_scores(graph)
    print("edges:", sorted(tuple(sorted(e)) for e in graph.edges()))
    print("exact maximum cut size:", int(scores.max()))


    # construct the same two-qubit error for two approaches: using an aer noise model and composing each CX with an explcit quantum channel 
    err2 = two_qubit_dephasing(lam)

    # first approach:  aer attaches error after every CX instruction.
    model = NoiseModel()
    model.add_all_qubit_quantum_error(err2, ["cx"])
    sim_model = AerSimulator(method="density_matrix", noise_model=model)

    # second: The circuit explicitly contains a noisy channel after every CX.
    channel_instruction = err2.to_instruction()
    sim_explicit = AerSimulator(method="density_matrix")  # no noise model
    sim_ideal = AerSimulator(method="density_matrix")

    # functions for computing variational objectives.
    def make_circuit(theta, explicit=False):
        return qaoa_circuit(
            graph,
            gammas=theta[:depth],
            betas=theta[depth:],
            after_cx=channel_instruction if explicit else None,
        )

    def objective(theta, simulator, explicit=False):
        circuit = make_circuit(theta, explicit=explicit)
        return -expected_cut(density_matrix(circuit, simulator), scores)

    theta0 = np.concatenate([np.full(depth, 0.6), np.full(depth, 0.35)])

    # verifying that the two methods agree for exactly the same angles.
    rho_a = density_matrix(make_circuit(theta0), sim_model)
    rho_b = density_matrix(make_circuit(theta0, explicit=True), sim_explicit)
    discrepancy = np.max(np.abs(rho_a - rho_b))
    print("Max |rho_noise_model - rho_explicit|:", discrepancy)
    np.testing.assert_allclose(rho_a, rho_b, rtol=1e-9, atol=1e-9)

    print("\nAt initial parameters:")
    print("Ideal expected cut:   ", -objective(theta0, sim_ideal))
    print("Noise-model cut:      ", -objective(theta0, sim_model))
    print("Explicit-channel cut: ", -objective(theta0, sim_explicit, True))

    # variationally optimize the noisy QAOA expectation value.
    result = minimize(
        lambda theta: objective(theta, sim_model),
        theta0,
        method="COBYLA",
        options={"maxiter": 1000, "tol": 1e-5},
    )
    theta_star = result.x
    print("\nNoisy optimized parameters [gammas..., betas...]:", theta_star)
    print("Ideal circuit at these parameters:", -objective(theta_star, sim_ideal))
    print("Noisy model expected cut:       ", -objective(theta_star, sim_model))
    print("Explicit-channel expected cut:  ", -objective(theta_star, sim_explicit, True))

    # Optional: sample a noisy QAOA circuit, as on an actual device.
    measured = make_circuit(theta_star)
    measured.measure_all()
    counts = sim_model.run(measured, shots=4096, seed_simulator=123).result().get_counts()
    print("\nFive most frequent measured bit strings (bit n-1 ... bit 0):")
    for bits, count in sorted(counts.items(), key=lambda item: -item[1])[:5]:
        print(f"  {bits}  count={count:4d}  cut={int(scores[int(bits, 2)])}")


if __name__ == "__main__":
    main()
