import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mqt.bench import get_benchmark
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise.errors import ReadoutError

# -------------------- Noise Model -------------------- #
def create_noise_model(platform_name):
    fidelity_params = {
        "IonQ_Aria": {"1q": 0.999, "2q": 0.99, "ro": 0.9999},
        "Quantinuum_H2": {"1q": 0.9997, "2q": 0.98, "ro": 0.9999},
        "IBM_Montreal": {"1q": 0.999, "2q": 0.985, "ro": 0.97},
    }
    f = fidelity_params[platform_name]
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(1 - f["1q"], 1), ['rx', 'rz', 'h', 'x', 'u3'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(1 - f["2q"], 2), ['cx', 'cz', 'rzz', 'xx'])

    readout_error = ReadoutError([[f["ro"], 1 - f["ro"]], [1 - f["ro"], f["ro"]]])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model

# -------------------- Fidelity Utilities -------------------- #
def total_variation_distance(P, Q):
    all_keys = set(P) | set(Q)
    return 0.5 * sum(abs(P.get(k, 0) - Q.get(k, 0)) for k in all_keys)

def normalize(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

# -------------------- Simulation Runner -------------------- #
def run_fidelity_simulation(benchmark_name, qubit_size, platform_name, noise_model, basis_gates, coupling_map, shots=1000):
    qc = get_benchmark(benchmark_name=benchmark_name, level="alg", circuit_size=qubit_size)
    sim = AerSimulator(noise_model=noise_model)
    transpiled = transpile(qc, basis_gates=basis_gates, coupling_map=coupling_map, optimization_level=0)
    result = sim.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    norm_counts = normalize(counts)

    ideal_key = max(counts, key=counts.get)
    ideal_dist = {ideal_key: 1.0}

    tvd = total_variation_distance(norm_counts, ideal_dist)
    return tvd, counts

platforms = {
    "IBM_Montreal": {
        "coupling_map": CouplingMap(edges_IBM_27),
        "basis_gates": ['rx', 'rz', 'cx', 'swap']
    },
    "IonQ_Aria": {
        "coupling_map": CouplingMap(IonQ_25),
        "basis_gates": ['u3', 'xx', 'cx', 'h', 'rx', 'rzz', 'cz', 'ry', 'swap']
    },
    "Quantinuum_H2": {
        "coupling_map": CouplingMap(Quantinuum_H2_32),
        "basis_gates": ['u3', 'zz', 'cx', 'h', 'rx', 'rzz', 'cz', 'ry', 'swap']
    }
}

benchmarks = ["ae", "graphstate", "portfolioqaoa", "portfoliovqe",
              "qaoa", "qft", "qnn", "vqe", "wstate"]
qubit_sizes = [5, 10]
shots = 1000

# -------------------- Run All Simulations -------------------- #
simulation_results = []

for benchmark in benchmarks:
    for size in qubit_sizes:
        for platform_name, platform in platforms.items():
            try:
                noise_model = create_noise_model(platform_name)
                tvd, counts = run_fidelity_simulation(
                    benchmark, size, platform_name,
                    noise_model,
                    platform["basis_gates"],
                    platform["coupling_map"],
                    shots=shots
                )
                simulation_results.append({
                    "Benchmark": benchmark,
                    "Qubits": size,
                    "Platform": platform_name,
                    "TVD": tvd,
                    "Counts": counts
                })
                print(f"✅ {benchmark} | {size} qubits | {platform_name} -> TVD = {tvd:.4f}")
            except Exception as e:
                print(f"❌ Failed for {benchmark} | {size} | {platform_name}: {e}")

# -------------------- Visualization -------------------- #
results_df = pd.DataFrame([{
    "Benchmark": r["Benchmark"],
    "Qubits": r["Qubits"],
    "Platform": r["Platform"],
    "TVD": r["TVD"]
} for r in simulation_results])

# Table summary
print("\n=== TVD Results Summary ===")
print(results_df.pivot_table(index=["Benchmark", "Qubits"], columns="Platform", values="TVD"))

# Plot results
plt.figure(figsize=(12, 6))
for platform in platforms:
    subset = results_df[results_df["Platform"] == platform]
    for benchmark in benchmarks:
        bdata = subset[subset["Benchmark"] == benchmark]
        plt.plot(bdata["Qubits"], bdata["TVD"], marker='o', label=f"{platform} - {benchmark}")

plt.xlabel("Qubits")
plt.ylabel("Total Variation Distance")
plt.title("TVD vs Qubits for Each Platform and Benchmark")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()