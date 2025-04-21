#pip install types-pkg-resources==0.1.3
#pip install git+https://github.com/jpmorganchase/QOKit.git@830adaf9b8e6ac43d4b100a83d0e1e47a51f9e6b
#pip install "docplex<=2.24.232" "qiskit[visualization]<2" qiskit-optimization qiskit-aer mqt.bench "qiskit-ibm-runtime<0.37.0"
#pip install numpy==1.24.4

import numpy as np
import pandas as pd
from mqt.bench import get_benchmark
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from IPython.display import display

Quantinuum_H2_32=[
 (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
 (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
 (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
 (3, 4), (3, 5), (3, 6), (3, 7),
 (4, 5), (4, 6), (4, 7),
 (5, 6), (5, 7),
 (6, 7),
 (7, 8),  # inter-group connection

 (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15),
 (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (9, 15),
 (10, 11), (10, 12), (10, 13), (10, 14), (10, 15),
 (11, 12), (11, 13), (11, 14), (11, 15),
 (12, 13), (12, 14), (12, 15),
 (13, 14), (13, 15),
 (14, 15),
 (15, 16),  # inter-group connection

 (16, 17), (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23),
 (17, 18), (17, 19), (17, 20), (17, 21), (17, 22), (17, 23),
 (18, 19), (18, 20), (18, 21), (18, 22), (18, 23),
 (19, 20), (19, 21), (19, 22), (19, 23),
 (20, 21), (20, 22), (20, 23),
 (21, 22), (21, 23),
 (22, 23),
 (23, 24),  # inter-group connection

 (24, 25), (24, 26), (24, 27), (24, 28), (24, 29), (24, 30), (24, 31),
 (25, 26), (25, 27), (25, 28), (25, 29), (25, 30), (25, 31),
 (26, 27), (26, 28), (26, 29), (26, 30), (26, 31),
 (27, 28), (27, 29), (27, 30), (27, 31),
 (28, 29), (28, 30), (28, 31),
 (29, 30), (29, 31),
 (30, 31)
]

IonQ_25=[
 (0, 1), (0, 2), (0, 3), (0, 4),
 (1, 2), (1, 3), (1, 4),
 (2, 3), (2, 4),
 (3, 4),
 (4, 5),

 (5, 6), (5, 7), (5, 8), (5, 9),
 (6, 7), (6, 8), (6, 9),
 (7, 8), (7, 9),
 (8, 9),
 (9, 10),

 (10, 11), (10, 12), (10, 13), (10, 14),
 (11, 12), (11, 13), (11, 14),
 (12, 13), (12, 14),
 (13, 14),
 (14, 15),

 (15, 16), (15, 17), (15, 18), (15, 19),
 (16, 17), (16, 18), (16, 19),
 (17, 18), (17, 19),
 (18, 19),
 (19, 20),

 (20, 21), (20, 22), (20, 23), (20, 24),
 (21, 22), (21, 23), (21, 24),
 (22, 23), (22, 24),
 (23, 24)
]


edges_IBM_27 = [
    (0, 1), (1, 2), (2, 3),
    (3, 4), (4, 5), (5, 6),
    (6, 7), (7, 8), (8, 9),
    (1, 10), (3, 11), (5, 12), (7, 13), (9, 14),
    (10, 11), (11, 12), (12, 13), (13, 14),
    (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
    (15, 16), (16, 17), (17, 18), (18, 19),
    (15, 20), (16, 21), (17, 22), (18, 23), (19, 24),
    (20, 21), (21, 22), (22, 23), (23, 24),
    (21, 25), (23, 26)
]

platforms = {
    "IBM_Montreal": {
        "coupling_map": CouplingMap(edges_IBM_27),
        "basis_gates": ['rz', 'sx', 'x', 'cx', 'measure']
    },
    "IonQ_Aria": {
        "coupling_map": CouplingMap(IonQ_25),
        "basis_gates": ['u3', 'xx', 'cx', 'h', 'rx', 'rzz', 'cz', 'ry']
    },
    "Quantinuum_H2": {
        "coupling_map": CouplingMap(Quantinuum_H2_32),
        "basis_gates": ['u3', 'zz', 'cx', 'h', 'rx', 'rzz', 'cz', 'ry']
    }
}

benchmarks = ["ae", "graphstate", "portfolioqaoa", "portfoliovqe",
              "qaoa", "qft", "qnn", "vqe", "wstate"]
sizes = [5, 10, 15, 25, 35, 45]
results = []
def run_suite(size):
  for bench in benchmarks:
    if (size > 15): # told to elide these as they are extensively long
        if bench in ["portfolioqaoa", "portfoliovqe", "vqe", "qaoa"]:
            continue
    try:
        qc = get_benchmark(benchmark_name=bench, level="alg", circuit_size=size)
    except Exception as e:
        print(f"Failed to load {bench} with size {size}: {e}")
        continue

    for platform_name, platform in platforms.items():
        try:
            transpiled = transpile(
                qc,
                basis_gates=platform["basis_gates"],
                #coupling_map=platform["coupling_map"], disable for now tech-agnostic
                optimization_level=0
            )

            ops = transpiled.count_ops()
            swap_count = ops.get('swap', 0)
            depth = transpiled.depth()
            total_gates = np.sum(list(ops.values()))

            results.append({
                "Benchmark": bench,
                "Qubits": size,
                "Platform": platform_name,
                "Depth": depth,
                "Total Gates": total_gates,
                "SWAP Count": swap_count
            })
            print(f"Transpiled {bench} with size {size} for {platform_name} successfully.")

        except Exception as e:
            print(f"Error during transpilation for {platform_name}, {bench}, {size}: {e}")

# Create DataFrame
df = pd.DataFrame(results)

depth_table = df.pivot_table(index=["Benchmark", "Qubits"], columns="Platform", values="Depth")
depth_table = depth_table.sort_index()
print("=== Circuit Depth by Platform ===")
display(depth_table)

gate_count_table = df.pivot_table(index=["Benchmark", "Qubits"], columns="Platform", values="Total Gates")
gate_count_table = gate_count_table.sort_index()
print("\n=== Gate Count by Platform ===")
display(gate_count_table)