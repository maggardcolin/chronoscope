
import numpy as np
import math as m
from mqt.bench import get_benchmark
from tabulate import tabulate
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.z(2)

for inst, qargs, cargs in qc.data:
    qubit_indices = [q._index for q in qargs]
    print(f"{inst.name} on qubits {qubit_indices}")
    
    
