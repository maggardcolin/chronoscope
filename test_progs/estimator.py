print("Entry")


import numpy as np
import math as m
from mqt.bench import get_benchmark
from tabulate import tabulate
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter


from qiskit.transpiler import CouplingMap

print("Imports complete...")


benchmark_list = [
    "ae", "graphstate", "qft", "qnn", "wstate"
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


def delayer_circuit():
    return 0

def critical_path_analyzer(circuit, qubit_count, single_gate_delay, double_gate_delay, readout_delay):
    """Determine the critical path (costliest qubit) of a quantum circuit for execution time
    
        Does not support circuits using more than 2-qubit gates
        Assumes multiple qubit operations are always the longer operation
    """
    
    #          1  2  $
    counts = [[0, 0, 0] for _ in range(qubit_count)]

    #Iterate over all gates
    for inst, qargs, cargs in circuit.data:
        qubit_indices = [q._index for q in qargs]
        if len(qubit_indices) > 1:                  #Case 2 qubit gate
            #equalize counts between the 2 qubits to max of the two (both single and double)
            #Then add one double to each
            
            for qi in qubit_indices:
                counts[qi][1] += 1
                counts[qi][2] =  counts[qi][0] * single_gate_delay + counts[qi][1] * double_gate_delay

            if counts[qubit_indices[0]][2] > counts[qubit_indices[1]][2]:
                counts[qubit_indices[1]][0] = counts[qubit_indices[0]][0]
                counts[qubit_indices[1]][1] = counts[qubit_indices[0]][1]
                counts[qubit_indices[1]][2] = counts[qubit_indices[0]][2]
            else:
                counts[qubit_indices[0]][0] = counts[qubit_indices[1]][0]
                counts[qubit_indices[0]][1] = counts[qubit_indices[1]][1]
                counts[qubit_indices[0]][2] = counts[qubit_indices[1]][2]
            

        else:                                       #Case 1 qubit gate
            #Add 1 to the single op count
            counts[qubit_indices[0]][0] += 1
            counts[qubit_indices[0]][2] += counts[qubit_indices[0]][0] * single_gate_delay + counts[qubit_indices[0]][1] * double_gate_delay
        
    max = 0
    for count in counts:
        if count[2] > max:
            max = count[2]

    return max

def make_bidirectional(edges):
    return edges + [(t, s) for (s, t) in edges if (t, s) not in edges]

def collect_benchmark_data_analytical (id, benchmark_name, benchmark, connectivity_map, force_bi, gateset, num_qubits, delays, result):
    """Performs analytical calculation of design characteristics

       benchmark:        pre-prepared benchmark circuit to run (type QuantumCircuit)
       connectivity_map: connectivity map to turn into coupling map (provider or customm)
       force_bi:         Leave at 0 - if fails, try 1
       gateset:          basis gates to use for benchmarking
       num_qubits:       # qubits to use (unstable at high amounts)
       delays:           array/list containing gate delays [single, double, readout]
    """
    
    ################
    #Problem set up#
    ################
    
    #Create our coupling map
    local_coupling_map = CouplingMap(make_bidirectional(connectivity_map) if force_bi else connectivity_map)
    
    #Transpile to specified connectivity
    transpiled_benchmark = transpile(benchmark, coupling_map=local_coupling_map)    
    
    swap_overhead = transpiled_benchmark.count_ops().get('swap', 0)

    transpiled_depth = transpiled_benchmark.depth()
    total_gatecount = 0
    for key in transpiled_benchmark.count_ops():
        total_gatecount += transpiled_benchmark.count_ops()[key]

    #Get the critical path (returns cost of the critical path)
    cost = critical_path_analyzer(transpiled_benchmark, num_qubits, delays[0], delays[1], delays[2])




    result.append([
        id,
        benchmark_name,
        num_qubits,
        total_gatecount,
        transpiled_depth,
        cost,
        swap_overhead
    ])

    return 0



print("Starting program...")
results = []

test_q_cnt = 5
benchmark = "ae"
test_mark = get_benchmark(benchmark_name=benchmark, level=2, circuit_size=test_q_cnt)
#print(critical_path_analyzer(test_mark, test_q_cnt, .001, 1, 1000))
#print(test_mark)
delay = [0, .15, .1]


collect_benchmark_data_analytical(
                                  id = 1,
                                  benchmark_name=benchmark, 
                                  benchmark = test_mark, 
                                  connectivity_map=edges_IBM_27, 
                                  force_bi=1, 
                                  gateset=['rz', 'sx', 'x', 'cx', 'measure'], 
                                  num_qubits=test_q_cnt, 
                                  delays=delay,
                                  result=results
                                  )

headers = ["ID", "Benchmark", "Qubit Count", "Gate Count", "Depth", "Exec Time", "SWAP overhead"]
print(tabulate(results, headers=headers, tablefmt="grid"))