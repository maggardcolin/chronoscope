import os
os.system('cls' if os.name == 'nt' else 'clear')

verbose = 1
if verbose:
    print("    +----------------------------------------------+")
    print("    |    Resource and Fidelity Utility for QAOA    |")
    print("    |                                              |")
    print("    |          CS639 FINAL COURSE PROJECT          |")
    print("    +----------------------------------------------+")
    print()
    print("Initializing...", end='', flush=True)

import numpy as np
import math as m
print(".", end = '', flush=True)
from mqt.bench import get_benchmark
from tabulate import tabulate
print(".", end = '', flush=True)
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
print(".", end = '', flush=True)
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
print(".", end = '', flush=True)
from collections import defaultdict
import warnings
from connectivity_maps import edges_mesh, edges_trapped_ion, edges_heavy_hex
print(".", end = '', flush=True)
from qiskit.transpiler import CouplingMap
import connectivity_maps as cn
print(".", end = '', flush=True)

os.system('cls' if os.name == 'nt' else 'clear')

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def make_parallel_copies(circuit, n):
    total_qubits = circuit.num_qubits * n
    qc = QuantumCircuit(total_qubits)
    for i in range(n):
        offset = i * circuit.num_qubits
        qc.compose(circuit, qubits=range(offset, offset + circuit.num_qubits), inplace=True)
    return qc

def calculate_idling_error(exec_time, t1, t2):
    px_y = (1-m.exp(-exec_time/t1))/4
    pz = ((1-m.exp(-exec_time/t2))/2) - px_y
    return 1- (px_y * pz)

def total_variation_distance(P, Q):
    all_keys = set(P) | set(Q)
    return 0.5 * sum(abs(P.get(k, 0) - Q.get(k, 0)) for k in all_keys)

def normalize(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

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

def calculate_features(circuit: QuantumCircuit, qubit_count: int):
    dag = circuit_to_dag(circuit)
    circuit_depth = circuit.depth()

    # Interaction and gate stats
    interaction_graph = defaultdict(set)
    total_gates = 0
    two_qubit_gates = 0
    measure_count = 0

    for gate in circuit.data:
        total_gates += 1
        if gate.operation.name == 'measure':
            measure_count += 1
        qargs = [circuit.qubits.index(q) for q in gate.qubits]
        if len(qargs) == 2:
            interaction_graph[qargs[0]].add(qargs[1])
            interaction_graph[qargs[1]].add(qargs[0])
            two_qubit_gates += 1

    # Communication feature
    total_degrees = sum(len(neighbors) for neighbors in interaction_graph.values())
    max_possible = qubit_count * (qubit_count - 1)
    communication = (total_degrees / max_possible) * 100 if max_possible > 0 else 0

    # Entanglement
    entanglement = (two_qubit_gates / total_gates) * 100 if total_gates > 0 else 0

    # Parallelism
    parallelism = ((total_gates / circuit_depth - 1) / (qubit_count - 1)) * 100 if qubit_count > 1 and circuit_depth > 0 else 0

    # Measurement
    measurement = (measure_count / circuit_depth) * 100 if circuit_depth > 0 else 0

    # Liveness (layer count may differ from circuit depth)
    dag_layers = list(dag.layers())
    actual_layer_count = len(dag_layers)
    liveness_matrix = [[0 for _ in range(actual_layer_count)] for _ in range(qubit_count)]

    for layer_index, layer in enumerate(dag_layers):
        gate_nodes = layer['graph'].op_nodes()
        for node in gate_nodes:
            for qubit in node.qargs:
                qid = circuit.qubits.index(qubit)
                liveness_matrix[qid][layer_index] = 1

    total_active = sum(sum(row) for row in liveness_matrix)
    liveness = (total_active / (qubit_count * actual_layer_count)) * 100 if actual_layer_count > 0 else 0

    # Critical depth proxy: layer with most 2q gates
    two_q_depthwise = [0] * actual_layer_count
    for layer_idx, layer in enumerate(dag_layers):
        for node in layer['graph'].op_nodes():
            if len(node.qargs) == 2:
                two_q_depthwise[layer_idx] += 1
    max_2q_layer = max(two_q_depthwise, default=0)
    critical_depth = (max_2q_layer / two_qubit_gates) * 100 if two_qubit_gates > 0 else 0

    # truncate all results to 2 decimal places
    communication = round(communication, 2)
    critical_depth = round(critical_depth, 2)
    entanglement = round(entanglement, 2)
    parallelism = round(parallelism, 2)
    liveness = round(liveness, 2)
    measurement = round(measurement, 2)

    return {
        "communication": communication,
        "critical-depth": critical_depth,
        "entanglement": entanglement,
        "parallelism": parallelism,
        "liveness": liveness,
        "measurement": measurement
    }

def collect_benchmark_data_analytical (id, 
                                       benchmark_name, 
                                       benchmark, 
                                       connectivity_map, 
                                       force_bi, 
                                       gateset, 
                                       benchmark_qubits, 
                                       connectivity_map_size, 
                                       delays,
                                       result,
                                       attempt_parallelism,
                                       parallelism_level,
                                       fidelities,
                                       coherence_times):
    """Optimistically performs analytical calculation of design characteristics

       benchmark:        pre-prepared benchmark circuit to run (type QuantumCircuit)
       connectivity_map: connectivity map to turn into coupling map (provider or customm)
       connectivity_map_size: number of qubits supported by the connectivity map
       force_bi:         Leave at 0 - if fails, try 1
       gateset:          basis gates to use for benchmarking
       num_qubits:       # qubits to use (unstable at high amounts)
       delays:           array/list containing gate delays [single, double, readout]
    """
    
    ################
    #Problem set up#
    ################
    if benchmark_qubits > connectivity_map_size:
        print("ERROR: Benchmark circuit of size" + str(benchmark_qubits) + "is too large for this connectivity map of size " +  str(connectivity_map_size))
        print("       Recheck your inputs and ensure all values are set correctly.")
        print("NOTE:  No modifications have been made to the results.")
        return
    
    #Create our coupling map
    local_coupling_map = CouplingMap(make_bidirectional(connectivity_map) if force_bi else connectivity_map)
    max_possible_copies = 1
    if attempt_parallelism:
        if int(parallelism_level) == -1:
            max_possible_copies = int(connectivity_map_size/benchmark_qubits) #Truncate decimal
        else:
            max_possible_copies = parallelism_level
        benchmark = make_parallel_copies(circuit=benchmark, n=max_possible_copies)  #Store parallel-ed version into 
    
    #Transpile to specified connectivity
    transpiled_benchmark = transpile(benchmark, coupling_map=local_coupling_map, optimization_level=2)
    num_qubits_transpiled = transpiled_benchmark.num_qubits
    
    swap_overhead = transpiled_benchmark.count_ops().get('swap', 0)
    cx_count = transpiled_benchmark.count_ops().get('cx', 0)

    transpiled_depth = transpiled_benchmark.depth()

    total_gatecount = 0
    for key in transpiled_benchmark.count_ops():
        total_gatecount += transpiled_benchmark.count_ops()[key]

    #Get the critical path (returns cost of the critical path)
    cost = critical_path_analyzer(transpiled_benchmark, num_qubits_transpiled, delays[0], delays[1], delays[2])
    
    #Estimate fidelity based on individual gate errors and execution time
    oneqgate_p_circ = (total_gatecount - cx_count - swap_overhead)
    
    
    #replace with TVD (?)
    estimated_average_fidelity = (fidelities[0]**(oneqgate_p_circ/max_possible_copies) * fidelities[1]**((cx_count + swap_overhead*3)/max_possible_copies) * fidelities[2]**(benchmark_qubits)) * calculate_idling_error(cost, coherence_times[0], coherence_times[1])**(benchmark_qubits/max_copies)

    result.append([
        id,
        benchmark_name,
        benchmark_qubits,
        total_gatecount,
        transpiled_depth,
        cost,
        attempt_parallelism,
        cost/max_possible_copies,
        max_possible_copies,
        swap_overhead,
        estimated_average_fidelity,
        #transpiled_benchmark            #save the benchmark 
    ])

    return 0

#noisy simulation for shots estimator

#num copies affect speedup time                         (INCOMPLETE)
#   Ratio of original time over speedup time

#As we change connectivity maps how does speedup change (INCOMPLETE)
#   Specifically make sure to do a trapped ion config 
#   (groups of 5 or so qubits with limited extracell 
#   connectivity)

#What compilation settings to use to optimize these     (INCOMPLETE)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

warnings.filterwarnings("ignore", category=DeprecationWarning)


print("    +----------------------------------------------+")
print("    |                                              |")
print("    |  ++Resource and Fidelity Utility for QAOA++  |")
print("    |                                              |")
print("    |          CS639 FINAL COURSE PROJECT          |")
print("    |                                              |")
print("    +----------------------------------------------+")
print()
results = []

benchmark = "qaoa"

test_q_cnt = 5      #Fixed to 5 do not change 
test_mark = get_benchmark(benchmark_name=benchmark, level=2, circuit_size=test_q_cnt)
delay = [0.02, .2, 200]        #us
fdlt = [0.999, .985, .97]   # %
ctimes = [.1, .1] #ms

#Example use of the program

max_copies = int(100/test_q_cnt)    #truncate

print("QAOA problem size is " + str(test_q_cnt) + " qubits running 1 - " + str(max_copies) + " circuits in parallel.")

test_number = 1
connectivity_maps = [edges_mesh, edges_trapped_ion, edges_heavy_hex]
connectivity_maps_ascii = ["Mesh", "Trapped Ion", "Heavy Hex"]
ascii_index = 0
for connectivity_map in connectivity_maps:
    print()
    print()
    print(connectivity_maps_ascii[ascii_index])
    ascii_index = ascii_index + 1
    #Benchmark data for no parallelism 
    collect_benchmark_data_analytical(
                                            id = test_number,                                               #An arbitrary id for use in identifying and ordering tests
                                            benchmark_name=benchmark,                             #An arbitrary string (but you should set it to the name of the benchmark)
                                            benchmark = test_mark,                                #the actual benchmark circuit
                                            benchmark_qubits=test_q_cnt,                          #Number of qubits in the benchmark circuit

                                            connectivity_map=connectivity_map,                        #edge map of the arch we are testing
                                            connectivity_map_size=100,                             #Maximum number of allowed qubits on the map
                                            force_bi=1,                                           #????? sometimes necessary to force bidrectionality of coupling  map

                                            gateset=['rz', 'sx', 'x', 'cx', 'measure'],           #Basis gates to use
                                            delays=delay,                                         #Gate delays in form of         [single, double, readout] (us)
                                            fidelities= fdlt,                                     #Fidelities in form of          [single, double, readout] (%)
                                            coherence_times=ctimes,                               #Coherence timee in form of     [t1, t2] (ms)

                                            attempt_parallelism=False,                             #Set 'True' to attempt adding copies to the circuit (will maximize number of copies)
                                            parallelism_level = -1,                               #-1 is maximum copies otherwise specify number of copies (0 copies not allowed) Only used when attempt_parallelism is true

                                            result=results                                       #The return array to which results are appended
                                            )

    test_number = test_number + 1

    #Parallel metrification
    for i in range(2, max_copies + 1):
        try:
            test_mark = get_benchmark(benchmark_name=benchmark, level=2, circuit_size=test_q_cnt)
            collect_benchmark_data_analytical(
                                            id = test_number,                                               #An arbitrary id for use in identifying and ordering tests
                                            benchmark_name=benchmark,                             #An arbitrary string (but you should set it to the name of the benchmark)
                                            benchmark = test_mark,                                #the actual benchmark circuit
                                            benchmark_qubits=test_q_cnt,                          #Number of qubits in the benchmark circuit

                                            connectivity_map=connectivity_map,                        #edge map of the arch we are testing
                                            connectivity_map_size=100,                             #Maximum number of allowed qubits on the map
                                            force_bi=1,                                           #????? sometimes necessary to force bidrectionality of coupling  map

                                            gateset=['rz', 'sx', 'x', 'cx', 'measure'],           #Basis gates to use
                                            delays=delay,                                         #Gate delays in form of         [single, double, readout] (us)
                                            fidelities= fdlt,                                     #Fidelities in form of          [single, double, readout] (%)
                                            coherence_times=ctimes,                               #Coherence timee in form of     [t1, t2] (ms)

                                            attempt_parallelism=True,                             #Set 'True' to attempt adding copies to the circuit (will maximize number of copies)
                                            parallelism_level = i,                               #-1 is maximum copies otherwise specify number of copies (0 copies not allowed) Only used when attempt_parallelism is true

                                            result=results                                       #The return array to which results are appended
                                            )
            test_number = test_number + 1
        except Exception as e:
            print(f"Error processing benchmark with {i} qubits: {e}")
            continue
            
    headers = ["ID", "Bnchmrk", "# Qubit", "# Gate", "Depth", "Cost (us)", "Prllsm?", "Copy Cost (us)", "# Prlll cps", "SWAP ovhd", "Net Fidelity"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

    #print(f"Analytical values calulated for connectivity {connectivity_map}...\n")
    print("Calculating runtime for 1024 shots...\n")

    runtime_headers = ["# Copies", "Runtime (us)", "Speedup (single/parallel)", "Fidelity"]
    runtime_results = []
    for run in results:
        runtime_no_copies = results[0][7] * 1024
        runtime_results.append([
            run[8],
            run[7] * 1024,
            runtime_no_copies/(run[7] * 1024),
            run[10]
            ])

    print(tabulate(runtime_results, headers=runtime_headers, tablefmt="grid"))
    runtime_results = []
    results = []

print("\nCompleted. Exiting...")