verbose = 0
if verbose:
    print()
    print()
    print("    +-------------------------------------+")
    print("    |                                     |")
    print("    |  ++Resource and Fidelity Utility++  |")
    print("    |                                     |")
    print("    |     CS639 FINAL COURSE PROJECT      |")
    print("    |                                     |")
    print("    +-------------------------------------+")
    print()
    print("Initializing RFU... ")

import numpy as np
import math as m
from mqt.bench import get_benchmark
from tabulate import tabulate
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_aer import AerSimulator


#Sim imports
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.noise.errors import ReadoutError

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from qiskit.transpiler import CouplingMap

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
    transpiled_benchmark = transpile(benchmark, coupling_map=local_coupling_map, optimization_level=3)
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
    
    
    #replace with TVD
    
    sim = AerSimulator(noise_model=create_noise_model("IBM_Montreal"))
    _result = sim.run(transpiled_benchmark, shots=128).result()
    _counts = _result.get_counts()
    _norm_counts = normalize(_counts)
    _ideal_key = max(_counts, key=_counts.get)
    _ideal_dist = {_ideal_key: 1.0}
    _tvd = total_variation_distance(_norm_counts, _ideal_dist)

    
    estimated_average_fidelity = (fidelities[0]**oneqgate_p_circ * fidelities[1]**(cx_count + swap_overhead*3) * fidelities[2]**(benchmark_qubits))**(1/max_possible_copies) * calculate_idling_error(cost, coherence_times[0], coherence_times[1])**max_possible_copies

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
        _tvd,
        #transpiled_benchmark            #save the benchmark 
    ])

    return 0

#noisy simulation for shots estimator



#START OF ACTUAL PROGRAM

if verbose:
    print("Initialized")
    print("Beginning execution...")






results = []

test_q_cnt = 5
benchmark = "ae"
test_mark = get_benchmark(benchmark_name=benchmark, level=2, circuit_size=test_q_cnt)
#print(critical_path_analyzer(test_mark, test_q_cnt, .001, 1, 1000))
#print(test_mark)
delay = [0.02, .2, 200]        #us
fdlt = [0.999, .985, .97]   #
ctimes = [.1, .1] #ms


#Example use of the program
for i in range(1, 2):
    test_q_cnt = 5*i
    test_mark = get_benchmark(benchmark_name=benchmark, level=2, circuit_size=test_q_cnt)
    collect_benchmark_data_analytical(
                                    id = i,                                               #An arbitrary id for use in identifying and ordering tests
                                    benchmark_name=benchmark,                             #An arbitrary string (but you should set it to the name of the benchmark)
                                    benchmark = test_mark,                                #the actual benchmark circuit
                                    benchmark_qubits=test_q_cnt,                          #Number of qubits in the benchmark circuit

                                    connectivity_map=edges_IBM_27,                        #edge map of the arch we are testing
                                    connectivity_map_size=27,                             #Maximum number of allowed qubits on the map
                                    force_bi=1,                                           #????? sometimes necessary to force bidrectionality of coupling  map

                                    gateset=['rz', 'sx', 'x', 'cx', 'measure'],           #Basis gates to use
                                    delays=delay,                                         #Gate delays in form of         [single, double, readout] (us)
                                    fidelities= fdlt,                                     #Fidelities in form of          [single, double, readout] (%)
                                    coherence_times=ctimes,                                #Coherence timee in form of     [t1, t2] (ms)

                                    attempt_parallelism=True,                            #Set 'True' to attempt adding copies to the circuit (will maximize number of copies)
                                    parallelism_level = 2,                                #-1 is maximum copies otherwise specify number of copies (0 copies not allowed) Only used when attempt_parallelism is true

                                    result=results                                       #The return array to which results are appended
                                    )

headers = ["ID", "Bnchmrk", "# Qubit", "# Gate", "Depth", "Cost (us)", "Prllsm?", "Copy Cost (us)", "# Prlll cps", "SWAP ovhd", "Net Fidelity", "TVD"]
print(tabulate(results, headers=headers, tablefmt="grid"))