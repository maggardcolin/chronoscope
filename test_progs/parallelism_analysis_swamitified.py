import os
os.system('cls' if os.name == 'nt' else 'clear')

verbose = 0
print("    +----------------------------------------------+")
print("    |    Resource and Fidelity Utility for QAOA    |")
print("    |                                              |")
print("    |          CS639 FINAL COURSE PROJECT          |")
print("    +----------------------------------------------+")
print()
print("Initializing...", end='', flush=True)


#These print statements are just to give an indicator of how far along the loading process we are
# not functional and probably slow things down but they're neat
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
from connectivity_maps import edges_mesh, edges_trapped_ion_5_20, edges_heavy_hex, edges_trapped_ion_10_10  
print(".", end = '', flush=True)
from qiskit.transpiler import CouplingMap
import connectivity_maps as cn
print(".", end = '', flush=True)
from datetime import datetime
import matplotlib.pyplot as plt

benchmark = "ae"

test_q_cnt = 7      ######################################################################################################

#print a log
fname = "logs/" +str(datetime.now())[:10] + '-' + str(datetime.now())[11:13]+ '-' + str(datetime.now())[14:16] + "-" + benchmark + "-" + str(test_q_cnt) +"-qubit"+ ".log"

#fname = "execute.log"

f = open(fname, 'w')
print("    +----------------------------------------------+", file=f)
print("    |    Resource and Fidelity Utility for QAOA    |", file=f)
print("    |                                              |", file=f)
print("    |          CS639 FINAL COURSE PROJECT          |", file=f)
print("    +----------------------------------------------+", file=f)
print('', file=f)
print("Verbose: " + str(verbose), file=f)
print('', file=f)

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
        
    cx_count = 0
    swap_overhead = 0
    transpiled_depth = 0
    total_gatecount = 0
    cost = 0
    
    num_to_average = 3
    
    for k in range(num_to_average):
        #Transpile to specified connectivity
        transpiled_benchmark = transpile(benchmark, coupling_map=local_coupling_map, optimization_level=1)
        num_qubits_transpiled = transpiled_benchmark.num_qubits
        
        swap_overhead += transpiled_benchmark.count_ops().get('swap', 0)
        cx_count += transpiled_benchmark.count_ops().get('cx', 0)

        transpiled_depth += transpiled_benchmark.depth()

        for key in transpiled_benchmark.count_ops():
            total_gatecount += transpiled_benchmark.count_ops()[key]

        #Get the critical path (returns cost of the critical path)
        cost += critical_path_analyzer(transpiled_benchmark, num_qubits_transpiled, delays[0], delays[1], delays[2])
        
    #Estimate fidelity based on individual gate errors and execution time
    oneqgate_p_circ = (total_gatecount - cx_count - swap_overhead)
    
    cx_count = np.ceil(cx_count/num_to_average)
    swap_overhead = np.ceil(swap_overhead/num_to_average)
    transpiled_depth = transpiled_depth/num_to_average
    total_gatecount = total_gatecount/num_to_average
    cost = cost/num_to_average
    
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
    ])

    return 0

#num copies affect speedup time                         (INCOMPLETE)
#   Ratio of original time over speedup time
#   Add plotting function

#As we change connectivity maps how does speedup change (COMPLETE)
#   Specifically make sure to do a trapped ion config 
#   (groups of 5 or so qubits with limited extracell 
#   connectivity)

#What compilation settings to use to optimize these     (COMPLETE)
#   Just let the compiler do its job

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


test_mark = get_benchmark(benchmark_name=benchmark, level=2, circuit_size=test_q_cnt)
delay = [0.02, .2, 200]        #us
fdlt = [0.999, .985, .97]   # %
ctimes = [.1, .1] #ms

#Example use of the program

max_copies = int(100/test_q_cnt)    #truncate

print("QAOA problem size is " + str(test_q_cnt) + " qubits running 1 - " + str(max_copies) + " circuits in parallel.")
print("QAOA problem size is " + str(test_q_cnt) + " qubits running 1 - " + str(max_copies) + " circuits in parallel.", file=f)

test_number = 1
connectivity_maps = [edges_mesh, edges_trapped_ion_10_10, edges_trapped_ion_5_20]
connectivity_maps_ascii = ["Mesh (100 qubit)", "Trapped Ion (10 qubit clusters)", "Trapped Ion (5 qubit clusters)"]
ascii_index = 0
for connectivity_map in connectivity_maps:
    print()
    print('', file = f)
    print()
    print('', file = f)
    print(connectivity_maps_ascii[ascii_index], file = f)
    print(connectivity_maps_ascii[ascii_index])
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
    if verbose:
        print(tabulate(results, headers=headers, tablefmt="grid"))
        print(tabulate(results, headers=headers, tablefmt="grid"), file=f)

    #print(f"Analytical values calulated for connectivity {connectivity_map}...\n")
    print("Calculating runtime for 1024 shots...\n")
    print("Calculating runtime for 1024 shots...\n", file=f)

    runtime_headers = ["# Copies", "Runtime (us)", "Speedup (single/parallel)", "Fidelity"]
    runtime_results = []
    for run in results:
        runtime_no_copies = results[0][7] * 1024
        runtime_results.append([
            run[8],
            run[5] * np.ceil(1024/run[8]),
            runtime_no_copies/(run[5] * np.ceil(1024/run[8])),
            run[10]
            ])
        
    print(tabulate(runtime_results, headers=runtime_headers, tablefmt="grid"))
    print(tabulate(runtime_results, headers=runtime_headers, tablefmt="grid"), file=f)
    ascii_index = ascii_index + 1


##Chat GPT stuff below
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools

# Split runtime_results into architecture blocks
architecture_names = connectivity_maps_ascii
grouped_runtime_results = []
cursor = 0

# Calculate how many runs belong to each architecture
runs_per_architecture = [1 + max_copies - 1 for _ in architecture_names]  # 1 no-parallel + (max_copies - 1) parallel runs

for count in runs_per_architecture:
    arch_block = runtime_results[cursor:cursor + count]
    grouped_runtime_results.append(arch_block)
    cursor += count

# Set up distinct colors
colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# Plot all architectures on one figure
plt.figure()
for i, arch_runtime in enumerate(grouped_runtime_results):
    if not arch_runtime:
        continue  # skip empty blocks
    copies = [row[0] for row in arch_runtime]
    speedup = [row[2] for row in arch_runtime]
    plt.plot(copies, speedup, marker='o', linestyle='-', label=architecture_names[i], color=next(colors))

pltname = "Parallelism Speedup (" + benchmark + " " + str(test_q_cnt) + " qubit)"
plt.xlabel("# Copies")
plt.ylabel("Speedup")
plt.title(pltname)
plt.xticks(sorted(set(row[0] for arch in grouped_runtime_results for row in arch)))
plt.grid(True)
plt.legend()
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.savefig('plots/' + benchmark + '/' + str(test_q_cnt) + '.png', dpi=300, bbox_inches='tight')
#plt.show()
    
    
print("Analysis done.", file=f)
f.flush()
f.close()
print()
print()
print("A log has been generated at " + fname)

print("\nCompleted. Exiting...")

