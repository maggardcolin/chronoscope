###########
## NOTES ##
###########
# 
# Currently runs using CPU but line 38 (sampler line) can be updated to use GPU if you
# have a compatible NVIDIA GPU.
#
# On CPU it's pretty slow (3-5 minutes) and don't know about runtime on GPU
#


#################################################
### Run commented lines before executing file ###
#################################################

#pip install git+https://github.com/jpmorganchase/QOKit.git
#pip install qiskit[visualization] qiskit-optimization qiskit-aer
#pip install qiskit-aer-gpu
#pip install pennylane

#For user CLI
import sys


from qiskit_optimization.applications import Maxcut
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_algorithms import NumPyMinimumEigensolver
import numpy as np
from qokit.parameter_utils import get_fixed_gamma_beta

from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA

from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import depolarizing_error

def get_initial_point(p):
    gamma, beta = get_fixed_gamma_beta(3, p)
    return np.concatenate([-np.array(beta), np.array(gamma) / 2]) # make up for different conventions


def run_qaoa(depth: int, noise_model: NoiseModel | None = None) -> float:
    sampler = Sampler(run_options={"shots": 10000, "seed": seed}, backend_options={"noise_model": noise_model, "device": "CPU"})

    #do the QAOA store in res
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), initial_point=get_initial_point(depth), reps=depth)
    result = qaoa.compute_minimum_eigenvalue(hamiltonian).optimal_value + offset

    return result/ground_state_energy



n = 10 # was 8?
seed = 6657

#Create graph of n nodes
graph = nx.random_regular_graph(3, n, seed=seed)
#nx.draw(graph)
#plt.show()

hamiltonian, offset = Maxcut(graph).to_quadratic_program().to_ising()

ground_state_energy = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian).eigenvalue.real + offset



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("running qaoa...")
run_qaoa(1)

# Change the noise model

qaoa_returns = ([], [], [])
qaoa_depths  = ([], [], [])
label = ("No Noise", "0.001", "0.1")

depol_error = depolarizing_error(0, 2), depolarizing_error(0.001, 2), depolarizing_error(0.1, 2)


print("running noise loop...")
for j in range(0, 3): #Noise model loop
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depol_error[j], "cx")

    print("looping...")
    for i in range(1, 7):  #1-6

        qaoa_depths[j].insert(i-1, i)
        qaoa_returns[j].insert(i-1, run_qaoa(i, noise_model=noise_model))

print("plotting...")
#plot the three graphs onto the same plot
for i in range(0, 3):
    plt.plot(qaoa_depths[i], qaoa_returns[i], label=label[i])
    plt.xlabel("QAOA Depth")
    plt.ylabel("Approximation Ratio")

print("Plotting...")
plt.title("Approximation Ratio vs QAOA Depth")
plt.legend()
plt.show()

#Need to add a way to save/restore data from our simulations