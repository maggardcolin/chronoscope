#################################################
### Run commented lines before executing file ###
#################################################

#pip install git+https://github.com/jpmorganchase/QOKit.git
#pip install qiskit[visualization] qiskit-optimization qiskit-aer
#pip install qiskit-aer-gpu
#pip install pennylane
from qiskit_optimization.applications import Maxcut
import networkx as nx
import matplotlib.pyplot as plt


n = 10 # was 8?
seed = 6657

graph = nx.random_regular_graph(3, n, seed=seed)
nx.draw(graph)
plt.show()