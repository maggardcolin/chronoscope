Semester project for CS 639: Systems Architecture for Quantum Computers

Main test program can be found under test_progs/parallelism_analysis_swamitified.py.

This script is a resource estimation utility that can be used to determine the optimal level of parallelism that can be achieved in a variety of quantum connectivity maps. The program does not take into account fidelity of the circuit due to the difficulty of modeling cross-talk and idling errors and analytical calcuation of fidelity leaves much to be desired for precision.
Single and CX gate counts are reported in log files and in the console output and critical path analysis is used to determine the execution times of an entire circuit or the execution time of a single 'copy' of a circuit - both of which are useful in determining maximum parallelisation of an algorithm.
'
