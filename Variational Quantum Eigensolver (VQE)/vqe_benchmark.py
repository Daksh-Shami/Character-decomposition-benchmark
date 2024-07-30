import numpy as np
import cupy as cp
import time
import warnings
import matplotlib.pyplot as plt
import psutil
import pickle
from itertools import product

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.visualization import circuit_drawer as qiskit_drawer

# Initialize CUDA device
cp.cuda.Device(0).use()

# Load the character table
with open('vqe_character_table.pkl', 'rb') as f:
    character_table = pickle.load(f)

def vqe_ansatz(params, num_qubits):
    if len(params) != 2 * num_qubits:
        raise ValueError(f"Expected {2 * num_qubits} parameters, but got {len(params)}")

    qc = QuantumCircuit(num_qubits)

    # Apply initial rotations
    for i in range(num_qubits):
        qc.rx(params[i], i)
        qc.rz(params[i + num_qubits], i)

    # Apply entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc

def get_character_decomposition(n):
    if n in character_table:
        return character_table[n]
    else:
        raise ValueError(f"Character decomposition for n={n} not found in the table")

def reconstruct_unitary(decomposition, n):
    I = cp.eye(2, dtype=cp.complex128)
    X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    Y = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    Z = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)
    
    pauli_group = [I, X, Y, Z]
    
    unitary = cp.zeros((2**n, 2**n), dtype=cp.complex128)
    
    for character, indices in decomposition:
        P = pauli_group[indices[0]]
        for idx in indices[1:]:
            P = cp.kron(P, pauli_group[idx])
        unitary += character * P
    
    return unitary

def save_circuit_diagram(circuit, filename):
    circuit_drawing = qiskit_drawer(circuit, output='mpl', plot_barriers=False)
    circuit_drawing.figure.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(circuit_drawing.figure)

def check_system_resources():
    if psutil.cpu_percent(interval=1) > 80 or psutil.virtual_memory().percent > 80:
        print("Warning: High system load. Results may be affected.")
        time.sleep(5)

def run_benchmark(circuit, backend, num_measurements, num_qubits):
    running_sum = 0.0
    running_sum_squared = 0.0
    
    for i in range(num_measurements):
        print(f"    Measurement {i+1}/{num_measurements}")
        start_time = time.time()
        
        transpiled_circuit = transpile(circuit, backend, optimization_level=0)
        result = backend.run(transpiled_circuit).result()
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        running_sum += execution_time
        running_sum_squared += execution_time ** 2

    mean_time = running_sum / num_measurements
    std_dev = np.sqrt((running_sum_squared / num_measurements) - (mean_time ** 2))
    
    return mean_time, std_dev

def run_original_benchmark(n, backend, num_measurements):
    print(f"  Running original benchmark for {n} qubits")
    params = np.random.uniform(0, 2 * np.pi, size=2 * n)
    vqe_circ = vqe_ansatz(params, n)
    return run_benchmark(vqe_circ, backend, num_measurements, n)

def run_optimized_benchmark(n, backend, num_measurements):
    print(f"  Running optimized benchmark for {n} qubits")
    
    decomposition = get_character_decomposition(n)
    unitary = reconstruct_unitary(decomposition, n)
    
    optimized_circuit = QuantumCircuit(n)
    unitary_gate = UnitaryGate(cp.asnumpy(unitary))  # Convert to NumPy array here
    optimized_circuit.append(unitary_gate, range(n))
    
    return run_benchmark(optimized_circuit, backend, num_measurements, n)

import gc
from contextlib import contextmanager

@contextmanager
def benchmark_context():
    try:
        yield
    finally:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

def wait_for_resources():
    while psutil.virtual_memory().percent > 90 or psutil.cpu_percent(interval=0.1) > 90:
        time.sleep(0.1)

def run_benchmark_suite(n, num_measurements, backend):
    print(f"\nStarting benchmark for {n} qubits")
    try:
        with benchmark_context():
            orig_mean, orig_std = run_original_benchmark(n, backend, num_measurements)
        print(f"  Original benchmark for {n} qubits completed")
        wait_for_resources()
        with benchmark_context():
            opt_mean, opt_std = run_optimized_benchmark(n, backend, num_measurements)
        print(f"  Optimized benchmark for {n} qubits completed")
        wait_for_resources()
        return n, orig_mean, orig_std, opt_mean, opt_std
    except Exception as e:
        print(f"Error in benchmark for {n} qubits: {str(e)}")
        return n, None, None, None, None

def main():
    qiskit_backend = AerSimulator(method='statevector', precision='single')
    
    qubit_range = list(range(2, 8, 1))
    num_measurements = 100
    num_suites = 3  # Number of times to run the entire benchmark suite

    all_results = []
    
    for suite in range(num_suites):
        print(f"Running benchmark suite {suite + 1}/{num_suites}")
        results = []
        for i, n in enumerate(qubit_range):
            print(f"Benchmark {i+1}/{len(qubit_range)} in suite {suite + 1}")
            result = run_benchmark_suite(n, num_measurements, qiskit_backend)
            results.append(result)
        all_results.append(results)

    # Process and plot results
    averaged_results = process_results(all_results, qubit_range)
    plot_results(averaged_results, num_measurements, num_suites)
    print_results(averaged_results)

def process_results(results, qubit_range):
    averaged_results = []
    for i in range(len(qubit_range)):
        results_for_qubit = [suite[i] for suite in results if suite[i][1] is not None]
        if not results_for_qubit:
            continue
        n = results_for_qubit[0][0]
        orig_means = [r[1] for r in results_for_qubit]
        orig_stds = [r[2] for r in results_for_qubit]
        opt_means = [r[3] for r in results_for_qubit]
        opt_stds = [r[4] for r in results_for_qubit]
        
        if orig_means and opt_means:
            avg_orig_mean = np.mean(orig_means)
            avg_orig_std = np.mean(orig_stds)
            avg_opt_mean = np.mean(opt_means)
            avg_opt_std = np.mean(opt_stds)
            averaged_results.append((n, avg_orig_mean, avg_orig_std, avg_opt_mean, avg_opt_std))

    return sorted(averaged_results, key=lambda x: x[0])

def plot_results(averaged_results, num_measurements, num_suites):
    qubit_numbers = [r[0] for r in averaged_results]
    original_means = [r[1] for r in averaged_results]
    original_stds = [r[2] for r in averaged_results]
    optimized_means = [r[3] for r in averaged_results]
    optimized_stds = [r[4] for r in averaged_results]

    plt.figure(figsize=(12, 6))
    plt.errorbar(qubit_numbers, original_means, yerr=original_stds, fmt='b-o', capsize=5, label='Original Circuit')
    plt.errorbar(qubit_numbers, optimized_means, yerr=optimized_stds, fmt='r-o', capsize=5, label='Optimized Circuit')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Mean Runtime (seconds)')
    plt.title(f'Runtime Scaling of Variational Quantum Eigensolver\n({num_measurements} measurements per data point, {num_suites} suites)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('vqe_runtime_scaling_robust.png')
    print("\nPlot saved as 'vqe_runtime_scaling_robust.png'")

def print_results(averaged_results):
    print("\nResults:")
    for result in averaged_results:
        n, orig_mean, orig_std, opt_mean, opt_std = result
        print(f"\nNumber of qubits: {n}")
        print(f"Original circuit - Mean time: {orig_mean:.6f} s, Std dev: {orig_std:.6f} s")
        print(f"Optimized circuit - Mean time: {opt_mean:.6f} s, Std dev: {opt_std:.6f} s")

if __name__ == "__main__":
    main()