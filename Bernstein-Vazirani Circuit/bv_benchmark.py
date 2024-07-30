import numpy as np
import cupy as cp
from sympy import Matrix, eye, zeros, Symbol, I
import time
import warnings
import matplotlib.pyplot as plt
import psutil
import pickle
from scipy.linalg import polar
from itertools import product

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator as QiskitOperator
from qiskit.visualization import circuit_drawer as qiskit_drawer
from qiskit.result import marginal_counts as qiskit_marginal_counts
from qiskit.visualization import plot_histogram

# Initialize CUDA device
cp.cuda.Device(0).use()

# Load the character table
with open('bv_character_table.pkl', 'rb') as f:
    character_table = pickle.load(f)

def bernstein_vazirani_circuit(n, a):
    if len(a) != n:
        raise ValueError(f"Length of 'a' ({len(a)}) must match the number of qubits ({n})")

    qr = QuantumRegister(n+1, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr)

    for i in range(n):
        qc.h(qr[i])
    qc.x(qr[n])
    qc.h(qr[n])

    for i in range(n):
        if a[i] == '1':
            qc.cx(qr[i], qr[n])

    for i in range(n):
        qc.h(qr[i])

    for i in range(n):
        qc.measure(qr[i], cr[i])

    return qc

def get_character_decomposition(n, a):
    if (n, a) in character_table:
        return character_table[(n, a)]
    else:
        raise ValueError(f"Character decomposition for n={n}, a={a} not found in the table")

def reconstruct_unitary(decomposition, n):
    I = cp.eye(2, dtype=cp.complex128)
    X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    Y = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    Z = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)
    
    pauli_group = [I, X, Y, Z]
    
    unitary = cp.zeros((2**(n+1), 2**(n+1)), dtype=cp.complex128)
    
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

def run_benchmark(circuit, backend, num_measurements, num_qubits, shots=1024):
    running_sum = 0.0
    running_sum_squared = 0.0
    
    for i in range(num_measurements):
        print(f"    Measurement {i+1}/{num_measurements}")
        start_time = time.time()
        
        transpiled_circuit = transpile(circuit, backend, optimization_level=0)
        result = backend.run(transpiled_circuit, shots=shots).result()
        counts = qiskit_marginal_counts(result.get_counts(), range(num_qubits))
        most_common = max(counts, key=counts.get)
        
        end_time = time.time()
        print(f"    Most common result: {most_common}")
        
        execution_time = end_time - start_time
        running_sum += execution_time
        running_sum_squared += execution_time ** 2

    mean_time = running_sum / num_measurements
    std_dev = np.sqrt((running_sum_squared / num_measurements) - (mean_time ** 2))
    
    return mean_time, std_dev, counts

def run_original_benchmark(n, a, backend, num_measurements):
    print(f"  Running original benchmark for {n} qubits")
    bv_circuit = bernstein_vazirani_circuit(n, a)
    return run_benchmark(bv_circuit, backend, num_measurements, n)

def run_optimized_benchmark(n, a, backend, num_measurements):
    print(f"  Running optimized benchmark for {n} qubits")
    
    decomposition = get_character_decomposition(n, a)
    unitary = reconstruct_unitary(decomposition, n)
    
    optimized_circuit = QuantumCircuit(n+1, n)
    unitary_gate = UnitaryGate(cp.asnumpy(unitary))  # Convert to NumPy array here
    optimized_circuit.append(unitary_gate, range(n+1))
    for i in range(n):
        optimized_circuit.measure(i, i)
    
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
        a = ''.join(np.random.choice(['0', '1']) for _ in range(n))
        with benchmark_context():
            orig_mean, orig_std, orig_counts = run_original_benchmark(n, a, backend, num_measurements)
        print(f"  Original benchmark for {n} qubits completed")
        wait_for_resources()
        with benchmark_context():
            opt_mean, opt_std, opt_counts = run_optimized_benchmark(n, a, backend, num_measurements)
        print(f"  Optimized benchmark for {n} qubits completed")
        wait_for_resources()
        return n, a, orig_mean, orig_std, opt_mean, opt_std, orig_counts, opt_counts
    except Exception as e:
        print(f"Error in benchmark for {n} qubits: {str(e)}")
        return n, None, None, None, None, None, None, None

def main():
    qiskit_backend = AerSimulator(method='statevector', precision='single')
    
    qubit_range = list(range(2, 6, 1))
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
    plot_aggregated_results(averaged_results)

def process_results(results, qubit_range):
    averaged_results = []
    for i in range(len(qubit_range)):
        results_for_qubit = [suite[i] for suite in results if suite[i][1] is not None]
        if not results_for_qubit:
            continue
        n = results_for_qubit[0][0]
        a = results_for_qubit[0][1]
        orig_means = [r[2] for r in results_for_qubit]
        orig_stds = [r[3] for r in results_for_qubit]
        opt_means = [r[4] for r in results_for_qubit]
        opt_stds = [r[5] for r in results_for_qubit]
        orig_counts = [r[6] for r in results_for_qubit]
        opt_counts = [r[7] for r in results_for_qubit]
        
        if orig_means and opt_means:
            avg_orig_mean = np.mean(orig_means)
            avg_orig_std = np.mean(orig_stds)
            avg_opt_mean = np.mean(opt_means)
            avg_opt_std = np.mean(opt_stds)
            merged_orig_counts = merge_counts(orig_counts)
            merged_opt_counts = merge_counts(opt_counts)
            averaged_results.append((n, a, avg_orig_mean, avg_orig_std, avg_opt_mean, avg_opt_std, merged_orig_counts, merged_opt_counts))

    return sorted(averaged_results, key=lambda x: x[0])

def merge_counts(counts_list):
    merged = {}
    for counts in counts_list:
        for key, value in counts.items():
            if key in merged:
                merged[key] += value
            else:
                merged[key] = value
    return merged

def plot_results(averaged_results, num_measurements, num_suites):
    qubit_numbers = [r[0] for r in averaged_results]
    original_means = [r[2] for r in averaged_results]
    original_stds = [r[3] for r in averaged_results]
    optimized_means = [r[4] for r in averaged_results]
    optimized_stds = [r[5] for r in averaged_results]

    plt.figure(figsize=(12, 6))
    plt.errorbar(qubit_numbers, original_means, yerr=original_stds, fmt='b-o', capsize=5, label='Original Circuit')
    plt.errorbar(qubit_numbers, optimized_means, yerr=optimized_stds, fmt='r-o', capsize=5, label='Optimized Circuit')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Mean Runtime (seconds)')
    plt.title(f'Runtime Scaling of Bernstein-Vazirani Algorithm\n({num_measurements} measurements per data point, {num_suites} suites)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('bv_runtime_scaling_robust.png')
    print("\nPlot saved as 'bv_runtime_scaling_robust.png'")

def print_results(averaged_results):
    print("\nResults:")
    for result in averaged_results:
        n, a, orig_mean, orig_std, opt_mean, opt_std, _, _ = result
        print(f"\nNumber of qubits: {n}")
        print(f"Secret string: {a}")
        print(f"Original circuit - Mean time: {orig_mean:.6f} s, Std dev: {orig_std:.6f} s")
        print(f"Optimized circuit - Mean time: {opt_mean:.6f} s, Std dev: {opt_std:.6f} s")

def plot_aggregated_results(averaged_results):
    for result in averaged_results:
        n, a, _, _, _, _, orig_counts, opt_counts = result
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        plot_histogram(orig_counts, ax=ax1, title=f"Original Circuit (n={n})")
        plot_histogram(opt_counts, ax=ax2, title=f"Optimized Circuit (n={n})")
        plt.tight_layout()
        plt.savefig(f'histogram_comparison_{n}_qubits.png')
        plt.close()

    print("\nHistograms saved")

if __name__ == "__main__":
    main()