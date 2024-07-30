import numpy as np
import cupy as cp
from sympy import Matrix, eye, zeros, Symbol, I
import time
import warnings
import matplotlib.pyplot as plt
import psutil
import pickle
from scipy.linalg import polar
import traceback
import gc
from contextlib import contextmanager

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
import networkx as nx
from qiskit.visualization import circuit_drawer as qiskit_drawer
from qiskit.result import marginal_counts as qiskit_marginal_counts

# Initialize CUDA device
cp.cuda.Device(0).use()

# Load the character table
with open('qaoa_character_table_symbolic.pkl', 'rb') as f:
    character_table = pickle.load(f)

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

def create_maxcut_problem(num_nodes):
    if num_nodes < 2:
        raise ValueError("Number of nodes must be at least 2")
    elif num_nodes == 2:
        G = nx.Graph()
        G.add_edge(0, 1)
    else:
        if num_nodes % 2 == 1:
            G = nx.cycle_graph(num_nodes)
        else:
            G = nx.complete_graph(num_nodes)
    
    max_cut = Maxcut(G)
    qubo = QuadraticProgramToQubo().convert(max_cut.to_quadratic_program())
    return qubo

def qaoa_circuit(qubo, p):
    ansatz = QAOAAnsatz(qubo.to_ising()[0], reps=p)
    return ansatz.decompose()

def get_character_decomposition(n, p):
    if (n, p) in character_table:
        return character_table[(n, p)]
    raise ValueError(f"Character decomposition for n={n}, p={p} not found in the table")

def generate_qaoa_params(p):
    return list(np.random.uniform(0, 2*np.pi, 2*p))

def reconstruct_unitary(decomposition, n, param_dict):
    pauli_group = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    
    unitary = np.zeros((2**n, 2**n), dtype=np.complex128)
    
    for character, indices in decomposition:
        g = pauli_group[indices[0]]
        for i in indices[1:]:
            g = np.kron(g, pauli_group[i])
        coefficient = complex(character.subs(param_dict).evalf())
        unitary += coefficient * g
    
    return unitary

def run_benchmark(circuit, backend, num_measurements, num_qubits, shots=1024):
    running_sum = running_sum_squared = 0.0
    
    for i in range(num_measurements):
        print(f"Measurement {i+1}/{num_measurements}")
        start_time = time.time()
        
        if circuit.num_parameters > 0:
            params = generate_qaoa_params(circuit.num_parameters // 2)
            bound_circuit = circuit.assign_parameters(params)
        else:
            bound_circuit = circuit
        
        transpiled_circuit = transpile(bound_circuit, backend, optimization_level=1)
        
        try:
            job = backend.run(transpiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
            marginal_counts = qiskit_marginal_counts(counts, range(num_qubits))
            most_common = max(marginal_counts, key=marginal_counts.get)
            
            execution_time = time.time() - start_time
            running_sum += execution_time
            running_sum_squared += execution_time ** 2
            
            print(f"Measurement {i+1} completed. Most common result: {most_common}")
        except Exception as e:
            print(f"Error in measurement {i+1}: {str(e)}")
            print(traceback.format_exc())
            return None, None, circuit

    mean_time = running_sum / num_measurements
    std_dev = np.sqrt((running_sum_squared / num_measurements) - (mean_time ** 2))
    
    return mean_time, std_dev, circuit

def run_original_benchmark(n, p, backend, num_measurements):
    print(f"Running original benchmark for {n} qubits and p={p}")
    qubo = create_maxcut_problem(n)
    qaoa_circ = qaoa_circuit(qubo, p)
    return run_benchmark(qaoa_circ, backend, num_measurements, n)

def run_optimized_benchmark(n, p, backend, num_measurements):
    print(f"Running optimized benchmark for {n} qubits and p={p}")
    
    try:
        decomposition, params = get_character_decomposition(n, p)
        param_values = generate_qaoa_params(len(params))
        param_dict = dict(zip(params, param_values))
        
        unitary = reconstruct_unitary(decomposition, n, param_dict)
        
        optimized_circuit = QuantumCircuit(n)
        optimized_circuit.unitary(unitary, range(n))
        optimized_circuit.measure_all()
        
        return run_benchmark(optimized_circuit, backend, num_measurements, n)
    except Exception as e:
        print(f"Error in optimized benchmark: {str(e)}")
        print(traceback.format_exc())
        return None, None, None

def run_benchmark_suite(n, p, num_measurements, backend):
    print(f"\nStarting benchmark for {n} qubits and p={p}")
    try:
        with benchmark_context():
            orig_mean, orig_std, orig_circuit = run_original_benchmark(n, p, backend, num_measurements)
        print(f"Original benchmark for {n} qubits and p={p} completed")
        
        wait_for_resources()
        
        with benchmark_context():
            opt_mean, opt_std, opt_circuit = run_optimized_benchmark(n, p, backend, num_measurements)
        
        if opt_mean is None or opt_std is None:
            print(f"Optimized benchmark for {n} qubits and p={p} failed")
            return n, p, orig_mean, orig_std, None, None, None
        
        print(f"Optimized benchmark for {n} qubits and p={p} completed")
        
        wait_for_resources()
        
        analysis = {
            'total_variation_distance': None,
            'orig_counts': None,
            'opt_counts': None
        }
        
        return n, p, orig_mean, orig_std, opt_mean, opt_std, analysis
    except Exception as e:
        print(f"Error in benchmark for {n} qubits and p={p}: {str(e)}")
        print(traceback.format_exc())
        return n, p, None, None, None, None, None

def plot_results(results, num_measurements):
    qubit_numbers = sorted(set(r[0] for r in results))
    p_values = sorted(set(r[1] for r in results))
    
    for p in p_values:
        plt.figure(figsize=(12, 6))
        original_means = [r[2] for r in results if r[1] == p and r[2] is not None]
        original_stds = [r[3] for r in results if r[1] == p and r[3] is not None]
        optimized_means = [r[4] for r in results if r[1] == p and r[4] is not None]
        optimized_stds = [r[5] for r in results if r[1] == p and r[5] is not None]
        valid_qubits = [r[0] for r in results if r[1] == p and r[2] is not None and r[4] is not None]
        
        plt.errorbar(valid_qubits, original_means, yerr=original_stds, fmt='b-o', capsize=5, label='Original Circuit')
        plt.errorbar(valid_qubits, optimized_means, yerr=optimized_stds, fmt='r-o', capsize=5, label='Optimized Circuit')
        plt.xlabel('Number of Qubits')
        plt.ylabel('Mean Runtime (seconds)')
        plt.title(f'Runtime Scaling of QAOA (p={p})\n({num_measurements} measurements per data point)')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'qaoa_runtime_scaling_p{p}.png')
        print(f"\nPlot saved as 'qaoa_runtime_scaling_p{p}.png'")
        plt.close()

def main():
    try:
        backend = AerSimulator(method='statevector', precision='single')
        print(f"Using backend: {backend.name}")
        
        qubit_range = list(range(2, 6))
        p_range = list(range(1, 4))
        num_measurements = 100

        results = []
        for n in qubit_range:
            for p in p_range:
                result = run_benchmark_suite(n, p, num_measurements, backend)
                results.append(result)
                print(f"Completed benchmark for n={n}, p={p}")

        plot_results(results, num_measurements)

        for result in results:
            n, p, orig_mean, orig_std, opt_mean, opt_std, analysis = result
            print(f"\nNumber of qubits: {n}, p: {p}")
            if orig_mean is not None:
                print(f"Original circuit - Mean time: {orig_mean:.6f} s, Std dev: {orig_std:.6f} s")
            if opt_mean is not None:
                print(f"Optimized circuit - Mean time: {opt_mean:.6f} s, Std dev: {opt_std:.6f} s")
    except Exception as e:
        print(f"Unexpected error in main: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()