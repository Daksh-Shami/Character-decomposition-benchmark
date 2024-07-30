import numpy as np
import cupy as cp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator as QiskitOperator
from itertools import product
import pickle
import os
import numpy as np
import random

# Add this at the beginning of your main function
np.random.seed(42)
random.seed(42)

# Initialize CUDA device
cp.cuda.Device(0).use()

def grover_circuit(n, marked_state):
    print(f"Creating Grover circuit for {n} qubits, marked state: {marked_state}")
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(n, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Initialize superposition
    circuit.h(qr)
    
    # Number of Grover iterations
    iterations = int(np.pi/4 * np.sqrt(2**n))
    print(f"Number of Grover iterations: {iterations}")
    
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}")
        # Oracle
        for j in range(n):
            if marked_state[j] == '0':
                circuit.x(qr[j])
        circuit.h(qr[-1])
        circuit.mcx(qr[:-1], qr[-1])
        circuit.h(qr[-1])
        for j in range(n):
            if marked_state[j] == '0':
                circuit.x(qr[j])
        
        # Diffusion operator
        circuit.h(qr)
        circuit.x(qr)
        circuit.h(qr[-1])
        circuit.mcx(qr[:-1], qr[-1])
        circuit.h(qr[-1])
        circuit.x(qr)
        circuit.h(qr)
    
    circuit.measure(qr, cr)
    
    print(f"Grover circuit created with {circuit.num_qubits} qubits and {circuit.size()} gates")
    return circuit

def grover_character_decomposition(circuit, n):
    print(f"Starting character decomposition for {n} qubits")
    circuit_no_measure = circuit.copy()
    circuit_no_measure.remove_final_measurements()
    u = cp.array(QiskitOperator(circuit_no_measure).data, dtype=cp.complex128)
    print(f"Unitary matrix shape: {u.shape}")

    I = cp.eye(2, dtype=cp.complex128)
    X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    Y = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    Z = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)
    
    pauli_group = [I, X, Y, Z]

    decomposition = []
    d_i = 2**n

    total_terms = 4**n
    print(f"Total number of terms to process: {total_terms}")

    for i, indices in enumerate(product(range(4), repeat=n)):
        if i % 1000 == 0:
            print(f"Processing term {i+1}/{total_terms}")
        g = pauli_group[indices[0]]
        for j in indices[1:]:
            g = cp.kron(g, pauli_group[j])
        
        chi_u = cp.trace(cp.matmul(g.conj().T, u))
        coefficient = chi_u / d_i
        
        if abs(coefficient) > 1e-10:  # Only store non-negligible terms
            decomposition.append((coefficient.get(), indices))

    print(f"Number of terms in decomposition: {len(decomposition)}")
    print("Top 5 terms by absolute coefficient value:")
    for coeff, indices in sorted(decomposition, key=lambda x: abs(x[0]), reverse=True)[:5]:
        print(f"  Coefficient: {coeff}, Indices: {indices}")

    return decomposition

def generate_character_table(max_qubits):
    character_table = {}
    for n in range(2, max_qubits + 1):
        print(f"\nGenerating character table for {n} qubits")
        marked_states = [''.join(x) for x in product('01', repeat=n)]
        for marked_state in marked_states:
            print(f"\n  Processing marked state: {marked_state}")
            circuit = grover_circuit(n, marked_state)
            decomposed = grover_character_decomposition(circuit, n)
            character_table[(n, marked_state)] = decomposed
    return character_table

def save_character_table(character_table, filename):
    print(f"Saving character table to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(character_table, f)
    print("Character table saved successfully")

if __name__ == "__main__":
    max_qubits = 7  # Adjust based on your resources
    print(f"Generating character table for up to {max_qubits} qubits")
    table = generate_character_table(max_qubits)
    save_character_table(table, 'grover_character_table.pkl')
    print(f"Character table generation completed")