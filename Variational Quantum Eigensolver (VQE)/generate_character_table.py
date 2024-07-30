import numpy as np
import cupy as cp
from qiskit import QuantumCircuit, QuantumRegister
import pickle
from itertools import product
import os

# Initialize CUDA device
cp.cuda.Device(0).use()

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

from qiskit.quantum_info import Operator

def character_decomposition(circuit, n):
    print(f"  Performing character decomposition for {n} qubits")

    I = cp.eye(2, dtype=cp.complex128)
    X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    Y = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    Z = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)

    pauli_group = [I, X, Y, Z]

    unitary = Operator(circuit).data
    unitary = cp.array(unitary, dtype=cp.complex128)

    decomposition = []
    d = 2**n  # Dimension of the Hilbert space

    for pauli_indices in product(range(4), repeat=n):
        P = pauli_group[pauli_indices[0]]
        for idx in pauli_indices[1:]:
            P = cp.kron(P, pauli_group[idx])
        
        character = cp.trace(cp.matmul(P.conj().T, unitary)) / d
        
        if cp.abs(character) > 1e-10:  # Only store non-negligible terms
            decomposition.append((character, pauli_indices))

    return decomposition

def generate_character_table(max_qubits):
    character_table = {}
    for n in range(2, max_qubits + 1):
        print(f"Generating character table for {n} qubits")
        
        # Generate random parameters for the ansatz
        params = np.random.uniform(0, 2 * np.pi, size=2 * n)
        
        circuit = vqe_ansatz(params, n)
        decomposed = character_decomposition(circuit, n)
        character_table[n] = decomposed
    return character_table

def save_character_table(character_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(character_table, f)

if __name__ == "__main__":
    max_qubits = 7  # Adjust this based on your needs and computational resources
    table = generate_character_table(max_qubits)
    save_character_table(table, 'vqe_character_table.pkl')
    print(f"Character table saved to vqe_character_table.pkl")