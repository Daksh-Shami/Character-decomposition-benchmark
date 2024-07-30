import numpy as np
import cupy as cp
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator as QiskitOperator
import pickle
from itertools import product
import os

# Initialize CUDA device
cp.cuda.Device(0).use()

def qft_circuit(n):
    qr = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qr)

    for i in range(n-1, -1, -1):
        qc.h(qr[i])
        for j in range(i-1, -1, -1):
            qc.cp(np.pi/2**(i-j), qr[i], qr[j])

    for i in range(n//2):
        qc.swap(qr[i], qr[n-i-1])

    return qc

def character_decomposition(circuit, n):
    print(f"  Performing character decomposition for {n} qubits")

    I = cp.eye(2, dtype=cp.complex128)
    X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    Y = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    Z = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)

    pauli_group = [I, X, Y, Z]

    unitary = cp.array(QiskitOperator(circuit).data, dtype=cp.complex128)

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
        circuit = qft_circuit(n)
        decomposed = character_decomposition(circuit, n)
        character_table[n] = decomposed
    return character_table

def save_character_table(character_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(character_table, f)

if __name__ == "__main__":
    max_qubits = 7  # Adjust this based on your needs and computational resources
    table = generate_character_table(max_qubits)
    save_character_table(table, 'qft_character_table.pkl')
    print(f"Character table saved to qft_character_table.pkl")