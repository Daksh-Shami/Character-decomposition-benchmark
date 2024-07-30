import numpy as np
import cupy as cp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator as QiskitOperator
import pickle
from itertools import product
import os

# Initialize CUDA device
cp.cuda.Device(0).use()

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

def character_decomposition(circuit, n):
    print(f"  Performing character decomposition for {n} qubits")

    I = cp.eye(2, dtype=cp.complex128)
    X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
    Y = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
    Z = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)

    pauli_group = [I, X, Y, Z]

    circuit_no_measure = circuit.copy()
    circuit_no_measure.remove_final_measurements()
    unitary = cp.array(QiskitOperator(circuit_no_measure).data, dtype=cp.complex128)

    decomposition = []
    d = 2**(n+1)  # Dimension of the Hilbert space (including ancilla)

    for pauli_indices in product(range(4), repeat=n+1):  # Include ancilla qubit
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
        a_values = [''.join(x) for x in np.array(np.meshgrid(*[[0,1]]*n)).T.reshape(-1, n).astype(str)]
        for a in a_values:
            circuit = bernstein_vazirani_circuit(n, a)
            decomposed = character_decomposition(circuit, n)
            character_table[(n, a)] = decomposed
    return character_table

def save_character_table(character_table, filename):
    with open(filename, 'wb') as f:
        pickle.dump(character_table, f)

if __name__ == "__main__":
    max_qubits = 5  # Adjust this based on your needs and computational resources
    table = generate_character_table(max_qubits)
    save_character_table(table, 'bv_character_table.pkl')
    print(f"Character table saved to bv_character_table.pkl")