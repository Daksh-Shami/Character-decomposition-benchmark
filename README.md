# Quantum Forge Benchmark Code

This repository contains the benchmark code used in the paper "Bridging Classical and Quantum: Group-Theoretic Approach to Quantum Circuit Simulation" by Daksh Shami. The code demonstrates the performance improvements achieved by the character function decomposition method for simulating quantum circuits.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Benchmarked Algorithms](#benchmarked-algorithms)
- [Results](#results)
- [Contributing](#contributing)

## Overview

The benchmark code compares the runtime of the character function decomposition method implemented in Quantum Forge with state-of-the-art simulators like Qiskit. The code covers several important quantum algorithms, each contained in its own directory:

- Bernstein-Vazirani
- Quantum Fourier Transform (QFT)
- Grover's Search
- Variational Quantum Eigensolver (VQE)

## Requirements

To run the benchmark code, you'll need:

- Python 3.7+
- Qiskit 1.1.0+
- Quantum Forge (latest version)
- NumPy
- Matplotlib

## Usage

Each benchmarked algorithm is contained in its own directory. To run the benchmarks for a specific algorithm:

1. Navigate to the algorithm's directory:


bash cd bernstein-vazirani

2. Generate the character table:


bash python generate_character_table.py

3. Run the benchmarks:


bash python run_benchmarks.py

Repeat these steps for each algorithm you want to benchmark.

## Benchmarked Algorithms

The following quantum algorithms are benchmarked:

1. **Bernstein-Vazirani**: A quantum algorithm for solving the Bernstein-Vazirani problem.
2. **Quantum Fourier Transform (QFT)**: A fundamental quantum algorithm for performing the discrete Fourier transform.
3. **Grover's Search**: A quantum search algorithm that provides a quadratic speedup over classical search algorithms.
4. **Variational Quantum Eigensolver (VQE)**: A hybrid quantum-classical algorithm for finding the lowest eigenvalue of a Hamiltonian.

## Results

The benchmark results, including performance comparison plots, can be found in the images within each algorithm's folder. Each benchmark script is configured to also generate graphs.

## Contributing

Contributions to the benchmark code are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
