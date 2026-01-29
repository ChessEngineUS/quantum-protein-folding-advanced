# Advanced Quantum Protein Folding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/quantum-protein-folding-advanced/blob/main/quantum_protein_folding_advanced.ipynb)

## Overview

This repository presents a **rigorous, state-of-the-art implementation** of quantum algorithms for protein structure prediction using the Hydrophobic-Polar (HP) lattice model. Building upon baseline benchmarks, this work advances the field through:

### Key Innovations

1. **Extended Sequence Length**: Support for 8-20 qubit proteins (vs. 8-14 baseline)
2. **Advanced Quantum Algorithms**:
   - Adaptive VQE (ADAPT-VQE) with operator pool selection
   - Quantum Natural Gradient Descent
   - Multi-start QAOA with parameter transfer
   - Hardware-efficient ansatz with entanglement strategies

3. **Enhanced Energy Landscape Analysis**:
   - Multiple local minima detection
   - Energy barrier mapping
   - Convergence trajectory visualization
   - Statistical significance testing (t-tests, ANOVA)

4. **Sophisticated Noise Mitigation**:
   - Zero-noise extrapolation
   - Probabilistic error cancellation
   - Measurement error mitigation
   - Realistic device noise models (IBM, Rigetti, IonQ)

5. **Comprehensive Benchmarking**:
   - 5,000+ experimental runs
   - Classical baselines: MCMC, simulated annealing, genetic algorithms
   - Performance metrics: energy gap, success rate, convergence speed
   - Reproducibility analysis with confidence intervals

## Benchmark Comparison

| Metric | Baseline Study | This Work |
|--------|----------------|------------|
| Total Experiments | 2,280 | 5,000+ |
| Sequence Length | 8-14 qubits | 8-20 qubits |
| Quantum Algorithms | 3 (QAOA, VQE, Hybrid) | 5 (+ ADAPT-VQE, QNG) |
| Classical Baselines | 1 method | 4 methods |
| Avg Energy Gap (Quantum) | 1.42 (QAOA) | **Target: <1.2** |
| Success Rate | 63.2% | **Target: >75%** |
| Runtime | Not specified | **<30 min per run** |

## Features

### Quantum Circuit Design
- **Parameterized ansatz**: Multiple entanglement strategies (linear, circular, full)
- **Adaptive depth**: Dynamic circuit construction based on convergence
- **Gradient-based optimization**: Classical and quantum natural gradients
- **Parameter initialization**: Warm-start from smaller systems

### Energy Model
- **HP lattice Hamiltonian**: Exact ground state calculation
- **Constraint handling**: Self-avoiding walk enforcement
- **Energy landscape**: Visualization and analysis tools

### Noise Analysis
- **Depolarizing noise**: Configurable error rates
- **Gate fidelity models**: T1/T2 decoherence
- **Readout errors**: Measurement calibration
- **Error mitigation**: Multiple strategies benchmarked

## Installation

```bash
git clone https://github.com/ChessEngineUS/quantum-protein-folding-advanced.git
cd quantum-protein-folding-advanced
pip install -r requirements.txt
```

## Quick Start

Open the Google Colab notebook for immediate execution:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChessEngineUS/quantum-protein-folding-advanced/blob/main/quantum_protein_folding_advanced.ipynb)

Or run locally:

```python
from quantum_protein_folding import AdvancedProteinFolder

# Define HP sequence
sequence = "HPHPPHHPHPPHPHHPPHPH"  # 20-mer

# Initialize folder with ADAPT-VQE
folder = AdvancedProteinFolder(
    sequence=sequence,
    algorithm='adapt-vqe',
    max_depth=8,
    noise_model='ibm_lagos',
    error_mitigation='zne'
)

# Run optimization
results = folder.fold(max_iterations=1000, time_limit=1800)

# Analyze results
folder.visualize_convergence()
folder.plot_structure(results['best_configuration'])
print(f"Energy gap: {results['energy_gap']:.3f}")
print(f"Success rate: {results['success_rate']:.2%}")
```

## Methodology

### HP Lattice Model
Proteins are represented as sequences of Hydrophobic (H) and Polar (P) residues on a 2D square lattice. The energy function minimizes H-H contacts:

```
E = -Σ c_ij
```

where c_ij = 1 if residues i and j are non-adjacent in sequence but adjacent on the lattice.

### Quantum Encoding
- **Position encoding**: Binary representation of lattice coordinates
- **Turn encoding**: Relative angles between consecutive residues
- **Constraint satisfaction**: Penalty terms for invalid configurations

### Optimization Algorithms

1. **QAOA**: p-layer alternating operator ansatz
2. **VQE**: Hardware-efficient ansatz with layered structure
3. **ADAPT-VQE**: Greedy operator pool selection
4. **Hybrid VQE-QAOA**: Combined approach with transfer learning
5. **QNG-VQE**: Quantum natural gradient descent

## Results

Detailed results are provided in the notebook, including:

- **Statistical Analysis**: Mean energy gaps with 95% confidence intervals
- **Scaling Behavior**: Performance vs. sequence length
- **Noise Impact**: Comparison across error rates and mitigation strategies
- **Classical Benchmarks**: Head-to-head comparisons with MCMC, SA, GA
- **Reproducibility**: Multiple runs with variance analysis

## Performance Targets

✅ **Runtime**: <30 minutes per protein on Google Colab (T4/A100 GPU)
✅ **Energy Gap**: <1.2 average across all sequences
✅ **Success Rate**: >75% finding optimal or near-optimal structures
✅ **Scalability**: Support for 20-qubit systems
✅ **Reproducibility**: Full experimental logs and random seeds

## Citation

If you use this code, please cite:

```bibtex
@software{marena2026quantum,
  author = {Marena, Tommaso R.},
  title = {Advanced Quantum Protein Folding: Beyond Baseline Benchmarks},
  year = {2026},
  url = {https://github.com/ChessEngineUS/quantum-protein-folding-advanced}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Builds upon foundational work in quantum protein folding benchmarking. Implements algorithms from Qiskit, PennyLane, and Cirq ecosystems.

## Contact

Tommaso R. Marena  
GitHub: [@ChessEngineUS](https://github.com/ChessEngineUS)

---

**Status**: Active Development | Last Updated: January 2026