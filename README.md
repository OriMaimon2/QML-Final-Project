# Quantum Generative Adversarial Networks (QGAN)

A complete implementation of Quantum Generative Adversarial Networks based on the paper "Quantum Generative Adversarial Networks" by Seth Lloyd and Christian Weedbrook.

## 📚 Overview

This project provides a full environment for training QGANs both numerically (using quantum simulators) and on real IBM quantum hardware. The implementation includes:

- **Quantum Generator**: Parameterized quantum circuits that generate samples
- **Quantum Discriminator**: Quantum circuits that distinguish between real and generated data
- **Training Loop**: Complete adversarial training implementation
- **Evaluation Tools**: Metrics and visualizations for assessing QGAN performance
- **Hardware Integration**: Support for IBM Quantum hardware

## 🚀 Features

- **Hybrid Quantum-Classical**: Combines quantum circuits with classical optimization
- **Flexible Architecture**: Configurable circuit depths and qubit counts
- **Multiple Backends**: Support for simulators and real quantum hardware
- **Comprehensive Evaluation**: Wasserstein distance, KL divergence, and distribution analysis
- **Visualization Tools**: Training curves, data distributions, and correlation analysis
- **Parameter Sensitivity**: Analysis tools for optimizing circuit parameters

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Installation

```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
# Core quantum computing packages
pip install qiskit==1.1.0 qiskit-aer==0.13.1 qiskit-algorithms==0.2.1

# Scientific computing
pip install numpy scipy matplotlib seaborn

# Machine learning
pip install torch torchvision

# Utilities
pip install tqdm pandas scikit-learn jupyter ipykernel
```

### IBM Quantum Hardware (Optional)

To use real quantum hardware:

1. Sign up at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API token
3. Install IBM runtime: `pip install qiskit-ibm-runtime`
4. Set environment variable: `export QISKIT_IBM_TOKEN=your_token`

## 🎯 Quick Start

### Basic Usage

```python
from qgan_implementation import QGAN, QGANConfig

# Configure QGAN
config = QGANConfig(
    num_qubits=4,
    generator_depth=3,
    discriminator_depth=2,
    num_epochs=100,
    batch_size=32,
    backend_type="aer_simulator",
    shots=1000
)

# Initialize and train
qgan = QGAN(config)
qgan.train()

# Generate samples
real_data = qgan.generate_real_data(1000)
generated_samples = qgan.generate_samples(1000)

# Evaluate performance
evaluation = qgan.evaluate(real_data, generated_samples)
print(f"Wasserstein Distance: {evaluation['wasserstein_distance']:.4f}")

# Visualize results
qgan.plot_training_history()
qgan.plot_data_distribution(real_data, generated_samples)
```

### Run Demo

```bash
# Run the complete demo
python QGAN_Notebook.py
```

## 📊 Configuration Options

The `QGANConfig` class allows you to customize various aspects:

```python
@dataclass
class QGANConfig:
    # Circuit parameters
    num_qubits: int = 4              # Number of qubits
    generator_depth: int = 3         # Generator circuit depth
    discriminator_depth: int = 2     # Discriminator circuit depth
    
    # Training parameters
    num_epochs: int = 100            # Training epochs
    batch_size: int = 32             # Batch size
    generator_lr: float = 0.01       # Generator learning rate
    discriminator_lr: float = 0.01   # Discriminator learning rate
    
    # Quantum backend
    backend_type: str = "aer_simulator"  # "aer_simulator" or "ibm_quantum"
    shots: int = 1000                # Number of shots for measurements
    
    # Data parameters
    data_dim: int = 4                # Data dimensionality
    noise_dim: int = 4               # Noise dimensionality
```

## 🔬 Advanced Usage

### Custom Data Distributions

```python
# Modify the generate_real_data method in QGAN class
def generate_real_data(self, num_samples: int) -> np.ndarray:
    """Generate your custom data distribution"""
    # Example: Mixture of Gaussians
    data = np.random.normal(0, 1, (num_samples, self.config.data_dim))
    data[:, 0] = np.sin(data[:, 1]) + 0.1 * np.random.normal(0, 1, num_samples)
    return data
```

### Custom Circuit Architectures

```python
# Modify the _build_circuit method in QuantumGenerator/QuantumDiscriminator
def _build_circuit(self) -> QuantumCircuit:
    """Build custom quantum circuit architecture"""
    # Your custom circuit design here
    pass
```

### Parameter Sensitivity Analysis

```python
# Test different parameters
depths = [1, 2, 3, 4]
results = {}

for depth in depths:
    config = QGANConfig(generator_depth=depth, discriminator_depth=depth)
    qgan = QGAN(config)
    qgan.train()
    # Evaluate and store results
```

## 📈 Evaluation Metrics

The implementation provides several evaluation metrics:

- **Wasserstein Distance**: Measures distribution similarity
- **KL Divergence**: Information-theoretic distance
- **Statistical Moments**: Mean, variance comparison
- **Correlation Analysis**: Feature relationship preservation
- **Training Stability**: Generator/Discriminator loss ratios

## 🔗 IBM Quantum Hardware

To use real quantum hardware:

```python
# Configure for IBM Quantum
config = QGANConfig(
    num_qubits=2,           # Small circuits for hardware constraints
    generator_depth=1,      # Shallow depth
    discriminator_depth=1,  # Shallow depth
    num_epochs=10,          # Few epochs
    batch_size=8,           # Small batch size
    backend_type="ibm_quantum",
    shots=100               # Fewer shots
)

qgan = QGAN(config)
qgan.train()
```

**Important Notes for Hardware:**
- Limited qubit count (typically 5-27 qubits)
- High noise levels
- Limited circuit depth
- Longer execution times
- Cost considerations

## 📚 Theoretical Background

### QGAN Architecture

The QGAN consists of two main components:

1. **Quantum Generator (G)**: A parameterized quantum circuit that generates samples from a learned distribution
2. **Quantum Discriminator (D)**: A quantum circuit that distinguishes between real and generated data

### Training Process

1. **Generator Training**: Optimize G to fool D
2. **Discriminator Training**: Optimize D to distinguish real from fake
3. **Adversarial Equilibrium**: Both networks improve until convergence

### Quantum Advantages

- **Expressiveness**: Quantum circuits can represent complex distributions
- **Entanglement**: Natural representation of correlated data
- **Quantum Features**: Potential quantum advantage for certain problems

## 🛠️ Troubleshooting

### Common Issues

1. **Qiskit Import Errors**
   ```bash
   pip uninstall qiskit qiskit-terra
   pip install qiskit==1.1.0
   ```

2. **Memory Issues**
   - Reduce `num_qubits` or `shots`
   - Use smaller `batch_size`

3. **Training Instability**
   - Adjust learning rates
   - Change circuit depth
   - Modify loss functions

4. **Hardware Connection Issues**
   - Verify IBM Quantum token
   - Check internet connection
   - Ensure account has credits

### Performance Tips

- Start with small circuits (2-4 qubits)
- Use simulators for development
- Increase shots for better statistics
- Monitor training stability

## 📖 References

- Lloyd, S., & Weedbrook, C. (2018). Quantum generative adversarial learning. Physical Review Letters, 121(4), 040502.
- Qiskit Documentation: https://qiskit.org/documentation/
- IBM Quantum: https://quantum-computing.ibm.com/

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- More sophisticated optimization algorithms
- Additional circuit architectures
- Error mitigation techniques
- Performance optimizations
- Documentation improvements

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Seth Lloyd and Christian Weedbrook for the original QGAN paper
- IBM Quantum team for Qiskit framework
- Quantum computing community for ongoing research

---

**Happy Quantum Computing! 🚀⚛️**
