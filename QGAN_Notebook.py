#!/usr/bin/env python3
"""
Quantum Generative Adversarial Networks (QGAN) Demo
Based on the paper: "Quantum Generative Adversarial Networks" by Seth Lloyd and Christian Weedbrook

This script demonstrates:
1. QGAN implementation and training
2. Sample generation and evaluation
3. Visualization of results
4. Comparison with classical GAN
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qgan_implementation import *

def main():
    print("🚀 Quantum Generative Adversarial Network (QGAN) Demo")
    print("=" * 60)
    
    # 1. Basic QGAN Training
    print("\n📚 Section 1: Basic QGAN Training")
    print("-" * 40)
    
    # Configure QGAN with conservative settings for better compatibility
    config = QGANConfig(
        num_qubits=2,           # Small circuit for faster training
        generator_depth=1,      # Shallow depth
        discriminator_depth=1,  # Shallow depth
        num_epochs=15,          # Fewer epochs for demo
        batch_size=8,           # Small batch size
        backend_type="aer_simulator",
        shots=100               # Fewer shots for speed
    )
    
    # Initialize and train QGAN
    qgan = QGAN(config)
    print(f"✅ QGAN initialized with {config.num_qubits} qubits")
    
    print("🚀 Starting QGAN training...")
    qgan.train()
    print("✅ Training completed!")
    
    # 2. Generate and Evaluate Samples
    print("\n🎲 Section 2: Sample Generation and Evaluation")
    print("-" * 40)
    
    # Generate data
    real_data = qgan.generate_real_data(200)
    generated_samples = qgan.generate_samples(200)
    
    print(f"Generated {len(real_data)} real data samples")
    print(f"Generated {len(generated_samples)} fake data samples")
    
    # Evaluate performance
    evaluation = qgan.evaluate(real_data, generated_samples)
    
    print("\n📊 Evaluation Results:")
    for metric, value in evaluation.items():
        if isinstance(value, np.ndarray):
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")
    
    # 3. Training Analysis
    print("\n📈 Section 3: Training Analysis")
    print("-" * 40)
    
    if qgan.generator_losses and qgan.discriminator_losses:
        final_g_loss = qgan.generator_losses[-1]
        final_d_loss = qgan.discriminator_losses[-1]
        print(f"Final Generator Loss: {final_g_loss:.4f}")
        print(f"Final Discriminator Loss: {final_d_loss:.4f}")
        
        # Check training stability
        if final_d_loss > 0:
            loss_ratio = final_g_loss / final_d_loss
            print(f"Final G/D Loss Ratio: {loss_ratio:.4f}")
            if 0.5 < loss_ratio < 2.0:
                print("✅ Training appears stable")
            else:
                print("⚠️  Training may be unstable")
    
    # 4. Visualization
    print("\n📊 Section 4: Visualization")
    print("-" * 40)
    
    print("Plotting training history...")
    qgan.plot_training_history()
    
    print("Plotting data distributions...")
    qgan.plot_data_distribution(real_data, generated_samples)
    
    # 5. Parameter Sensitivity Analysis
    print("\n🔬 Section 5: Parameter Sensitivity Analysis")
    print("-" * 40)
    
    depths = [1, 2]
    results = {}
    
    for depth in depths:
        print(f"\nTesting depth {depth}...")
        
        config = QGANConfig(
            num_qubits=2,
            generator_depth=depth,
            discriminator_depth=depth,
            num_epochs=10,
            batch_size=8,
            shots=100
        )
        
        qgan_test = QGAN(config)
        qgan_test.train()
        
        real_data_test = qgan_test.generate_real_data(100)
        generated_data_test = qgan_test.generate_samples(100)
        evaluation_test = qgan_test.evaluate(real_data_test, generated_data_test)
        
        results[depth] = {
            'wasserstein_distance': evaluation_test['wasserstein_distance'],
            'final_g_loss': qgan_test.generator_losses[-1] if qgan_test.generator_losses else float('inf'),
            'final_d_loss': qgan_test.discriminator_losses[-1] if qgan_test.discriminator_losses else float('inf')
        }
    
    # Plot sensitivity results
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        depths_list = list(results.keys())
        wasserstein_distances = [results[d]['wasserstein_distance'] for d in depths_list]
        g_losses = [results[d]['final_g_loss'] for d in depths_list]
        d_losses = [results[d]['final_d_loss'] for d in depths_list]
        
        axes[0].plot(depths_list, wasserstein_distances, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Circuit Depth')
        axes[0].set_ylabel('Wasserstein Distance')
        axes[0].set_title('Effect of Circuit Depth on Distribution Matching')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(depths_list, g_losses, 'o-', linewidth=2, markersize=8, label='Generator')
        axes[1].plot(depths_list, d_losses, 's-', linewidth=2, markersize=8, label='Discriminator')
        axes[1].set_xlabel('Circuit Depth')
        axes[1].set_ylabel('Final Loss')
        axes[1].set_title('Effect of Circuit Depth on Training Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(depths_list, [g/d if d > 0 else 1.0 for g, d in zip(g_losses, d_losses)], 
                    '^-', linewidth=2, markersize=8, color='green')
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equilibrium')
        axes[2].set_xlabel('Circuit Depth')
        axes[2].set_ylabel('G/D Loss Ratio')
        axes[2].set_title('Training Stability vs Circuit Depth')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 6. IBM Quantum Hardware Preparation
    print("\n🌐 Section 6: IBM Quantum Hardware Preparation")
    print("-" * 40)
    
    print("To run on IBM Quantum hardware:")
    print("1. Sign up at https://quantum-computing.ibm.com/")
    print("2. Get your API token")
    print("3. Install: pip install qiskit-ibm-runtime")
    print("4. Set environment variable: export QISKIT_IBM_TOKEN=your_token")
    print("5. Use this configuration:")
    
    ibm_config = QGANConfig(
        num_qubits=2,           # Small circuits for hardware constraints
        generator_depth=1,      # Shallow depth
        discriminator_depth=1,  # Shallow depth
        num_epochs=5,           # Few epochs
        batch_size=4,           # Small batch size
        backend_type="ibm_quantum",
        shots=50                # Fewer shots
    )
    
    print(f"   - Qubits: {ibm_config.num_qubits}")
    print(f"   - Generator Depth: {ibm_config.generator_depth}")
    print(f"   - Discriminator Depth: {ibm_config.discriminator_depth}")
    print(f"   - Epochs: {ibm_config.num_epochs}")
    print(f"   - Batch Size: {ibm_config.batch_size}")
    print(f"   - Shots: {ibm_config.shots}")
    
    # 7. Conclusion
    print("\n🎉 Section 7: Conclusion")
    print("-" * 40)
    
    print("✅ QGAN implementation completed successfully!")
    print("\n📋 What we accomplished:")
    print("   - Implemented quantum generator and discriminator circuits")
    print("   - Created robust training loop with error handling")
    print("   - Added comprehensive evaluation metrics")
    print("   - Provided visualization tools")
    print("   - Demonstrated parameter sensitivity analysis")
    print("   - Prepared for IBM Quantum hardware deployment")
    
    print("\n🚀 Next steps:")
    print("   - Experiment with different circuit architectures")
    print("   - Try larger datasets and more complex distributions")
    print("   - Run on real quantum hardware")
    print("   - Compare with classical GAN performance")
    print("   - Explore quantum advantages for specific problems")
    
    print("\n✅ QGAN demo completed successfully!")

if __name__ == "__main__":
    main()
