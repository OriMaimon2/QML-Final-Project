#!/usr/bin/env python3
"""
Quick QGAN Test Script
Run this to test if everything is working correctly.
"""

try:
    from qgan_implementation import QGAN, QGANConfig
    import numpy as np
    
    print("🚀 Testing QGAN Implementation...")
    
    # Simple configuration for testing
    config = QGANConfig(
        num_qubits=2,
        generator_depth=1,
        discriminator_depth=1,
        num_epochs=5,
        batch_size=4,
        shots=50
    )
    
    # Initialize QGAN
    qgan = QGAN(config)
    print("✅ QGAN initialized successfully")
    
    # Quick training test
    print("🔄 Running quick training test...")
    qgan.train()
    print("✅ Training completed successfully")
    
    # Generate samples
    real_data = qgan.generate_real_data(20)
    generated_samples = qgan.generate_samples(20)
    print(f"✅ Generated {len(generated_samples)} samples")
    
    # Evaluate
    evaluation = qgan.evaluate(real_data, generated_samples)
    print(f"✅ Evaluation completed. Wasserstein distance: {evaluation['wasserstein_distance']:.4f}")
    
    print("\n🎉 All tests passed! QGAN is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please check that all dependencies are installed correctly.")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    print("Please check the implementation and try again.")
