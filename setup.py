#!/usr/bin/env python3
"""
Setup script for Quantum Generative Adversarial Networks (QGAN)
This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core quantum computing packages
    packages = [
        ("qiskit==1.1.0", "Qiskit quantum computing framework"),
        ("qiskit-aer==0.13.1", "Qiskit Aer simulator"),
        ("qiskit-algorithms==0.2.1", "Qiskit algorithms"),
        ("numpy>=1.21.0", "Numerical computing"),
        ("scipy>=1.7.0", "Scientific computing"),
        ("matplotlib>=3.5.0", "Plotting library"),
        ("seaborn>=0.11.0", "Statistical visualization"),
        ("torch>=1.9.0", "PyTorch deep learning"),
        ("torchvision>=0.10.0", "PyTorch computer vision"),
        ("tqdm>=4.62.0", "Progress bars"),
        ("pandas>=1.3.0", "Data manipulation"),
        ("scikit-learn>=1.0.0", "Machine learning"),
        ("jupyter>=1.0.0", "Jupyter notebooks"),
        ("ipykernel>=6.0.0", "Jupyter kernel")
    ]
    
    success_count = 0
    for package, description in packages:
        if run_command(f"{sys.executable} -m pip install {package}", f"Installing {description}"):
            success_count += 1
    
    print(f"\n📊 Installation Summary: {success_count}/{len(packages)} packages installed successfully")
    return success_count == len(packages)

def create_example_script():
    """Create an example script for quick testing"""
    example_script = '''#!/usr/bin/env python3
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
        batch_size=8,
        shots=100
    )
    
    # Initialize QGAN
    qgan = QGAN(config)
    print("✅ QGAN initialized successfully")
    
    # Quick training test
    print("🔄 Running quick training test...")
    qgan.train()
    print("✅ Training completed successfully")
    
    # Generate samples
    real_data = qgan.generate_real_data(100)
    generated_samples = qgan.generate_samples(100)
    print(f"✅ Generated {len(generated_samples)} samples")
    
    # Evaluate
    evaluation = qgan.evaluate(real_data, generated_samples)
    print(f"✅ Evaluation completed. Wasserstein distance: {evaluation['wasserstein_distance']:.4f}")
    
    print("\\n🎉 All tests passed! QGAN is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please check that all dependencies are installed correctly.")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("Please check the implementation and try again.")
'''
    
    with open("test_qgan.py", "w") as f:
        f.write(example_script)
    
    # Make executable on Unix systems
    try:
        os.chmod("test_qgan.py", 0o755)
    except:
        pass
    
    print("✅ Created test_qgan.py - run 'python test_qgan.py' to test the installation")

def setup_ibm_quantum():
    """Setup instructions for IBM Quantum"""
    print("\n🔗 IBM Quantum Setup Instructions:")
    print("=" * 50)
    print("1. Sign up at https://quantum-computing.ibm.com/")
    print("2. Get your API token from the IBM Quantum dashboard")
    print("3. Install IBM runtime: pip install qiskit-ibm-runtime")
    print("4. Set environment variable:")
    print("   - Windows: set QISKIT_IBM_TOKEN=your_token")
    print("   - Linux/Mac: export QISKIT_IBM_TOKEN=your_token")
    print("5. Test connection with the provided examples")

def main():
    """Main setup function"""
    print("🚀 Quantum Generative Adversarial Networks (QGAN) Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Some dependencies failed to install.")
        print("You may need to install them manually or check your internet connection.")
    
    # Create test script
    create_example_script()
    
    # IBM Quantum setup
    setup_ibm_quantum()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run 'python test_qgan.py' to test the installation")
    print("2. Run 'python QGAN_Notebook.py' for the full demo")
    print("3. Check the README.md for detailed usage instructions")
    print("4. Set up IBM Quantum if you want to use real hardware")
    
    print("\n📚 Files created:")
    print("- qgan_implementation.py: Main QGAN implementation")
    print("- QGAN_Notebook.py: Complete demo script")
    print("- test_qgan.py: Quick test script")
    print("- requirements.txt: Dependencies list")
    print("- README.md: Detailed documentation")

if __name__ == "__main__":
    main()
