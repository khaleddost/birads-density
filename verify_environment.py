#!/usr/bin/env python3
"""
Environment Verification Script for BIRADS Mammogram Classification Project
Tests all installed packages and their core functionality for Phase 1 setup.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all core library imports."""
    print("=" * 60)
    print("PHASE 1 ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    imports_to_test = [
        # Core Python utilities
        ("numpy", "np"),
        ("pandas", "pd"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("h5py", "h5py"),
        ("PIL", "PIL"),
        ("einops", "einops"),
        
        # Medical imaging
        ("pydicom", "pydicom"),
        ("SimpleITK", "sitk"),
        ("nibabel", "nib"),
        ("cv2", "cv2"),
        
        # Core ML
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        
        # HuggingFace ecosystem
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        
        # Additional ML tools
        ("sklearn", "sklearn"),
        ("scipy", "scipy"),
        ("bitsandbytes", "bnb"),
        
        # Visualization
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("plotly", "plotly"),
        
        # Experiment tracking
        ("wandb", "wandb"),
        ("tensorboard", "tensorboard"),
        ("gradio", "gr"),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, alias in imports_to_test:
        try:
            exec(f"import {module_name} as {alias}")
            successful_imports.append(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            failed_imports.append((module_name, str(e)))
            print(f"✗ {module_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"IMPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Successful imports: {len(successful_imports)}/{len(imports_to_test)}")
    
    if failed_imports:
        print("\nFailed imports:")
        for module, error in failed_imports:
            print(f"  - {module}: {error}")
    
    return len(failed_imports) == 0

def test_core_functionality():
    """Test basic functionality of key libraries."""
    print(f"\n{'='*60}")
    print("CORE FUNCTIONALITY TESTS")
    print(f"{'='*60}")
    
    try:
        # Test PyTorch
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ PyTorch device: {device}")
        
        # Test small tensor operations
        x = torch.randn(2, 3).to(device)
        y = torch.matmul(x, x.T)
        print(f"✓ PyTorch tensor operations work")
        
        # Test Transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        text = "This is a test."
        tokens = tokenizer(text, return_tensors="pt")
        print(f"✓ Transformers tokenization works")
        
        # Test medical imaging
        import pydicom
        import SimpleITK as sitk
        print(f"✓ Medical imaging libraries ready")
        
        # Test PEFT
        import peft
        print(f"✓ PEFT library ready")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def print_system_info():
    """Print system and environment information."""
    print(f"\n{'='*60}")
    print("SYSTEM INFORMATION")
    print(f"{'='*60}")
    
    print(f"Python version: {sys.version}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not available")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not available")

def main():
    """Main verification function."""
    print("Starting BIRADS project environment verification...\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test functionality if imports succeed
    if imports_ok:
        functionality_ok = test_core_functionality()
    else:
        functionality_ok = False
    
    # Print system info
    print_system_info()
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL VERIFICATION RESULT")
    print(f"{'='*60}")
    
    if imports_ok and functionality_ok:
        print("🎉 ENVIRONMENT SETUP SUCCESSFUL!")
        print("Your Phase 1 environment is ready for BIRADS mammogram classification.")
        print("\nNext steps:")
        print("- Phase 2: Set up data preprocessing pipeline")
        print("- Phase 3: Begin model fine-tuning")
        return 0
    else:
        print("❌ ENVIRONMENT SETUP INCOMPLETE")
        print("Please address the failed imports/tests above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)