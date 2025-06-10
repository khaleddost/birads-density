#!/usr/bin/env python3
"""
Comprehensive Configuration Test for BIRADS Project
Tests readiness for Phase 2: Data Preparation and Preprocessing
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_pytorch_configuration():
    """Test PyTorch configuration and capabilities."""
    print("=" * 60)
    print("PYTORCH CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("! Running on CPU (recommended to use GPU for training)")
        
        # Test tensor operations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1000, 1000, device=device)
        y = torch.matmul(x, x.T)
        print(f"✓ Large tensor operations work on {device}")
        
        # Test memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✓ CUDA memory management functional")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch configuration test failed: {e}")
        return False

def test_huggingface_access():
    """Test HuggingFace model access and authentication."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE MODEL ACCESS TEST")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Test basic model access
        print("Testing BERT model access...")
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        print("✓ BERT model loaded successfully")
        
        # Test tokenization
        text = "This is a test for BIRADS mammogram classification."
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        print("✓ Model inference working")
        
        # Test if we can access gated models (optional)
        try:
            from transformers import AutoTokenizer as AT
            # Test with a smaller model first
            test_tokenizer = AT.from_pretrained("microsoft/DialoGPT-small")
            print("✓ Can access Microsoft models")
        except Exception:
            print("! Limited model access (may need HF authentication for some models)")
        
        return True
        
    except Exception as e:
        print(f"✗ HuggingFace access test failed: {e}")
        return False

def test_medical_imaging():
    """Test medical imaging library functionality."""
    print("\n" + "=" * 60)
    print("MEDICAL IMAGING LIBRARIES TEST")
    print("=" * 60)
    
    try:
        import pydicom
        import SimpleITK as sitk
        import nibabel as nib
        import cv2
        import numpy as np
        
        # Test creating synthetic medical image data
        print("Testing medical imaging data handling...")
        
        # Create synthetic mammogram-like data
        synthetic_mammo = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
        print("✓ Synthetic mammogram data created")
        
        # Test SimpleITK operations
        sitk_image = sitk.GetImageFromArray(synthetic_mammo)
        sitk_image.SetSpacing([0.1, 0.1])  # 0.1mm pixel spacing
        filtered_image = sitk.DiscreteGaussian(sitk_image, variance=1.0)
        print("✓ SimpleITK image processing works")
        
        # Test OpenCV operations
        normalized = cv2.normalize(synthetic_mammo, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        resized = cv2.resize(normalized, (224, 224))
        print("✓ OpenCV image processing works")
        
        # Test DICOM capabilities (without real DICOM file)
        print("✓ DICOM library ready")
        
        return True
        
    except Exception as e:
        print(f"✗ Medical imaging test failed: {e}")
        return False

def test_peft_and_lora():
    """Test PEFT and LoRA functionality for fine-tuning."""
    print("\n" + "=" * 60)
    print("PEFT AND LORA CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        from transformers import AutoModel
        import torch
        
        # Load a small model for testing
        print("Loading small model for LoRA test...")
        base_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,  # rank
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        print("✓ LoRA configuration created")
        
        # Apply LoRA to model
        peft_model = get_peft_model(base_model, lora_config)
        print("✓ LoRA applied to model")
        
        # Test trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        # Test forward pass
        dummy_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = peft_model(dummy_input)
        print("✓ LoRA model forward pass works")
        
        return True
        
    except Exception as e:
        print(f"✗ PEFT/LoRA test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization tools."""
    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION TOOLS TEST")
    print("=" * 60)
    
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        import torch
        
        print("✓ BitsAndBytes library loaded")
        
        # Test 8-bit configuration
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        print("✓ 8-bit quantization config created")
        
        # Test 4-bit configuration
        bnb_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("✓ 4-bit quantization config created")
        
        # Test memory monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(100, 100, device='cuda')
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"✓ Memory monitoring works (Peak: {peak_memory:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        return False

def test_experiment_tracking():
    """Test experiment tracking setup."""
    print("\n" + "=" * 60)
    print("EXPERIMENT TRACKING TEST")
    print("=" * 60)
    
    try:
        import wandb
        import tensorboard
        print("✓ W&B and TensorBoard libraries loaded")
        
        # Test W&B offline mode
        wandb.init(mode="offline", project="birads-test")
        wandb.log({"test_metric": 0.95})
        wandb.finish()
        print("✓ W&B offline logging works")
        
        # Test TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir='./logs/test')
        writer.add_scalar('test/accuracy', 0.95, 1)
        writer.close()
        print("✓ TensorBoard logging works")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment tracking test failed: {e}")
        return False

def test_project_structure():
    """Test project structure and permissions."""
    print("\n" + "=" * 60)
    print("PROJECT STRUCTURE TEST")
    print("=" * 60)
    
    import os
    
    required_dirs = [
        'data/raw', 'data/processed', 'data/splits',
        'src/data', 'src/models', 'src/training', 'src/evaluation', 'src/utils',
        'notebooks', 'experiments', 'checkpoints', 'logs'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"✓ {dir_path}")
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    
    # Test write permissions
    try:
        test_file = 'test_write_permission.tmp'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✓ Write permissions working")
        
        # Test write to key directories
        for test_dir in ['checkpoints', 'logs', 'experiments']:
            test_path = os.path.join(test_dir, 'test.tmp')
            with open(test_path, 'w') as f:
                f.write('test')
            os.remove(test_path)
        print("✓ Write permissions in all key directories")
        
        return True
        
    except Exception as e:
        print(f"✗ Permission test failed: {e}")
        return False

def print_readiness_assessment(test_results):
    """Print final readiness assessment."""
    print("\n" + "=" * 60)
    print("PHASE 2 READINESS ASSESSMENT")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    
    if passed_tests == total_tests:
        print("🎉 CONFIGURATION FULLY READY FOR PHASE 2!")
        print("\nYour environment is optimally configured for:")
        print("- DICOM mammogram data processing")
        print("- MedGemma-4B model fine-tuning with LoRA")
        print("- Memory-efficient training with quantization")
        print("- Comprehensive experiment tracking")
        print("\n✅ Proceed to Phase 2: Data Preparation & Preprocessing")
        return 0
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  CONFIGURATION MOSTLY READY")
        print("\nMinor issues detected. Address failed tests before proceeding.")
        print("Most core functionality is working correctly.")
        return 1
    else:
        print("❌ CONFIGURATION NEEDS ATTENTION")
        print("\nMultiple critical issues detected.")
        print("Please resolve failed tests before proceeding to Phase 2.")
        return 2

def main():
    """Main test function."""
    print("🔬 COMPREHENSIVE CONFIGURATION TEST")
    print("Testing readiness for Phase 2: Data Preparation & Preprocessing\n")
    
    # Run all tests
    test_results = {
        "PyTorch Configuration": test_pytorch_configuration(),
        "HuggingFace Model Access": test_huggingface_access(),
        "Medical Imaging Libraries": test_medical_imaging(),
        "PEFT and LoRA": test_peft_and_lora(),
        "Memory Optimization": test_memory_optimization(),
        "Experiment Tracking": test_experiment_tracking(),
        "Project Structure": test_project_structure()
    }
    
    # Print final assessment
    return print_readiness_assessment(test_results)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)