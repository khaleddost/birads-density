# BIRADS Mammogram Classification

AI model to classify mammograms according to their BIRADS breast density score (A, B, C, or D) using Google's MedGemma-4B multimodal model.

## Overview

This project implements a fine-tuned version of MedGemma-4B for automated BIRADS breast density classification from mammogram images. The model leverages the multimodal capabilities of MedGemma to process both imaging data and clinical context.

## Project Status

🚧 **In Development** - Currently in planning and setup phase

## Key Features

- **Multimodal Classification**: Combines mammogram images with textual instructions
- **Medical AI Foundation**: Built on MedGemma-4B, pre-trained on medical data
- **Efficient Fine-tuning**: Uses LoRA/QLoRA for parameter-efficient training
- **Clinical Integration**: Designed for DICOM workflow compatibility
- **Explainable AI**: Includes visualization capabilities for clinical decision support

## Target Performance

- Classification accuracy: >90%
- Inference time: <100ms per image
- BIRADS categories: A, B, C, D density classifications

## Development Roadmap

See [project-plan.md](project-plan.md) for detailed implementation timeline and technical specifications.

## Requirements

- GPU with 24GB+ VRAM (or 16GB with QLoRA)
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Medical imaging libraries (SimpleITK, pydicom)

## License

[To be determined]

## Contact

[Project team information to be added]