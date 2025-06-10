# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a BIRADS mammogram classification project using Google's MedGemma-4B multimodal model. The goal is to train an AI model to classify mammograms according to their BIRADS breast density score (A, B, C, or D).

## Architecture

This project follows a medical AI fine-tuning approach:

- **Base Model**: MedGemma-4B (multimodal vision + text model pre-trained on medical data)
- **Fine-tuning Strategy**: LoRA/QLoRA for efficient parameter-efficient fine-tuning
- **Input Format**: Mammogram images + instruction text ("Classify the breast density according to BIRADS categories.")
- **Output Format**: "BIRADS [A/B/C/D]" classification

## Key Technical Components

### Data Pipeline
- DICOM to standard image format conversion using SimpleITK/pydicom
- Preprocessing for MedGemma input standardization
- Train/validation/test splits (70/15/15)
- Class-balanced sampling for handling imbalanced BIRADS categories

### Training Infrastructure
- Uses Hugging Face Transformers ecosystem
- PEFT library for LoRA implementation
- Accelerate for distributed training
- PyTorch as base framework
- Requires 24GB+ VRAM (or 16GB with QLoRA)

### Expected Datasets
- CBIS-DDSM, INbreast, VinDr-Mammo public datasets
- DICOM format medical images with BIRADS density labels

## Development Phases

1. **Phase 1 (Weeks 1-2)**: Data preparation and preprocessing pipeline
2. **Phase 2 (Weeks 3-6)**: Model fine-tuning with LoRA adapters  
3. **Phase 3 (Weeks 7-8)**: Evaluation and clinical validation

## Performance Requirements

- Target: >90% classification accuracy
- Inference: <100ms per image
- Must handle DICOM workflow integration
- Requires explainability (Grad-CAM visualizations)

## Compliance Considerations

- HIPAA compliance for medical data handling
- FDA regulatory pathway planning (510(k) submission)
- Bias auditing across demographics and equipment manufacturers
- Audit trails for all predictions