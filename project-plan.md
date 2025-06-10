# BIRADS Mammogram Classification with MedGemma-4B Project Plan

## Project Overview
Train an AI model to classify mammograms according to their BIRADS breast density score (A, B, C, or D) using Google's MedGemma-4B multimodal model as the base.

## 1. Understanding MedGemma-4B
- **Model capabilities**: Multimodal (vision + text) medical AI model
- **Architecture**: Based on Gemma with vision encoder
- **Advantages**: Pre-trained on medical data, understands medical terminology
- **Fine-tuning approach**: Will need to adapt for specific mammogram → BIRADS classification task

## 2. Data Preparation

### Dataset Requirements
- Mammogram images (DICOM or standard formats)
- BIRADS density labels (A, B, C, D)
- Consider including text reports if available (leverage multimodal nature)

### Data Sources
- **Public Datasets**:
  - CBIS-DDSM
  - INbreast
  - VinDr-Mammo
- May need institutional data for better performance

### Preprocessing Steps
- Standardize image sizes/formats for MedGemma input
- Create instruction-following format for fine-tuning
- DICOM to standard image format conversion
- Normalization and standardization
- Train/validation/test split (70/15/15 recommended)

## 3. Fine-tuning Strategy

### Instruction Tuning Approach
```
Input: [Mammogram Image] + "Classify the breast density according to BIRADS categories."
Output: "BIRADS [A/B/C/D]"
```

### Training Configurations
- LoRA or QLoRA for efficient fine-tuning
- Gradient checkpointing for memory efficiency
- Mixed precision training
- Class-balanced sampling
- Appropriate loss functions for multi-class classification

## 4. Environment Setup

### Hardware Requirements
- GPU with 24GB+ VRAM (for 4B model fine-tuning)
- Can use QLoRA for lower VRAM requirements (16GB possible)

### Software Stack
- Hugging Face Transformers
- PEFT for efficient fine-tuning
- Accelerate for distributed training
- PyTorch as base framework
- SimpleITK/pydicom for medical image handling

## 5. Implementation Steps

### Phase 1: Setup & Data (Weeks 1-2)
- Set up MedGemma-4B from Hugging Face
- Prepare mammogram dataset with BIRADS labels
- Create data loaders for image-text pairs
- Implement train/val/test splits
- Develop preprocessing pipeline

### Phase 2: Fine-tuning (Weeks 3-6)
- Configure LoRA/QLoRA adapters
- Design prompt templates for classification
- Implement training loop with:
  - Class-balanced sampling
  - Appropriate loss functions
  - Validation monitoring
  - Early stopping
  - Learning rate scheduling

### Phase 3: Evaluation (Weeks 7-8)
- Calculate performance metrics
- Perform clinical validation
- Analyze failure cases
- Generate performance reports

## 6. Evaluation Metrics

### Classification Metrics
- Overall accuracy
- Per-class precision/recall/F1-score
- Confusion matrix analysis
- Cohen's kappa score (inter-rater agreement)
- AUC-ROC curves

### Clinical Validation
- Compare with radiologist assessments
- Inter-observer variability analysis
- Analyze failure cases and edge cases
- Performance across different mammogram views (CC/MLO)

## 7. Optimization Strategies

### Efficient Fine-tuning
- LoRA rank optimization
- Quantization (4-bit, 8-bit)
- Gradient accumulation
- Mixed precision training (FP16/BF16)

### Data Efficiency
- Few-shot learning experiments
- Active learning for labeling
- Data augmentation strategies (careful with medical images)
- Synthetic data generation considerations

## 8. Deployment Considerations

### Inference Optimization
- Model quantization for deployment
- Batch processing for efficiency
- API endpoint design
- Response time optimization

### Integration Requirements
- DICOM workflow integration
- PACS compatibility
- Report generation capabilities
- Confidence scores for predictions
- Audit trail implementation

## 9. Leveraging Multimodal Capabilities

### Enhanced Inputs
- Combine mammogram with patient history text
- Include prior mammogram comparisons
- Integrate clinical notes
- Incorporate technical parameters (kVp, compression force)

### Rich Outputs
- Generate explanations for classifications
- Highlight regions contributing to density assessment
- Provide detailed reports with medical terminology
- Confidence intervals for predictions

## 10. Ethical & Regulatory Considerations

### Bias Mitigation
- Ensure diverse training data across:
  - Demographics (age, ethnicity)
  - Equipment manufacturers
  - Imaging protocols
- Test across different populations
- Monitor for demographic disparities
- Regular bias audits

### Compliance Requirements
- HIPAA compliance for data handling
- FDA regulatory pathway planning (510(k) submission)
- Audit trails for all predictions
- Data governance protocols
- Patient consent considerations

### Explainability
- Implement Grad-CAM or similar visualization techniques
- Feature importance analysis
- Generate human-readable explanations
- Document model limitations

## 11. Project Timeline

### Weeks 1-2: Data Collection and Preprocessing
- Acquire and organize datasets
- Implement preprocessing pipeline
- Set up data versioning

### Weeks 3-4: MedGemma Setup and Initial Experiments
- Configure development environment
- Run baseline experiments
- Optimize hyperparameters

### Weeks 5-6: Fine-tuning and Optimization
- Full model fine-tuning
- Performance optimization
- Ablation studies

### Weeks 7-8: Evaluation and Validation
- Comprehensive evaluation
- Clinical validation
- Error analysis

### Weeks 9-10: Deployment Preparation
- Model optimization for inference
- API development
- Documentation and reporting

## 12. Risk Management

### Technical Risks
- Model overfitting on limited data
- Computational resource constraints
- Integration challenges with existing systems

### Clinical Risks
- Misclassification consequences
- Edge case handling
- Radiologist acceptance

### Mitigation Strategies
- Robust validation protocols
- Fail-safe mechanisms
- Clear confidence thresholds
- Human-in-the-loop workflows

## 13. Success Criteria

### Technical Metrics
- >90% classification accuracy
- <100ms inference time per image
- Stable performance across datasets

### Clinical Acceptance
- Comparable to radiologist performance
- High user satisfaction scores
- Successful pilot deployment

## 14. Future Enhancements

### Model Improvements
- Multi-task learning (density + other findings)
- Temporal analysis (comparing with priors)
- 3D mammography support

### System Enhancements
- Real-time processing capabilities
- Mobile deployment options
- Integration with AI-assisted reporting tools

## 15. Resources and References

### Key Papers
- Original BIRADS density classification guidelines
- Recent AI mammography classification studies
- MedGemma model documentation

### Tools and Libraries
- Hugging Face Model Hub
- MONAI for medical imaging
- TorchXRayVision for baseline comparisons

### Support Communities
- Hugging Face forums
- Medical imaging AI communities
- Radiological societies' AI working groups

---

## Next Steps

1. **Secure data access**: Obtain necessary datasets and agreements
2. **Set up compute resources**: Provision appropriate GPU infrastructure
3. **Create project repository**: Initialize version control and documentation
4. **Begin Phase 1**: Start with data preparation and initial experiments

## Contact and Collaboration

- Define project team roles
- Establish communication channels
- Set up regular progress reviews
- Identify clinical advisors