# Skin Cancer Classification - Lightweight Models Implementation Plan

## Overview
This plan outlines the implementation of lightweight models for skin cancer classification with a smaller subset of data. The implementation allows for easy adjustment of data size and model selection while maintaining compatibility with the existing project structure.

## Key Components

### 1. Lightweight Models
The following lightweight models have been implemented:

- **EfficientNet Family**
  - EfficientNetB0 (224x224, 5.3M parameters)
  - EfficientNetB1 (240x240, 7.8M parameters)
  - EfficientNetB2 (260x260, 9.2M parameters)

- **MobileNet Family**
  - MobileNetV2 (224x224, 3.5M parameters)
  - MobileNetV3Small (224x224, 2.5M parameters)

- **Additional Optimal Models**
  - DenseNet121 (224x224, 8M parameters) - Very parameter-efficient with good performance
  - Xception (299x299, 22.9M parameters) - Advanced depthwise separable convolutions

### 2. Model Architecture
- **Using the project's existing architecture**:
  - Base model for feature extraction (various backbones)
  - Flatten layer (instead of GlobalAveragePooling)
  - Dense layer with 500 units and ReLU activation
  - Dense layer with 256 units and ReLU activation
  - Output layer with softmax activation for 9 skin lesion classes

### 3. Data Subset Handling
- Configurable subset size (default: 20% of total data)
- Stratified sampling to maintain class distribution
- Compatible with the existing data preprocessing pipeline

### 4. Preprocessing
- Model-specific preprocessing functions for each architecture family
- Consistent augmentation techniques across models:
  - Rotation, flipping, zooming, brightness adjustments
  - Shearing and distortion
  - Proper normalization per model requirements

### 5. Training Strategy
- Transfer learning with ImageNet pre-trained weights
- Two-phase training:
  1. Train with frozen backbone (only top layers trained)
  2. Optional fine-tuning with selective unfreezing (keeping BatchNorm layers frozen)
- Early stopping and learning rate scheduling for optimal convergence
- Model checkpointing to save best weights

### 6. Hyperparameters
- Batch size: 16 (adjustable)
- Learning rates: Model-specific
  - 1e-4 for EfficientNets B0-B2
  - 5e-5 for MobileNets and DenseNet121
  - 1e-5 for Xception (lower due to larger model size)
- Fine-tuning learning rate: 1/10th of initial rate
- Epochs: 15 for initial training, 5 for fine-tuning
- Early stopping patience: 5 epochs
- Learning rate reduction: Factor of 0.2 after 3 epochs without improvement

## Implementation Details

### Script Structure
The implementation consists of a single script (`train_lightweight_models.py`) with the following components:

1. **Configuration System**
   - Centralized configuration dictionary with all parameters
   - Easy adjustment of models, data paths, and training parameters

2. **Model Creation**
   - Function to create models with proper preprocessing and input dimensions
   - Consistent model architecture matching the project's existing approach
   - Selective layer unfreezing during fine-tuning

3. **Data Pipeline**
   - Functions to load and preprocess data appropriately for each model
   - Subset creation with stratified sampling

4. **Training Functions**
   - Training with callbacks for monitoring and optimization
   - Fine-tuning option with BatchNorm layers kept frozen

5. **Visualization**
   - Training history plots saved for each model
   - Separate plots for initial training and fine-tuning

### User Interface
The script provides an interactive interface to:
- Select which model(s) to train
- Specify the data subset size
- Choose whether to fine-tune after initial training

## Usage Instructions

1. Run the script:
   ```
   python3 train_lightweight_models.py
   ```

2. Follow the prompts to select:
   - Which model to train (or leave blank to train all)
   - Subset size (0.0-1.0, default 0.2)
   - Fine-tuning option (after initial training)

3. Results will be saved as:
   - Model weights: `[model_name]_best.keras`
   - Fine-tuned weights: `[model_name]_finetuned.keras`
   - Training logs: `[model_name]_training_log.csv`
   - Training plots: `[model_name]_training_history.png`

## Benefits of This Approach

1. **Architecture Consistency**: Uses the same model architecture as the main project
2. **Lightweight Models**: Significantly lower computational requirements compared to larger models
3. **Model Diversity**: Includes models with different architectural approaches (EfficientNet, MobileNet, DenseNet, Xception)
4. **Flexibility**: Easy selection of models and data subset size
5. **Efficiency**: Two-phase training strategy with proper layer freezing for optimal performance
6. **Documentation**: Training logs and visualizations for all models

## Next Steps

1. Evaluate model performance on test set
2. Compare lightweight models to larger models (EfficientNetB4/B5/etc.)
3. Consider ensemble approaches combining multiple lightweight models
4. Explore quantization and pruning for further model optimization 