# Conditional Random Field for Medical Image Classification

## Introduction
This project implements a Conditional Random Field (CRF) for classifying pneumonia from chest X-ray images using the PneumoniaMNIST dataset. CRF is a probabilistic model excels in structured prediction tasks, making it well-suited for medical image classification where spatial relationships and contextual information play a critical role. The implementation emphasizes robust classification by modeling dependencies between output variables, leveraging efficient training techniques and comprehensive evaluation metrics.

## Overview
The project showcases the application of Conditional Random Fields to medical image classification, highlighting the model's ability to capture contextual dependencies in structured data. Key aspects include:

- **Conditional Random Field** with L-BFGS optimization for efficient training
- **Feature Engineering** converting pixel intensities into feature dictionaries
- **Memory-Efficient Implementation** for processing large-scale medical datasets
- **Detailed Performance Analysis** with visualizations
- **Model Persistence** for reproducibility
- **Real-Time Prediction Capabilities** for practical use

## Performance Analysis
### Metrics Implementation
- **Accuracy**: Overall classification performance using `accuracy_score`
- **Precision/Recall**: Assessment of the model's ability to identify pneumonia cases
- **F1-Score**: Balanced measure of precision and recall
- **Confusion Matrix**: Detailed breakdown of classification outcomes using `seaborn.heatmap`

### Results
The CRF model was evaluated on the PneumoniaMNIST test set, achieving the following performance:

- **Accuracy**: 83.81%
- **Precision**: 81.62%
- **Recall**: 95.64%
- **F1-Score**: 88.08%
- **Confusion Matrix**:
  ```
  [[150  84]
   [ 17 373]]
  ```

These results highlight the model's high recall, which is critical for medical diagnostics to minimize false negatives, alongside a solid overall performance.

### Visualization
- **Model Comparison Plots**: Bar plots of accuracy and F1-score for the CRF model
- **Confusion Matrix Heatmap**: Visual representation of true positives, false positives, true negatives, and false negatives

## Code Efficiency
### Time Complexity
- **Training**: The L-BFGS algorithm provides an efficient optimization approach, with a time complexity of approximately \(O(n \cdot m)\), where \(n\) is the number of training samples and \(m\) is the number of features.
- **Prediction**: Linear time complexity relative to the number of test samples, enabling real-time inference.

### Space Complexity
- **Feature Storage**: Each 28x28 image is converted into a 784-entry feature dictionary, designed for memory efficiency.
- **Model Storage**: The trained CRF model is persisted using `sklearn-crfsuite` for compact storage and rapid loading.

### Optimization Techniques
- **Batch Processing**: Manages memory usage by processing data in batches.
- **Feature Engineering**: Direct use of pixel intensities as features simplifies preprocessing.
- **Convergence Monitoring**: Training halts after a maximum of 100 iterations or upon convergence, preventing overfitting.

### Scalability
- Tested successfully on the PneumoniaMNIST dataset (5,856 training images)
- Capable of handling larger datasets with adaptive memory management
- Compatible with distributed computing frameworks if needed

## Documentation
### Data Processing
- **src/**
  - **data/**
    - **preprocess/**
        - `conditional_random_field.py`: Model loading and data preprocessing for CRFs
### Code Structure
```
src/models/conditional_random_field/
├── core/                  # Core CRF implementations
├── persistence.py         # Model serialization utilities
├── model_tuning.py        # Training and tuning logic
├── main.py                # Main execution script
├── predict.py             # Prediction utilities
└── test.py                # Testing and evaluation script
```

### Key Features
- **Feature Extraction**: Converts flattened image arrays into CRF-compatible feature dictionaries
- **Model Training**: Uses `sklearn-crfsuite` with L-BFGS for efficient parameter estimation
- **Prediction**: Enables real-time classification of new images
- **Evaluation**: Provides accuracy, precision, recall, F1-score, and confusion matrix
- **Logging**: Tracks training and inference processes

## Model Analysis
### Conditional Random Field Theory
- **Structured Prediction**: Models the conditional probability \(P(Y|X)\), capturing label dependencies.
- **Feature Functions**: Maps input-output pairs to learn image patterns.
- **Regularization**: Employs L1 (`c1=0.1`) and L2 (`c2=0.1`) penalties to enhance generalization.
- **Optimization**: L-BFGS ensures efficient convergence during training.

### Feature Importance
- **Pixel Intensity Patterns**: Leverages raw pixel values to detect pneumonia indicators.
- **Spatial Relationships**: Implicitly modeled through CRF's structure.
- **Normalization**: Applied to maintain consistent feature scales.

### Model Behavior
- **Decision Boundary**: Based on conditional probabilities of labels given features.
- **Generalization**: Strong performance on unseen test data, as shown by evaluation metrics.
- **Uncertainty Quantification**: Offers probability estimates for confidence assessment.

### Limitations
- **Training Time**: Can be intensive for very large datasets, mitigated by L-BFGS.
- **Feature Engineering**: Relies on manual feature design, potentially missing deeper patterns.
- **Hyperparameter Sensitivity**: Performance varies with regularization parameters.
- **Class Imbalance**: May need adjustments for highly imbalanced data, though results suggest balance.

### Theoretical Foundation
- **Probabilistic Graphical Models**: Discriminative approach focusing on conditional distributions.
- **Maximum Likelihood**: Parameters optimized via conditional log-likelihood.
- **Regularization**: Balances model complexity and training error.

### Use Case Analysis
- **Medical Diagnosis Support**: High recall ensures reliable detection of pneumonia.
- **Real-Time Classification**: Fast inference suits clinical applications.
- **Small Dataset Performance**: Effective with moderate data sizes like PneumoniaMNIST.
- **Interpretability**: Feature functions offer insights into influential patterns.

## Hyperparameter Tuning
### Configuration
- **Algorithm**: L-BFGS
- **Regularization**: `c1=0.1` (L1), `c2=0.1` (L2)
- **Max Iterations**: 100
- **All Possible Transitions**: Enabled

### Optimization Strategy
- **Scoring Metrics**: Evaluated using accuracy, precision, recall, and F1-score
- **Validation**: Performance assessed on a hold-out test set

## Memory Management
### Large Dataset Handling
- **Feature Dictionaries**: Efficiently stores image features for CRF input.
- **Batch Processing**: Reduces memory load during prediction.
- **Model Persistence**: Saves and loads models compactly with `sklearn-crfsuite`.

### Storage Optimization
- **Model Compression**: Retains only essential parameters.
- **Result Persistence**: Stores metrics and visualizations for future use.

## Conclusion
This CRF implementation effectively classifies pneumonia from chest X-ray images, achieving strong performance with an emphasis on high recall—crucial for medical diagnostics. The project balances theoretical depth with practical efficiency, offering a robust solution for the PneumoniaMNIST dataset. Future improvements could involve integrating CRF with deep learning feature extractors, exploring advanced feature representations, or extending the approach to other medical imaging tasks. The modular codebase supports such enhancements, making it a versatile tool for further research and application.