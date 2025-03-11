# Machine Learning Models: Implementation and Comparison

This repository contains implementations and comparative analyses of various machine learning models, focusing on pneumonia classification using the MedMNIST dataset. Each implementation demonstrates different approaches to the same problem, allowing for direct comparison of model characteristics, strengths, and limitations.

## Project Overview

This project implements and compares the following machine learning models:

- **Bayesian Models** (Gaussian NB, Multinomial NB, Bernoulli NB)
- **Decision Trees**
- **Neural Networks** (Vision Transformer)
- **Graphical Models** (Bayesian Networks, Augmented Naive Bayes, Hidden Markov Models)
- **Genetic Algorithm-based Ensemble Model**

All models are trained and evaluated on the **PneumoniaMNIST** dataset, a binary classification task to identify pneumonia in pediatric chest X-rays.

## Repository Structure
```
├── data/                   # Dataset storage
│   ├── external/           # Data from third-party sources
│   ├── processed/          # Cleaned and transformed data
│   └── raw/                # Original, immutable data
├── models/                 # Saved model files
│   ├── experiments/        # Experimental configurations
│   └── trained/            # Trained model files
│     └── augmentated_naive_bayes_graphical/
├── notebooks/              # Jupyter notebooks
│   ├── assignment1/        # Analysis notebooks
│   │   └── model/          # Model implementation notebooks
│   ├── assignment2/
│   └── final_project/
├── reports/                # Generated analysis reports
│   ├── figures/            # Generated graphics and figures
│   └── final_project/      # Final project reports
├── src/                    # Source code
│   ├── data/               # Scripts to download or generate data
│   ├── features/           # Feature engineering scripts
│   ├── models/             # Model implementations
│   │   ├── bayesian/       # Naive Bayes implementations
│   │   ├── decision_tree/  # Decision Tree implementations
│   │   ├── genetic_algorithm/ # Genetic Algorithm ensemble
│   │   ├── graphical_model/ # Graphical model implementations
│   │   └── neural_network/ # Neural network implementations
│   └── visualization/      # Visualization tools
├── tests/                  # Test files
├── README.md               # Project description
└── requirements.txt        # Package dependencies
```

## Models Implemented

### 1. Bayesian Models
- **Implementation**: Gaussian, Multinomial, and Bernoulli Naive Bayes classifiers
- **Key Features**: Probabilistic approach with different distribution assumptions
- **Performance**: Gaussian NB achieved best results (83.3% accuracy)
- **Strengths**: Simple, fast, and interpretable

### 2. Decision Trees
- **Implementation**: Standard decision tree with optimized depth
- **Key Features**: Hierarchical decision rules
- **Performance**: 80.1% accuracy on test set
- **Strengths**: Interpretability and feature importance ranking

### 3. Neural Networks
- **Implementation**: Vision Transformer model
- **Key Features**: Attention-based architecture for image processing
- **Performance**: 87.9% accuracy on test set
- **Strengths**: Strong feature extraction capabilities for complex patterns

### 4. Graphical Models
- **Implementation**: Bayesian Networks, Augmented Naive Bayes, Hidden Markov Models
- **Key Features**: Structured probabilistic relationships
- **Performance**: Augmented Naive Bayes achieved 85% accuracy
- **Strengths**: Captures complex conditional dependencies

### 5. Genetic Algorithm Ensemble
- **Implementation**: Weighted ensemble combining Decision Tree, CNN, and Naive Bayes
- **Key Features**: Optimization of model weights using genetic algorithm
- **Performance**: Improved accuracy over individual models
- **Strengths**: Automated model combination optimization

## Dataset

The **PneumoniaMNIST** dataset consists of **5,856** pediatric chest X-ray images formatted as **28×28 grayscale images**. The task is binary classification (normal vs. pneumonia). The dataset is split as follows:

- **Training**: 4,708 images
- **Testing**: 624 images

## Performance Comparison

| Model                     | Accuracy | F1-Score | Key Characteristics |
|---------------------------|----------|----------|----------------------|
| Gaussian NB              | 83.3%    | 0.87     | Fast training, probabilistic output |
| Bernoulli NB             | 80.1%    | 0.84     | Binary feature handling |
| Multinomial NB           | 82.7%    | 0.86     | Good with discrete features |
| Decision Tree            | 80.1%    | 0.85     | Interpretable rules |
| Vision Transformer       | 87.9%    | 0.90     | Complex pattern recognition |
| Augmented Naive Bayes    | 85.0%    | 0.89     | Structured probabilistic model |
| Hidden Markov Model      | 72.0%    | 0.71     | Sequential data modeling |
| Genetic Algorithm Ensemble | 88.5%  | 0.91     | Optimized model combination |

## Getting Started

### Prerequisites
Ensure you have the following installed:

- Python 3.11+
- PyTorch and torchvision
- medmnist
- scikit-learn
- pgmpy (for graphical models)
- numpy, pandas
- matplotlib, seaborn

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/ml-course-01.git
cd ml-course-01
```
Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Models
To train all models sequentially:
```bash
python src/models/train_model.py --models all
```
To train specific models:
```bash
python src/models/train_model.py --models bayesian neural_network
```

## Results and Analysis
Each model implementation includes detailed analysis in both code comments and notebook documentation. Key visualizations include:

- Confusion matrices
- F1-score heatmaps
- Performance comparison charts
- ROC curves (where applicable)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/yourfeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/yourfeature`)
5. Create a new Pull Request

## Acknowledgments

- **PneumoniaMNIST dataset** from the MedMNIST collection
- **Inspiration** from various machine learning courses and textbooks

