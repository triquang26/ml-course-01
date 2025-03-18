# Machine Learning Models: Implementation and Comparison

This repository contains implementations and comparative analyses of various machine learning models, focusing on pneumonia classification using the MedMNIST dataset. Each implementation demonstrates different approaches to the same problem, allowing for direct comparison of model characteristics, strengths, and limitations.

## Project Overview

This project implements and compares the following machine learning models:

- **Bayesian Models** (Gaussian NB, Multinomial NB, Bernoulli NB)
- **Decision Trees**
- **Neural Networks** (Vision Transformer)
- **Graphical Models** (Bayesian Networks, Augmented Naive Bayes, Hidden Markov Models)
- **Genetic Algorithm-based Ensemble Model** (Convolutional Neural Network, Decision Tree, Naive Bayes)

All models are trained and evaluated on the **PneumoniaMNIST** dataset, a binary classification task to identify pneumonia in pediatric chest X-rays.

## Repository Structure

```
├── notebooks/                # Jupyter notebooks for analysis and model development
│   ├── assignment1/          # Main assignment notebooks
│   │   ├── model/            # Individual model implementation notebooks
│   │   │   ├── bayesian_models.ipynb
│   │   │   ├── decision_tree.ipynb
│   │   │   ├── genetic_algorithm.ipynb
│   │   │   ├── graphical_model.ipynb
│   │   │   └── neural_network.ipynb
│   │   ├── exploration.ipynb  # Dataset exploration
│   │   ├── submission.ipynb   # Final submission notebook
│   │   └── README.md          # Assignment-specific documentation
│   ├── assignment2/          # Future assignment work
│   └── final_project/        # Final project notebooks
├── src/                      # Source code (core implementation)
│   ├── data/                 # Data processing scripts
│   │   ├── preprocess/       # Model-specific preprocessing
│   │   │   ├── bayesian.py
│   │   │   ├── decision_tree.py
│   │   │   ├── genetic_algorithm.py
│   │   │   ├── graphical_model.py
│   │   │   └── neural_network.py
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── features/             # Feature engineering scripts
│   │   ├── features_selection/  # Model-specific feature selection
│   │   │   ├── bayesian.py
│   │   │   ├── decision_tree.py
│   │   │   ├── genetic_algorithm.py
│   │   │   ├── graphical_model.py
│   │   │   └── neural_network.py
│   │   └── build_features.py
│   ├── models/               # Model implementations
│   │   ├── bayesian/         # Naive Bayes implementations
│   │   │   ├── core/         # Core implementations
│   │   │   ├── figures/      # Generated visualizations
│   │   │   ├── trained/      # Trained model files
│   │   │   ├── main.py       # Training entry point
│   │   │   ├── test.py       # Evaluation script
│   │   │   ├── predict.py    # Prediction script
│   │   │   └── README.md     # Model-specific documentation
│   │   ├── decision_tree/    # Decision Tree implementations
│   │   │   ├── figures/
│   │   │   ├── trained/
│   │   │   ├── main.py
│   │   │   └── README.md
│   │   ├── genetic_algorithm/  # Genetic Algorithm ensemble
│   │   │   ├── main.py
│   │   │   └── genetic_algorithm.py
│   │   ├── graphical_model/  # Graphical model implementations
│   │   │   ├── core/
│   │   │   ├── figures/
│   │   │   ├── trained/
│   │   │   ├── main.py
│   │   │   └── README.md
│   │   ├── neural_network/   # Neural network implementations
│   │   │   ├── reports/figures/
│   │   │   ├── trained/
│   │   │   ├── main.py
│   │   │   └── README.md
│   │   ├── train_model.py    # Main training script
│   │   └── predict_model.py  # Main prediction script
│   └── visualization/        # Visualization tools
├── data/                     # Dataset storage
│   ├── external/             # Data from third-party sources
│   ├── processed/            # Cleaned and transformed data
│   └── raw/                  # Original, immutable data
├── models/                   # High-level model storage
│   ├── experiments/          # Experimental configurations
│   └── trained/              # Top-level trained model files
├── reports/                  # Generated analysis reports
│   ├── figures/              # Generated graphics and figures
│   └── final_project/        # Final project reports
├── tests/                    # Test files
├── README.md                 # Project description
└── requirements.txt          # Package dependencies
```

## Model Organization

Each model is structured consistently across the repository:

1. **Notebook Implementation**
   - Located in `notebooks/assignment1/model/` 
   - Contains exploratory implementation, experiments, and visualizations
   - Serves as a tutorial and explanation of the model approach

2. **Source Code Implementation**
   - Located in `src/models/{model_type}/`
   - Contains modular, production-ready implementation
   - Includes separate files for:
     - Core model logic
     - Training entry points
     - Testing/evaluation
     - Model persistence
   
3. **Documentation**
   - Each model has a dedicated README.md explaining:
     - Implementation details
     - Unique aspects of the approach
     - Performance characteristics
     - Usage instructions

4. **Artifacts**
   - Trained models stored in `src/models/{model_type}/trained/`
   - Visualizations stored in `src/models/{model_type}/figures/` or reports directories

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
![Model Performance Comparison](/figures/model-comparison.png)

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
