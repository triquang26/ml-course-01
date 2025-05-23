# MACHINE LEARNING ASSIGNMENT

## Ho Chi Minh City University of Technology
### Faculty of Computer Science and Engineering
### Instructor: Nguyễn An Khương
## Team Members:
| **Name**              | **Student ID** | **Email**                                                      | **Role**              |
|-----------------------|----------------|----------------------------------------------------------------|-----------------------|
| **Phạm Trí Quang**    | 2352976        | [quang.phamtri@hcmut.edu.vn](mailto:quang.phamtri@hcmut.edu.vn) | Graphical Model,SVM       |
| **Nguyễn Thiện Anh**  | 2352053        | [anh.nguyencse2350253@hcmut.edu.vn](mailto:anh.nguyencse2350253@hcmut.edu.vn) | Genetic Algorithm, CRF     |
| **Lê Nhân Văn**       | 2252899        | [van.lenhanvan369@hcmut.edu.vn](mailto:van.lenhanvan369@hcmut.edu.vn) | Decision Tree, PCA         |
| **Nguyễn Anh Kiệt**   | 2252403        | [kiet.nguyenanh@hcmut.edu.vn](mailto:kiet.nguyenanh@hcmut.edu.vn) | Bayesian Models, Boosting       |
| **Nguyễn Hữu Cường** | 2252098        | [cuong.nguyen1nos1mp4@hcmut.edu.vn](mailto:cuong.nguyen1nos1mp4@hcmut.edu.vn) | Neural Network, Garphical Model        |

## Repository Link
You can access the repository [here](https://github.com/triquang26/ml-course-01).

## Project Overview

This project implements and compares the following machine learning models:

- **Bayesian Models** (Gaussian NB, Multinomial NB, Bernoulli NB)
- **Decision Trees**
- **Neural Networks** (Vision Transformer)
- **Graphical Models** (Bayesian Networks, Augmented Naive Bayes, Hidden Markov Models, Conditional Random Fields (CRF))
- **Genetic Algorithm-based Ensemble Model** (Convolutional Neural Network, Decision Tree, Naive Bayes)
- **Support Vector Machines (SVM)** (Linear and RBF kernels)
- **Principal Component Analysis (PCA)** (for dimensionality reduction)
- **Boosting Methods** (AdaBoost, Gradient Boosting, Bagging)

All models are trained and evaluated on the **PneumoniaMNIST** dataset, a binary classification task to identify pneumonia in pediatric chest X-rays.

All models are trained and evaluated on the **PneumoniaMNIST** dataset, a binary classification task to identify pneumonia in pediatric chest X-rays.

I'll create an updated Repository Structure in markdown format that includes the Genetic Algorithm (GA) section. Let me first analyze the current structure to make sure I understand the existing organization correctly.

Read file: README.md
Looking at the current repository structure, I notice you'd like to update it to better reflect the Genetic Algorithm (GA) section. Here's the updated repository structure in markdown format:

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
│   │   │   ├── core/         # Core GA implementation
│   │   │   ├── figures/      # GA evolution and performance visualizations
│   │   │   ├── trained/      # Trained ensemble models and weights
│   │   │   ├── main.py       # Training entry point
│   │   │   ├── genetic_algorithm.py  # GA implementation
│   │   │   ├── ensemble.py   # Model ensemble integration
│   │   │   ├── test.py       # Evaluation script
│   │   │   ├── predict.py    # Prediction script
│   │   │   └── README.md     # GA-specific documentation
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
│   ├── genetic_algorithm/    # GA experimental results
│   │   ├── evolution/        # Evolution progress tracking
│   │   ├── configurations/   # Different GA parameter configurations
│   │   └── ensemble_weights/ # Optimized ensemble weights
│   └── trained/              # Top-level trained model files
├── reports/                  # Generated analysis reports
│   ├── figures/              # Generated graphics and figures
│   ├── genetic_algorithm/    # GA-specific reports and analysis
│   │   ├── evolution_analysis.md  # Analysis of GA evolution process
│   │   ├── ensemble_performance.md  # Ensemble performance analysis
│   │   └── figures/          # GA-specific visualizations
│   └── final_project/        # Final project reports
├── tests/                    # Test files
│   ├── test_genetic_algorithm.py  # GA-specific tests
│   └── test_ensemble.py      # Ensemble model tests  
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
- **Strengths**: Simple, fast, and interpretable

### 2. Decision Trees
- **Implementation**: Standard decision tree with optimized depth
- **Key Features**: Hierarchical decision rules
- **Strengths**: Interpretability and feature importance ranking

### 3. Neural Networks
- **Implementation**: Vision Transformer model
- **Key Features**: Attention-based architecture for image processing
- **Strengths**: Strong feature extraction capabilities for complex patterns

### 4. Graphical Models
- **Implementation**: Bayesian Networks, Augmented Naive Bayes, Hidden Markov Models, Conditional Random Fields (CRF)
- **Key Features**: Structured probabilistic relationships, sequence modeling (CRF)
- **Strengths**: Captures complex conditional dependencies, effective for structured prediction

### 5. Genetic Algorithm Ensemble
- **Implementation**: Weighted ensemble combining Decision Tree, CNN, and Naive Bayes
- **Key Features**: Optimization of model weights using genetic algorithm
- **Strengths**: Automated model combination optimization

### 6. Support Vector Machines (SVM)
- **Implementation**: SVM with linear and RBF kernels
- **Key Features**: Maximum margin classification
- **Strengths**: Effective in high-dimensional spaces, robust to overfitting

### 7. Principal Component Analysis (PCA)
- **Implementation**: Dimensionality reduction for feature preprocessing
- **Key Features**: Orthogonal transformation to reduce feature space
- **Strengths**: Improves model efficiency, reduces noise

### 8. Boosting Methods
- **Implementation**: AdaBoost, Gradient Boosting, Bagging
- **Key Features**: Ensemble methods with iterative weak learner improvement (AdaBoost, Gradient Boosting) and parallel weak learners (Bagging)
- **Strengths**: High predictive power, reduces variance and bias

## Dataset

The **PneumoniaMNIST** dataset consists of **5,856** pediatric chest X-ray images formatted as **28×28 grayscale images**. The task is binary classification (normal vs. pneumonia). The dataset is split as follows:

- **Training**: 4,708 images
- **Testing**: 624 images

## Performance Comparison
![Model Performance Comparison](/figures/model-comparison.png)
![Model Performance Comparison](/figures/model-comparison-2.png)
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
- optuna
### Installation
Clone the repository:
```bash
git clone https://github.com/triquang26/ml-course-01
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
