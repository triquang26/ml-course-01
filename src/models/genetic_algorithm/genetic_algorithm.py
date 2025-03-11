import random
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

class GeneticAlgorithm:
    """
    Genetic Algorithm implementation for optimizing ensemble model weights.
    """
    
    def __init__(self, pop_size=20, generations=30, mutation_rate=0.3, 
                 mutation_strength=0.1, tournament_size=3, alpha=0.5, 
                 reg_lambda=0.01, k_folds=5):
        """
        Initialize the genetic algorithm parameters.
        
        Args:
            pop_size: Size of population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            mutation_strength: Maximum change during mutation
            tournament_size: Number of individuals in tournament selection
            alpha: Blending factor for crossover
            reg_lambda: Regularization strength for fitness function
            k_folds: Number of folds for cross-validation
        """
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.k_folds = k_folds
    
    def initialize_population(self, n_models):
        """
        Initialize a population with random weights for each model.
        
        Args:
            n_models: Number of models in the ensemble
            
        Returns:
            List of individuals, where each individual is a list of weights
        """
        # Initialize individuals with weights in range [0.1, 1.0]
        return [[random.uniform(0.1, 1.0) for _ in range(n_models)] 
                for _ in range(self.pop_size)]
    
    def ensemble_prediction(self, weights, model_probs):
        """
        Make ensemble predictions using weighted average of model probabilities.
        
        Args:
            weights: List of weights for each model
            model_probs: List of probability outputs from each model
            
        Returns:
            Binary predictions (0 or 1)
        """
        total_weight = sum(weights)
        # Weighted average of probabilities
        ensemble_probs = sum(w*p for w, p in zip(weights, model_probs)) / total_weight
        # Convert to binary predictions
        return (ensemble_probs > 0.5).astype(int)
    
    def ensemble_probabilities(self, weights, model_probs):
        """
        Calculate ensemble probability estimates using weighted average.
        
        Args:
            weights: List of weights for each model
            model_probs: List of probability outputs from each model
            
        Returns:
            Probability estimates
        """
        total_weight = sum(weights)
        return sum(w*p for w, p in zip(weights, model_probs)) / total_weight
    
    def fitness_cv(self, weights, model_probs, true_labels):
        """
        Calculate fitness using cross-validation and AUC.
        
        Args:
            weights: List of weights for each model
            model_probs: List of probability outputs from each model
            true_labels: True class labels
            
        Returns:
            Fitness value (higher is better)
        """
        # Prepare cross-validation
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        auc_scores = []
        
        # Convert all inputs to numpy arrays
        model_probs = [np.array(probs) for probs in model_probs]
        true_labels = np.array(true_labels)
        
        for train_idx, val_idx in kf.split(true_labels):
            # Get probabilities and labels for this fold
            fold_probs = [probs[val_idx] for probs in model_probs]
            labels_fold = true_labels[val_idx]
            
            # Calculate ensemble probabilities
            ensemble_probs = self.ensemble_probabilities(weights, fold_probs)
            
            # Compute AUC for this fold if there are both classes present
            if len(np.unique(labels_fold)) > 1:
                auc = roc_auc_score(labels_fold, ensemble_probs)
                auc_scores.append(auc)
        
        # Average AUC across folds
        avg_auc = np.mean(auc_scores) if auc_scores else 0.0
        
        # Regularization to penalize extreme weights
        reg_penalty = self.reg_lambda * np.sum(np.square(weights))
        
        # Final fitness value
        return avg_auc - reg_penalty
    
    def tournament_selection(self, population, fitnesses):
        """
        Select individuals using tournament selection.
        
        Args:
            population: List of individuals
            fitnesses: List of fitness values
            
        Returns:
            Selected individuals
        """
        selected = []
        for _ in range(self.pop_size):
            # Randomly choose individuals for tournament
            candidates = [random.randrange(self.pop_size) 
                         for _ in range(self.tournament_size)]
            # Select the best one
            best = max(candidates, key=lambda idx: fitnesses[idx])
            selected.append(population[best])
        return selected
    
    def blend_crossover(self, parent1, parent2):
        """
        Perform blend crossover between two parents.
        
        Args:
            parent1: First parent's weights
            parent2: Second parent's weights
            
        Returns:
            Two children produced by crossover
        """
        child1, child2 = [], []
        for gene1, gene2 in zip(parent1, parent2):
            d = abs(gene1 - gene2)
            lower = min(gene1, gene2) - self.alpha * d
            upper = max(gene1, gene2) + self.alpha * d
            # Ensure we stay within valid range [0.1, 1.0]
            lower = max(0.1, lower)
            upper = min(1.0, upper)
            
            c1 = random.uniform(lower, upper)
            c2 = random.uniform(lower, upper)
            child1.append(c1)
            child2.append(c2)
        return child1, child2
    
    def mutate(self, individual):
        """
        Mutate an individual by randomly changing weights.
        
        Args:
            individual: List of weights
            
        Returns:
            Mutated individual
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += random.uniform(-self.mutation_strength, self.mutation_strength)
                # Ensure weight stays in valid range [0.1, 1.0]
                individual[i] = max(0.1, min(1.0, individual[i]))
        return individual
    
    def run_ga(self, model_probs, true_labels):
        """
        Run the genetic algorithm to find optimal weights.
        
        Args:
            model_probs: List of probability outputs from each model
            true_labels: True class labels
            
        Returns:
            Tuple of (best_weights, best_fitness)
        """
        n_models = len(model_probs)
        population = self.initialize_population(n_models)
        best_individual = None
        best_fitness = -np.inf

        print(f"Starting genetic algorithm optimization with {self.generations} generations...")
        for gen in range(self.generations):
            # Evaluate fitness for each individual
            fitnesses = [self.fitness_cv(ind, model_probs, true_labels) 
                        for ind in population]
            
            # Track best solution
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            print(f"Generation {gen+1}: Best Fitness = {fitnesses[gen_best_idx]:.4f}, "
                 f"Weights = {[round(w, 3) for w in population[gen_best_idx]]}")
            
            # Selection
            selected = self.tournament_selection(population, fitnesses)
            
            # Crossover
            next_population = []
            for i in range(0, self.pop_size - 1, 2):
                child1, child2 = self.blend_crossover(selected[i], selected[i+1])
                next_population.append(child1)
                next_population.append(child2)
            
            # If odd population size, add last individual
            if len(next_population) < self.pop_size:
                next_population.append(selected[-1])
            
            # Mutation
            next_population = [self.mutate(ind) for ind in next_population]
            population = next_population
        
        return best_individual, best_fitness