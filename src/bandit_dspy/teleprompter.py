import os
import tempfile

# Ensure DSPy uses a writable disk cache directory even in sandboxed environments
if "DSPY_CACHEDIR" not in os.environ:
    os.environ["DSPY_CACHEDIR"] = os.path.join(tempfile.gettempdir(), ".dspy_cache")

import random
from typing import Any, Callable, Dict, Optional

from dspy.teleprompt import LabeledFewShot, Teleprompter

from .metric import create_bandit_metric  # <--- IMPORT THE FACTORY

try:
    from dspy.teleprompt import MIPRO

    MIPRO_AVAILABLE = True
except ImportError:
    MIPRO_AVAILABLE = False


class GeneticOptimizer:
    """Genetic algorithm for optimizing few-shot example selection."""

    def __init__(
        self, population_size=20, generations=10, mutation_rate=0.1, elite_ratio=0.2
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = int(population_size * elite_ratio)

    def optimize(self, trainset, k, evaluator):
        """Run genetic algorithm to find best few-shot combinations."""
        if len(trainset) < k:
            return list(trainset)

        # Initialize population with random combinations
        population = [random.sample(trainset, k) for _ in range(self.population_size)]

        for generation in range(self.generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                score = evaluator(individual)
                fitness_scores.append((score, individual))

            # Sort by fitness (higher is better)
            fitness_scores.sort(reverse=True, key=lambda x: x[0])

            # Select elite individuals
            elite = [individual for _, individual in fitness_scores[: self.elite_size]]

            # Generate new population
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(fitness_scores)
                parent2 = self._tournament_select(fitness_scores)

                # Crossover and mutation
                child = self._crossover(parent1, parent2, trainset, k)
                child = self._mutate(child, trainset)

                new_population.append(child)

            population = new_population

        # Return best individual
        final_scores = [
            (evaluator(individual), individual) for individual in population
        ]
        best_score, best_individual = max(final_scores, key=lambda x: x[0])
        return best_individual

    def _tournament_select(self, fitness_scores, tournament_size=3):
        """Select individual using tournament selection."""
        tournament = random.sample(
            fitness_scores, min(tournament_size, len(fitness_scores))
        )
        return max(tournament, key=lambda x: x[0])[1]

    def _crossover(self, parent1, parent2, trainset, k):
        """Create child by combining parents."""
        combined = list(set(parent1 + parent2))
        if len(combined) >= k:
            return random.sample(combined, k)
        else:
            # Fill with random examples if not enough unique ones
            remaining = [ex for ex in trainset if ex not in combined]
            return combined + random.sample(remaining, k - len(combined))

    def _mutate(self, individual, trainset):
        """Mutate individual by replacing some examples."""
        if random.random() < self.mutation_rate:
            available = [ex for ex in trainset if ex not in individual]
            if available:
                replace_idx = random.randint(0, len(individual) - 1)
                individual[replace_idx] = random.choice(available)
        return individual


class BayesianOptimizer:
    """Simple Bayesian optimization for hyperparameter tuning."""

    def __init__(self, n_calls=20, random_state=None):
        self.n_calls = n_calls
        self.random_state = random_state
        if random_state:
            random.seed(random_state)

    def optimize_k(self, trainset, max_k, evaluator):
        """Find optimal k using Bayesian optimization."""
        if len(trainset) <= 1:
            return 1

        max_k = min(max_k, len(trainset))

        # Simple grid search for k (since it's discrete and small range)
        best_k = 1
        best_score = 0

        for k in range(1, max_k + 1):
            scores = []
            # Multiple evaluations for each k to reduce noise
            for _ in range(min(5, self.n_calls // max_k)):
                examples = random.sample(trainset, k)
                score = evaluator(examples)
                scores.append(score)

            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_k = k

        return best_k


class BanditTeleprompter(Teleprompter):
    def __init__(
        self,
        metric: Optional[Callable] = None,
        k=3,
        num_candidates=10,
        optimization_method="random",
        train_val_split=0.8,
        genetic_config=None,
        bayesian_config=None,
        mipro_config=None,
        bandit_config: Optional[Dict[str, Any]] = None,
    ):
        # If no metric is provided, create a default one using the factory
        if metric is None:
            self.metric = create_bandit_metric(bandit_config=bandit_config)
        else:
            self.metric = metric  # Use the provided metric

        self.k = k
        self.num_candidates = num_candidates
        self.optimization_method = optimization_method
        self.train_val_split = train_val_split
        self.mipro_config = mipro_config or {}
        self.bandit_config = (
            bandit_config  # Store for potential future use or debugging
        )

        # Initialize optimizers
        self.genetic_optimizer = GeneticOptimizer(**(genetic_config or {}))
        self.bayesian_optimizer = BayesianOptimizer(**(bayesian_config or {}))

    def compile(self, student, *, trainset):
        # Split training set for proper evaluation
        split_idx = int(len(trainset) * self.train_val_split)
        train_subset = trainset[:split_idx]
        val_subset = trainset[split_idx:] if split_idx < len(trainset) else trainset

        if self.optimization_method == "genetic":
            return self._compile_genetic(student, train_subset, val_subset)
        elif self.optimization_method == "bayesian":
            return self._compile_bayesian(student, train_subset, val_subset)
        elif self.optimization_method == "mipro" and MIPRO_AVAILABLE:
            return self._compile_mipro(student, train_subset, val_subset)
        else:
            return self._compile_random(student, train_subset, val_subset)

    def _compile_random(self, student, train_subset, val_subset):
        """Original random search approach."""
        best_program = None
        best_score = -1

        for i in range(self.num_candidates):
            if len(train_subset) < self.k:
                candidate_examples = train_subset
            else:
                candidate_examples = random.sample(train_subset, self.k)

            optimizer = LabeledFewShot(k=len(candidate_examples))
            candidate_program = optimizer.compile(student, trainset=candidate_examples)

            # Evaluate on validation set
            total_score = 0
            for dev_example in val_subset:
                prediction = candidate_program(**dev_example.inputs())
                metric_result = self.metric(dev_example, prediction)
                total_score += metric_result["score"]

            score = total_score / len(val_subset) if val_subset else 0

            if score > best_score:
                best_score = score
                best_program = candidate_program

        return best_program

    def _compile_genetic(self, student, train_subset, val_subset):
        """Genetic algorithm optimization."""

        def evaluator(examples):
            optimizer = LabeledFewShot(k=len(examples))
            candidate_program = optimizer.compile(student, trainset=examples)

            total_score = 0
            for dev_example in val_subset:
                prediction = candidate_program(**dev_example.inputs())
                metric_result = self.metric(dev_example, prediction)
                total_score += metric_result["score"]

            return total_score / len(val_subset) if val_subset else 0

        best_examples = self.genetic_optimizer.optimize(train_subset, self.k, evaluator)
        optimizer = LabeledFewShot(k=len(best_examples))
        return optimizer.compile(student, trainset=best_examples)

    def _compile_bayesian(self, student, train_subset, val_subset):
        """Bayesian optimization for k selection."""

        def evaluator(examples):
            optimizer = LabeledFewShot(k=len(examples))
            candidate_program = optimizer.compile(student, trainset=examples)

            total_score = 0
            for dev_example in val_subset:
                prediction = candidate_program(**dev_example.inputs())
                metric_result = self.metric(dev_example, prediction)
                total_score += metric_result["score"]

            return total_score / len(val_subset) if val_subset else 0

        # Find optimal k
        optimal_k = self.bayesian_optimizer.optimize_k(train_subset, self.k, evaluator)

        # Use genetic algorithm with optimal k
        best_examples = self.genetic_optimizer.optimize(
            train_subset, optimal_k, evaluator
        )
        optimizer = LabeledFewShot(k=len(best_examples))
        return optimizer.compile(student, trainset=best_examples)

    def _compile_mipro(self, student, train_subset, val_subset):
        """MIPRO-based gradient optimization."""
        if not MIPRO_AVAILABLE:
            return self._compile_random(student, train_subset, val_subset)

        # Create security-aware metric wrapper for MIPRO
        def security_metric_wrapper(example, pred, trace=None):
            result = self.metric(example, pred, trace)
            # MIPRO expects a score that higher is better
            return result["score"]

        # Configure MIPRO with security metric
        mipro_optimizer = MIPRO(metric=security_metric_wrapper, **self.mipro_config)

        # Combine train and validation for MIPRO (it handles splitting internally)
        full_trainset = train_subset + val_subset

        return mipro_optimizer.compile(student, trainset=full_trainset)
