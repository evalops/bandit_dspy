"""
Additional tests to improve coverage of advanced features.
"""

import pytest
import dspy
from unittest.mock import patch, MagicMock
from bandit_dspy import BanditTeleprompter, bandit_metric, BanditRunner
from bandit_dspy.core import run_bandit


def test_bandit_runner_syntax_error_fallback():
    """Test BanditRunner fallback with actual syntax error."""
    runner = BanditRunner()
    
    # Use actual syntax error code
    code = "def invalid_syntax("
    issues = runner.analyze_code(code)
    # Should handle gracefully
    assert isinstance(issues, list)


def test_run_bandit_with_config():
    """Test run_bandit with custom config."""
    # Test with custom config dict
    config = {}  # Empty config should work
    code = "def test(): pass"
    issues = run_bandit(code, config)
    assert isinstance(issues, list)


def test_mipro_integration():
    """Test MIPRO integration when available."""
    # Test when MIPRO is not available (mocked)
    with patch('bandit_dspy.teleprompter.MIPRO_AVAILABLE', False):
        teleprompter = BanditTeleprompter(
            metric=bandit_metric,
            optimization_method="mipro"
        )
        
        trainset = [
            dspy.Example(description="test", code="def test(): pass").with_inputs('description')
        ]
        
        class DummyModule(dspy.Module):
            def forward(self, description):
                return dspy.Example(code="def safe(): return 1")
        
        student = DummyModule()
        
        # Should fallback to random optimization
        compiled_program = teleprompter.compile(student, trainset=trainset)
        assert compiled_program is not None


def test_genetic_optimizer_edge_cases():
    """Test genetic optimizer edge cases."""
    from bandit_dspy.teleprompter import GeneticOptimizer
    
    optimizer = GeneticOptimizer(population_size=2, generations=1, mutation_rate=1.0)
    
    trainset = [
        dspy.Example(description="test1", code="def test1(): pass").with_inputs('description'),
        dspy.Example(description="test2", code="def test2(): pass").with_inputs('description'),
    ]
    
    def evaluator(examples):
        return len(examples) * 0.3
    
    # Test with high mutation rate
    result = optimizer.optimize(trainset, k=1, evaluator=evaluator)
    assert len(result) == 1
    assert result[0] in trainset


def test_bayesian_optimizer_edge_cases():
    """Test Bayesian optimizer edge cases."""
    from bandit_dspy.teleprompter import BayesianOptimizer
    
    optimizer = BayesianOptimizer(n_calls=2, random_state=42)
    
    trainset = [
        dspy.Example(description="test", code="def test(): pass").with_inputs('description')
    ]
    
    def evaluator(examples):
        return 0.7
    
    # Test with max_k larger than trainset
    optimal_k = optimizer.optimize_k(trainset, max_k=5, evaluator=evaluator)
    assert optimal_k == 1


def test_context_creation():
    """Test Context creation in BanditRunner."""
    runner = BanditRunner()
    
    # Test that Context is created properly
    code = "import os"
    issues = runner.analyze_code(code)
    assert isinstance(issues, list)
    assert len(issues) == 0  # No security issues in this code


def test_genetic_crossover_edge_cases():
    """Test genetic algorithm crossover with edge cases."""
    from bandit_dspy.teleprompter import GeneticOptimizer
    
    optimizer = GeneticOptimizer()
    
    trainset = [
        dspy.Example(description="test1", code="def test1(): pass").with_inputs('description'),
        dspy.Example(description="test2", code="def test2(): pass").with_inputs('description'),
        dspy.Example(description="test3", code="def test3(): pass").with_inputs('description'),
    ]
    
    # Test crossover when combined has exactly k elements
    parent1 = [trainset[0]]
    parent2 = [trainset[1]]
    child = optimizer._crossover(parent1, parent2, trainset, k=2)
    assert len(child) == 2
    
    # Test crossover when combined has fewer than k elements
    child = optimizer._crossover(parent1, parent2, trainset, k=1)
    assert len(child) == 1


def test_tournament_selection():
    """Test tournament selection in genetic optimizer."""
    from bandit_dspy.teleprompter import GeneticOptimizer
    
    optimizer = GeneticOptimizer()
    
    fitness_scores = [
        (0.8, "individual1"),
        (0.6, "individual2"), 
        (0.9, "individual3"),
        (0.4, "individual4")
    ]
    
    # Test tournament selection
    selected = optimizer._tournament_select(fitness_scores, tournament_size=2)
    assert selected in ["individual1", "individual2", "individual3", "individual4"]
    
    # Test with tournament size larger than population
    selected = optimizer._tournament_select(fitness_scores, tournament_size=10)
    assert selected in ["individual1", "individual2", "individual3", "individual4"]