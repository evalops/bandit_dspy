"""
Tests for GEPA (Reflective Prompt Evolution) optimizer.
"""

import pytest
import dspy
from unittest.mock import patch, MagicMock
from bandit_dspy import (
    SecurityGEPAOptimizer, 
    GEPATeleprompter, 
    SecurityReflector, 
    ParetoSelector,
    create_bandit_metric
)
from bandit_dspy.gepa_optimizer import GEPACandidate, SecurityReflection


class DummyModule(dspy.Module):
    """Dummy DSPy module for testing."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("description -> code")
    
    def forward(self, description):
        return dspy.Example(code=f"def test(): # {description}\n    return 'secure_code'")
    
    def named_modules(self):
        return [('predictor', self.predictor)]


@pytest.fixture
def sample_trainset():
    """Sample training set for testing."""
    return [
        dspy.Example(description="create a function that adds numbers", code="def add(a, b): return a + b").with_inputs('description'),
        dspy.Example(description="create a secure hash function", code="import hashlib\ndef hash_string(s): return hashlib.sha256(s.encode()).hexdigest()").with_inputs('description'),
        dspy.Example(description="create a file reader", code="def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()").with_inputs('description'),
    ]


@pytest.fixture
def security_metric():
    """Security metric for testing."""
    return create_bandit_metric()


class TestGEPACandidate:
    """Test GEPACandidate class."""
    
    def test_candidate_creation(self):
        candidate = GEPACandidate(
            instruction_prompt="Generate secure code",
            few_shot_examples=[],
            security_guidelines="Follow OWASP",
            performance_config={'temp': 0.3},
            candidate_id="test_1"
        )
        
        assert candidate.instruction_prompt == "Generate secure code"
        assert candidate.candidate_id == "test_1"
        assert candidate.scores is None
    
    def test_candidate_hash(self):
        candidate = GEPACandidate(
            instruction_prompt="Test",
            few_shot_examples=[],
            security_guidelines="Guidelines",
            performance_config={},
            candidate_id="test_hash"
        )
        
        assert hash(candidate) == hash("test_hash")


class TestParetoSelector:
    """Test Pareto selector for multi-objective optimization."""
    
    def test_pareto_selector_initialization(self):
        selector = ParetoSelector()
        assert selector.objectives == ['security_score', 'performance_score', 'functionality_score']
        assert len(selector.pareto_front) == 0
    
    def test_add_dominated_candidate(self):
        selector = ParetoSelector()
        
        # Add first candidate
        candidate1 = GEPACandidate("", [], "", {}, "c1")
        candidate1.scores = {'security_score': 0.8, 'performance_score': 0.6, 'functionality_score': 0.7}
        
        assert selector.add_candidate(candidate1) == True
        assert len(selector.pareto_front) == 1
        
        # Add dominated candidate
        candidate2 = GEPACandidate("", [], "", {}, "c2")
        candidate2.scores = {'security_score': 0.7, 'performance_score': 0.5, 'functionality_score': 0.6}
        
        assert selector.add_candidate(candidate2) == False
        assert len(selector.pareto_front) == 1
    
    def test_add_dominating_candidate(self):
        selector = ParetoSelector()
        
        # Add first candidate
        candidate1 = GEPACandidate("", [], "", {}, "c1")
        candidate1.scores = {'security_score': 0.6, 'performance_score': 0.5, 'functionality_score': 0.6}
        selector.add_candidate(candidate1)
        
        # Add dominating candidate
        candidate2 = GEPACandidate("", [], "", {}, "c2")
        candidate2.scores = {'security_score': 0.8, 'performance_score': 0.7, 'functionality_score': 0.8}
        
        assert selector.add_candidate(candidate2) == True
        assert len(selector.pareto_front) == 1
        assert selector.pareto_front[0].candidate_id == "c2"
    
    def test_get_best_candidates(self):
        selector = ParetoSelector()
        
        candidates = []
        for i, scores in enumerate([
            {'security_score': 0.9, 'performance_score': 0.5, 'functionality_score': 0.7},
            {'security_score': 0.6, 'performance_score': 0.9, 'functionality_score': 0.7},
            {'security_score': 0.7, 'performance_score': 0.7, 'functionality_score': 0.9}
        ]):
            candidate = GEPACandidate("", [], "", {}, f"c{i}")
            candidate.scores = scores
            candidates.append(candidate)
            selector.add_candidate(candidate)
        
        best = selector.get_best_candidates(k=2)
        assert len(best) == 2
        assert all(c in selector.pareto_front for c in best)


class TestSecurityReflector:
    """Test SecurityReflector class."""
    
    @patch('dspy.settings.lm')
    def test_reflector_initialization(self, mock_lm):
        reflector = SecurityReflector()
        assert reflector.reflection_lm == mock_lm
        assert reflector.reflector is not None
    
    def test_format_security_issues(self):
        reflector = SecurityReflector()
        
        issues = [
            {'test_id': 'B101', 'description': 'Hardcoded password'},
            {'test_id': 'B602', 'description': 'Shell injection'}
        ]
        
        formatted = reflector._format_security_issues(issues)
        assert "B101: Hardcoded password" in formatted
        assert "B602: Shell injection" in formatted
    
    def test_format_security_issues_empty(self):
        reflector = SecurityReflector()
        formatted = reflector._format_security_issues([])
        assert formatted == "No security issues detected."
    
    def test_format_examples(self):
        reflector = SecurityReflector()
        
        examples = [
            dspy.Example(description="test1", code="def test1(): pass"),
            dspy.Example(description="test2", code="def test2(): pass"),
        ]
        
        formatted = reflector._format_examples(examples)
        assert "Example 1: test1" in formatted
        assert "Example 2: test2" in formatted


class TestSecurityGEPAOptimizer:
    """Test SecurityGEPAOptimizer class."""
    
    def test_optimizer_initialization(self):
        optimizer = SecurityGEPAOptimizer(
            max_iterations=5,
            population_size=4,
            reflection_frequency=2
        )
        
        assert optimizer.max_iterations == 5
        assert optimizer.population_size == 4
        assert optimizer.reflection_frequency == 2
        assert isinstance(optimizer.reflector, SecurityReflector)
        assert isinstance(optimizer.pareto_selector, ParetoSelector)
    
    def test_initialize_population(self, sample_trainset):
        optimizer = SecurityGEPAOptimizer(population_size=3)
        
        population = optimizer._initialize_population(
            "Generate secure code",
            sample_trainset[:1],
            sample_trainset
        )
        
        assert len(population) == 3
        assert all(isinstance(c, GEPACandidate) for c in population)
        assert population[0].candidate_id == "seed_0"
    
    def test_mutate_candidate(self, sample_trainset):
        optimizer = SecurityGEPAOptimizer()
        
        base_candidate = GEPACandidate(
            instruction_prompt="Original prompt",
            few_shot_examples=sample_trainset[:1],
            security_guidelines="Original guidelines",
            performance_config={'temperature': 0.3},
            candidate_id="base"
        )
        
        mutated = optimizer._mutate_candidate(base_candidate, 1, sample_trainset)
        
        assert mutated.candidate_id.startswith("mutation_1_")
        assert isinstance(mutated, GEPACandidate)
    
    @patch('dspy.settings.lm')
    def test_configure_program(self, mock_lm):
        optimizer = SecurityGEPAOptimizer()
        student = DummyModule()
        
        candidate = GEPACandidate(
            instruction_prompt="Secure coding instruction",
            few_shot_examples=[],
            security_guidelines="Follow OWASP",
            performance_config={},
            candidate_id="test"
        )
        
        configured = optimizer._configure_program(student, candidate)
        assert configured is not None
    
    def test_create_optimized_program(self):
        optimizer = SecurityGEPAOptimizer()
        student = DummyModule()
        
        candidate = GEPACandidate(
            instruction_prompt="Best instruction",
            few_shot_examples=[],
            security_guidelines="Best guidelines",
            performance_config={'temp': 0.2},
            candidate_id="best"
        )
        candidate.scores = {'security_score': 0.9, 'performance_score': 0.8}
        
        optimized = optimizer._create_optimized_program(student, candidate)
        
        assert hasattr(optimized, '_gepa_metadata')
        assert optimized._gepa_metadata['best_candidate_id'] == "best"
        assert optimized._gepa_metadata['optimization_type'] == 'GEPA'
    
    def test_extract_improved_instruction(self):
        optimizer = SecurityGEPAOptimizer()
        
        reflection = SecurityReflection(
            code="test code",
            security_issues=[],
            security_score=0.5,
            performance_score=0.7,
            reflection_text="Analysis",
            proposed_improvements=["Instruction: Use secure coding practices"],
            trajectory_summary="Summary"
        )
        
        improved = optimizer._extract_improved_instruction(reflection)
        assert "Use secure coding practices" in improved
    
    def test_extract_improved_guidelines(self):
        optimizer = SecurityGEPAOptimizer()
        
        reflection = SecurityReflection(
            code="test code",
            security_issues=[],
            security_score=0.5,
            performance_score=0.7,
            reflection_text="Analysis",
            proposed_improvements=["Guidelines: Enhanced security rules"],
            trajectory_summary="Summary"
        )
        
        improved = optimizer._extract_improved_guidelines(reflection)
        assert "Enhanced security rules" in improved


class TestGEPATeleprompter:
    """Test GEPATeleprompter integration class."""
    
    def test_teleprompter_initialization(self, security_metric):
        teleprompter = GEPATeleprompter(
            metric=security_metric,
            max_iterations=3,
            population_size=2
        )
        
        assert teleprompter.metric == security_metric
        assert isinstance(teleprompter.optimizer, SecurityGEPAOptimizer)
        assert teleprompter.optimizer.max_iterations == 3
        assert teleprompter.optimizer.population_size == 2
    
    @patch('builtins.print')
    @patch.object(SecurityGEPAOptimizer, 'optimize')
    def test_compile(self, mock_optimize, mock_print, security_metric, sample_trainset):
        # Mock the optimizer to return a dummy result
        dummy_program = DummyModule()
        mock_optimize.return_value = (dummy_program, {'iterations': []})
        
        teleprompter = GEPATeleprompter(metric=security_metric)
        student = DummyModule()
        
        result = teleprompter.compile(student, trainset=sample_trainset)
        
        assert result == dummy_program
        mock_optimize.assert_called_once()
        mock_print.assert_called()


class TestGEPAIntegration:
    """Integration tests for GEPA components."""
    
    def test_end_to_end_small_scale(self, sample_trainset, security_metric):
        """Test GEPA optimization with minimal scale."""
        
        # Use very small parameters for fast testing
        optimizer = SecurityGEPAOptimizer(
            max_iterations=1,
            population_size=2,
            reflection_frequency=1
        )
        
        student = DummyModule()
        
        # Mock the LLM calls to avoid external dependencies
        with patch('dspy.context'), \
             patch.object(optimizer.reflector, 'forward') as mock_reflect:
            
            # Mock reflection to return basic response
            mock_reflect.return_value = SecurityReflection(
                code="def test(): pass",
                security_issues=[],
                security_score=0.8,
                performance_score=0.7,
                reflection_text="Good security practices observed",
                proposed_improvements=["Continue with current approach"],
                trajectory_summary="Secure implementation"
            )
            
            # This should run without errors
            optimized_program, history = optimizer.optimize(
                student_program=student,
                trainset=sample_trainset[:1],  # Use minimal trainset
                security_metric=security_metric
            )
            
            assert optimized_program is not None
            assert 'iterations' in history