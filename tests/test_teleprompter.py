import pytest

# Handle missing dependencies gracefully
try:
    import dspy
    from bandit_dspy import (BanditTeleprompter, BayesianOptimizer,
                             GeneticOptimizer, create_bandit_metric)
    HAS_DEPS = True
except (ImportError, ModuleNotFoundError):
    HAS_DEPS = False
    # Create minimal mocks
    class MockDSPy:
        class Signature: pass
        class Module: 
            def __init__(self): pass
        class Predict:
            def __init__(self, sig): pass
        class InputField: pass
        class OutputField: pass
    dspy = MockDSPy()
    BanditTeleprompter = type('BanditTeleprompter', (), {})
    BayesianOptimizer = type('BayesianOptimizer', (), {})
    GeneticOptimizer = type('GeneticOptimizer', (), {})
    create_bandit_metric = lambda: None

# Skip all tests if dependencies missing
if not HAS_DEPS:
    pytestmark = pytest.mark.skip(reason="Dependencies (dspy-ai, bandit) not available")


class SimpleCodeGen(dspy.Signature):
    """Generate a short Python code snippet."""

    description = dspy.InputField()
    code = dspy.OutputField()


class CodeGenModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleCodeGen)

    def forward(self, description):
        return self.predictor(description=description)


def test_bandit_teleprompter_basic(secure_llm, sample_trainset):
    """Test basic BanditTeleprompter functionality."""
    dspy.settings.configure(lm=secure_llm)

    student = CodeGenModule()
    teleprompter = BanditTeleprompter(
        metric=create_bandit_metric(), k=2, num_candidates=2
    )
    compiled_program = teleprompter.compile(student, trainset=sample_trainset)

    assert compiled_program is not None
    assert isinstance(compiled_program, CodeGenModule)


def test_train_validation_split(mixed_llm, sample_trainset):
    """Test train/validation split functionality."""
    dspy.settings.configure(lm=mixed_llm)

    student = CodeGenModule()

    # Test different split ratios
    for split_ratio in [0.6, 0.8, 0.9]:
        teleprompter = BanditTeleprompter(
            metric=create_bandit_metric(),
            k=2,
            num_candidates=2,
            train_val_split=split_ratio,
        )

        compiled_program = teleprompter.compile(student, trainset=sample_trainset)
        assert compiled_program is not None


class TestOptimizationMethods:
    """Test different optimization methods."""

    def test_random_optimization(self, mixed_llm, sample_trainset):
        """Test random optimization method."""
        dspy.settings.configure(lm=mixed_llm)

        student = CodeGenModule()
        teleprompter = BanditTeleprompter(
            metric=create_bandit_metric(),
            k=2,
            num_candidates=3,
            optimization_method="random",
        )

        compiled_program = teleprompter.compile(student, trainset=sample_trainset)
        assert compiled_program is not None

    def test_genetic_optimization(self, mixed_llm, sample_trainset):
        """Test genetic algorithm optimization."""
        dspy.settings.configure(lm=mixed_llm)

        student = CodeGenModule()
        teleprompter = BanditTeleprompter(
            metric=create_bandit_metric(),
            k=2,
            num_candidates=3,
            optimization_method="genetic",
            genetic_config={
                "population_size": 4,
                "generations": 2,
                "mutation_rate": 0.2,
            },
        )

        compiled_program = teleprompter.compile(student, trainset=sample_trainset)
        assert compiled_program is not None

    def test_bayesian_optimization(self, mixed_llm, sample_trainset):
        """Test Bayesian optimization method."""
        dspy.settings.configure(lm=mixed_llm)

        student = CodeGenModule()
        teleprompter = BanditTeleprompter(
            metric=create_bandit_metric(),
            k=2,
            num_candidates=3,
            optimization_method="bayesian",
            bayesian_config={"n_calls": 4},
        )

        compiled_program = teleprompter.compile(student, trainset=sample_trainset)
        assert compiled_program is not None

    def test_invalid_optimization_method(self, mixed_llm, sample_trainset):
        """Test that invalid optimization method falls back to random."""
        dspy.settings.configure(lm=mixed_llm)

        student = CodeGenModule()
        teleprompter = BanditTeleprompter(
            metric=create_bandit_metric(),
            k=2,
            num_candidates=2,
            optimization_method="invalid_method",
        )

        # Should fallback to random and still work
        compiled_program = teleprompter.compile(student, trainset=sample_trainset)
        assert compiled_program is not None


class TestGeneticOptimizer:
    """Test the GeneticOptimizer class."""

    def test_basic_optimization(self, sample_trainset):
        """Test basic genetic optimization."""
        optimizer = GeneticOptimizer(population_size=4, generations=2)

        def dummy_evaluator(examples):
            # Return random score for testing
            return len(examples) * 0.1

        result = optimizer.optimize(sample_trainset, k=2, evaluator=dummy_evaluator)
        assert len(result) == 2
        assert all(ex in sample_trainset for ex in result)

    def test_insufficient_examples(self):
        """Test genetic optimizer with insufficient training examples."""
        optimizer = GeneticOptimizer(population_size=4, generations=2)
        small_trainset = [
            dspy.Example(description="test", code="test").with_inputs("description")
        ]

        def dummy_evaluator(examples):
            return 0.5

        result = optimizer.optimize(small_trainset, k=3, evaluator=dummy_evaluator)
        # Should return all available examples when k > len(trainset)
        assert len(result) == 1


class TestBayesianOptimizer:
    """Test the BayesianOptimizer class."""

    def test_k_optimization(self, sample_trainset):
        """Test k parameter optimization."""
        optimizer = BayesianOptimizer(n_calls=6)

        def dummy_evaluator(examples):
            # Prefer k=2 for testing
            return 0.8 if len(examples) == 2 else 0.5

        optimal_k = optimizer.optimize_k(
            sample_trainset, max_k=3, evaluator=dummy_evaluator
        )
        assert optimal_k == 2

    def test_single_example_trainset(self):
        """Test Bayesian optimizer with single example."""
        optimizer = BayesianOptimizer(n_calls=4)
        single_trainset = [
            dspy.Example(description="test", code="test").with_inputs("description")
        ]

        def dummy_evaluator(examples):
            return 0.5

        optimal_k = optimizer.optimize_k(
            single_trainset, max_k=5, evaluator=dummy_evaluator
        )
        assert optimal_k == 1
