import json
import pytest

# Handle missing dependencies gracefully
try:
    import dspy
    from bandit_dspy import (BanditTeleprompter, SecurityMetric,
                             create_bandit_metric)
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
        class LM:
            def __init__(self, name="test"): pass
        class Example:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            def with_inputs(self, *args): return self
    dspy = MockDSPy()
    BanditTeleprompter = type('BanditTeleprompter', (), {})
    SecurityMetric = type('SecurityMetric', (), {})
    create_bandit_metric = lambda: None

# Skip all tests if dependencies missing
if not HAS_DEPS:
    pytestmark = pytest.mark.skip(reason="Dependencies (dspy-ai, bandit) not available")


class SimpleCodeGen(dspy.Signature):
    """Generate a short Python code snippet."""

    description = dspy.InputField()
    code = dspy.OutputField()


class AdaptiveLLM(dspy.LM):
    """LLM that adapts behavior based on few-shot examples."""

    def __init__(self):
        super().__init__("adaptive-llm")

    def __call__(self, messages, **kwargs):
        # Check if this is within the compile context (during optimization)
        # by checking for the actual message content or prompt structure
        message_text = str(messages) if messages else ""

        # If the message mentions secure examples or contains secure code patterns
        if any(
            secure_pattern in message_text.lower()
            for secure_pattern in [
                "def add",
                "def subtract",
                "def multiply",
                "def divide",
            ]
        ):
            # Generate secure code when examples are present
            response = {
                "code": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
            }
        else:
            # Generate insecure code otherwise
            response = {"code": 'import os\npassword = "hardcoded_password"'}
        return [json.dumps(response)]

    def basic_request(self, prompt, **kwargs):
        pass


class SecureLLM(dspy.LM):
    """LLM that always generates secure code."""

    def __init__(self):
        super().__init__("secure-llm")

    def __call__(self, messages, **kwargs):
        response = {
            "code": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
        }
        return [json.dumps(response)]

    def basic_request(self, prompt, **kwargs):
        pass


class CodeGenModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleCodeGen)

    def forward(self, description):
        return self.predictor(description=description)


def test_teleprompter_optimization():
    """Test that teleprompter optimizes for security."""
    # Configure DSPy to use the adaptive LLM
    dspy.settings.configure(lm=AdaptiveLLM())

    trainset = [
        dspy.Example(
            description="a function that adds two numbers",
            code="def add(a, b): return a + b",
        ).with_inputs("description"),
        dspy.Example(
            description="a function that subtracts two numbers",
            code="def subtract(a, b): return a - b",
        ).with_inputs("description"),
    ]

    student = CodeGenModule()

    # The student program with the bad LLM should generate insecure code
    prediction = student(
        description="a function that calculates the factorial of a number"
    )
    assert "hardcoded_password" in prediction.code

    teleprompter = BanditTeleprompter(
        metric=create_bandit_metric(), k=1, num_candidates=2
    )
    compiled_program = teleprompter.compile(student, trainset=trainset)

    # The compiled program should now generate secure code
    prediction = compiled_program(
        description="a function that calculates the factorial of a number"
    )
    # Check that the optimization improved security (higher score than raw insecure code)
    security_result = create_bandit_metric()(None, prediction)
    assert (
        security_result["score"] > 0.1
    )  # Should be better than completely insecure code


def test_optimization_methods_comparison():
    """Test that different optimization methods work and can be compared."""
    dspy.settings.configure(lm=AdaptiveLLM())

    trainset = [
        dspy.Example(
            description="a function that adds two numbers",
            code="def add(a, b): return a + b",
        ).with_inputs("description"),
        dspy.Example(
            description="a function that subtracts two numbers",
            code="def subtract(a, b): return a - b",
        ).with_inputs("description"),
        dspy.Example(
            description="a function that multiplies two numbers",
            code="def multiply(a, b): return a * b",
        ).with_inputs("description"),
    ]

    student = CodeGenModule()
    results = {}

    # Test different optimization methods
    methods = ["random", "genetic", "bayesian"]

    for method in methods:
        config = {}
        if method == "genetic":
            config = {
                "genetic_config": {
                    "population_size": 4,
                    "generations": 2,
                    "mutation_rate": 0.2,
                }
            }
        elif method == "bayesian":
            config = {"bayesian_config": {"n_calls": 6}}

        teleprompter = BanditTeleprompter(
            metric=create_bandit_metric(),
            k=2,
            num_candidates=3,
            optimization_method=method,
            **config,
        )

        compiled_program = teleprompter.compile(student, trainset=trainset)

        # Test the compiled program
        prediction = compiled_program(
            description="a function that validates user input"
        )
        security_result = create_bandit_metric()(None, prediction)

        results[method] = {
            "score": security_result["score"],
            "issues": len(security_result["issues"]),
        }

        # All methods should produce reasonable results
        assert security_result["score"] >= 0.5  # Should be reasonably secure


def test_security_metric_integration():
    """Test integration with different SecurityMetric configurations."""
    dspy.settings.configure(lm=AdaptiveLLM())

    trainset = [
        dspy.Example(
            description="a function that adds two numbers",
            code="def add(a, b): return a + b",
        ).with_inputs("description"),
        dspy.Example(
            description="a function that subtracts two numbers",
            code="def subtract(a, b): return a - b",
        ).with_inputs("description"),
    ]

    student = CodeGenModule()

    # Test with strict security metric
    strict_metric = SecurityMetric(
        severity_weights={"HIGH": 10.0, "MEDIUM": 5.0, "LOW": 1.0},
        confidence_weights={"HIGH": 10.0, "MEDIUM": 5.0, "LOW": 1.0},
    )

    def strict_metric_wrapper(example, pred, trace=None):
        result = strict_metric.evaluate(pred.code)
        return {
            "score": result["score"],
            "issues": result["issues"],
            "issue_counts": result["issue_counts"],
        }

    teleprompter = BanditTeleprompter(
        metric=strict_metric_wrapper, k=1, num_candidates=2
    )

    compiled_program = teleprompter.compile(student, trainset=trainset)
    prediction = compiled_program(description="a function that processes data")

    # Should work with custom metric
    assert prediction.code is not None
    assert len(prediction.code) > 0


def test_edge_cases():
    """Test edge cases in teleprompter optimization."""
    dspy.settings.configure(lm=SecureLLM())

    # Test with single training example
    small_trainset = [
        dspy.Example(description="a function", code="def func(): return 1").with_inputs(
            "description"
        )
    ]

    student = CodeGenModule()
    teleprompter = BanditTeleprompter(
        metric=create_bandit_metric(), k=1, num_candidates=2
    )

    compiled_program = teleprompter.compile(student, trainset=small_trainset)
    assert compiled_program is not None

    # Test with k larger than trainset
    teleprompter_large_k = BanditTeleprompter(
        metric=create_bandit_metric(), k=5, num_candidates=2
    )
    compiled_program_large_k = teleprompter_large_k.compile(
        student, trainset=small_trainset
    )
    assert compiled_program_large_k is not None


def test_different_train_val_splits():
    """Test various train/validation split ratios."""
    dspy.settings.configure(lm=AdaptiveLLM())

    trainset = [
        dspy.Example(
            description=f"function {i}", code=f"def func_{i}(): return {i}"
        ).with_inputs("description")
        for i in range(10)
    ]

    student = CodeGenModule()

    # Test different split ratios
    for split_ratio in [0.5, 0.7, 0.8, 0.9]:
        teleprompter = BanditTeleprompter(
            metric=create_bandit_metric(),
            k=2,
            num_candidates=2,
            train_val_split=split_ratio,
        )

        compiled_program = teleprompter.compile(student, trainset=trainset)
        assert compiled_program is not None

        # Verify it can generate predictions
        prediction = compiled_program(description="test function")
        assert prediction.code is not None
