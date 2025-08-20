"""
Pytest configuration and shared fixtures for bandit_dspy tests.
"""

from unittest.mock import MagicMock
import pytest

# Handle missing dependencies gracefully
try:
    import dspy
    from bandit_dspy import BanditRunner, SecurityMetric
    HAS_DEPS = True
except (ImportError, ModuleNotFoundError):
    HAS_DEPS = False
    # Create minimal mock classes
    class MockDSPy:
        class LM:
            def __init__(self, name="test"): self.name = name
        class Example:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
            def with_inputs(self, *args): return self
            def inputs(self): return {}
        class settings:
            lm = None
            @classmethod
            def configure(cls, **kwargs):
                cls.lm = kwargs.get('lm')
    
    dspy = MockDSPy()
    BanditRunner = type('BanditRunner', (), {})
    SecurityMetric = type('SecurityMetric', (), {})

# Skip all tests if dependencies missing
if not HAS_DEPS:
    pytestmark = pytest.mark.skip(reason="Dependencies (dspy-ai, bandit) not available")


@pytest.fixture(autouse=True)
def dspy_lm_fixture():
    """Fixture to manage dspy.settings.lm for tests."""
    if not HAS_DEPS:
        yield MagicMock()
        return
        
    original_lm = dspy.settings.lm
    mock_lm = MagicMock()
    dspy.settings.configure(lm=mock_lm)
    yield mock_lm
    dspy.settings.configure(lm=original_lm)


if HAS_DEPS:
    class TestLLM(dspy.LM):
        """Test LLM that returns predictable outputs for testing."""

        def __init__(self, responses=None):
            super().__init__("test-llm")
            self.responses = responses or [
                '{"code": "def add(a, b): return a + b"}',  # Secure code
                '{"code": "password = "hardcoded""}',  # Insecure code
            ]
            self.call_count = 0

        def __call__(self, messages, **kwargs):
            response = self.responses[self.call_count % len(self.responses)]
            self.call_count += 1
            return [response]

        def basic_request(self, prompt, **kwargs):
            pass
else:
    TestLLM = type('TestLLM', (), {})


@pytest.fixture
def secure_llm():
    """LLM that generates secure code."""
    return TestLLM(
        [
            '{"code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"}',
            '{"code": "import hashlib\ndata = "test"\nhash_val = hashlib.sha256(data.encode()).hexdigest()"}',
        ]
    )


@pytest.fixture
def insecure_llm():
    """LLM that generates insecure code."""
    return TestLLM(
        [
            '{"code": "password = "hardcoded_password""}',
            '{"code": "import subprocess\nsubprocess.call("rm -rf /", shell=True)"}',
        ]
    )


@pytest.fixture
def mixed_llm():
    """LLM that generates both secure and insecure code."""
    return TestLLM(
        [
            '{"code": "def add(a, b): return a + b"}',  # Secure
            '{"code": "password = "hardcoded""}',  # Insecure
            '{"code": "import math\nresult = math.sqrt(16)"}',  # Secure
            '{"code": "exec("print(hello)")"}',  # Insecure
        ]
    )


@pytest.fixture
def sample_trainset():
    """Sample training dataset for testing."""
    return [
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
        dspy.Example(
            description="a function that divides two numbers",
            code="def divide(a, b): return a / b if b != 0 else None",
        ).with_inputs("description"),
    ]


@pytest.fixture
def security_metric_default():
    """Default security metric instance."""
    return SecurityMetric()


@pytest.fixture
def security_metric_strict():
    """Strict security metric with high penalties."""
    return SecurityMetric(
        severity_weights={"HIGH": 10.0, "MEDIUM": 5.0, "LOW": 1.0},
        confidence_weights={"HIGH": 10.0, "MEDIUM": 5.0, "LOW": 1.0},
        penalty_threshold=0.5,
    )


@pytest.fixture
def bandit_runner():
    """BanditRunner instance for testing."""
    return BanditRunner()


@pytest.fixture
def secure_code():
    """Sample secure Python code."""
    return """
def calculate_hash(data):
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()

def safe_divide(a, b):
    return a / b if b != 0 else None
"""


@pytest.fixture
def insecure_code():
    """Sample insecure Python code."""
    return """
import subprocess
import os

password = "hardcoded_password"
api_key = "sk-1234567890abcdef"

def unsafe_command(user_input):
    subprocess.call(user_input, shell=True)
    
def eval_user_input(expr):
    return eval(expr)
"""


@pytest.fixture
def code_with_mixed_issues():
    """Code with various security issue severities."""
    return """
import subprocess
password = "hardcoded"  # B105 - LOW severity
subprocess.call("ls", shell=True)  # B602 - HIGH severity
assert True  # B101 - LOW severity
"""
