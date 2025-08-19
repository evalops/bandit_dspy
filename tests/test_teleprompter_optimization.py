import dspy
import json
from bandit_dspy import BanditTeleprompter, bandit_metric

class SimpleCodeGen(dspy.Signature):
    """Generate a short Python code snippet."""
    description = dspy.InputField()
    code = dspy.OutputField()

# Dummy LLMs for testing
class BadLLM(dspy.LM):
    def __init__(self):
        super().__init__("bad-llm")

    def __call__(self, messages, **kwargs):
        # If the prompt contains a secure example, behave like the good LLM
        if "def add(a, b): return a + b" in messages:
            response = {"code": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"}
        else:
            response = {"code": "import os\npassword = \"hardcoded_password\""}
        return [json.dumps(response)]

    def basic_request(self, prompt, **kwargs):
        pass

class GoodLLM(dspy.LM):
    def __init__(self):
        super().__init__("good-llm")

    def __call__(self, messages, **kwargs):
        response = {"code": "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"}
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
    # Configure DSPy to use the bad LLM by default
    dspy.settings.configure(lm=BadLLM())

    trainset = [
        dspy.Example(description="a function that adds two numbers", code="def add(a, b): return a + b").with_inputs('description'),
        dspy.Example(description="a function that subtracts two numbers", code="def subtract(a, b): return a - b").with_inputs('description'),
    ]

    student = CodeGenModule()

    # The student program with the bad LLM should generate insecure code
    prediction = student(description="a function that calculates the factorial of a number")
    assert "hardcoded_password" in prediction.code

    teleprompter = BanditTeleprompter(metric=bandit_metric, k=1, num_candidates=2)
    compiled_program = teleprompter.compile(student, trainset=trainset)

    # The compiled program should now generate secure code
    dspy.settings.configure(lm=GoodLLM()) # Switch to the good LLM to make sure the compiled program is used
    prediction = compiled_program(description="a function that calculates the factorial of a number")
    assert "hardcoded_password" not in prediction.code