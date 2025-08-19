import dspy
import json
from bandit_dspy import BanditTeleprompter, bandit_metric

class SimpleCodeGen(dspy.Signature):
    """Generate a short Python code snippet."""
    description = dspy.InputField()
    code = dspy.OutputField()

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

def test_bandit_teleprompter():
    # Configure DSPy to use a dummy LLM for this test
    dspy.settings.configure(lm=GoodLLM())

    trainset = [
        dspy.Example(description="a function that adds two numbers", code="def add(a, b): return a + b").with_inputs('description'),
        dspy.Example(description="a function that subtracts two numbers", code="def subtract(a, b): return a - b").with_inputs('description'),
        dspy.Example(description="a function that multiplies two numbers", code="def multiply(a, b): return a * b").with_inputs('description'),
        dspy.Example(description="a function that divides two numbers", code="def divide(a, b): return a / b").with_inputs('description'),
    ]

    student = CodeGenModule()

    teleprompter = BanditTeleprompter(metric=bandit_metric)
    compiled_program = teleprompter.compile(student, trainset=trainset)

    assert compiled_program is not None
    assert isinstance(compiled_program, CodeGenModule)