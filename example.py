import dspy
import json
from bandit_dspy import BanditTeleprompter, bandit_metric

class SimpleCodeGen(dspy.Signature):
    """Generate a short Python code snippet."""
    description = dspy.InputField()
    code = dspy.OutputField()

# Dummy LLM for testing
class DummyLLM(dspy.LM):
    def __init__(self):
        super().__init__("dummy")

    def __call__(self, messages, **kwargs):
        # This dummy LLM always returns the same insecure code
        # wrapped in a JSON object with a "code" field.
        response = {"code": "import os\npassword = \"hardcoded_password\""}
        return [json.dumps(response)]

    def basic_request(self, prompt, **kwargs):
        pass

class CodeGenModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleCodeGen)

    def forward(self, description):
        return self.predictor(description=description)

if __name__ == "__main__":
    # Configure DSPy to use the dummy LLM
    dspy.settings.configure(lm=DummyLLM())

    trainset = [
        dspy.Example(description="a function that adds two numbers", code="def add(a, b): return a + b").with_inputs('description'),
        dspy.Example(description="a function that subtracts two numbers", code="def subtract(a, b): return a - b").with_inputs('description'),
        dspy.Example(description="a function that multiplies two numbers", code="def multiply(a, b): return a * b").with_inputs('description'),
        dspy.Example(description="a function that divides two numbers", code="def divide(a, b): return a / b").with_inputs('description'),
    ]

    student = CodeGenModule()

    teleprompter = BanditTeleprompter(metric=bandit_metric)
    compiled_program = teleprompter.compile(student, trainset=trainset)

    # Use the compiled program to generate code
    prediction = compiled_program(description="a function that calculates the factorial of a number")
    print(prediction.code)
