# bandit_dspy

A library to integrate Bandit and DSPy for security-aware LLM development.

## 🚀 FEATURES

*   **Security-Aware LLM Development:** Integrate static analysis directly into your DSPy optimization loops.
*   **Bandit Integration:** Leverage the power of Bandit to identify common Python security vulnerabilities in generated code.
*   **DSPy Metric:** A custom DSPy metric (`bandit_metric`) that quantifies the security posture of generated code.
*   **DSPy Teleprompter:** A specialized DSPy teleprompter (`BanditTeleprompter`) that optimizes LLM programs to produce more secure outputs.
*   **Configurable Optimization:** Adjust the teleprompter's parameters (e.g., number of candidates, few-shot examples) to fine-tune the optimization process.

## 🔧 PREREQUISITES

Before running this project, ensure you have the following installed:

*   Python 3.8+
*   `pip` (Python package installer)
*   A configured DSPy Language Model (e.g., `dspy.OllamaLocal`, `dspy.OpenAI`)

## 📦 INSTALLATION

1.  Clone the repository:
    ```bash
    git clone https://github.com/evalops/bandit_dspy.git
    cd bandit_dspy
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -e .
    ```

## 🚦 QUICK START

Here's a quick example of how to use the `BanditTeleprompter` to optimize a simple code generation program for security.

```python
import dspy
import json
from bandit_dspy import BanditTeleprompter, bandit_metric

# 1. Define your code generation signature
class SimpleCodeGen(dspy.Signature):
    """Generate a short Python code snippet."""
    description = dspy.InputField()
    code = dspy.OutputField()

# 2. Define your DSPy module
class CodeGenModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleCodeGen)

    def forward(self, description):
        return self.predictor(description=description)

if __name__ == "__main__":
    # Configure DSPy with your Language Model (replace with your actual LLM)
    # For demonstration, we'll use a dummy LLM that always returns insecure code
    class DummyInsecureLLM(dspy.LM):
        def __init__(self):
            super().__init__("dummy-insecure")
        def __call__(self, messages, **kwargs):
            return [json.dumps({"code": "import os\npassword = \"hardcoded_password\""})]
        def basic_request(self, prompt, **kwargs): pass

    dspy.settings.configure(lm=DummyInsecureLLM())

    # 3. Prepare a training set (examples of desired code behavior)
    # For security optimization, these examples should ideally be secure code snippets.
    trainset = [
        dspy.Example(description="a function that adds two numbers", code="def add(a, b): return a + b").with_inputs('description'),
        dspy.Example(description="a function that subtracts two numbers", code="def subtract(a, b): return a - b").with_inputs('description'),
    ]

    # 4. Instantiate your student program
    student_program = CodeGenModule()

    # 5. Create the BanditTeleprompter
    teleprompter = BanditTeleprompter(metric=bandit_metric, k=1, num_candidates=2)

    # 6. Compile the program for security optimization
    print("Compiling the program for security optimization...")
    compiled_program = teleprompter.compile(student_program, trainset=trainset)
    print("Compilation complete!")

    # 7. Use the compiled program to generate code
    print("\nGenerating code with the compiled program:")
    prediction = compiled_program(description="a function that calculates the factorial of a number")
    print(prediction.code)

    # You can also evaluate the security of the generated code directly
    security_results = bandit_metric(None, prediction)
    print(f"\nSecurity Score: {security_results['score']:.2f}")
    print(f"Issues Found: {len(security_results['issues'])}")
    if security_results['issues']:
        print("Issue Details:")
        for issue in security_results['issues']:
            print(f"  - {issue.test_id}: {issue.issue_text} (Severity: {issue.issue_severity}, Confidence: {issue.issue_confidence})")
```

## 💡 HOW IT WORKS

`bandit_dspy` enhances your DSPy development workflow by integrating static application security testing (SAST) directly into the optimization process.

1.  **`bandit_metric`**: This custom DSPy metric wraps the Bandit security scanner. When evaluating generated code, it runs Bandit and calculates a security score based on the severity and confidence of detected vulnerabilities. A higher score indicates more secure code. It also provides detailed information about the issues found.
2.  **`BanditTeleprompter`**: This teleprompter uses the `bandit_metric` to guide the optimization of your DSPy program. It works by:
    *   Generating multiple candidate programs, each potentially using different few-shot examples from your training set.
    *   Evaluating each candidate program's generated code using the `bandit_metric`.
    *   Selecting the candidate program that consistently produces the most secure code (highest `bandit_metric` score).
    *   The current implementation uses a random search approach to explore different few-shot combinations.

## 🔧 CONFIGURATION

### `BanditTeleprompter` Parameters

*   `metric`: The DSPy metric to use for evaluation (e.g., `bandit_metric`).
*   `k` (int): The number of few-shot examples to randomly select from the training set for each candidate program. (Default: 3)
*   `num_candidates` (int): The total number of candidate programs (combinations of few-shot examples) to evaluate during the optimization process. A higher number may lead to better results but will take longer. (Default: 10)

## 📝 USAGE EXAMPLES

### EVALUATING CODE SECURITY DIRECTLY

You can use `bandit_metric` independently to assess the security of any Python code string:

```python
from bandit_dspy import bandit_metric
import dspy

# Example insecure code
insecure_code = """
import subprocess
subprocess.call("ls -l", shell=True) # B602: subprocess call with shell=True
password = "my_secret_password" # B105: hardcoded password
"""

# Create a dummy prediction object
prediction = dspy.Example(code=insecure_code)

# Get security results
security_results = bandit_metric(None, prediction)

print(f"Security Score: {security_results['score']:.2f}")
print(f"Issues Found: {len(security_results['issues'])}")
if security_results['issues']:
    print("Issue Details:")
    for issue in security_results['issues']:
        print(f"  - {issue.test_id}: {issue.issue_text} (Severity: {issue.issue_severity}, Confidence: {issue.issue_confidence})")

```

## 🐛 TROUBLESHOOTING

### Common Issues

**`No LM is loaded` Error:**
*   Ensure you have configured a Language Model for DSPy using `dspy.settings.configure(lm=...)`. Refer to the DSPy documentation for details on configuring various LLMs.

**Bandit Warnings in Console:**
*   Bandit may output warnings (e.g., about deprecated AST nodes). These are typically informational and do not prevent the library from functioning.

**Low Security Scores:**
*   Your LLM might be generating insecure code. Consider providing more diverse and secure examples in your `trainset`.
*   Increase `num_candidates` in `BanditTeleprompter` to explore more optimization possibilities.

## 🔬 DEVELOPMENT

### Project Structure

```
bandit_dspy/
├── src/
│   └── bandit_dspy/
│       ├── __init__.py      # Package initialization
│       ├── core.py          # Core Bandit integration (run_bandit function)
│       ├── metric.py        # DSPy bandit_metric implementation
│       └── teleprompter.py  # DSPy BanditTeleprompter implementation
├── tests/                   # Unit and integration tests
├── example.py               # Quick start example script
├── pyproject.toml           # Project metadata and dependencies
├── README.md                # This file
└── LICENSE                  # Project license
```

### Running Tests

To run the test suite:

```bash
source venv/bin/activate
python3 -m pytest
```

## 🤝 CONTRIBUTING

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure tests pass.
4.  Write clear, conventional commit messages.
5.  Submit a pull request.

## 📄 LICENSE

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🔗 DEPENDENCIES

*   `dspy-ai`: The DSPy framework.
*   `bandit`: The Python security linter.

## 📚 ADDITIONAL RESOURCES

*   [DSPy Documentation](https://dspy.ai/docs/)
*   [Bandit Documentation](https://bandit.readthedocs.io/)