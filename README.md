# bandit_dspy

A production-ready library that integrates Bandit static analysis with DSPy for security-aware LLM development and optimization.

## 🚀 FEATURES

### Core Security Integration
- **Security-Aware LLM Development:** Integrate static analysis directly into your DSPy optimization loops
- **Advanced Bandit Integration:** Leverage Bandit to identify Python security vulnerabilities in generated code with caching and performance optimizations
- **Configurable Security Metrics:** Customizable security scoring with severity/confidence weighting and penalty thresholds

### Optimization Algorithms
- **Multi-Algorithm Support:** Choose from random search, genetic algorithms, Bayesian optimization, or MIPRO
- **GEPA Optimizer:** State-of-the-art Reflective Prompt Evolution using LLM feedback for iterative security improvements
- **Multi-Objective Optimization:** Balance security, performance, and functionality using Pareto frontiers

### Production Features
- **Performance Optimizations:** Bounded LRU caching for faster repeated analysis
- **Robust Error Handling:** Graceful fallbacks and a comprehensive test suite
- **Type Hints:** Typed core APIs and mypy-friendly usage
- **Configurable Pipeline:** Flexible train/validation splits and optimization parameters

## 🔧 PREREQUISITES

- Python 3.8+
- A configured DSPy Language Model (e.g., `dspy.OllamaLocal`, `dspy.OpenAI`)

## 📦 INSTALLATION

### From Source
```bash
git clone https://github.com/evalops/bandit_dspy.git
cd bandit_dspy
pip install -e .
```

### Development Installation
```bash
pip install -e .[dev]  # Includes pytest, mypy, ruff, etc.
```

## 🚦 QUICK START

### Basic Security Optimization

```python
import dspy
from bandit_dspy import BanditTeleprompter, create_bandit_metric

class CodeGen(dspy.Signature):
    """Generate secure Python code for the given task."""
    description = dspy.InputField()
    code = dspy.OutputField()

class SecureCodeGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(CodeGen)
    def forward(self, description):
        return self.predictor(description=description)

dspy.settings.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo"))

trainset = [
    dspy.Example(description="hash a password securely", code="import bcrypt
def hash_password(password): return bcrypt.hashpw(password.encode(), bcrypt.gensalt())").with_inputs('description'),
    dspy.Example(description="validate user input", code="import re
def validate_email(email): return bool(re.match(r'^[^@]+@[^@]+\.[^@]+$', email))").with_inputs('description'),
]

student = SecureCodeGenerator()
teleprompter = BanditTeleprompter(metric=create_bandit_metric(), optimization_method="genetic", k=3, num_candidates=10)
compiled = teleprompter.compile(student, trainset=trainset)
res = compiled(description="create a secure file upload function")
print(res.code)
```

### GEPA Optimization

```python
from bandit_dspy import GEPATeleprompter, create_bandit_metric

gepa = GEPATeleprompter(metric=create_bandit_metric(), max_iterations=4, population_size=3)
compiled = gepa.compile(student, trainset=trainset)
```

### Performance Monitoring

```python
import time
from bandit_dspy import BanditRunner

runner = BanditRunner(cache_maxsize=2000)

# Measure cache performance
code = "import os\npassword = 'hardcoded'"

# Cold cache
start = time.time()
result1 = runner.analyze_code(code)
cold_time = time.time() - start

# Warm cache  
start = time.time()
result2 = runner.analyze_code(code)
warm_time = time.time() - start

print(f"Speedup: {cold_time/warm_time:.1f}x")
```

## 🧪 TESTING

### Run All Tests
```bash
export DSPY_CACHEDIR="$(pwd)/.dspy_cache"  # ensure writable cache\npytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=bandit_dspy --cov-report=html
```

### Run Performance Tests
```bash
pytest tests/test_performance.py -v
```

### Run GEPA Tests (requires dependencies)
```bash
pytest tests/test_gepa_optimizer.py -v
```

## 🐛 TROUBLESHOOTING

### Common Issues

**Import Errors:**
```bash
# Install all dependencies
pip install -e .[dev]

# Or install individual packages
pip install dspy-ai bandit
```

**sqlite3 OperationalError: attempt to write a readonly database**
```bash
# Set DSPy cache directory to a writable location
export DSPY_CACHEDIR="$(pwd)/.dspy_cache"
```

**Memory Issues with Large Codebases:**
```python
# Reduce cache size
runner = BanditRunner(cache_maxsize=500)

# Use streaming evaluation
for code_chunk in large_codebase:
    result = runner.analyze_code(code_chunk)
```

**Slow Optimization:**
```python
# Reduce candidates for faster iteration
teleprompter = BanditTeleprompter(
    num_candidates=5,          # Reduce from default 10
    optimization_method="random"  # Fastest method
)

# Use smaller population for genetic algorithm
teleprompter = BanditTeleprompter(
    optimization_method="genetic",
    genetic_config={"population_size": 10, "generations": 5}
)
```

## 🔬 DEVELOPMENT

### Project Structure

```
bandit_dspy/
├── src/bandit_dspy/
│   ├── __init__.py           # Package exports and compatibility shims
│   ├── core.py              # BanditRunner with caching and AST analysis  
│   ├── metric.py            # SecurityMetric and create_bandit_metric factory
│   ├── teleprompter.py      # BanditTeleprompter with multiple algorithms
│   └── gepa_optimizer.py    # GEPA implementation for reflective optimization
├── tests/                   # Comprehensive test suite
│   ├── conftest.py         # Shared fixtures and test configuration
│   ├── test_*.py           # Individual test modules
│   └── test_gepa_*.py      # GEPA-specific tests
├── example.py              # Quick start example
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install dev dependencies: `pip install -e .[dev]`
4. Make changes and add tests
5. Run the test suite: `export DSPY_CACHEDIR="$(pwd)/.dspy_cache"  # ensure writable cache\npytest tests/ -v`
6. Run type checking: `mypy src/bandit_dspy`
7. Run linting: `ruff check src/ tests/`
8. Format code: `black src/ tests/`
9. Commit with conventional messages: `git commit -m "feat: add amazing feature"`
10. Push and create a pull request

### Release Process

```bash
# Update version in pyproject.toml
# Create release notes
git tag v0.1.0
git push origin v0.1.0

# Build and publish
python -m build
twine upload dist/*
```

## 📄 LICENSE

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🔗 DEPENDENCIES

### Required
- **dspy-ai**: The DSPy framework for language model programming
- **bandit**: Python security linter for static analysis

### Optional Development
- **pytest**: Testing framework with coverage and fixtures
- **mypy**: Static type checking
- **ruff**: Fast Python linter and formatter  
- **black**: Code formatting
- **pre-commit**: Git hooks for code quality

## 📚 RESOURCES

- [DSPy Documentation](https://dspy.ai/docs/) - DSPy framework guide
- [Bandit Documentation](https://bandit.readthedocs.io/) - Security scanning tool
- [GEPA Paper](https://arxiv.org/abs/2404.00757) - Reflective Prompt Evolution research
- [OWASP Guidelines](https://owasp.org/www-project-top-ten/) - Security best practices

## 🏆 ACKNOWLEDGMENTS

- **DSPy Team** for the innovative language model programming framework
- **Bandit Contributors** for the robust security analysis tool
- **GEPA Researchers** for the reflective prompt evolution methodology

---

**Built with ❤️ for secure AI development**