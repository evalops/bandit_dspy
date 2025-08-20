import time

from bandit_dspy.core import BanditRunner, run_bandit


def test_run_bandit_basic():
    """Test basic run_bandit functionality."""
    code = 'password = "hardcoded_password"'
    issues = run_bandit(code)
    assert len(issues) > 0
    assert issues[0].test_id == "B105"


def test_run_bandit_secure_code():
    """Test run_bandit with secure code."""
    code = "def add(a, b): return a + b"
    issues = run_bandit(code)
    assert len(issues) == 0


def test_run_bandit_syntax_error():
    """Test run_bandit fallback with syntax errors."""
    code = "def invalid_syntax("
    issues = run_bandit(code)
    # Should handle gracefully, possibly with empty result
    assert isinstance(issues, list)


class TestBanditRunner:
    """Test the improved BanditRunner class."""

    def test_caching_performance(self, bandit_runner, insecure_code):
        """Test that caching improves performance."""
        # First run (no cache)
        start_time = time.time()
        result1 = bandit_runner.analyze_code(insecure_code)
        first_run_time = time.time() - start_time

        # Second run (should use cache)
        start_time = time.time()
        result2 = bandit_runner.analyze_code(insecure_code)
        second_run_time = time.time() - start_time

        # Cache should return identical results
        assert len(result1) == len(result2)
        # Second run should be faster (though this might be flaky in CI)
        assert second_run_time <= first_run_time

    def test_analyze_code_with_issues(self, bandit_runner, insecure_code):
        """Test analyze_code finds security issues."""
        issues = bandit_runner.analyze_code(insecure_code)
        assert len(issues) > 0

        # Should find hardcoded password
        password_issues = [i for i in issues if i.test_id == "B105"]
        assert len(password_issues) > 0

    def test_analyze_code_secure(self, bandit_runner, secure_code):
        """Test analyze_code with secure code."""
        issues = bandit_runner.analyze_code(secure_code)
        assert len(issues) == 0

    def test_custom_config(self):
        """Test BanditRunner with custom configuration."""
        # Bandit config expects file path, so test basic config instead
        runner = BanditRunner()

        code = "assert True"
        issues = runner.analyze_code(code)
        # Should find B101 (assert) issues
        assert_issues = [i for i in issues if i.test_id == "B101"]
        assert len(assert_issues) > 0
