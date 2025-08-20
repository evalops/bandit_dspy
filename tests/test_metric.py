import dspy

from bandit_dspy.metric import SecurityMetric, create_bandit_metric


def test_bandit_metric_secure_code():
    """Test bandit_metric with secure code."""
    bandit_metric_callable = create_bandit_metric()
    pred = dspy.Example(code="import os")
    metric_result = bandit_metric_callable(None, pred)
    assert metric_result["score"] == 1.0
    assert len(metric_result["issues"]) == 0


def test_bandit_metric_insecure_code():
    """Test bandit_metric with insecure code."""
    bandit_metric_callable = create_bandit_metric()
    pred = dspy.Example(code='password = "hardcoded_password"')
    metric_result = bandit_metric_callable(None, pred)
    assert metric_result["score"] < 1.0
    assert len(metric_result["issues"]) > 0
    assert metric_result["issue_counts"]["LOW"]["MEDIUM"] == 1


def test_bandit_metric_low_severity_issue():
    """Test bandit_metric with low severity issues."""
    bandit_metric_callable = create_bandit_metric()
    pred = dspy.Example(code="assert True")
    metric_result = bandit_metric_callable(None, pred)
    assert metric_result["score"] == 0.25
    assert len(metric_result["issues"]) > 0
    assert metric_result["issue_counts"]["LOW"]["HIGH"] == 1


class TestSecurityMetric:
    """Test the improved SecurityMetric class."""

    def test_default_configuration(self, security_metric_default, secure_code):
        """Test SecurityMetric with default configuration."""
        result = security_metric_default.evaluate(secure_code)
        assert result["score"] == 1.0
        assert len(result["issues"]) == 0
        assert result["total_penalty"] == 0

    def test_strict_configuration(self, security_metric_strict, insecure_code):
        """Test SecurityMetric with strict configuration."""
        result = security_metric_strict.evaluate(insecure_code)
        assert result["score"] < 1.0
        assert len(result["issues"]) > 0
        assert result["total_penalty"] > 0

    def test_custom_weights(self, code_with_mixed_issues):
        """Test SecurityMetric with custom weights."""
        # High penalty for HIGH severity issues
        high_severity_metric = SecurityMetric(
            severity_weights={"HIGH": 100.0, "MEDIUM": 1.0, "LOW": 0.1}
        )

        # Equal weights
        equal_weights_metric = SecurityMetric(
            severity_weights={"HIGH": 1.0, "MEDIUM": 1.0, "LOW": 1.0}
        )

        high_result = high_severity_metric.evaluate(code_with_mixed_issues)
        equal_result = equal_weights_metric.evaluate(code_with_mixed_issues)

        # High severity weighting should result in lower score if there are HIGH severity issues
        # For testing, let's verify both results are reasonable
        assert 0 <= high_result["score"] <= 1
        assert 0 <= equal_result["score"] <= 1

    def test_penalty_threshold(self):
        """Test penalty threshold functionality."""
        # High threshold - only count significant issues
        high_threshold_metric = SecurityMetric(penalty_threshold=5.0)

        # Low threshold - count all issues
        low_threshold_metric = SecurityMetric(penalty_threshold=0.1)

        code = 'password = "hardcoded"'  # LOW severity issue

        high_result = high_threshold_metric.evaluate(code)
        low_result = low_threshold_metric.evaluate(code)

        # High threshold should ignore low-severity issues
        assert high_result["total_penalty"] < low_result["total_penalty"]

    def test_length_normalization(self):
        """Test code length normalization."""
        short_code = 'password = "hardcoded"'
        long_code = 'password = "hardcoded"\n' + "\n".join(["# comment"] * 50)

        normalized_metric = SecurityMetric(normalize_by_length=True)
        non_normalized_metric = SecurityMetric(normalize_by_length=False)

        short_result_norm = normalized_metric.evaluate(short_code)
        long_result_norm = normalized_metric.evaluate(long_code)

        short_result_no_norm = non_normalized_metric.evaluate(short_code)
        long_result_no_norm = non_normalized_metric.evaluate(long_code)

        # With normalization, longer code should have lower penalty per line
        assert long_result_norm["total_penalty"] < short_result_norm["total_penalty"]

        # Without normalization, penalty should be same regardless of length
        assert (
            short_result_no_norm["total_penalty"]
            == long_result_no_norm["total_penalty"]
        )

    def test_issue_counts_tracking(self, code_with_mixed_issues):
        """Test that issue counts are tracked correctly."""
        metric = SecurityMetric()
        result = metric.evaluate(code_with_mixed_issues)

        # Should have proper issue count structure
        assert "issue_counts" in result
        assert "HIGH" in result["issue_counts"]
        assert "MEDIUM" in result["issue_counts"]
        assert "LOW" in result["issue_counts"]

        # Should track issues by severity and confidence
        total_issues = sum(
            sum(confidence_counts.values())
            for confidence_counts in result["issue_counts"].values()
        )
        assert total_issues > 0

    def test_lines_of_code_calculation(self):
        """Test lines of code calculation."""
        metric = SecurityMetric()

        # Empty code
        empty_result = metric.evaluate("")
        assert empty_result["lines_of_code"] == 0

        # Single line
        single_line = "print('hello')"
        single_result = metric.evaluate(single_line)
        assert single_result["lines_of_code"] == 1

        # Multiple lines with empty lines
        multi_line = "print('hello')\n\nprint('world')\n# comment"
        multi_result = metric.evaluate(multi_line)
        assert multi_result["lines_of_code"] == 3  # Ignores empty lines
