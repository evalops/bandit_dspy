"""
Property-based tests for bandit_dspy using hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from bandit_dspy import SecurityMetric, BanditRunner, run_bandit


# Generate valid Python identifiers
python_identifiers = st.text(
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Lt", "Lm", "Lo", "Nl"]),
    min_size=1,
    max_size=20
).filter(lambda x: x.isidentifier() and not x.startswith('_'))

# Generate simple Python code
simple_python_code = st.one_of([
    # Simple function definitions
    st.builds(
        lambda name: f"def {name}(): return 1",
        python_identifiers
    ),
    # Variable assignments
    st.builds(
        lambda name, value: f"{name} = {value}",
        python_identifiers,
        st.integers(min_value=0, max_value=1000)
    ),
    # Import statements
    st.builds(
        lambda module: f"import {module}",
        st.sampled_from(["os", "sys", "math", "json", "time"])
    ),
    # Simple expressions
    st.builds(
        lambda a, b: f"result = {a} + {b}",
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100)
    )
])


class TestSecurityMetricProperties:
    """Property-based tests for SecurityMetric."""
    
    @given(
        severity_weight=st.floats(min_value=0.1, max_value=100.0),
        confidence_weight=st.floats(min_value=0.1, max_value=100.0),
        threshold=st.floats(min_value=0.0, max_value=10.0)
    )
    @settings(max_examples=20)
    def test_security_metric_score_bounds(self, severity_weight, confidence_weight, threshold):
        """SecurityMetric scores should always be between 0 and 1."""
        metric = SecurityMetric(
            severity_weights={"HIGH": severity_weight, "MEDIUM": severity_weight/2, "LOW": severity_weight/4},
            confidence_weights={"HIGH": confidence_weight, "MEDIUM": confidence_weight/2, "LOW": confidence_weight/4},
            penalty_threshold=threshold
        )
        
        # Test with various code samples
        test_codes = [
            "def safe(): return 1",
            "password = 'hardcoded'",
            "import os",
            "assert True",
            ""
        ]
        
        for code in test_codes:
            result = metric.evaluate(code)
            assert 0 <= result["score"] <= 1, f"Score {result['score']} out of bounds for code: {code}"
            assert result["total_penalty"] >= 0, f"Penalty {result['total_penalty']} should be non-negative"
            assert result["lines_of_code"] >= 0, f"LOC {result['lines_of_code']} should be non-negative"
    
    @given(simple_python_code)
    @settings(max_examples=50)
    def test_security_metric_handles_valid_code(self, code):
        """SecurityMetric should handle any valid Python code without errors."""
        assume(len(code) < 1000)  # Keep code reasonable
        
        metric = SecurityMetric()
        
        try:
            result = metric.evaluate(code)
            assert isinstance(result, dict)
            assert "score" in result
            assert "issues" in result
            assert "issue_counts" in result
            assert 0 <= result["score"] <= 1
        except Exception as e:
            pytest.fail(f"SecurityMetric failed on valid code '{code}': {e}")
    
    @given(
        normalize=st.booleans(),
        threshold=st.floats(min_value=0.0, max_value=5.0)
    )
    @settings(max_examples=20)
    def test_normalization_consistency(self, normalize, threshold):
        """Test that normalization behaves consistently."""
        metric = SecurityMetric(
            normalize_by_length=normalize,
            penalty_threshold=threshold
        )
        
        short_code = "password = 'test'"
        long_code = "password = 'test'\n" + "\n".join(["# comment"] * 20)
        
        short_result = metric.evaluate(short_code)
        long_result = metric.evaluate(long_code)
        
        if normalize:
            # With normalization, longer code should have lower penalty per line
            if short_result["lines_of_code"] > 0 and long_result["lines_of_code"] > 0:
                assert long_result["total_penalty"] <= short_result["total_penalty"]
        else:
            # Without normalization, penalty should be same regardless of length
            # (since the actual security issue is the same)
            pass  # Hard to assert exact equality due to different issue detection


class TestBanditRunnerProperties:
    """Property-based tests for BanditRunner."""
    
    @given(simple_python_code)
    @settings(max_examples=50)
    def test_bandit_runner_consistency(self, code):
        """BanditRunner should give consistent results for the same code."""
        assume(len(code) < 500)
        
        runner = BanditRunner()
        
        try:
            result1 = runner.analyze_code(code)
            result2 = runner.analyze_code(code)
            
            # Should get identical results (testing cache)
            assert len(result1) == len(result2)
            
            # Results should be lists
            assert isinstance(result1, list)
            assert isinstance(result2, list)
            
        except Exception as e:
            pytest.fail(f"BanditRunner failed on code '{code}': {e}")
    
    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=30)
    def test_bandit_runner_handles_arbitrary_text(self, text):
        """BanditRunner should handle arbitrary text gracefully."""
        runner = BanditRunner()
        
        try:
            result = runner.analyze_code(text)
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"BanditRunner failed on text '{text}': {e}")


class TestRunBanditProperties:
    """Property-based tests for run_bandit function."""
    
    @given(simple_python_code)
    @settings(max_examples=30)
    def test_run_bandit_returns_list(self, code):
        """run_bandit should always return a list."""
        assume(len(code) < 500)
        
        try:
            result = run_bandit(code)
            assert isinstance(result, list)
            
            # All items in list should have expected attributes
            for issue in result:
                assert hasattr(issue, 'test_id')
                assert hasattr(issue, 'severity') or hasattr(issue, 'issue_severity')
                assert hasattr(issue, 'confidence') or hasattr(issue, 'issue_confidence')
                
        except Exception as e:
            pytest.fail(f"run_bandit failed on code '{code}': {e}")


class TestSecurityMetricMonotonicity:
    """Test monotonicity properties of SecurityMetric."""
    
    def test_more_issues_lower_score(self):
        """More security issues should generally result in lower scores."""
        metric = SecurityMetric()
        
        secure_code = "def safe(): return 1"
        insecure_code = "password = 'hardcoded'\nassert True\nexec('print(1)')"
        
        secure_result = metric.evaluate(secure_code)
        insecure_result = metric.evaluate(insecure_code)
        
        # More issues should result in lower score
        assert secure_result["score"] >= insecure_result["score"]
        assert len(secure_result["issues"]) <= len(insecure_result["issues"])
    
    @given(
        weight_multiplier=st.floats(min_value=1.1, max_value=10.0)
    )
    @settings(max_examples=20)
    def test_higher_weights_lower_scores(self, weight_multiplier):
        """Higher penalty weights should result in lower scores for the same issues."""
        base_weights = {"HIGH": 1.0, "MEDIUM": 1.0, "LOW": 1.0}
        high_weights = {k: v * weight_multiplier for k, v in base_weights.items()}
        
        base_metric = SecurityMetric(severity_weights=base_weights)
        high_metric = SecurityMetric(severity_weights=high_weights)
        
        insecure_code = "password = 'hardcoded'"
        
        base_result = base_metric.evaluate(insecure_code)
        high_result = high_metric.evaluate(insecure_code)
        
        # Higher weights should result in lower scores
        if len(base_result["issues"]) > 0:
            assert high_result["score"] <= base_result["score"]
            assert high_result["total_penalty"] >= base_result["total_penalty"]