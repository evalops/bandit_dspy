from bandit_dspy.metric import bandit_metric
import dspy

def test_bandit_metric_secure_code():
    pred = dspy.Example(code="import os")
    metric_result = bandit_metric(None, pred)
    assert metric_result["score"] == 1.0
    assert len(metric_result["issues"]) == 0

def test_bandit_metric_insecure_code():
    pred = dspy.Example(code='password = "hardcoded_password"')
    metric_result = bandit_metric(None, pred)
    assert metric_result["score"] < 1.0
    assert len(metric_result["issues"]) > 0
    assert metric_result["issue_counts"]["LOW"]["MEDIUM"] == 1

def test_bandit_metric_low_severity_issue():
    pred = dspy.Example(code='assert True')
    metric_result = bandit_metric(None, pred)
    assert metric_result["score"] == 0.25
    assert len(metric_result["issues"]) > 0
    assert metric_result["issue_counts"]["LOW"]["HIGH"] == 1