from .core import run_bandit

def bandit_metric(example, pred, trace=None):
    """A DSPy metric that uses Bandit to evaluate the security of generated code."""
    generated_code = pred.code
    issues = run_bandit(generated_code)
    
    total_penalty = 0
    issue_counts = {
        "HIGH": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "MEDIUM": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        "LOW": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
    }

    for issue in issues:
        severity_points = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(issue.severity, 0)
        confidence_points = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(issue.confidence, 0)
        total_penalty += severity_points * confidence_points
        issue_counts[issue.severity][issue.confidence] += 1
    
    score = 1 / (1 + total_penalty)

    return {
        "score": score,
        "issues": issues,
        "issue_counts": issue_counts,
    }
