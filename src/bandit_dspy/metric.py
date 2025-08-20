from typing import Any, Callable, Dict, List, Optional

from .core import run_bandit


class SecurityMetric:
    """Configurable security metric for evaluating code safety."""

    def __init__(
        self,
        severity_weights: Optional[Dict[str, float]] = None,
        confidence_weights: Optional[Dict[str, float]] = None,
        penalty_threshold: float = 0.1,
        normalize_by_length: bool = True,
        bandit_config: Optional[Dict[str, Any]] = None,
    ):
        self.severity_weights = severity_weights or {
            "HIGH": 3.0,
            "MEDIUM": 2.0,
            "LOW": 1.0,
        }
        self.confidence_weights = confidence_weights or {
            "HIGH": 3.0,
            "MEDIUM": 2.0,
            "LOW": 1.0,
        }
        self.penalty_threshold = penalty_threshold
        self.normalize_by_length = normalize_by_length
        self.bandit_config = bandit_config

    def evaluate(self, code: str, issues: Optional[List] = None) -> Dict[str, Any]:
        """Evaluate security score for given code."""
        if issues is None:
            issues = run_bandit(code, config_dict=self.bandit_config)

        total_penalty = 0.0
        issue_counts = {
            "HIGH": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "MEDIUM": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "LOW": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        }

        for issue in issues:
            # Bandit issues may expose attributes as either `.severity`/`.confidence`
            # or `.issue_severity`/`.issue_confidence` depending on path.
            severity_str = str(getattr(issue, "severity", getattr(issue, "issue_severity", "")))
            confidence_str = str(getattr(issue, "confidence", getattr(issue, "issue_confidence", "")))

            severity_weight = self.severity_weights.get(severity_str, 0)
            confidence_weight = self.confidence_weights.get(confidence_str, 0)
            penalty = severity_weight * confidence_weight

            # Apply penalty threshold - only count significant issues
            if penalty >= self.penalty_threshold:
                total_penalty += penalty
                if severity_str in issue_counts and confidence_str in issue_counts[severity_str]:
                    issue_counts[severity_str][confidence_str] += 1

        # Normalize by code length if enabled
        if self.normalize_by_length and code.strip():
            lines_of_code = len([line for line in code.split("\n") if line.strip()])
            total_penalty = total_penalty / max(lines_of_code, 1)

        # Calculate score using sigmoid-like function for better distribution
        score: float = 1 / (1 + total_penalty)

        return {
            "score": score,
            "issues": issues,
            "issue_counts": issue_counts,
            "total_penalty": total_penalty,
            "lines_of_code": (
                len([line for line in code.split("\n") if line.strip()])
                if code.strip()
                else 0
            ),
        }


def create_bandit_metric(
    severity_weights: Optional[Dict[str, float]] = None,
    confidence_weights: Optional[Dict[str, float]] = None,
    penalty_threshold: float = 0.1,
    normalize_by_length: bool = True,
    bandit_config: Optional[Dict[str, Any]] = None,
) -> Callable[[Any, Any, Any], Dict[str, Any]]:
    """
    Factory function to create a bandit metric callable for DSPy.
    """
    metric_instance = SecurityMetric(
        severity_weights=severity_weights,
        confidence_weights=confidence_weights,
        penalty_threshold=penalty_threshold,
        normalize_by_length=normalize_by_length,
        bandit_config=bandit_config,
    )

    def _bandit_metric_callable(example, pred, trace=None):
        generated_code = pred.code
        return metric_instance.evaluate(generated_code)

    return _bandit_metric_callable


# Default instance for backward compatibility with existing imports
_default_metric = SecurityMetric()


def bandit_metric(example, pred, trace=None):
    """A DSPy metric that uses Bandit to evaluate the security of generated code.

    Backwards-compatible default metric used across tests and examples.
    """
    generated_code = pred.code
    return _default_metric.evaluate(generated_code)
