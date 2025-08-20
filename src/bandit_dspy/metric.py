from typing import Dict, Any, Optional, List
from .core import run_bandit

class SecurityMetric:
    """Configurable security metric for evaluating code safety."""
    
    def __init__(self, 
                 severity_weights: Optional[Dict[str, float]] = None,
                 confidence_weights: Optional[Dict[str, float]] = None,
                 penalty_threshold: float = 0.1,
                 normalize_by_length: bool = True):
        self.severity_weights = severity_weights or {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        self.confidence_weights = confidence_weights or {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}
        self.penalty_threshold = penalty_threshold
        self.normalize_by_length = normalize_by_length
    
    def evaluate(self, code: str, issues: Optional[List] = None) -> Dict[str, Any]:
        """Evaluate security score for given code."""
        if issues is None:
            issues = run_bandit(code)
        
        total_penalty = 0
        issue_counts = {
            "HIGH": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "MEDIUM": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "LOW": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
        }
        
        for issue in issues:
            severity_weight = self.severity_weights.get(issue.severity, 0)
            confidence_weight = self.confidence_weights.get(issue.confidence, 0)
            penalty = severity_weight * confidence_weight
            
            # Apply penalty threshold - only count significant issues
            if penalty >= self.penalty_threshold:
                total_penalty += penalty
                issue_counts[issue.severity][issue.confidence] += 1
        
        # Normalize by code length if enabled
        if self.normalize_by_length and code.strip():
            lines_of_code = len([line for line in code.split('\n') if line.strip()])
            total_penalty = total_penalty / max(lines_of_code, 1)
        
        # Calculate score using sigmoid-like function for better distribution
        score = 1 / (1 + total_penalty)
        
        return {
            "score": score,
            "issues": issues,
            "issue_counts": issue_counts,
            "total_penalty": total_penalty,
            "lines_of_code": len([line for line in code.split('\n') if line.strip()]) if code.strip() else 0
        }

# Default instance for backward compatibility
_default_metric = SecurityMetric()

def bandit_metric(example, pred, trace=None):
    """A DSPy metric that uses Bandit to evaluate the security of generated code."""
    generated_code = pred.code
    return _default_metric.evaluate(generated_code)
