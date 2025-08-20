import os
import tempfile

# Ensure DSPy disk cache is writable in constrained environments
if "DSPY_CACHEDIR" not in os.environ:
    os.environ["DSPY_CACHEDIR"] = os.path.join(tempfile.gettempdir(), ".dspy_cache")

from .core import BanditRunner, run_bandit
from .gepa_optimizer import (
    GEPATeleprompter,
    ParetoSelector,
    SecurityGEPAOptimizer,
    SecurityReflector,
)
from .metric import SecurityMetric, bandit_metric, create_bandit_metric
from .teleprompter import (
    BanditTeleprompter,
    BayesianOptimizer,
    GeneticOptimizer,
)

# Compatibility shim for varying DSPy Example APIs across versions
try:
    import dspy  # type: ignore
    Example = getattr(dspy, "Example", None)
    if Example is not None and not hasattr(Example, "with_inputs"):
        def _bd_with_inputs(self, *keys):
            try:
                setattr(self, "_bd_input_keys", tuple(keys))
            except Exception:
                pass
            return self

        def _bd_inputs(self):
            keys = getattr(self, "_bd_input_keys", ())
            result = {}
            for k in keys:
                if hasattr(self, k):
                    result[k] = getattr(self, k)
                else:
                    try:
                        store = getattr(self, "_store", {})
                        if k in store:
                            result[k] = store[k]
                    except Exception:
                        pass
            return result

        setattr(Example, "with_inputs", _bd_with_inputs)
        if not hasattr(Example, "inputs"):
            setattr(Example, "inputs", _bd_inputs)
except Exception:
    pass

__all__ = [
    "run_bandit",
    "BanditRunner",
    "bandit_metric",
    "create_bandit_metric",
    "SecurityMetric",
    "BanditTeleprompter",
    "GeneticOptimizer",
    "BayesianOptimizer",
    "SecurityGEPAOptimizer",
    "GEPATeleprompter",
    "SecurityReflector",
    "ParetoSelector",
]
