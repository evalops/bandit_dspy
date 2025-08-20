import ast
import hashlib
import logging
import tempfile
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Optional, cast

from bandit.core import config, manager, test_set
from bandit.core.node_visitor import BanditNodeVisitor

logging.getLogger("bandit").setLevel(logging.ERROR)


class BanditRunner:
    """Improved Bandit integration with caching and direct AST analysis.

    Maintains a bounded LRU cache of analysis results to prevent unbounded
    memory growth while speeding up repeat analyses on identical code.
    """

    def __init__(
        self, config_dict: Optional[Dict[str, Any]] = None, *, cache_maxsize: int = 1000
    ):
        self.config = config.BanditConfig(config_dict or {})
        self.test_set = test_set.BanditTestSet(self.config)
        self._cache: Dict[str, List[Any]] = OrderedDict()
        self._cache_maxsize = cache_maxsize

    @lru_cache(maxsize=1000)
    def _get_code_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()

    def analyze_code(self, code: str) -> List[Any]:
        """Run bandit directly on code string using AST analysis."""
        code_hash = self._get_code_hash(code)
        if code_hash in self._cache:
            # Move to end to mark as recently used
            issues = self._cache.pop(code_hash)
            self._cache[code_hash] = issues
            return issues

        try:
            tree = ast.parse(code)

            # Create metaast - this is required for BanditNodeVisitor
            from bandit.core.meta_ast import BanditMetaAst

            metaast = BanditMetaAst()

            visitor = BanditNodeVisitor(
                fname="<generated_code>",
                fdata=code,
                metaast=metaast,
                testset=self.test_set,
                debug=False,
                nosec_lines=set(),
                metrics=None,
            )

            visitor.visit(tree)
            issues = visitor.tester.results

            # Insert and enforce LRU bound
            self._cache[code_hash] = issues
            if len(self._cache) > self._cache_maxsize:
                cast(OrderedDict, self._cache).popitem(last=False)
            return issues

        except (SyntaxError, Exception):
            # Fall back to file-based analysis for any issues
            return self._fallback_analysis(code)

    def _fallback_analysis(self, code: str) -> List[Any]:
        """Fallback to file-based analysis for problematic code."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(code)
            f.flush()

            b_mgr = manager.BanditManager(self.config, "file")
            b_mgr.discover_files([f.name])
            b_mgr.run_tests()
            issues = b_mgr.get_issue_list()
            # Insert and enforce LRU bound
            code_hash = self._get_code_hash(code)
            self._cache[code_hash] = issues
            if len(self._cache) > self._cache_maxsize:
                cast(OrderedDict, self._cache).popitem(last=False)
            return issues


_default_runner = BanditRunner()


def run_bandit(code: str, config_dict: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Run bandit analysis on Python code string.

    Args:
        code: Python code string to analyze
        config_dict: Optional bandit configuration

    Returns:
        List of bandit issues found
    """
    if config_dict:
        runner = BanditRunner(config_dict)
        return runner.analyze_code(code)
    else:
        return _default_runner.analyze_code(code)
