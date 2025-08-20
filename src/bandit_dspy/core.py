import ast
import tempfile
import hashlib
import logging
from typing import List, Dict, Any, Optional
from bandit.core import config, manager, test_set
from bandit.core.node_visitor import BanditNodeVisitor
from bandit.core.context import Context
from functools import lru_cache

logging.getLogger('bandit').setLevel(logging.ERROR)

class BanditRunner:
    """Improved Bandit integration with caching and direct AST analysis."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.config = config.BanditConfig(config_dict or {})
        self.test_set = test_set.BanditTestSet(self.config)
        self._cache = {}
    
    @lru_cache(maxsize=1000)
    def _get_code_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()
    
    def analyze_code(self, code: str) -> List[Any]:
        """Run bandit directly on code string using AST analysis."""
        code_hash = self._get_code_hash(code)
        if code_hash in self._cache:
            return self._cache[code_hash]
        
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
                metrics=None
            )
            
            visitor.visit(tree)
            issues = visitor.tester.results
            
            self._cache[code_hash] = issues
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
            return b_mgr.get_issue_list()

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