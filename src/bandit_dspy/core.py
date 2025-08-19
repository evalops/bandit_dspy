import tempfile
from bandit.core import config, manager

def run_bandit(code: str) -> list:
    """Runs bandit on a string of python code and returns the results."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        b_config = config.BanditConfig()
        b_mgr = manager.BanditManager(b_config, "file")
        b_mgr.discover_files([f.name])
        b_mgr.run_tests()
        return b_mgr.get_issue_list()