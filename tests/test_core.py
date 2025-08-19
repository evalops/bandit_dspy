from bandit_dspy.core import run_bandit

def test_run_bandit():
    code = "password = \"hardcoded_password\""
    issues = run_bandit(code)
    assert len(issues) > 0
    assert issues[0].test_id == "B105"

