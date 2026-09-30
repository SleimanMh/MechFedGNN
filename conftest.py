# Repo-root conftest: puts the repo root on sys.path so tests import data/, signatures, scores, kernel.


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: spawns real processes; excluded with -m 'not slow'")
