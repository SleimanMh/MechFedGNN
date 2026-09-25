"""§13 tests whose code under test is built in Stage 2. Kept visible as skips."""
import pytest


@pytest.mark.skip(reason="Stage 2: headroom.py not built yet")
def test_headroom_detects_ok_and_no_headroom():
    pass


@pytest.mark.skip(reason="Stage 2: loop/run scripts that write CSVs not built yet")
def test_determinism_same_seed_identical_csvs():
    pass
