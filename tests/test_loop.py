from loop import for_dataset


def test_for_dataset_deep_merges_without_mutating():
    cfg = {"headroom": {"none_max": 0.02, "early_stopping": {"patience": 50, "max_steps": 5000}},
           "dataset_overrides": {"protein": {"headroom": {"early_stopping": {"max_steps": 20000}}}}}
    p = for_dataset(cfg, "protein")
    assert p["headroom"]["early_stopping"] == {"patience": 50, "max_steps": 20000}
    assert p["headroom"]["none_max"] == 0.02
    assert for_dataset(cfg, "wine")["headroom"]["early_stopping"]["max_steps"] == 5000
    assert cfg["headroom"]["early_stopping"]["max_steps"] == 5000        # original untouched
