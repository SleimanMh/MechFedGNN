"""Separate randomness for partitioning, mask generation and model training.

The research code derived everything from ONE root seed via
``loop._seed(seed, purpose, k)``. Splitting the streams is required for
reproducibility work, but changing the default derivation would silently change
every historical result - so:

  * ``SeedBundle.legacy(seed)``  reproduces the historical derivation EXACTLY
    (all three streams equal the root seed). This is the default, so existing
    experiments are bit-identical.
  * ``SeedBundle(partition=..., mask=..., train=...)`` lets a new experiment
    vary one stream while holding the others fixed - e.g. re-draw masks without
    moving the partition.

Which mode was used is recorded in provenance, so a run is never ambiguous.
"""
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedBundle:
    partition: int
    mask: int
    train: int
    mode: str = "independent"

    @classmethod
    def legacy(cls, seed: int) -> "SeedBundle":
        """Historical behaviour: one root seed feeds all three streams."""
        return cls(partition=int(seed), mask=int(seed), train=int(seed), mode="legacy")

    @classmethod
    def from_config(cls, cfg: dict, seed: int) -> "SeedBundle":
        s = (cfg or {}).get("seed_streams")
        if not s:
            return cls.legacy(seed)
        return cls(partition=int(s.get("partition", seed)), mask=int(s.get("mask", seed)),
                   train=int(s.get("train", seed)), mode="independent")

    def derive(self, stream: str, *parts: Any) -> int:
        """A deterministic child seed from one named stream."""
        root = {"partition": self.partition, "mask": self.mask, "train": self.train}[stream]
        return int(np.random.default_rng([root, *[int(p) for p in parts]]).integers(2**31 - 1))

    def as_dict(self) -> dict:
        return asdict(self)
