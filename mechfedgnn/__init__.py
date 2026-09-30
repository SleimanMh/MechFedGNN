"""MechFedGNN federated-learning research framework.

Component boundaries (docs/ARCHITECTURE.md). Score calculation, weight
normalization and parameter aggregation are three separate operations behind
three separate interfaces.

Importing this package registers the built-in strategies.
"""
from mechfedgnn import aggregation, learner, scoring, weighting  # noqa: F401  (registration)
from mechfedgnn.registry import LEARNERS, SCORES, WEIGHTING

__all__ = ["SCORES", "WEIGHTING", "LEARNERS", "scoring", "weighting", "aggregation", "learner"]
