"""Signature provider: marginal rates, H, C, J and permitted population summaries.

Delegates to the research modules so values are unchanged. Conventions:
M = 1 observed; H joint absence; C binary association (phi, 0 for constant
features with rates kept separately in r); J joint observation.

``exclude`` drops columns from the POPULATION characteristics only - it never
changes the mask statistics. E5's robustness configuration uses it to remove
the partition characteristic.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from signatures import histograms, signature


@dataclass
class MaskSignatureProvider:
    name: str = "mask-signatures"

    def summarize(self, client: Mapping[str, Any], roles: Mapping[str, Any],
                  exclude: Sequence[int] = ()) -> dict:
        tr = client["train"]
        cols = [i for i in roles["always_observed"] if i not in set(exclude)]
        names = [roles["features"][i] for i in cols]
        return {"sig": signature(client["M"][tr]),
                "hist": histograms(client["X"][tr][:, cols], names, roles["bin_edges"]),
                "n_train": len(tr)}

    def aggregate_only(self, summary: Mapping[str, Any]) -> dict:
        """What may leave a client for the server: aggregate mask statistics and
        the declared sample count. NEVER per-row masks, features or labels."""
        return {"r": summary["sig"]["r"], "H": summary["sig"]["H"],
                "J": summary["sig"]["J"], "C": summary["sig"]["C"],
                "hist": summary["hist"], "n_train": summary["n_train"]}
