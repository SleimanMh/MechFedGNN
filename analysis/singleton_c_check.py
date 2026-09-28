"""EXPLORATORY check of the singleton explanation for s's failed group recovery.

Rebuilds E1's clients and masks for all 10 seeds (no training; deterministic)
and reports, per latent group, the mean phi (C) entries among maskable
features: within each panel, and across panels - in particular every entry
involving a single-feature panel. Also the mean pairwise s and rate
similarity within vs between groups.

Run:  python -m analysis.singleton_c_check
"""
import numpy as np
import yaml

from data.clients import build_clients
from data.design import load_dataset
from data.inject import load_or_build_roles
from loop import for_dataset
from scores import missingness_similarity, rate_similarity
from signatures import signature


def main():
    cfg = yaml.safe_load(open("configs/defaults.yaml"))
    for name in cfg["datasets"]:
        df, _ = load_dataset(name, verbose=False)
        roles = load_or_build_roles(name, df)
        cfg_d = for_dataset(cfg, name)
        feats, panels = roles["features"], roles["panels"]
        pairs = {}
        for a in range(len(panels)):
            for b in range(a, len(panels)):
                for f in panels[a]:
                    for g in panels[b]:
                        if f < g or (a != b and f != g):
                            key = f"P{a}xP{b}" if a != b else f"within P{a}"
                            pairs.setdefault(key, set()).add((min(f, g), max(f, g)))
        acc = {k: {0: [], 1: []} for k in pairs}
        sim = {"s": {"within": [], "between": []}, "rate": {"within": [], "between": []}}
        for seed in cfg_d["seeds"]:
            clients, groups = build_clients(df, roles, cfg_d, seed)
            sigs = [signature(c["M"][c["train"]]) for c in clients]
            for c, sg in zip(clients, sigs):
                for k, ps in pairs.items():
                    acc[k][c["group"]].append(np.mean([sg["C"][f, g] for f, g in ps]))
            for i in range(len(clients)):
                for j in range(i + 1, len(clients)):
                    kind = "within" if groups[i] == groups[j] else "between"
                    sim["s"][kind].append(missingness_similarity(sigs[i]["C"], sigs[j]["C"])[0])
                    sim["rate"][kind].append(rate_similarity(sigs[i]["r"], sigs[j]["r"])[0])
        print(f"\n######## {name}  panels: " + ", ".join(f"P{k}={[feats[i] for i in P]}" for k, P in enumerate(panels)))
        print("  group profiles: A rarely orders P0, B rarely orders the last panel (§4.1)")
        for k in sorted(acc):
            a, b = np.mean(acc[k][0]), np.mean(acc[k][1])
            sd = np.std(acc[k][0] + acc[k][1])
            print(f"  {k:12s} mean C  group A {a:+.3f}   group B {b:+.3f}   (sd over clients x seeds {sd:.3f})")
        for sc in ["s", "rate"]:
            print(f"  {sc:4s} similarity  within-group {np.mean(sim[sc]['within']):.4f}   "
                  f"between-group {np.mean(sim[sc]['between']):.4f}")


if __name__ == "__main__":
    main()
