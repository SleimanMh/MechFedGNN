"""Assemble docs/RESULTS.md from the saved run artefacts (no recomputation).

Tables are copied verbatim from each run's REPORT.md and from the saved
analysis outputs; the narrative sections are written here. Re-run after any
new run to refresh.

Run:  python -m analysis.build_results_md
"""
import glob
import json
import os
import re

OUT = "docs/RESULTS.md"
E1_ORDER = ["concrete", "wine", "kin8nm", "protein"]


def read(path):
    return open(path, encoding="utf-8", errors="replace").read()


def block(text, header):
    """The bold-headed block starting with `header`, up to the next bold header or section."""
    start = text.find(header)
    if start < 0:
        return f"*(block '{header}' not found)*"
    rest = text[start:]
    m = re.search(r"\n(\*\*[A-Z]|## )", rest[len(header):])
    return rest[: len(header) + m.start()].strip() if m else rest.strip()


def headroom_table(text):
    sec = text[text.find("## 5."):text.find("## 6.")]
    lines = [l for l in sec.splitlines() if l.startswith("|")]
    return "\n".join(lines[: next((i for i, l in enumerate(lines) if i > 1 and not l.startswith("| c")), len(lines))])


def runs(exp):
    out = {}
    for r in sorted(glob.glob(f"results/{exp}/*/")):
        cfg = json.load(open(os.path.join(r, "config.json")))
        key = cfg["dataset"] if exp == "e1" else (cfg["dataset"], cfg["e1m"]["condition"])
        out[key] = (r.replace("\\", "/"), read(os.path.join(r, "REPORT.md")))
    return out


def code(text):
    return "```text\n" + text.strip() + "\n```"


def main():
    e1, e1m = runs("e1"), runs("e1m")
    L = [open("analysis/results_narrative.md", encoding="utf-8").read().strip(), ""]

    L += ["## 5. E1 — per-dataset results (verbatim from the run reports)", ""]
    for name in E1_ORDER:
        path, rep = e1[name]
        L += [f"### 5.{E1_ORDER.index(name) + 1} {name}", "", f"Run: `{path}` — full report `{path}REPORT.md`.", ""]
        for h in ["**Paired contrasts**", "**Harm from averaging, per receiver**",
                  "**Receivers harmed relative to local-only, per arm**", "**Validation-selected candidate**",
                  "**RMSE**", "**Score-vs-U agreement**", "**Group recovery**", "**Compute**"]:
            L += [block(rep, h), ""]
        L += ["**Headroom (reference, not a gate)**", "", headroom_table(rep), ""]

    L += ["## 6. E1 — exploratory post-hoc analyses (not pre-registered)", "",
          "### 6.1 Ranking vs weighting, and exact score-vs-uniform intervals", "",
          code(read("results/e1/exploratory_ranking_vs_weighting.txt")), "",
          "### 6.2 Singleton check (C entries around single-feature panels)", "",
          code(read("results/e1/exploratory_singleton_c_check.txt")), ""]

    L += ["## 7. E1M — matched-marginal test", "",
          "### 7.1 Construction checks (before any training)", "", code(read("results/e1m/checks.txt")), "",
          "### 7.2 Analysis (same- vs other-partition U; every arm vs uniform-donor; top donor; "
          "oracle vs validation; parameter geometry)", "", code(read("results/e1m/analysis.txt")), "",
          "### 7.3 Run index", "", "| dataset | condition | run |", "|---|---|---|"]
    for (name, cond), (path, _) in sorted(e1m.items()):
        L.append(f"| {name} | {cond} | `{path}` |")
    L += ["", "## 8. Appendix — pilots and validation (verbatim)", "",
          "### 8.1 Budget and headroom pilot", "", code(read("results/pilots/pilots.txt")), "",
          "### 8.2 Power pilot", "", code(read("results/pilots/pilots_power.txt")), ""]
    for name in E1_ORDER:
        p = f"results/injector_validation/{name}.txt"
        if os.path.exists(p):
            L += [f"### 8.{3 + E1_ORDER.index(name)} Injector validation — {name}", "", code(read(p)), ""]
    L += ["## 9. §16 correction study (EXPLORATORY)", "",
          "Duplicate-group splitting, one frozen shared standardisation, and a target-independent "
          "maskable-assignment control. Sections 5–7 above are **left exactly as run**; where the "
          "corrected runs disagree, the superseded claim is flagged in §9.1 and in the banner at the "
          "top of this file. Declared in `CLAUDE.md` §16 before the runs; criteria (a) and (b) and the "
          "user's expectation were recorded there in advance.", "",
          "### 9.1 Survival of each conclusion, functional comparison, geometry", "",
          code(read("results/correction_study.txt")), "",
          "### 9.2 Wine decomposition — duplicates vs shared scaler vs the role flip", "",
          "The f7/f2 role flip is not an independent factor: it is a downstream consequence of "
          "duplicate grouping changing the design split, hence the |corr(f,target)| ranking on the "
          "design rows. It is isolated as the D → E step below.", "",
          code(read("results/wine_decomposition.txt")), "",
          "### 9.3 Setup: duplicate rates, rows moved, corrected roles, E1M tolerance", "",
          code(read("results/correction_setup.txt")), "",
          "### 9.4 Corrected run index", "", "| experiment | dataset | condition | run |", "|---|---|---|---|"]
    for exp in sorted(glob.glob("results/e1*_corrected_*/")):
        for r in sorted(glob.glob(exp + "*/")):
            cfg = json.load(open(os.path.join(r, "config.json")))
            L.append(f"| {os.path.basename(exp.rstrip('/'))} | {cfg['dataset']} | "
                     f"{cfg.get('e1m', {}).get('condition', '-')} | `{r.replace(chr(92), '/')}` |")
    os.makedirs("docs", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {OUT}: {sum(1 for _ in open(OUT, encoding='utf-8'))} lines")


if __name__ == "__main__":
    main()
