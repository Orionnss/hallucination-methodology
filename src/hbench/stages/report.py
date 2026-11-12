"""Stage `report` (prompt §3.6): one table per metric (rows = detectors, columns = (generator, dataset),
cells = mean ± std over the outer folds), the labelling statistics, the significance tests, and
pooled AUROC in a separate table (D57). Output: CSV + LaTeX (booktabs) + a Markdown summary."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from hbench.config import ExperimentCfg
from hbench.io import PartTable
from hbench.metrics import METRICS
from hbench.paths import RunPaths
from hbench.stats import corrected_resampled_ttest, holm, min_wilcoxon_p, wilcoxon_across_cells

TEST_METRICS = ("auroc", "mcc")  # D13
ALPHA = 0.05
METRIC_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "auroc": "AUROC",
    "wacc": "Weighted accuracy",
}


# ---------- loading ----------


def load_metrics(cfg: ExperimentCfg, paths: RunPaths) -> pd.DataFrame:
    frames = []
    for det in cfg.detectors:
        for g in cfg.generators:
            for d in cfg.datasets:
                p = paths.metrics(det.name, g.name, d.name)
                if not p.exists():
                    raise RuntimeError(f"report: missing metrics for {det.name} on {g.name}/{d.name}; run `evaluate` first")
                frames.append(pd.read_parquet(p))
    m = pd.concat(frames, ignore_index=True)
    # Paired design: all detectors of a cell must use the same labels and folds.
    for (g, d), c in m.groupby(["generator", "dataset"]):
        if c["labels_hash"].nunique() != 1:
            raise RuntimeError(f"report: detectors on {g}/{d} were evaluated on different labels; re-run `evaluate`")
    return m


def cell_columns(cfg: ExperimentCfg) -> list[tuple[str, str]]:
    return [(g.name, d.name) for g in cfg.generators for d in cfg.datasets]


# ---------- tables ----------


def metric_table(m: pd.DataFrame, metric: str, detectors: list[str], cells: list[tuple[str, str]]) -> pd.DataFrame:
    """Tidy table: detector, generator, dataset, mean, std, n_folds."""
    agg = m.groupby(["detector", "generator", "dataset"])[metric].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"count": "n_folds"})
    order = {d: i for i, d in enumerate(detectors)}
    corder = {c: i for i, c in enumerate(cells)}
    agg["_r"] = agg["detector"].map(order)
    agg["_c"] = [corder[(g, d)] for g, d in zip(agg["generator"], agg["dataset"])]
    return agg.sort_values(["_r", "_c"]).drop(columns=["_r", "_c"]).reset_index(drop=True)


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def wide_strings(tidy: pd.DataFrame, detectors: list[str], cells: list[tuple[str, str]], bold: str) -> pd.DataFrame:
    """Wide table of formatted cells; the best mean of each column is bold (D57). `bold` is a format
    string with one `{}` (LaTeX: \\textbf{{{}}}, Markdown: **{}**)."""
    out = pd.DataFrame(index=detectors, columns=pd.MultiIndex.from_tuples(cells, names=["generator", "dataset"]), dtype=object)
    for (g, d), c in tidy.groupby(["generator", "dataset"]):
        best = c["mean"].max()
        for _, r in c.iterrows():
            s = _fmt(r["mean"], r["std"])
            out.loc[r["detector"], (g, d)] = bold.format(s) if np.isclose(r["mean"], best) else s
    return out


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("±", r"$\pm$")):
        s = s.replace(a, b)
    return s


GROUP_ORDER = ("published", "reimplemented", "reference")
GROUP_TITLES = {
    "published": "Published methods (original code)",
    "reimplemented": "Reimplemented from the paper (no official code)",
    "reference": "Reference probes (pipeline baselines)",
}


def detector_groups(cfg: ExperimentCfg) -> dict[str, str]:
    from hbench.pool.registry import get_detector

    return {d.name: get_detector(d.name).group for d in cfg.detectors}


def order_by_group(detectors: list[str], groups: dict[str, str]) -> list[str]:
    """Report rows: published methods, then reimplementations (ADDING_A_METHOD.md case D: separate
    group in all tables), then reference probes; config order inside a group."""
    return sorted(detectors, key=lambda d: GROUP_ORDER.index(groups.get(d, "published")))


def to_latex_booktabs(wide: pd.DataFrame, caption: str, label: str, groups: dict[str, str] | None = None) -> str:
    cols = list(wide.columns)
    gens = []
    for g, _ in cols:
        if not gens or gens[-1][0] != g:
            gens.append([g, 0])
        gens[-1][1] += 1
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{_tex_escape(g)}}}" for g, n in gens) + r" \\",
    ]
    start = 2
    rules = []
    for _, n in gens:
        rules.append(rf"\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n
    lines.append(" ".join(rules))
    lines.append("Detector & " + " & ".join(_tex_escape(d) for _, d in cols) + r" \\")
    lines.append(r"\midrule")
    prev = None
    for det, row in wide.iterrows():
        g = (groups or {}).get(det)
        if groups and g != prev:
            if prev is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{_tex_escape(GROUP_TITLES[g])}}}}} \\")
            prev = g
        cells = [str(v).replace("±", r"$\pm$") for v in row.tolist()]
        lines.append(_tex_escape(det) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def to_markdown(wide: pd.DataFrame, groups: dict[str, str] | None = None) -> str:
    head = "| Detector | " + " | ".join(f"{g} / {d}" for g, d in wide.columns) + " |"
    sep = "|---|" + "---|" * len(wide.columns)
    rows, prev = [], None
    for det, row in wide.iterrows():
        g = (groups or {}).get(det)
        if groups and g != prev:
            rows.append(f"| *{GROUP_TITLES[g]}* |" + " |" * len(wide.columns))
            prev = g
        rows.append(f"| {det} | " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join([head, sep, *rows])


# ---------- significance ----------


def ttests(m: pd.DataFrame, detectors: list[str]) -> pd.DataFrame:
    """Corrected resampled t-test for each detector pair, cell and metric; Holm per (metric, cell) (D54)."""
    rows = []
    for (g, d), c in m.groupby(["generator", "dataset"]):
        ratio = float((c["n_test"] / c["n_train"]).mean())
        for metric in TEST_METRICS:
            piv = c.pivot_table(index=["repeat", "fold"], columns="detector", values=metric)
            fam = []
            for a, b in combinations(detectors, 2):
                diff = (piv[a] - piv[b]).to_numpy()
                r = corrected_resampled_ttest(diff, ratio)
                fam.append({"metric": metric, "generator": g, "dataset": d, "detector_a": a, "detector_b": b,
                            "mean_diff": float(diff.mean()), "t": r.statistic, "df": r.n - 1, "p": r.p, "flag": r.flag})
            for row, ph in zip(fam, holm([x["p"] for x in fam])):
                row["p_holm"] = ph
                row["significant"] = bool(ph < ALPHA) if not np.isnan(ph) else False
            rows += fam
    return pd.DataFrame(rows)


def wilcoxons(m: pd.DataFrame, detectors: list[str]) -> pd.DataFrame:
    """Wilcoxon across cells on the per-cell mean differences; Holm per metric (D54, D55)."""
    means = m.groupby(["detector", "generator", "dataset"])[list(TEST_METRICS)].mean()
    rows = []
    for metric in TEST_METRICS:
        piv = means[metric].unstack("detector")
        fam = []
        for a, b in combinations(detectors, 2):
            diff = (piv[a] - piv[b]).to_numpy()
            r = wilcoxon_across_cells(diff)
            fam.append({"metric": metric, "detector_a": a, "detector_b": b, "n_cells": r.n, "mean_diff": float(diff.mean()),
                        "W": r.statistic, "p": r.p, "min_attainable_p": min_wilcoxon_p(int((diff != 0).sum())), "flag": r.flag})
        for row, ph in zip(fam, holm([x["p"] for x in fam])):
            row["p_holm"] = ph
            row["significant"] = bool(ph < ALPHA)
        rows += fam
    return pd.DataFrame(rows)


# ---------- feature × classifier grid (D89) ----------


def cross_section(m: pd.DataFrame, cells: list[tuple[str, str]], tables: Path) -> list[str]:
    """The 3 × 3 grid of vector features × vector classifiers: the diagonal is the published
    detectors, the rest the combinations of group "cross". Descriptive only: no significance tests.
    Returns the Markdown lines of the section ([] when no combination was evaluated)."""
    from hbench.pool.combos import HEADS, axes

    rows = [(det, *axes(det)) for det in m["detector"].unique() if axes(det) is not None]
    if not any("+" in det for det, _, _ in rows):
        return []
    feats = [v for v in ("lapeigvals_feat", "icr_feat", "hidden_feat") if any(r[1] == v for r in rows)]
    heads = [h for h in HEADS if any(r[2] == h for r in rows)]
    name = {(v, h): det for det, v, h in rows}
    lines = ["## Feature × classifier grid", "",
             "Each row = one stored feature (with its own selection: k and layer for lapeigvals, layer for the hidden state) "
             "fed to one published classifier. Diagonal (†) = the published detector. Mean ± std over the outer folds; "
             "best mean per column in bold. Descriptive only: no significance tests (D89).", ""]
    for metric in TEST_METRICS:
        order = [name[(v, h)] for v in feats for h in heads if (v, h) in name]
        tidy = metric_table(m[m["detector"].isin(order)], metric, order, cells)
        tidy.insert(1, "feature", [axes(d)[0] for d in tidy["detector"]])
        tidy.insert(2, "classifier", [axes(d)[1] for d in tidy["detector"]])
        tidy.to_csv(tables / f"cross_{metric}.csv", index=False)
        wide = wide_strings(tidy, order, cells, "**{}**")
        wide.index = [f"{axes(d)[0]} × {axes(d)[1]}" + (" †" if "+" not in d else "") for d in wide.index]
        wide_tex = wide_strings(tidy, order, cells, r"\textbf{{{}}}")
        wide_tex.index = wide.index
        (tables / f"cross_{metric}.tex").write_text(
            to_latex_booktabs(wide_tex, f"{METRIC_NAMES[metric]}, feature × classifier grid († = published detector)", f"tab:cross_{metric}"))
        # Matrix of the mean over cells (equal weight per cell).
        cm = tidy.groupby(["feature", "classifier"])["mean"].mean().unstack("classifier").reindex(index=feats, columns=heads)
        cm.to_csv(tables / f"cross_{metric}_matrix.csv")
        mat = ["| feature / classifier | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
        for v in feats:
            mat.append(f"| {v} | " + " | ".join(
                ("—" if pd.isna(cm.loc[v, h]) else f"{cm.loc[v, h]:.3f}" + (" †" if (v, h) in name and "+" not in name[(v, h)] else ""))
                for h in heads) + " |")
        lines += [f"### {METRIC_NAMES[metric]}", "", to_markdown(wide), "",
                  f"Mean over the {len(cells)} cells:", "", *mat, ""]
    return lines


# ---------- other tables ----------


def pooled_auroc(cfg: ExperimentCfg, paths: RunPaths) -> pd.DataFrame:
    """AUROC on the predictions pooled over the outer folds of one cell (per repeat, then mean).
    Not the main aggregate (prompt §2.7)."""
    rows = []
    for det in cfg.detectors:
        for g in cfg.generators:
            for d in cfg.datasets:
                p = pd.read_parquet(paths.predictions(det.name, g.name, d.name))
                vals = [roc_auc_score(x["y"], x["score"]) for _, x in p.groupby("repeat")]
                rows.append({"detector": det.name, "generator": g.name, "dataset": d.name, "pooled_auroc": float(np.mean(vals))})
    return pd.DataFrame(rows)


def judge_stats(cfg: ExperimentCfg, paths: RunPaths) -> pd.DataFrame:
    """Per judge and cell: vote counts, parse errors, forced closes of thinking judges (current settings only)."""
    from hbench.judging.judges import judge_prompt_hash
    from hbench.stages.judge import vote_key_suffix

    jh = judge_prompt_hash(cfg)
    rows = []
    for j in cfg.judges:
        suf = vote_key_suffix(cfg, j, jh)
        for g in cfg.generators:
            for d in cfg.datasets:
                v = PartTable(paths.votes(j.name, g.name, d.name)).read()
                v = v[v["key"] == v["gen_key"] + suf]
                r = {"judge": j.name, "generator": g.name, "dataset": d.name, "n_votes": len(v)}
                r.update({f"n_{k.lower()}": int((v["vote"] == k).sum()) for k in ("CORRECT", "INCORRECT", "REFUSAL", "PARSE_ERROR")})
                if "forced_close" in v and v["forced_close"].notna().any():
                    r["forced_close"] = int(v["forced_close"].fillna(False).astype(bool).sum())
                    r["median_reasoning_tokens"] = float(v["reasoning_tokens"].median())
                rows.append(r)
    return pd.DataFrame(rows)


def flag_counts(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (det, g, d), c in m.groupby(["detector", "generator", "dataset"]):
        flags = [f for fl in c["flags"] for f in fl]
        rows.append({"detector": det, "generator": g, "dataset": d, "n_flags": len(flags), "flags": ", ".join(sorted(set(flags)))})
    return pd.DataFrame(rows)


# ---------- stage ----------


def _write(df: pd.DataFrame, out: Path, name: str, latex: str | None = None) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    if latex is not None:
        (out / f"{name}.tex").write_text(latex)


def run(cfg: ExperimentCfg) -> Path:
    paths = RunPaths(cfg)
    out = paths.root / "report"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    groups = detector_groups(cfg)
    # Main tables and tests: the published, reimplemented and reference detectors. The feature ×
    # classifier combinations (group "cross", D89) get their own section.
    detectors = order_by_group([d.name for d in cfg.detectors if groups[d.name] != "cross"], groups)
    cells = cell_columns(cfg)
    m_all = load_metrics(cfg, paths)
    m = m_all[m_all["detector"].isin(detectors)]

    md_tables = {}
    for metric in METRICS:
        tidy = metric_table(m, metric, detectors, cells)
        tidy.to_csv(tables / f"{metric}.csv", index=False)
        wide_tex = wide_strings(tidy, detectors, cells, r"\textbf{{{}}}")
        (tables / f"{metric}.tex").write_text(
            to_latex_booktabs(wide_tex, f"{METRIC_NAMES[metric]} (mean ± std over outer folds; hallucination = positive class)", f"tab:{metric}", groups)
        )
        md_tables[metric] = to_markdown(wide_strings(tidy, detectors, cells, "**{}**"), groups)

    pooled = pooled_auroc(cfg, paths)
    pooled = pooled[pooled["detector"].isin(detectors)].reset_index(drop=True)
    pooled.to_csv(tables / "pooled_auroc.csv", index=False)
    pw = pooled.pivot_table(index="detector", columns=["generator", "dataset"], values="pooled_auroc").reindex(index=detectors, columns=cells)
    pw_s = pw.map(lambda v: f"{v:.3f}")
    (tables / "pooled_auroc.tex").write_text(
        to_latex_booktabs(pw_s, "Pooled AUROC (predictions pooled over the outer folds of each cell; not the main aggregate)", "tab:pooled_auroc", groups)
    )

    lab = pd.read_parquet(paths.label_stats())
    lab_cols = ["generator", "dataset", "n", "n_kept", "refusal_rate", "tie_rate", "no_valid_vote_rate", "parse_error_count",
                "empty_answer_count", "n_hallucination", "n_correct", "hallucination_ratio", "mean_pairwise_cohen_kappa"]
    lab = lab[lab_cols]
    lab.to_csv(tables / "labelling.csv", index=False)
    lab_tex = lab.copy()
    for c in ("refusal_rate", "tie_rate", "no_valid_vote_rate", "hallucination_ratio", "mean_pairwise_cohen_kappa"):
        lab_tex[c] = lab_tex[c].map(lambda v: f"{v:.3f}")
    (tables / "labelling.tex").write_text(_labelling_latex(lab_tex))

    tt = ttests(m, detectors) if len(detectors) > 1 else pd.DataFrame()
    wx = wilcoxons(m, detectors) if len(detectors) > 1 else pd.DataFrame()
    tt.to_csv(tables / "significance_ttest.csv", index=False)
    wx.to_csv(tables / "significance_wilcoxon.csv", index=False)
    js = judge_stats(cfg, paths)
    js.to_csv(tables / "judges.csv", index=False)
    fl = flag_counts(m_all)
    fl.to_csv(tables / "metric_flags.csv", index=False)
    cross_md = cross_section(m_all, cells, tables)

    (out / "summary.md").write_text(_summary(cfg, paths, m_all, md_tables, pooled, lab, tt, wx, js, fl, detectors, cross_md))
    logger.info(f"report: written to {out}")
    return out


def _labelling_latex(lab: pd.DataFrame) -> str:
    head = ["Generator", "Dataset", "$n$", "Kept", "Refusal", "Tie", "No vote", "Parse err.", "Empty", "Halluc.", "Correct", "Halluc. ratio", r"Cohen's $\kappa$"]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Labelling statistics (rates over $n$; class ratio over the kept samples; mean pairwise Cohen's $\kappa$)}",
             r"\label{tab:labelling}", r"\begin{tabular}{ll" + "r" * (len(head) - 2) + "}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for _, r in lab.iterrows():
        lines.append(" & ".join(_tex_escape(str(v)) for v in r.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _md_df(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_none_"
    cols = list(df.columns)
    def f(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v) if np.isfinite(v) else str(v)
        return str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _summary(cfg, paths, m, md_tables, pooled, lab, tt, wx, js, fl, detectors, cross_md) -> str:
    prov = sorted((paths.root / "provenance").glob("evaluate-*.json"))
    git = json.loads(prov[-1].read_text())["git"] if prov else {}
    meta = {p.name.removesuffix(".meta.json"): json.loads(p.read_text()) for p in sorted((paths.root / "samples").glob("*.meta.json"))}
    k = m.groupby(["generator", "dataset"]).size().iloc[0] // max(1, len(cfg.detectors))
    s = [
        f"# Report: {cfg.run_id}",
        "",
        f"- Generators: {', '.join(g.name for g in cfg.generators)}",
        f"- Datasets: {', '.join(d.name for d in cfg.datasets)} (n = {cfg.n} each, seed {cfg.seed})",
        f"- Judges (R = {len(cfg.judges)}): {', '.join(j.name for j in cfg.judges)}",
        f"- Detectors: {', '.join(detectors)}",
        f"- CV: {cfg.cv.outer_folds} outer × {cfg.cv.inner_folds} inner folds, {cfg.cv.repeats} repeat(s) → k = {k} paired folds per cell",
        f"- Git commit: `{git.get('commit', '?')}`{' (dirty)' if git.get('dirty') else ''}",
        "",
        "Positive class = hallucination. Cells = mean ± std over the outer folds; best mean per column in bold.",
        "",
        "## Labelling",
        "",
        _md_df(lab),
        "",
        "Dataset filtering before sampling:",
        "",
        _md_df(pd.DataFrame([{"dataset": d, **v} for d, v in meta.items()])),
        "",
        "Judges (votes with the current settings):",
        "",
        _md_df(js),
        "",
    ]
    for metric in METRICS:
        s += [f"## {METRIC_NAMES[metric]}", "", md_tables[metric], ""]
    s += [
        "## Pooled AUROC (not the main aggregate)",
        "",
        "Predictions pooled over the outer folds of each cell. Never pooled across cells or datasets.",
        "",
        _md_df(pooled),
        "",
        "## Significance tests",
        "",
        f"Corrected resampled t-test (Nadeau & Bengio, 2003) per cell, Holm per (metric, cell); Wilcoxon signed-rank across cells, Holm per metric. α = {ALPHA}.",
        "",
        "### Per cell",
        "",
        _md_df(tt[["metric", "generator", "dataset", "detector_a", "detector_b", "mean_diff", "t", "p", "p_holm", "significant"]] if not tt.empty else tt),
        "",
        "### Across cells",
        "",
        _md_df(wx[["metric", "detector_a", "detector_b", "n_cells", "mean_diff", "p", "min_attainable_p", "p_holm", "significant"]] if not wx.empty else wx),
        "",
        *cross_md,
        "## Undefined-metric flags",
        "",
        f"Total: {int(fl['n_flags'].sum())}.",
        "",
        _md_df(fl[fl["n_flags"] > 0]) if (fl["n_flags"] > 0).any() else "_none_",
        "",
    ]
    return "\n".join(s)
