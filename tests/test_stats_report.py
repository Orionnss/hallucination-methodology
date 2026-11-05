import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from hbench.stats import corrected_resampled_ttest, holm, min_wilcoxon_p, wilcoxon_across_cells


def test_corrected_ttest_formula():
    d = np.array([0.02, 0.05, -0.01, 0.04, 0.03])
    r = corrected_resampled_ttest(d, test_train_ratio=0.25)
    t = d.mean() / math.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
    assert r.statistic == pytest.approx(t) and r.p == pytest.approx(2 * st.t.sf(abs(t), 4))
    # The correction makes the test more conservative than the plain paired t-test.
    assert r.p > st.ttest_1samp(d, 0).pvalue


def test_corrected_ttest_zero_variance():
    assert corrected_resampled_ttest(np.zeros(5), 0.25).p == 1.0
    r = corrected_resampled_ttest(np.full(5, 0.1), 0.25)
    assert r.p == 0.0 and r.flag == "zero_variance"


def test_holm():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    out = holm([0.2, float("nan"), 0.01])
    assert out[0] == pytest.approx(0.2) and math.isnan(out[1]) and out[2] == pytest.approx(0.02)


def test_wilcoxon_exact_small_n():
    assert wilcoxon_across_cells(np.array([0.1, 0.2, 0.05, 0.3])).p == pytest.approx(0.125)
    assert min_wilcoxon_p(4) == pytest.approx(0.125) and min_wilcoxon_p(6) == pytest.approx(0.03125)
    assert wilcoxon_across_cells(np.zeros(4)).p == 1.0


def test_report_stage(make_cfg):
    from hbench.config import load_config
    from hbench.paths import RunPaths
    from hbench.stages import aggregate, evaluate, extract, generate, judge, report, sample

    dets = [{"name": "mean_hidden_probe", "grid": {"C": [1.0]}}, {"name": "saplma_style_probe"}]
    judges = [{"name": f"m{i}", "hf_id": "mock", "kind": "mock", "mock": {"seed": 5, "flip_prob": 0.5 if i == 0 else 0.1}} for i in range(3)]
    cfg = load_config(make_cfg(n=90, judges=judges, detectors=dets, extraction={"batch_size": 16, "device": "cpu"}))
    for s in (sample, generate, judge, aggregate, extract, evaluate):
        s.run(cfg)
    out = report.run(cfg)
    t = out / "tables"
    for m in ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "wacc"):
        assert (t / f"{m}.csv").exists()
        tex = (t / f"{m}.tex").read_text()
        assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex and r"\textbf{" in tex
    auroc = pd.read_csv(t / "auroc.csv")
    m = pd.read_parquet(RunPaths(cfg).metrics("mean_hidden_probe", "tiny0", "toy"))
    assert auroc.loc[auroc["detector"] == "mean_hidden_probe", "mean"].iloc[0] == pytest.approx(m["auroc"].mean())
    tt = pd.read_csv(t / "significance_ttest.csv")
    assert set(tt["metric"]) == {"auroc", "mcc"} and len(tt) == 2  # one pair × one cell × 2 metrics
    assert (pd.read_csv(t / "significance_wilcoxon.csv")["n_cells"] == 1).all()
    s = (out / "summary.md").read_text()
    assert "## Labelling" in s and "Pooled AUROC (not the main aggregate)" in s and "**" in s
    assert (pd.read_csv(t / "labelling.csv")["n"] == 90).all()
    assert "Reference probes" in (t / "auroc.tex").read_text() and "| *Reference probes" in s


def test_report_groups_order():
    from hbench.stages.report import order_by_group, to_markdown

    groups = {"a": "reference", "b": "published", "c": "reimplemented"}
    assert order_by_group(["a", "b", "c"], groups) == ["b", "c", "a"]
    wide = pd.DataFrame({("g", "d"): ["1", "2", "3"]}, index=["b", "c", "a"])
    md = to_markdown(wide, groups)
    assert md.index("Published") < md.index("Reimplemented") < md.index("Reference")
