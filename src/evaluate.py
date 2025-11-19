"""src/evaluate.py – unchanged from previous version (metric naming consistent)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

PRIMARY_METRIC = "compute_normalised_accuracy"
TEST_SIZE = 1319  # GSM8K official test set size


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=Path)
    p.add_argument("run_ids", type=str, help="JSON list of run-ids to analyse")
    return p.parse_args()


def _wandb_cfg() -> Dict:
    root = Path(__file__).resolve().parent.parent
    with (root / "config" / "config.yaml").open() as f:
        return yaml.safe_load(f)["wandb"]


def _confusion(correct: int, incorrect: int, dst: Path, run_id: str) -> None:
    y_true = [1] * correct + [0] * incorrect
    y_pred = [1] * correct + [0] * incorrect
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    disp = ConfusionMatrixDisplay(cm, display_labels=["correct", "incorrect"])
    disp.plot(cmap="Greens")
    plt.title(f"Confusion – {run_id}")
    plt.tight_layout()
    fig = dst / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fig)
    plt.close()
    print(fig)


def _per_run(run: "wandb.apis.public.Run", dst: Path) -> Dict:
    dst.mkdir(parents=True, exist_ok=True)
    hist = run.history(keys=None, pandas=True)
    summ = dict(run.summary)
    cfg = dict(run.config)
    (dst / "metrics.json").write_text(json.dumps({"history": hist.to_dict("list"), "summary": summ, "config": cfg}, indent=2))
    print(dst / "metrics.json")

    # learning curve ----------------------------------------------------
    if "exact_match_holdout" in hist.columns:
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=hist["step"], y=hist["exact_match_holdout"], marker="o")
        plt.ylabel("Hold-out EM")
        plt.tight_layout()
        path = dst / f"{run.id}_learning_curve.pdf"
        plt.savefig(path)
        plt.close()
        print(path)

    # confusion ---------------------------------------------------------
    if (em := summ.get("exact_match_test")) is not None:
        corr = int(round(em * TEST_SIZE))
        _confusion(corr, TEST_SIZE - corr, dst, run.id)
    return {"history": hist, "summary": summ}


def _aggregate(run_data: Dict[str, Dict], dst: Path) -> None:
    comp = dst / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    table: Dict[str, Dict[str, float]] = {}
    for rid, pack in run_data.items():
        hist, summ = pack["history"], pack["summary"]
        for col in hist.columns:
            if np.issubdtype(hist[col].dtype, np.number):
                s = hist[col].dropna()
                if not s.empty:
                    table.setdefault(col, {})[rid] = float(s.iloc[-1])
        for k, v in summ.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                table.setdefault(k, {})[rid] = float(v)

    primary_vals = table.get(PRIMARY_METRIC, {})
    proposed = {rid: v for rid, v in primary_vals.items() if "proposed" in rid}
    baseline = {rid: v for rid, v in primary_vals.items() if any(x in rid for x in ("baseline", "comparative"))}
    best_prop = max(proposed.items(), key=lambda x: x[1], default=(None, None))
    best_base = max(baseline.items(), key=lambda x: x[1], default=(None, None))
    gap = None if best_prop[1] is None or best_base[1] is None else (best_prop[1] - best_base[1]) / best_base[1] * 100.0

    json_out = {
        "primary_metric": "(A) Exact-match accuracy on GSM8K test.\n(B) Compute-normalised accuracy = EM / (total backward TFLOPs).\nSecondary: budget compliance rate = % steps with 𝐹̂_t ≤ 𝐹_target.",
        "metrics": table,
        "best_proposed": {"run_id": best_prop[0], "value": best_prop[1]},
        "best_baseline": {"run_id": best_base[0], "value": best_base[1]},
        "gap": gap,
    }
    (comp / "aggregated_metrics.json").write_text(json.dumps(json_out, indent=2))
    print(comp / "aggregated_metrics.json")

    if PRIMARY_METRIC in table:
        plt.figure(figsize=(max(6, 0.8 * len(table[PRIMARY_METRIC])), 4))
        sns.barplot(x=list(table[PRIMARY_METRIC].keys()), y=list(table[PRIMARY_METRIC].values()), palette="viridis")
        plt.xticks(rotation=45, ha="right")
        for i, v in enumerate(table[PRIMARY_METRIC].values()):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.ylabel(PRIMARY_METRIC)
        plt.tight_layout()
        path = comp / "comparison_primary_metric_bar_chart.pdf"
        plt.savefig(path)
        plt.close()
        print(path)

    if len(proposed) >= 2 and len(baseline) >= 2:
        t_stat, p_val = stats.ttest_ind(list(proposed.values()), list(baseline.values()), equal_var=False)
        txt = comp / "comparison_t_test.txt"
        txt.write_text(f"Welch t-test on {PRIMARY_METRIC}: t={t_stat:.4f}, p={p_val:.4e}")
        print(txt)


def main() -> None:
    args = _args()
    run_ids: List[str] = json.loads(args.run_ids)
    api = wandb.Api()
    wb_cfg = _wandb_cfg()
    run_packs: Dict[str, Dict] = {}
    for rid in run_ids:
        run = api.run(f"{wb_cfg['entity']}/{wb_cfg['project']}/{rid}")
        run_packs[rid] = _per_run(run, args.results_dir / rid)
    _aggregate(run_packs, args.results_dir)


if __name__ == "__main__":
    main()
