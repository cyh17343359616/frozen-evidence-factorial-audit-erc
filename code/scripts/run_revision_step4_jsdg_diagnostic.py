#!/usr/bin/env python3
"""Run the bounded Step 4 diagnostic from frozen Step 29 artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ieee_revision/step4_jsdg"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib_step4"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
SEEDS = [42, 43, 44]
TRANSITIONS = ["correct_to_error", "error_to_correct", "both_correct", "both_error"]
MECHANISMS = ["js_divergence", "normal_gate", "disagreement_gate"]
PROB_COLS = [f"prob_{label}" for label in LABELS]
PREDICTIONS = ROOT / "outputs/phase3_ieee_access/step29_robustness_stress/predictions.csv"
MECHANISM = ROOT / "outputs/phase3_ieee_access/step29_robustness_stress/mechanism_distributions.csv"
PROTOCOL = OUT / "protocol.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, lineterminator="\n")


def write_json(value: Any, name: str) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def holm(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def summary(values: pd.Series) -> dict[str, Any]:
    x = values.dropna().to_numpy(dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else math.nan,
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else math.nan,
        "min": float(np.min(x)) if len(x) else math.nan,
        "q25": float(np.quantile(x, 0.25)) if len(x) else math.nan,
        "median": float(np.median(x)) if len(x) else math.nan,
        "q75": float(np.quantile(x, 0.75)) if len(x) else math.nan,
        "max": float(np.max(x)) if len(x) else math.nan,
    }


def transition_name(a1_correct: pd.Series, a3_correct: pd.Series) -> np.ndarray:
    return np.select(
        [a1_correct & ~a3_correct, ~a1_correct & a3_correct, a1_correct & a3_correct],
        ["correct_to_error", "error_to_correct", "both_correct"],
        default="both_error",
    )


def load_joined() -> pd.DataFrame:
    pred_cols = [
        "experiment", "seed", "split", "scenario", "family", "severity", "sample_index",
        "dialogue_id", "utterance_id", "speaker", "gold", "gold_label", "prediction",
        "row_corrupted", "feature_fraction_corrupted", *PROB_COLS,
    ]
    pred = pd.read_csv(PREDICTIONS, usecols=pred_cols)
    pred = pred[pred["experiment"].isin(["A1", "A3"])].copy()
    if set(pred["seed"].unique()) != set(SEEDS) or set(pred["split"].unique()) != {"test"}:
        raise ValueError("unexpected Step 29 seed or split coverage")
    key = ["seed", "scenario", "sample_index"]
    a1 = pred[pred["experiment"] == "A1"].drop(columns="experiment")
    a3 = pred[pred["experiment"] == "A3"].drop(columns="experiment")
    if a1.duplicated(key).any() or a3.duplicated(key).any():
        raise ValueError("duplicate A1/A3 prediction identity")
    joined = a1.merge(a3, on=key, suffixes=("_a1", "_a3"), validate="one_to_one")
    identity = ["family", "severity", "dialogue_id", "utterance_id", "speaker", "gold", "gold_label"]
    for col in identity:
        left, right = joined[f"{col}_a1"], joined[f"{col}_a3"]
        if not left.equals(right):
            raise ValueError(f"A1/A3 identity mismatch: {col}")
        joined[col] = left
    mech = pd.read_csv(
        MECHANISM,
        usecols=["experiment", "seed", "scenario", "sample_index", *MECHANISMS],
    )
    mech = mech[mech["experiment"] == "A3"].drop(columns="experiment")
    if mech[MECHANISMS].isna().any().any() or mech.duplicated(key).any():
        raise ValueError("A3 mechanism fields are incomplete or duplicated")
    joined = joined.merge(mech, on=key, validate="one_to_one")
    expected = len(SEEDS) * 11 * 2593
    if len(joined) != expected:
        raise ValueError(f"expected {expected} matched rows, observed {len(joined)}")
    joined["a1_correct"] = joined["prediction_a1"] == joined["gold"]
    joined["a3_correct"] = joined["prediction_a3"] == joined["gold"]
    joined["transition"] = transition_name(joined["a1_correct"], joined["a3_correct"])
    joined["a1_error"] = (~joined["a1_correct"]).astype(int)
    joined["a3_error"] = (~joined["a3_correct"]).astype(int)
    a1_probs = joined[[f"{c}_a1" for c in PROB_COLS]].to_numpy(dtype=float)
    joined["a1_confidence"] = a1_probs.max(axis=1)
    joined["confidence_bin"] = pd.cut(
        joined["a1_confidence"], bins=[0.0, 0.5, 0.75, 1.0000001],
        labels=["[0,0.50)", "[0.50,0.75)", "[0.75,1.00]"], include_lowest=True, right=False,
    ).astype(str)
    train_speakers = set(np.load(ROOT / "datasets/MELD/features_v2/train_speakers.npy", allow_pickle=False).astype(str))
    joined["speaker_coverage"] = np.where(joined["speaker"].astype(str).isin(train_speakers), "seen_in_train", "unseen_in_train")
    return joined


def group_distributions(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = clean[clean["seed"] == seed]
        for group in TRANSITIONS:
            selected = seed_frame[seed_frame["transition"] == group]
            for metric in MECHANISMS:
                rows.append({"seed": seed, "transition": group, "metric": metric, **summary(selected[metric])})
        for metric in MECHANISMS:
            for group_a, group_b in combinations(TRANSITIONS, 2):
                x = seed_frame.loc[seed_frame["transition"] == group_a, metric].to_numpy(dtype=float)
                y = seed_frame.loc[seed_frame["transition"] == group_b, metric].to_numpy(dtype=float)
                result = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
                cliffs_delta = 2.0 * float(result.statistic) / (len(x) * len(y)) - 1.0
                tests.append({
                    "seed": seed, "metric": metric, "group_a": group_a, "group_b": group_b,
                    "n_a": len(x), "n_b": len(y), "median_a": float(np.median(x)),
                    "median_b": float(np.median(y)), "cliffs_delta_a_minus_b": cliffs_delta,
                    "p_raw": float(result.pvalue),
                })
    test_frame = pd.DataFrame(tests)
    test_frame["p_holm_within_seed"] = np.nan
    for seed in SEEDS:
        mask = test_frame["seed"] == seed
        test_frame.loc[mask, "p_holm_within_seed"] = holm(test_frame.loc[mask, "p_raw"])
    test_frame["reject_holm_0p05"] = test_frame["p_holm_within_seed"] < 0.05
    return pd.DataFrame(rows), test_frame


def metric_triplet(y: np.ndarray, score: np.ndarray) -> tuple[float, float, float, float]:
    rho, p_value = spearmanr(score, y)
    return (
        float(roc_auc_score(y, score)),
        float(average_precision_score(y, score)),
        float(rho),
        float(p_value),
    )


def bootstrap_predictive(clean: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        frame = clean[clean["seed"] == seed].reset_index(drop=True)
        y = frame["a1_error"].to_numpy(dtype=int)
        score = frame["js_divergence"].to_numpy(dtype=float)
        point = metric_triplet(y, score)
        dialogue_groups = [idx.to_numpy(dtype=int) for _, idx in frame.groupby("dialogue_id", sort=True).groups.items()]
        rng = np.random.Generator(np.random.PCG64(404000 + seed))
        boot = np.empty((replicates, 3), dtype=float)
        for rep in range(replicates):
            choices = rng.integers(0, len(dialogue_groups), size=len(dialogue_groups))
            sampled = np.concatenate([dialogue_groups[i] for i in choices])
            boot[rep, :] = metric_triplet(y[sampled], score[sampled])[:3]
        intervals = np.quantile(boot, [0.025, 0.975], axis=0)
        for pos, metric in enumerate(["auroc", "auprc", "spearman_rho"]):
            rows.append({
                "seed": seed, "scenario": "video_missing_r0", "target": "A1_baseline_error",
                "metric": metric, "estimate": point[pos], "ci_low": float(intervals[0, pos]),
                "ci_high": float(intervals[1, pos]), "bootstrap_unit": "dialogue_id",
                "bootstrap_replicates": replicates, "bootstrap_rng_seed": 404000 + seed,
                "baseline_error_prevalence": float(y.mean()),
                "p_raw": point[3] if metric == "spearman_rho" else math.nan,
            })
    result = pd.DataFrame(rows)
    rho_mask = result["metric"] == "spearman_rho"
    result["p_holm_across_seed_spearman"] = np.nan
    result.loc[rho_mask, "p_holm_across_seed_spearman"] = holm(result.loc[rho_mask, "p_raw"])
    return result


def stratified(clean_and_stress: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specifications = [
        ("emotion", ["seed", "gold_label"]),
        ("missingness", ["seed", "scenario", "family", "severity"]),
        ("speaker_coverage", ["seed", "speaker_coverage"]),
        ("confidence", ["seed", "confidence_bin"]),
    ]
    clean = clean_and_stress[clean_and_stress["scenario"] == "video_missing_r0"]
    for stratum_type, keys in specifications:
        frame = clean_and_stress if stratum_type == "missingness" else clean
        for values, group in frame.groupby(keys, observed=True, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            base = {key: value for key, value in zip(keys, values)}
            counts = group["transition"].value_counts()
            row = {
                "stratum_type": stratum_type, **base, "n": len(group),
                "a1_error_rate": float(group["a1_error"].mean()),
                "a3_error_rate": float(group["a3_error"].mean()),
                "mean_js": float(group["js_divergence"].mean()),
                "mean_normal_gate": float(group["normal_gate"].mean()),
                "mean_disagreement_gate": float(group["disagreement_gate"].mean()),
                **{f"n_{transition}": int(counts.get(transition, 0)) for transition in TRANSITIONS},
            }
            frames.append(pd.DataFrame([row]))
    return pd.concat(frames, ignore_index=True)


def ece(prob: np.ndarray, gold: np.ndarray, bins: int = 10) -> float:
    confidence = prob.max(axis=1)
    prediction = prob.argmax(axis=1)
    correct = prediction == gold
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for idx in range(bins):
        mask = (confidence >= edges[idx]) & (confidence < edges[idx + 1] if idx < bins - 1 else confidence <= edges[idx + 1])
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def final_metrics(clean: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        frame = clean[clean["seed"] == seed]
        gold = frame["gold"].to_numpy(dtype=int)
        one_hot = np.eye(len(LABELS))[gold]
        for experiment, suffix in [("A1", "a1"), ("A3", "a3")]:
            prob = frame[[f"{c}_{suffix}" for c in PROB_COLS]].to_numpy(dtype=float)
            sorted_prob = np.sort(prob, axis=1)
            entropy = -np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0)), axis=1)
            rows.append({
                "seed": seed, "experiment": experiment, "scenario": "video_missing_r0", "n": len(frame),
                "accuracy": float((prob.argmax(axis=1) == gold).mean()),
                "mean_margin": float(np.mean(sorted_prob[:, -1] - sorted_prob[:, -2])),
                "mean_entropy": float(np.mean(entropy)), "ece_10_equal_width": ece(prob, gold),
                "multiclass_brier": float(np.mean(np.sum((prob - one_hot) ** 2, axis=1))),
            })
    return pd.DataFrame(rows)


def bins_and_figure(clean: pd.DataFrame) -> pd.DataFrame:
    edges = np.arange(0.0, 0.7000001, 0.05)
    work = clean.copy()
    work["js_bin"] = pd.cut(work["js_divergence"], bins=edges, include_lowest=True, right=False)
    grouped = work.groupby(["seed", "js_bin"], observed=True, sort=True).agg(
        n=("sample_index", "size"), js_mean=("js_divergence", "mean"),
        a1_error_rate=("a1_error", "mean"), a3_error_rate=("a3_error", "mean"),
        disagreement_gate_mean=("disagreement_gate", "mean"), normal_gate_mean=("normal_gate", "mean"),
    ).reset_index()
    grouped["js_bin"] = grouped["js_bin"].astype(str)
    write_csv(grouped, "js_bins.csv")
    aggregate = grouped.groupby("js_bin", sort=False).agg(
        js_mean=("js_mean", "mean"), a1_error_rate=("a1_error_rate", "mean"),
        a3_error_rate=("a3_error_rate", "mean"), disagreement_gate_mean=("disagreement_gate_mean", "mean"),
        n_rows=("n", "sum"), seeds=("seed", "nunique"),
    ).reset_index().sort_values("js_mean")
    fig, left = plt.subplots(figsize=(6.8, 3.7))
    left.plot(aggregate["js_mean"], aggregate["a1_error_rate"], marker="o", label="A1 baseline error rate", color="#377eb8")
    left.plot(aggregate["js_mean"], aggregate["a3_error_rate"], marker="s", label="A3 final error rate", color="#e41a1c")
    left.set_xlabel("JS divergence (fixed-width bin mean)")
    left.set_ylabel("Error rate")
    left.set_ylim(0.0, max(0.7, float(aggregate[["a1_error_rate", "a3_error_rate"]].to_numpy().max()) + 0.05))
    right = left.twinx()
    right.plot(aggregate["js_mean"], aggregate["disagreement_gate_mean"], marker="^", linestyle="--", label="A3 disagreement gate", color="#4daf4a")
    right.set_ylabel("Mean disagreement gate")
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8)
    left.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "js_binned_error_gate.png", dpi=220)
    fig.savefig(OUT / "js_binned_error_gate.pdf")
    plt.close(fig)
    return aggregate


def parse_latex(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.is_file():
        return {"compiled": False, "status": "NOT_RUN"}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    errors = len(re.findall(r"^! ", text, flags=re.MULTILINE))
    refs = len(re.findall(r"undefined references|Reference .* undefined", text, flags=re.IGNORECASE))
    cites = len(re.findall(r"undefined citations|Citation .* undefined", text, flags=re.IGNORECASE))
    page_match = re.search(r"Output written on .*?\((\d+) pages?", text, flags=re.DOTALL)
    return {
        "compiled": True, "status": "PASS" if errors == refs == cites == 0 else "FAIL",
        "log_sha256": sha256(log_path), "errors": errors, "undefined_references": refs,
        "undefined_citations": cites, "overfull_hbox_warnings": len(re.findall(r"Overfull \\hbox", text)),
        "overfull_vbox_warnings": len(re.findall(r"Overfull \\vbox", text)),
        "pages": int(page_match.group(1)) if page_match else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--latex-log")
    parser.add_argument("--visual-review-pass", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    expected_hashes = {
        PREDICTIONS: "22256d3bd4018ad73d00cb72192e88df9cd5ae7d4abff5dcc2059c993436bd13",
        MECHANISM: "4559bea75d7466dc2c151b6a288922a03e5421c4258ed3ed072e4d824eecf59e",
    }
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {path}")
    joined = load_joined()
    clean = joined[joined["scenario"] == "video_missing_r0"].copy()
    if len(clean) != len(SEEDS) * 2593:
        raise ValueError("clean matched row count mismatch")
    distributions, pairwise = group_distributions(clean)
    predictive = bootstrap_predictive(clean, args.bootstrap_replicates)
    strata = stratified(joined)
    finals = final_metrics(clean)
    bin_aggregate = bins_and_figure(clean)
    write_csv(distributions, "transition_group_summary.csv")
    write_csv(pairwise, "transition_pairwise_tests.csv")
    write_csv(predictive, "js_error_predictive_metrics.csv")
    write_csv(strata, "stratified_summary.csv")
    write_csv(finals, "final_output_metrics.csv")

    transition_counts = clean.groupby(["seed", "transition"], observed=True).size().unstack(fill_value=0)
    auroc = predictive[predictive["metric"] == "auroc"].set_index("seed")["estimate"]
    auprc = predictive[predictive["metric"] == "auprc"].set_index("seed")["estimate"]
    rho = predictive[predictive["metric"] == "spearman_rho"].set_index("seed")["estimate"]
    report = f"""# Step 4 bounded frozen JS-DG diagnostic

## Decision

`BOUNDED_FROZEN_DIAGNOSTIC_WITH_BRANCH_LOGIT_NO_GO`. The A1/A3 row join, JS/gate distributions, final-output diagnostics, stratification, fixed-bin figure, effect sizes, Holm correction, and dialogue-cluster bootstrap are executable from frozen artifacts. Normal-branch and robust-branch logits/hidden vectors are absent, so branch-specific margin, entropy, ECE/Brier, and branch error transfers were not computed.

## Frozen clean transition counts

| Seed | correct→error | error→correct | both correct | both error |
|---:|---:|---:|---:|---:|
"""
    for seed in SEEDS:
        row = transition_counts.loc[seed]
        report += f"| {seed} | {row.get('correct_to_error', 0)} | {row.get('error_to_correct', 0)} | {row.get('both_correct', 0)} | {row.get('both_error', 0)} |\n"
    report += "\n## JS association with A1 baseline error\n\n"
    report += "Each interval is a 2,000-replicate dialogue-cluster bootstrap conditional on one frozen checkpoint. AUPRC must be interpreted against the seed-specific baseline-error prevalence recorded in the CSV.\n\n"
    report += "| Seed | AUROC | AUPRC | Spearman rho |\n|---:|---:|---:|---:|\n"
    for seed in SEEDS:
        report += f"| {seed} | {auroc[seed]:.4f} | {auprc[seed]:.4f} | {rho[seed]:.4f} |\n"
    report += f"""

These are associations, not causal evidence that the disagreement gate detects or repairs errors. Full confidence intervals and corrected p-values are in `js_error_predictive_metrics.csv`; transition-group distributions and Holm-corrected Mann--Whitney/Cliff's-delta contrasts are in `transition_group_summary.csv` and `transition_pairwise_tests.csv`.

## Stratification and final-output scope

`stratified_summary.csv` reports n and all four transition counts for every gold emotion, pre-registered missingness/corruption scenario, seen/unseen training-speaker coverage, and fixed A1-confidence bin. `final_output_metrics.csv` compares only saved A1 and A3 final probabilities for margin, entropy, ECE, Brier, and accuracy. It does not relabel final outputs as normal/robust branch outputs.

The fixed-width JS plot is `js_binned_error_gate.png`/`.pdf`; its source is `js_bins.csv`. It shows descriptive clean-scenario error rates and gate response across {len(bin_aggregate)} populated bins, averaged over the three mechanism seeds.

## Inherited performance boundary

Step 2 remains authoritative for training-seed variability: the A1→A3 sign varies across the eight preregistered seeds for Accuracy, weighted F1, Macro F1, and F1_FDS. This three-seed mechanism diagnostic cannot upgrade that mixed-sign performance evidence or establish a general mechanism.
"""
    (OUT / "diagnostic_report.md").write_text(report, encoding="utf-8")

    manuscript = (ROOT / "paper/submission/source/main.tex").read_text(encoding="utf-8")
    manuscript_lower = manuscript.lower()
    wording_checks = {
        "main_method_compressed": "(515 inputs)" not in manuscript and "259-input" not in manuscript,
        "figure_caption_neutral": "The diagram specifies the tested mechanism but does not imply empirical benefit." in manuscript,
        "negative_result_in_results_discussion": "the sign varies across the eight preregistered seeds" in manuscript,
        "no_primary_contribution_implication": "JS-DG is an experimental diagnostic extension rather than a validated algorithmic contribution" in manuscript,
        "branch_logit_no_go_stated": "normal- and robust-branch logits were not saved" in manuscript_lower,
    }
    latex = parse_latex(Path(args.latex_log) if args.latex_log else None)
    visual = {
        "performed": bool(args.visual_review_pass),
        "status": "PASS" if args.visual_review_pass and latex.get("pages") else "NOT_RUN",
        "rendered_pages": latex.get("pages") if args.visual_review_pass else 0,
    }
    output_names = [
        "protocol.md", "diagnostic_report.md", "transition_group_summary.csv",
        "transition_pairwise_tests.csv", "js_error_predictive_metrics.csv", "stratified_summary.csv",
        "final_output_metrics.csv", "js_bins.csv", "js_binned_error_gate.png", "js_binned_error_gate.pdf",
    ]
    manifest = {
        "step": 4, "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "BOUNDED_FROZEN_DIAGNOSTIC_WITH_BRANCH_LOGIT_NO_GO",
        "input_sources": [source(PREDICTIONS), source(MECHANISM)],
        "row_contract": {"matched_all_scenarios": len(joined), "matched_clean": len(clean), "seeds": SEEDS, "scenarios": 11},
        "selection_or_training": {
            "training": False, "new_inference": False, "frozen_test_artifacts_read": True,
            "new_test_result_generated": False, "test_selection": False,
        },
        "outputs": [source(OUT / name) for name in output_names],
        "runner": source(Path(__file__).resolve()),
    }
    write_json(manifest, "diagnostic_manifest.json")
    finalized = args.finalize and all(wording_checks.values()) and latex.get("status") == "PASS" and visual["status"] == "PASS"
    gate = {
        "step": 4, "status": "complete" if finalized else "analysis_complete_manuscript_pending",
        "decision": "BOUNDED_FROZEN_DIAGNOSTIC_WITH_BRANCH_LOGIT_NO_GO",
        "route": "bounded_frozen_diagnostic_plus_method_compression",
        "training_performed": False, "new_inference_performed": False,
        "frozen_test_artifacts_read": True, "new_test_result_generated": False,
        "new_ablation_performed": False, "mechanism_seeds": SEEDS,
        "step2_performance_seeds": list(range(42, 50)),
        "supported_diagnostics": [
            "A1-to-A3 correct/error transitions", "JS and gate distributions", "JS association with A1 error",
            "dialogue-cluster bootstrap confidence intervals", "emotion/missingness/speaker-coverage/confidence strata",
            "A1/A3 final-output margin, entropy, ECE, Brier, and error transfer", "fixed-bin JS/error/gate plot",
        ],
        "no_go_diagnostics": [
            "normal-branch margin/entropy/ECE/Brier", "robust-branch margin/entropy/ECE/Brier",
            "normal-to-robust branch error transfer", "causal gate mechanism", "mechanism evidence for seeds 45-49",
        ],
        "wording_checks": wording_checks, "latex_acceptance": latex, "visual_review": visual,
        "manifest": "outputs/ieee_revision/step4_jsdg/diagnostic_manifest.json",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(gate, "gate.json")
    handoff = f"""# IEEE Access revision Step 4 handoff

## Outcome

Step 4 completed the bounded frozen diagnostic route with `{gate['decision']}`. It read only the already-frozen Step 29 test CSVs; no training, new inference, ablation, candidate selection, or new test result generation occurred. The mechanism artifacts cover seeds 42--44 only; Step 2's seeds 42--49 remain the binding performance evidence.

## Supported diagnostics

""" + "\n".join(f"- {item}" for item in gate["supported_diagnostics"]) + "\n\n## No-Go diagnostics and unproved explanations\n\n" + "\n".join(f"- {item}" for item in gate["no_go_diagnostics"]) + f"""

Associations among JS, gates, confidence, and errors do not show that JS-DG detects causal modality conflict, adapts reliability, or causes corrections/failures. The frozen corruption scenarios act on cached target-utterance features, not raw frames.

## Manuscript and supplementary inheritance

The main text retains the JS definition, normal/robust branch inputs and outputs, and final gate merge, while the 515-D concatenation, 259-D gate, temperature/floor, and secondary construction details remain in `paper/submission/supplementary/step27/method_spec.md`. Figure 3's caption is information-flow-only. Negative evidence is concentrated in Results and Discussion. JS-DG must be described as an experimental diagnostic extension, not a validated primary contribution.

Step 5 must inherit the Step 3 pooling No-Go, Step 2's exact eight-seed sign wording, and this Step 4 branch-logit No-Go. It may not infer a mechanism from the Step 4 associations.

## Evidence

- Protocol: `outputs/ieee_revision/step4_jsdg/protocol.md`
- Diagnostic report: `outputs/ieee_revision/step4_jsdg/diagnostic_report.md`
- Machine manifest: `outputs/ieee_revision/step4_jsdg/diagnostic_manifest.json`
- Gate: `outputs/ieee_revision/step4_jsdg/gate.json`
- Source predictions SHA-256: `{sha256(PREDICTIONS)}`
- Source mechanisms SHA-256: `{sha256(MECHANISM)}`
- Runner SHA-256: `{sha256(Path(__file__).resolve())}`
- LaTeX: `{latex.get('status')}`, errors `{latex.get('errors')}`, undefined references `{latex.get('undefined_references')}`, undefined citations `{latex.get('undefined_citations')}`, pages `{latex.get('pages')}`.

## Reproducible commands

```bash
cd {ROOT}
python3 scripts/run_revision_step4_jsdg_diagnostic.py --bootstrap-replicates {args.bootstrap_replicates}
mkdir -p tmp/pdfs/step4_jsdg_qa
(cd paper/submission/source && latexmk -pdf -interaction=nonstopmode -halt-on-error \\
  -outdir=../../../tmp/pdfs/step4_jsdg_qa main.tex)
pdftoppm -png -r 100 tmp/pdfs/step4_jsdg_qa/main.pdf tmp/pdfs/step4_jsdg_qa/page
# Inspect every rendered page, then finalize:
python3 scripts/run_revision_step4_jsdg_diagnostic.py --bootstrap-replicates {args.bootstrap_replicates} \\
  --latex-log tmp/pdfs/step4_jsdg_qa/main.log --visual-review-pass --finalize
```
"""
    (OUT / "handoff.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({
        "decision": gate["decision"], "status": gate["status"], "matched_rows": len(joined),
        "clean_rows": len(clean), "mechanism_seeds": SEEDS, "wording_checks": wording_checks,
    }, indent=2))
    if args.finalize and not finalized:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
