#!/usr/bin/env python3
"""IEEE revision Step 2: preregistered seed expansion for A0/A1/A3/A4/A6/A7.

Stages are hard-separated:
  smoke    synthetic metric/bootstrap/schema checks only;
  dev      import frozen 42--44, train/evaluate dev for 45--49, freeze gate;
  test     require dev gate, infer test once for 45--49, import 42--44;
  finalize hash already-written evidence without inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_phase3_step28_factorial as step28  # noqa: E402

EXPERIMENTS = ("A0", "A1", "A3", "A4", "A6", "A7")
HISTORICAL_SEEDS = (42, 43, 44)
NEW_SEEDS = (45, 46, 47, 48, 49)
ALL_SEEDS = HISTORICAL_SEEDS + NEW_SEEDS
CORRECTED_KEYS = (("A6", 44), ("A7", 44))
NEW_TRAIN_KEYS = tuple((experiment, seed) for experiment in EXPERIMENTS for seed in NEW_SEEDS) + CORRECTED_KEYS
SCENARIOS = ("full", "missing_video", "random_missing")
METRICS = ("accuracy", "weighted_f1", "macro_f1", "f1_fds")
CONTRASTS = (
    ("masking", "A0", "A1"),
    ("js_dg_matched_masking", "A1", "A3"),
    ("speaker_prototype_memory", "A0", "A4"),
    ("s05", "A6", "A7"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(ROOT.resolve()))
    except ValueError:
        return str(value)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def verify_manifest(manifest: Mapping[str, Any], sections: Sequence[str]) -> None:
    for section in sections:
        for path, expected in manifest[section].items():
            actual = sha256(resolve(path))
            if actual != expected:
                raise ValueError(f"hash mismatch: {path}: {actual} != {expected}")


def normalize_metric(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    result["seed"] = int(result["seed"])
    if "minority_f1" in result and "f1_fds" not in result:
        result["f1_fds"] = result.pop("minority_f1")
    for metric in METRICS:
        result[metric] = float(result[metric])
    return result


def normalize_per_class(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    result["seed"] = int(result["seed"])
    result["support"] = int(result["support"])
    for metric in ("precision", "recall", "f1"):
        result[metric] = float(result[metric])
    return result


def normalize_prediction(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    result["seed"] = int(result["seed"])
    result["sample_index"] = int(result["sample_index"])
    result["gold"] = int(result["gold"])
    result["prediction"] = int(result["prediction"])
    value = str(result.get("video_masked", "")).lower()
    result["video_masked"] = value in ("true", "1")
    for label in step28.LABELS:
        result[f"prob_{label}"] = float(result[f"prob_{label}"])
    return result


def historical_rows(cfg: Mapping[str, Any], kind: str) -> List[Dict[str, Any]]:
    base = resolve(cfg["historical_dir"])
    filename = {
        "dev_metrics": "confirmation_dev_metrics.csv",
        "dev_per_class": "confirmation_dev_per_class.csv",
        "dev_predictions": "confirmation_dev_predictions.csv",
        "test_metrics": "confirmation_test_metrics.csv",
        "test_per_class": "confirmation_test_per_class.csv",
        "test_predictions": "confirmation_test_predictions.csv",
    }[kind]
    rows = read_csv(base / filename)
    selected = [
        row for row in rows
        if row["experiment"] in EXPERIMENTS
        and int(row["seed"]) in HISTORICAL_SEEDS
        and (row["experiment"], int(row["seed"])) not in CORRECTED_KEYS
    ]
    if kind.endswith("metrics"):
        return [normalize_metric(row) for row in selected]
    if kind.endswith("per_class"):
        return [normalize_per_class(row) for row in selected]
    return [normalize_prediction(row) for row in selected]


def historical_runs(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    base = resolve(cfg["historical_dir"])
    rows = read_csv(base / "checkpoint_index.csv")
    selected: List[Dict[str, Any]] = []
    for row in rows:
        key = (row["experiment"], int(row["seed"]))
        if row["experiment"] not in EXPERIMENTS or int(row["seed"]) not in HISTORICAL_SEEDS or key in CORRECTED_KEYS:
            continue
        item: Dict[str, Any] = dict(row)
        item["seed"] = int(item["seed"])
        item["success"] = str(item["success"]).lower() == "true"
        item["source"] = "historical_step28"
        selected.append(item)
    expected = {(experiment, seed) for experiment in EXPERIMENTS for seed in HISTORICAL_SEEDS} - set(CORRECTED_KEYS)
    actual = {(row["experiment"], row["seed"]) for row in selected}
    if actual != expected:
        raise ValueError(f"historical checkpoint set mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    for row in selected:
        if not row["success"]:
            raise ValueError(f"historical run was unsuccessful: {row['experiment']} seed {row['seed']}")
        for field, hash_field in (("config", "config_sha256"), ("checkpoint", "checkpoint_sha256")):
            if sha256(resolve(row[field])) != row[hash_field]:
                raise ValueError(f"historical {field} changed: {row['experiment']} seed {row['seed']}")
    return selected


def step28_args(cfg: Mapping[str, Any], smoke: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        feature_dir=cfg["feature_dir"], structure_config=cfg["structure_config"],
        augmentation_bundle=cfg["augmentation_bundle"], epochs=1 if smoke else int(cfg["epochs"]),
        batch_size=min(8, int(cfg["batch_size"])) if smoke else int(cfg["batch_size"]),
        hidden_dim=64 if smoke else int(cfg["hidden_dim"]), num_heads=int(cfg["num_heads"]),
        dropout=float(cfg["dropout"]), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]), patience=1 if smoke else int(cfg["patience"]),
        smoke_test=smoke, screen_seed=42,
    )


def train_new(experiment: str, seed: int, args: argparse.Namespace, out: Path) -> Dict[str, Any]:
    if experiment == "A7":
        source = out / "speaker_prototypes" / f"A6_seed{seed}_slots4.npz"
        target = out / "speaker_prototypes" / f"A7_seed{seed}_slots4.npz"
        if not source.is_file():
            raise FileNotFoundError(f"A7 requires frozen A6 prototype first: {source}")
        shutil.copyfile(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"shared A6/A7 prototype copy mismatch for seed {seed}")
    run = step28.train_row(experiment, seed, args, out)
    if experiment == "A7":
        source = out / "speaker_prototypes" / f"A6_seed{seed}_slots4.npz"
        target = out / "speaker_prototypes" / f"A7_seed{seed}_slots4.npz"
        if sha256(source) != sha256(target):
            raise RuntimeError(f"A7 modified its shared prototype for seed {seed}")
    return run


def evaluate_new(run: Mapping[str, Any], args: argparse.Namespace, split: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        metric, per_class, samples = step28.evaluate(run, args, split, scenario)
        metrics.append(normalize_metric(metric))
        classes.extend(normalize_per_class(row) for row in per_class)
        predictions.extend(normalize_prediction(row) for row in samples)
    return metrics, classes, predictions


def run_fields() -> List[str]:
    return ["experiment", "seed", "source", "success", "config", "config_sha256", "checkpoint", "checkpoint_sha256", "log", "train_seconds", "command"]


def metric_fields() -> List[str]:
    return ["experiment", "seed", "split", "scenario", *METRICS]


def class_fields() -> List[str]:
    return ["experiment", "seed", "split", "scenario", "label", "precision", "recall", "f1", "support"]


def prediction_fields() -> List[str]:
    return ["experiment", "seed", "split", "scenario", "sample_index", "dialogue_id", "utterance_id", "speaker", "gold", "prediction", "gold_label", "prediction_label", "video_masked", *[f"prob_{label}" for label in step28.LABELS]]


def journal(out: Path, runs: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> None:
    write_csv(out / "checkpoint_index.csv", runs, run_fields())
    write_csv(out / "failure_index.csv", failures, ["time", "experiment", "seed", "stage", "error_type", "message"])


def validate_group_counts(rows: Sequence[Mapping[str, Any]], split: str, cfg: Mapping[str, Any]) -> bool:
    counts: Dict[Tuple[str, int, str], int] = defaultdict(int)
    identities: Dict[Tuple[str, int, str], set] = defaultdict(set)
    probability_ok = True
    for row in rows:
        key = (str(row["experiment"]), int(row["seed"]), str(row["scenario"]))
        counts[key] += 1
        identities[key].add((int(row["sample_index"]), str(row["dialogue_id"]), str(row["utterance_id"]), str(row["speaker"]), int(row["gold"])))
        probabilities = np.asarray([float(row[f"prob_{label}"]) for label in step28.LABELS])
        probability_ok = probability_ok and bool(np.isfinite(probabilities).all() and np.isclose(probabilities.sum(), 1.0, atol=2e-6))
    expected_keys = {(experiment, seed, scenario) for experiment in EXPERIMENTS for seed in ALL_SEEDS for scenario in SCENARIOS}
    expected_rows = int(cfg["split_rows"][split])
    if set(counts) != expected_keys or not probability_ok:
        return False
    reference = identities[next(iter(expected_keys))]
    return all(counts[key] == expected_rows and len(identities[key]) == expected_rows and identities[key] == reference for key in expected_keys)


def dev_stage(cfg: Mapping[str, Any], out: Path) -> None:
    gate = read_json(out / "initial_gate.json")
    if gate.get("decision") != "GO_STEP2_DEV_ONLY" or gate.get("route") != "seed_expansion" or gate.get("test_evaluation_unlocked") is not False:
        raise ValueError("initial gate does not authorize frozen Step 2 dev stage")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Step 2 dev training requires a Slurm CUDA allocation")
    manifest = read_json(out / "pre_run_manifest.json")
    verify_manifest(manifest, ("train_dev_inputs", "historical_dev_inputs", "code_and_protocol"))
    checkpoint_path = out / "checkpoint_index.csv"
    if checkpoint_path.is_file() and any(row.get("source") == "step2_new" for row in read_csv(checkpoint_path)):
        raise RuntimeError("partial/new Step 2 runs already exist; preserve them and perform an audited common-cause recovery instead of silently rerunning")
    runs = historical_runs(cfg)
    failures: List[Dict[str, Any]] = []
    failure_path = out / "failure_index.csv"
    if failure_path.is_file():
        failures.extend(read_csv(failure_path))
    journal(out, runs, failures)
    args = step28_args(cfg)
    for experiment in EXPERIMENTS:
        seeds_to_train = (44, *NEW_SEEDS) if experiment in ("A6", "A7") else NEW_SEEDS
        for seed in seeds_to_train:
            try:
                run = train_new(experiment, seed, args, out)
                run["seed"] = int(run["seed"])
                run["source"] = "step2_new"
                if not run.get("success"):
                    raise RuntimeError(f"training returned unsuccessful run: {run}")
                run["config_sha256"] = sha256(resolve(run["config"]))
                run["checkpoint_sha256"] = sha256(resolve(run["checkpoint"]))
                runs.append(run)
                journal(out, runs, failures)
            except Exception as exc:
                failures.append({"time": utc_now(), "experiment": experiment, "seed": seed, "stage": "dev_train", "error_type": type(exc).__name__, "message": str(exc)})
                journal(out, runs, failures)
                raise
    expected = {(experiment, seed) for experiment in EXPERIMENTS for seed in ALL_SEEDS}
    actual = {(row["experiment"], int(row["seed"])) for row in runs}
    if actual != expected:
        raise ValueError(f"checkpoint set incomplete: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    metrics = historical_rows(cfg, "dev_metrics")
    classes = historical_rows(cfg, "dev_per_class")
    predictions = historical_rows(cfg, "dev_predictions")
    for run in runs:
        if (run["experiment"], int(run["seed"])) not in NEW_TRAIN_KEYS:
            continue
        run_metrics, run_classes, run_predictions = evaluate_new(run, args, "dev")
        metrics.extend(run_metrics); classes.extend(run_classes); predictions.extend(run_predictions)
    write_csv(out / "dev_metrics.csv", metrics, metric_fields())
    write_csv(out / "dev_per_class.csv", classes, class_fields())
    write_csv(out / "dev_predictions.csv", predictions, prediction_fields())
    identity_ok = validate_group_counts(predictions, "dev", cfg)
    dev_gate = {
        "step": 2, "stage": "dev", "status": "complete",
        "decision": "GO_FROZEN_STEP2_TEST" if identity_ok and actual == expected else "NO_GO_STEP2_TEST",
        "route": "seed_expansion", "route_change_allowed": False,
        "experiments": list(EXPERIMENTS), "seeds": list(ALL_SEEDS),
        "historical_model_seed_pairs_imported": 16, "historical_pairs_excluded": ["A6:44", "A7:44"],
        "new_seeds_trained": list(NEW_SEEDS), "corrected_retrain_pairs": ["A6:44", "A7:44"],
        "selection_split": "dev", "selection_metric": "weighted_f1",
        "checkpoint_set_complete": actual == expected, "dev_prediction_groups_complete": identity_ok,
        "failed_attempts_recorded": len(failures),
        "all_failed_attempts_retained": True,
        "failed_model_seed_pairs_missing": [f"{experiment}:{seed}" for experiment, seed in sorted(expected - actual)],
        "test_evaluated": False,
        "test_evaluation_unlocked": identity_ok and actual == expected,
        "candidate_reselection_allowed": False, "created_at": utc_now(),
    }
    write_json(out / "dev_selection_gate.json", dev_gate)
    print(json.dumps(dev_gate, indent=2))


def load_runs(out: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in read_csv(out / "checkpoint_index.csv"):
        item: Dict[str, Any] = dict(row)
        item["seed"] = int(item["seed"])
        item["success"] = str(item["success"]).lower() == "true"
        rows.append(item)
    return rows


def summarize(metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        groups[(str(row["experiment"]), str(row["split"]), str(row["scenario"]))].append(row)
    result: List[Dict[str, Any]] = []
    for key in sorted(groups):
        rows = groups[key]
        item: Dict[str, Any] = {"experiment": key[0], "split": key[1], "scenario": key[2], "seed_count": len(rows)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            item.update({f"{metric}_mean": float(values.mean()), f"{metric}_sd": float(values.std(ddof=1)), f"{metric}_median": float(np.median(values)), f"{metric}_min": float(values.min()), f"{metric}_max": float(values.max())})
        result.append(item)
    return result


def contrasts(metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = {(str(row["experiment"]), int(row["seed"]), str(row["split"]), str(row["scenario"])): row for row in metrics}
    result: List[Dict[str, Any]] = []
    for factor, control, treatment in CONTRASTS:
        for split in ("dev", "test"):
            for scenario in SCENARIOS:
                for seed in ALL_SEEDS:
                    c = keyed[(control, seed, split, scenario)]; t = keyed[(treatment, seed, split, scenario)]
                    result.append({"factor": factor, "control": control, "treatment": treatment, "seed": seed, "split": split, "scenario": scenario, **{f"delta_{metric}": float(t[metric]) - float(c[metric]) for metric in METRICS}})
    return result


def metric_from_confusion(confusion: np.ndarray) -> Dict[str, float]:
    support = confusion.sum(axis=1); predicted = confusion.sum(axis=0); tp = np.diag(confusion)
    denom = 2 * tp + predicted - tp + support - tp
    f1 = np.divide(2 * tp, denom, out=np.zeros(7, dtype=float), where=denom != 0)
    total = support.sum()
    return {"accuracy": float(tp.sum() / total), "weighted_f1": float(np.dot(f1, support) / total), "macro_f1": float(f1.mean()), "f1_fds": float(f1[[2, 5, 3]].mean())}


def paired_bootstrap(predictions: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row["split"] == "test":
            groups[(str(row["experiment"]), int(row["seed"]), str(row["scenario"]))].append(row)
    replicates = int(cfg["bootstrap"]["replicates"]); base_seed = int(cfg["bootstrap"]["seed"])
    output: List[Dict[str, Any]] = []
    for contrast_index, (factor, control, treatment) in enumerate(CONTRASTS):
        for seed in ALL_SEEDS:
            c = sorted(groups[(control, seed, "full")], key=lambda row: int(row["sample_index"]))
            t = sorted(groups[(treatment, seed, "full")], key=lambda row: int(row["sample_index"]))
            c_ids = [(row["sample_index"], row["dialogue_id"], row["utterance_id"], row["speaker"], row["gold"]) for row in c]
            t_ids = [(row["sample_index"], row["dialogue_id"], row["utterance_id"], row["speaker"], row["gold"]) for row in t]
            if c_ids != t_ids:
                raise ValueError(f"paired bootstrap identity mismatch: {factor} seed {seed}")
            gold = np.asarray([int(row["gold"]) for row in c]); cp = np.asarray([int(row["prediction"]) for row in c]); tp = np.asarray([int(row["prediction"]) for row in t])
            codes = gold * 49 + cp * 7 + tp
            counts = np.bincount(codes, minlength=343); probabilities = counts / counts.sum()
            rng = np.random.default_rng(base_seed + contrast_index * 100 + seed)
            values: Dict[str, List[float]] = {metric: [] for metric in METRICS}
            for sampled in rng.multinomial(len(gold), probabilities, size=replicates):
                cube = sampled.reshape(7, 7, 7)
                cm_c = cube.sum(axis=2); cm_t = cube.sum(axis=1)
                mc = metric_from_confusion(cm_c); mt = metric_from_confusion(cm_t)
                for metric in METRICS:
                    values[metric].append(mt[metric] - mc[metric])
            row: Dict[str, Any] = {"factor": factor, "control": control, "treatment": treatment, "seed": seed, "scenario": "full", "unit": "utterance", "replicates": replicates, "bootstrap_seed": base_seed + contrast_index * 100 + seed, "interpretation": "conditional_on_frozen_seed_checkpoint_not_training_variability"}
            for metric in METRICS:
                array = np.asarray(values[metric]); row[f"delta_{metric}_ci_low"] = float(np.quantile(array, 0.025)); row[f"delta_{metric}_ci_high"] = float(np.quantile(array, 0.975))
            output.append(row)
    return output


def test_stage(cfg: Mapping[str, Any], out: Path) -> None:
    if (out / "result_gate.json").is_file() or (out / "test_predictions.csv").is_file():
        raise RuntimeError("Step 2 test artifacts already exist; repeated test inference is forbidden")
    gate_path = out / "dev_selection_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("test locked: dev_selection_gate.json is absent")
    gate = read_json(gate_path)
    if gate.get("decision") != "GO_FROZEN_STEP2_TEST" or gate.get("test_evaluation_unlocked") is not True or gate.get("test_evaluated") is not False:
        raise ValueError("test remains locked by dev selection gate")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Step 2 test requires a Slurm CUDA allocation")
    manifest = read_json(out / "pre_run_manifest.json")
    verify_manifest(manifest, ("test_inputs", "historical_test_inputs", "code_and_protocol"))
    runs = load_runs(out)
    expected = {(experiment, seed) for experiment in EXPERIMENTS for seed in ALL_SEEDS}
    actual = {(row["experiment"], int(row["seed"])) for row in runs}
    if actual != expected:
        raise ValueError(f"frozen checkpoint set changed: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    for run in runs:
        if sha256(resolve(run["config"])) != run["config_sha256"] or sha256(resolve(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise ValueError(f"frozen artifact changed: {run['experiment']} seed {run['seed']}")
    args = step28_args(cfg)
    test_metrics = historical_rows(cfg, "test_metrics")
    test_classes = historical_rows(cfg, "test_per_class")
    test_predictions = historical_rows(cfg, "test_predictions")
    for run in runs:
        if (run["experiment"], int(run["seed"])) not in NEW_TRAIN_KEYS:
            continue
        metrics, classes, predictions = evaluate_new(run, args, "test")
        test_metrics.extend(metrics); test_classes.extend(classes); test_predictions.extend(predictions)
    if not validate_group_counts(test_predictions, "test", cfg):
        raise ValueError("test prediction group or identity completeness failed")
    write_csv(out / "test_metrics.csv", test_metrics, metric_fields())
    write_csv(out / "test_per_class.csv", test_classes, class_fields())
    write_csv(out / "test_predictions.csv", test_predictions, prediction_fields())
    dev_metrics = [normalize_metric(row) for row in read_csv(out / "dev_metrics.csv")]
    all_metrics = dev_metrics + test_metrics
    write_csv(out / "per_seed_metrics.csv", all_metrics, metric_fields())
    summary = summarize(all_metrics)
    summary_fields = ["experiment", "split", "scenario", "seed_count", *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "sd", "median", "min", "max")]]
    write_csv(out / "summary_metrics.csv", summary, summary_fields)
    delta_rows = contrasts(all_metrics)
    write_csv(out / "matched_contrasts.csv", delta_rows, ["factor", "control", "treatment", "seed", "split", "scenario", *[f"delta_{metric}" for metric in METRICS]])
    bootstrap_rows = paired_bootstrap(test_predictions, cfg)
    bootstrap_fields = ["factor", "control", "treatment", "seed", "scenario", "unit", "replicates", "bootstrap_seed", "interpretation", *[f"delta_{metric}_{bound}" for metric in METRICS for bound in ("ci_low", "ci_high")]]
    write_csv(out / "utterance_bootstrap.csv", bootstrap_rows, bootstrap_fields)
    result = {"step": 2, "stage": "test", "status": "complete", "decision": "STEP2_RESULTS_READY_FOR_LOCAL_AUDIT", "route": "seed_expansion", "route_changed": False, "experiments": list(EXPERIMENTS), "seeds": list(ALL_SEEDS), "historical_test_inference_repeated": False, "new_test_inference_passes_per_checkpoint_scenario": 1, "test_used_for_selection": False, "training_seed_variability_file": relative(out / "summary_metrics.csv"), "utterance_bootstrap_file": relative(out / "utterance_bootstrap.csv"), "uncertainty_types_kept_separate": True, "created_at": utc_now()}
    write_json(out / "result_gate.json", result)
    finalize(out)
    print(json.dumps(result, indent=2))


def finalize(out: Path) -> None:
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "final_manifest.json":
            files.append({"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(out / "final_manifest.json", {"step": 2, "created_at": utc_now(), "manual_numeric_edits": False, "files": files})


def smoke_stage(cfg: Mapping[str, Any]) -> None:
    rng = np.random.default_rng(20260812)
    gold = np.arange(28) % 7; control = gold.copy(); treatment = gold.copy()
    control[rng.choice(len(gold), 8, replace=False)] = rng.integers(0, 7, 8)
    treatment[rng.choice(len(gold), 6, replace=False)] = rng.integers(0, 7, 6)
    cm = np.zeros((7, 7), dtype=int)
    np.add.at(cm, (gold, control), 1)
    metrics = metric_from_confusion(cm)
    assert set(metrics) == set(METRICS) and all(math.isfinite(value) for value in metrics.values())
    assert tuple(cfg["seeds"]) == ALL_SEEDS and tuple(cfg["new_seeds"]) == NEW_SEEDS
    assert tuple(cfg["experiments"]) == EXPERIMENTS
    print(json.dumps({"status": "PASS", "synthetic_only": True, "dataset_files_opened": False, "route": cfg["route"], "experiments": list(EXPERIMENTS), "seeds": list(ALL_SEEDS), "formal_training_runs": len(NEW_TRAIN_KEYS), "corrected_retrain_keys": [f"{experiment}:{seed}" for experiment, seed in CORRECTED_KEYS], "metric_schema": list(metrics), "prediction_schema": prediction_fields()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "dev", "test", "finalize"), required=True)
    parser.add_argument("--config", default="outputs/ieee_revision/step2_seeds/config.json")
    parser.add_argument("--output-dir", default="outputs/ieee_revision/step2_seeds")
    args = parser.parse_args()
    cfg = read_json(resolve(args.config)); out = resolve(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # The formal trainer writes these paths directly.  Create them here so a
    # clean Step 2 output tree cannot fail before the first training epoch.
    for child in ("logs", "runs", "runtime", "slurm", "speaker_prototypes"):
        (out / child).mkdir(parents=True, exist_ok=True)
    if cfg.get("route") != "seed_expansion" or tuple(cfg["experiments"]) != EXPERIMENTS or tuple(cfg["seeds"]) != ALL_SEEDS:
        raise ValueError("Step 2 route/experiment/seed preregistration changed")
    if args.stage == "smoke": smoke_stage(cfg)
    elif args.stage == "dev": dev_stage(cfg, out)
    elif args.stage == "test": test_stage(cfg, out)
    else: finalize(out)


if __name__ == "__main__":
    main()
