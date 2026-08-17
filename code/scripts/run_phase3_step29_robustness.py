#!/usr/bin/env python3
"""Phase III Step 29 frozen-checkpoint robustness stress evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_phase3_step28_factorial import LABELS, MINORITY, dataset, eval_args, sha256
from training.train import build_model

STEP28_REL = Path("outputs/phase3_ieee_access/step28_factorial_ablation")
STEP29_REL = Path("outputs/phase3_ieee_access/step29_robustness_stress")
METRICS = ("accuracy", "weighted_f1", "macro_f1", "minority_f1")
MECHANISMS = (
    "normal_gate", "disagreement_gate", "js_divergence", "kl_text_video",
    "kl_video_text", "projection_cosine", "projection_video_norm", "projection_text_norm",
)
PRIMARY_ROWS = ("A1", "A3", "A6", "A7")
SEEDS = (42, 43, 44)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(ROOT.resolve()))
    except ValueError:
        return str(value)


def map_step28_path(path: str | Path, step28: Path) -> Path:
    value = Path(path)
    if value.exists():
        return value
    marker = "outputs/phase3_ieee_access/step28_factorial_ablation/"
    text = str(value)
    if marker in text:
        return step28 / text.split(marker, 1)[1]
    return resolve(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def scenario_specs(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in config["scenarios"]]


def validate_pre_run_manifest(out: Path) -> Dict[str, Any]:
    manifest = json.loads((out / "pre_run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_before_formal_run" or manifest.get("formal_run_performed") is not False:
        raise ValueError("invalid Step 29 pre-run manifest state")
    for section in ("upstream", "data", "protocol_and_gate", "code", "corruption_manifests"):
        for name, expected in manifest.get(section, {}).items():
            path = resolve(name)
            if not path.is_file():
                raise FileNotFoundError(f"pre-run manifest input missing: {name}")
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"pre-run manifest hash mismatch for {name}: {actual} != {expected}")
    return manifest


def validate_protocol_gate(out: Path, step28: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    if gate.get("decision") != "GO_STEP29_FROZEN_EVALUATION":
        raise ValueError("Step 29 protocol gate does not authorize frozen evaluation")
    if gate.get("training_unlocked") is not False or gate.get("evaluation_unlocked") is not True:
        raise ValueError("Step 29 must remain evaluation-only")
    if gate.get("candidate_reselection_allowed") is not False or gate.get("test_curve_tuning_allowed") is not False:
        raise ValueError("test reselection/tuning must be disabled")
    upstream = json.loads((step28 / "attribution_gate.json").read_text(encoding="utf-8"))
    required = gate["upstream_gate"]["required_decision"]
    if upstream.get("decision") != required:
        raise ValueError(f"unexpected Step 28 decision: {upstream.get('decision')} != {required}")
    if sha256(step28 / "attribution_gate.json") != gate["upstream_gate"]["sha256"]:
        raise ValueError("Step 28 attribution gate hash mismatch")
    if sha256(step28 / "checkpoint_index.csv") != gate["checkpoint_index"]["sha256"]:
        raise ValueError("Step 28 checkpoint index hash mismatch")
    for name, expected in gate["input_hashes"].items():
        actual = sha256(resolve(name))
        if actual != expected:
            raise ValueError(f"input hash mismatch for {name}: {actual} != {expected}")
    if config.get("training_allowed") is not False or config.get("candidate_reselection_allowed") is not False:
        raise ValueError("config violates evaluation-only boundary")
    return gate


def load_frozen_runs(step28: Path) -> List[Dict[str, Any]]:
    rows = read_csv(step28 / "checkpoint_index.csv")
    selected: List[Dict[str, Any]] = []
    expected = {(experiment, seed) for experiment in PRIMARY_ROWS for seed in SEEDS}
    for row in rows:
        key = (row["experiment"], int(row["seed"]))
        if key not in expected:
            continue
        checkpoint = map_step28_path(row["checkpoint"], step28)
        config = map_step28_path(row["config"], step28)
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError(f"missing frozen artifact: {checkpoint} / {config}")
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch for {key}")
        if sha256(config) != row["config_sha256"]:
            raise ValueError(f"config hash mismatch for {key}")
        selected.append({
            "experiment": row["experiment"], "seed": int(row["seed"]),
            "checkpoint": checkpoint, "checkpoint_sha256": row["checkpoint_sha256"],
            "config": config, "config_sha256": row["config_sha256"],
        })
    actual = {(row["experiment"], row["seed"]) for row in selected}
    if actual != expected:
        raise ValueError(f"frozen checkpoint set mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    return sorted(selected, key=lambda row: (row["experiment"], row["seed"]))


def prepare_cfg(run: Mapping[str, Any], step28: Path) -> SimpleNamespace:
    cfg = eval_args(Path(run["config"]))
    if cfg.speaker_prototype_path:
        cfg.speaker_prototype_path = str(map_step28_path(cfg.speaker_prototype_path, step28))
    return cfg


def build_manifests(
    out: Path, feature_dir: Path, seeds: Sequence[int], sample_count: int, feat_dim: int,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray, Dict[str, Any]]:
    manifest_dir = out / "mask_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    per_seed: Dict[int, Dict[str, np.ndarray]] = {}
    file_hashes: Dict[str, str] = {}
    missing_rates = (0.0, 0.25, 0.5, 0.75, 1.0)
    for seed in seeds:
        missing_rng = np.random.default_rng(290000 + seed)
        permutation = missing_rng.permutation(sample_count)
        rank = np.empty(sample_count, dtype=np.int64)
        rank[permutation] = np.arange(sample_count, dtype=np.int64)
        block_rng = np.random.default_rng(291000 + seed)
        block_start = block_rng.integers(0, feat_dim, size=sample_count, dtype=np.int64)
        masks = {rate: rank < int(round(rate * sample_count)) for rate in missing_rates}
        rows = []
        for idx in range(sample_count):
            rows.append({
                "sample_index": idx, "missing_rank": int(rank[idx]),
                "mask_r0": int(masks[0.0][idx]), "mask_r025": int(masks[0.25][idx]),
                "mask_r050": int(masks[0.5][idx]), "mask_r075": int(masks[0.75][idx]),
                "mask_r100": int(masks[1.0][idx]), "block_start": int(block_start[idx]),
                "gaussian_noise_seed": 292000 + seed,
            })
        path = manifest_dir / f"seed_{seed}_manifest.csv"
        write_csv(path, rows, list(rows[0]))
        file_hashes[rel(path)] = sha256(path)
        per_seed[seed] = {"rank": rank, "block_start": block_start}
    train_video_path = feature_dir / "train_video_features.npy"
    train_video = np.load(train_video_path, mmap_mode="r")
    if train_video.ndim != 2 or train_video.shape[1] != feat_dim:
        raise ValueError(f"unexpected train video shape: {train_video.shape}")
    train_std = np.std(train_video, axis=0, dtype=np.float64).astype(np.float32)
    if not np.isfinite(train_std).all() or np.any(train_std < 0):
        raise ValueError("invalid train-derived noise scale")
    scale_path = manifest_dir / "train_video_feature_std.npy"
    np.save(scale_path, train_std)
    file_hashes[rel(scale_path)] = sha256(scale_path)
    summary = {
        "algorithm": "numpy.random.Generator(PCG64)",
        "sample_count": sample_count, "feature_dim": feat_dim,
        "missing_seed_offset": 290000, "block_seed_offset": 291000,
        "noise_seed_offset": 292000, "nested_missing_masks": True,
        "train_scale_source": rel(train_video_path),
        "train_scale_source_sha256": sha256(train_video_path),
        "train_std_sha256": sha256(scale_path),
        "train_std_min": float(train_std.min()), "train_std_mean": float(train_std.mean()),
        "train_std_median": float(np.median(train_std)), "train_std_max": float(train_std.max()),
        "manifest_hashes": file_hashes,
    }
    write_json(manifest_dir / "manifest.json", summary)
    return per_seed, train_std, summary


def noise_for_seed(seed: int, sample_count: int, feat_dim: int) -> np.ndarray:
    rng = np.random.default_rng(292000 + seed)
    return rng.standard_normal((sample_count, feat_dim), dtype=np.float32)


def apply_corruption(
    video: torch.Tensor, text: torch.Tensor, indices: np.ndarray, scenario: Mapping[str, Any],
    seed_manifest: Mapping[str, np.ndarray], train_std: torch.Tensor, noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    family = str(scenario["family"])
    severity = float(scenario["severity"])
    corrupted_video = video.clone()
    corrupted_text = text.clone()
    device = video.device
    idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    row_corrupted = torch.zeros(len(indices), dtype=torch.bool, device=device)
    feature_fraction = torch.zeros(len(indices), dtype=torch.float32, device=device)
    if family == "video_missing":
        count = int(round(severity * len(seed_manifest["rank"])))
        selected = seed_manifest["rank"][indices] < count
        row_corrupted = torch.as_tensor(selected, dtype=torch.bool, device=device)
        corrupted_video[row_corrupted] = 0.0
        feature_fraction[row_corrupted] = 1.0
    elif family == "gaussian_noise":
        corrupted_video = corrupted_video + severity * train_std.unsqueeze(0) * noise.index_select(0, idx_tensor)
        row_corrupted[:] = True
        feature_fraction[:] = 1.0
    elif family == "block_occlusion":
        block_len = int(round(severity * video.shape[1]))
        starts = torch.as_tensor(seed_manifest["block_start"][indices], dtype=torch.long, device=device)
        dims = torch.arange(video.shape[1], device=device).unsqueeze(0)
        block_mask = torch.remainder(dims - starts.unsqueeze(1), video.shape[1]) < block_len
        corrupted_video = torch.where(block_mask, torch.zeros_like(corrupted_video), corrupted_video)
        row_corrupted[:] = True
        feature_fraction[:] = block_len / video.shape[1]
    elif family == "text_missing_diagnostic":
        corrupted_text.zero_()
    else:
        raise ValueError(f"unknown corruption family: {family}")
    return corrupted_video, corrupted_text, row_corrupted, feature_fraction


def vector_from_output(output: Mapping[str, Any], key: str, batch_size: int) -> np.ndarray:
    value = output.get(key)
    if value is None and isinstance(output.get("attention_weights"), Mapping):
        attention = output["attention_weights"]
        aliases = {"normal_gate": ("projection_gate", "gate")}
        for candidate in aliases.get(key, (key,)):
            if candidate in attention:
                value = attention[candidate]
                break
    if value is None:
        return np.full(batch_size, np.nan, dtype=np.float64)
    array = value.detach().float().cpu().numpy()
    if array.ndim > 1:
        array = array.reshape(array.shape[0], -1).mean(axis=1)
    return array.astype(np.float64)


def metric_rows(
    experiment: str, seed: int, scenario: Mapping[str, Any], gold: Sequence[int], pred: Sequence[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=range(len(LABELS)), zero_division=0,
    )
    per_class = [{
        "experiment": experiment, "seed": seed, "split": "test", "scenario": scenario["name"],
        "family": scenario["family"], "severity": scenario["severity"], "label": label,
        "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
        "support": int(support[i]),
    } for i, label in enumerate(LABELS)]
    metric = {
        "experiment": experiment, "seed": seed, "split": "test", "scenario": scenario["name"],
        "family": scenario["family"], "severity": scenario["severity"],
        "accuracy": float(accuracy_score(gold, pred)),
        "weighted_f1": float(f1_score(gold, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "minority_f1": float(np.mean([row["f1"] for row in per_class if row["label"] in MINORITY])),
    }
    return metric, per_class


def clean_reference(step28: Path) -> Dict[Tuple[str, int, int], int]:
    rows = read_csv(step28 / "confirmation_test_predictions.csv")
    return {
        (row["experiment"], int(row["seed"]), int(row["sample_index"])): int(row["prediction"])
        for row in rows if row["experiment"] in PRIMARY_ROWS and row["scenario"] == "full"
    }


@torch.no_grad()
def evaluate_run(
    run: Mapping[str, Any], args: argparse.Namespace, step28: Path, specs: Sequence[Mapping[str, Any]],
    manifests: Mapping[int, Mapping[str, np.ndarray]], train_std_np: np.ndarray,
    clean_predictions: Mapping[Tuple[str, int, int], int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    cfg = prepare_cfg(run, step28)
    data = dataset(resolve(args.feature_dir), "test", cfg)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(Path(run["checkpoint"]), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    train_std = torch.as_tensor(train_std_np, dtype=torch.float32, device=device)
    noise_np = noise_for_seed(run["seed"], len(data), cfg.feat_dim)
    noise = torch.as_tensor(noise_np, dtype=torch.float32, device=device)
    all_metrics: List[Dict[str, Any]] = []
    all_classes: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    all_mechanisms: List[Dict[str, Any]] = []
    clean_mismatch = 0
    for scenario in specs:
        gold: List[int] = []
        pred: List[int] = []
        for batch in loader:
            indices = batch["idx"].numpy().astype(int)
            raw_video = batch["video_feat"].to(device)
            raw_text = batch["text_feat"].to(device)
            video, text, row_corrupted, fraction = apply_corruption(
                raw_video, raw_text, indices, scenario, manifests[run["seed"]], train_std, noise,
            )
            output = model(
                video_feat=video, text_feat=text, return_attention=True,
                context_video_feat=batch["context_video_feat"].to(device),
                context_text_feat=batch["context_text_feat"].to(device),
                context_mask=batch["context_mask"].to(device),
                context_same_speaker=batch["context_same_speaker"].to(device),
                context_turn_distance=batch["context_turn_distance"].to(device),
                speaker_memory_video_feat=batch["speaker_memory_video_feat"].to(device),
                speaker_memory_text_feat=batch["speaker_memory_text_feat"].to(device),
                speaker_memory_mask=batch["speaker_memory_mask"].to(device),
            )
            probs = torch.softmax(output["logits"], dim=-1).cpu().numpy()
            batch_pred = probs.argmax(axis=1).astype(int)
            batch_gold = batch["label"].numpy().astype(int)
            text_probs = output.get("text_probs")
            video_probs = output.get("video_probs")
            text_pred = text_probs.argmax(dim=-1).cpu().numpy().astype(int) if text_probs is not None else None
            video_pred = video_probs.argmax(dim=-1).cpu().numpy().astype(int) if video_probs is not None else None
            mechanism_values = {key: vector_from_output(output, key, len(indices)) for key in MECHANISMS}
            raw_norm = raw_video.norm(dim=-1).cpu().numpy()
            corrupted_norm = video.norm(dim=-1).cpu().numpy()
            for pos, idx in enumerate(indices):
                row = {
                    "experiment": run["experiment"], "seed": run["seed"], "split": "test",
                    "scenario": scenario["name"], "family": scenario["family"],
                    "severity": scenario["severity"], "sample_index": int(idx),
                    "dialogue_id": str(data.dialogue_ids[idx]) if data.dialogue_ids is not None else "",
                    "utterance_id": str(data.utterance_ids[idx]) if data.utterance_ids is not None else "",
                    "speaker": str(data.speaker_ids[idx]) if data.speaker_ids is not None else "",
                    "gold": int(batch_gold[pos]), "prediction": int(batch_pred[pos]),
                    "gold_label": LABELS[int(batch_gold[pos])], "prediction_label": LABELS[int(batch_pred[pos])],
                    "row_corrupted": bool(row_corrupted[pos].item()),
                    "feature_fraction_corrupted": float(fraction[pos].item()),
                    "video_norm_before": float(raw_norm[pos]), "video_norm_after": float(corrupted_norm[pos]),
                    "text_prediction": int(text_pred[pos]) if text_pred is not None else "",
                    "video_prediction": int(video_pred[pos]) if video_pred is not None else "",
                    **{f"prob_{label}": float(probs[pos, j]) for j, label in enumerate(LABELS)},
                }
                all_predictions.append(row)
                mech = {
                    "experiment": run["experiment"], "seed": run["seed"], "scenario": scenario["name"],
                    "family": scenario["family"], "severity": scenario["severity"], "sample_index": int(idx),
                    **{key: ("" if math.isnan(mechanism_values[key][pos]) else float(mechanism_values[key][pos])) for key in MECHANISMS},
                }
                all_mechanisms.append(mech)
                if scenario["name"] == "video_missing_r0":
                    expected = clean_predictions.get((run["experiment"], run["seed"], int(idx)))
                    if expected is None or expected != int(batch_pred[pos]):
                        clean_mismatch += 1
            gold.extend(batch_gold.tolist())
            pred.extend(batch_pred.tolist())
        metric, classes = metric_rows(run["experiment"], run["seed"], scenario, gold, pred)
        all_metrics.append(metric)
        all_classes.extend(classes)
    reproduction = {
        "experiment": run["experiment"], "seed": run["seed"], "sample_count": len(data),
        "prediction_mismatches_vs_step28_full": clean_mismatch, "passes": clean_mismatch == 0,
    }
    if clean_mismatch:
        raise ValueError(f"clean reproduction mismatch for {run['experiment']} seed {run['seed']}: {clean_mismatch}")
    return all_metrics, all_classes, all_predictions, all_mechanisms, reproduction


def add_clean_drops(metrics: List[Dict[str, Any]]) -> None:
    clean = {
        (row["experiment"], row["seed"]): row
        for row in metrics if row["scenario"] == "video_missing_r0"
    }
    for row in metrics:
        base = clean[(row["experiment"], row["seed"])]
        for metric in METRICS:
            drop = float(base[metric]) - float(row[metric])
            row[f"absolute_drop_{metric}"] = drop
            row[f"relative_drop_{metric}"] = drop / max(abs(float(base[metric])), 1e-12)


def aggregate_curves(metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[(str(row["experiment"]), str(row["scenario"]))].append(row)
    result: List[Dict[str, Any]] = []
    for (experiment, scenario), rows in sorted(grouped.items()):
        first = rows[0]
        out: Dict[str, Any] = {
            "experiment": experiment, "scenario": scenario, "family": first["family"],
            "severity": first["severity"], "seeds": "/".join(str(row["seed"]) for row in sorted(rows, key=lambda r: int(r["seed"]))),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            drops = [float(row[f"absolute_drop_{metric}"]) for row in rows]
            relative = [float(row[f"relative_drop_{metric}"]) for row in rows]
            out[f"{metric}_mean"] = float(np.mean(values)); out[f"{metric}_std"] = sample_std(values)
            out[f"absolute_drop_{metric}_mean"] = float(np.mean(drops)); out[f"absolute_drop_{metric}_std"] = sample_std(drops)
            out[f"relative_drop_{metric}_mean"] = float(np.mean(relative)); out[f"relative_drop_{metric}_std"] = sample_std(relative)
        result.append(out)
    return result


def auc(points: Sequence[Tuple[float, float]], max_severity: float) -> float:
    ordered = sorted(points)
    x = np.asarray([row[0] for row in ordered], dtype=np.float64)
    y = np.asarray([row[1] for row in ordered], dtype=np.float64)
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(y, x)
    else:
        integral = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5)
    return float(integral / max_severity)


def compute_family_auc(metrics: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    keyed = {(row["experiment"], int(row["seed"]), row["scenario"]): row for row in metrics}
    families = {
        "video_missing": [(0.0, "video_missing_r0"), (0.25, "video_missing_r025"), (0.5, "video_missing_r050"), (0.75, "video_missing_r075"), (1.0, "video_missing_r100")],
        "gaussian_noise": [(0.0, "video_missing_r0"), (0.25, "video_noise_s025"), (0.5, "video_noise_s050"), (1.0, "video_noise_s100")],
        "block_occlusion": [(0.0, "video_missing_r0"), (0.25, "video_block_b025"), (0.5, "video_block_b050")],
    }
    auc_rows: List[Dict[str, Any]] = []
    for experiment in PRIMARY_ROWS:
        for seed in SEEDS:
            for family, points in families.items():
                row: Dict[str, Any] = {"experiment": experiment, "seed": seed, "family": family}
                for metric in METRICS:
                    row[f"{metric}_auc"] = auc(
                        [(severity, float(keyed[(experiment, seed, scenario)][metric])) for severity, scenario in points],
                        max(severity for severity, _ in points),
                    )
                auc_rows.append(row)
    deltas: List[Dict[str, Any]] = []
    for family in families:
        row: Dict[str, Any] = {"factor": "dg_matched_masking", "control": "A1", "treatment": "A3", "family": family}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                control = next(x for x in auc_rows if x["experiment"] == "A1" and x["seed"] == seed and x["family"] == family)
                treatment = next(x for x in auc_rows if x["experiment"] == "A3" and x["seed"] == seed and x["family"] == family)
                values.append(float(treatment[f"{metric}_auc"]) - float(control[f"{metric}_auc"]))
            row[f"delta_{metric}_auc_mean"] = float(np.mean(values))
            row[f"delta_{metric}_auc_std"] = sample_std(values)
            row[f"{metric}_positive_seeds"] = int(sum(value > 0 for value in values))
            for seed, value in zip(SEEDS, values):
                row[f"delta_{metric}_auc_seed{seed}"] = value
        deltas.append(row)
    family_primary_positive = {
        row["family"]: (row["delta_macro_f1_auc_mean"] > 0 or row["delta_minority_f1_auc_mean"] > 0)
        for row in deltas
    }
    consistent_families = 0
    consistency: Dict[str, Any] = {}
    for row in deltas:
        macro_ok = row["delta_macro_f1_auc_mean"] > 0 and row["macro_f1_positive_seeds"] >= 2
        minority_ok = row["delta_minority_f1_auc_mean"] > 0 and row["minority_f1_positive_seeds"] >= 2
        consistency[row["family"]] = {"macro": macro_ok, "minority": minority_ok}
        if macro_ok or minority_ok:
            consistent_families += 1
    aggregate_floor = all(
        row["delta_accuracy_auc_mean"] >= -0.01 and row["delta_weighted_f1_auc_mean"] >= -0.01
        for row in deltas
    )
    passed = all(family_primary_positive.values()) and consistent_families >= 2 and aggregate_floor
    audit = {
        "all_family_primary_auc_positive": all(family_primary_positive.values()),
        "family_primary_positive": family_primary_positive,
        "seed_consistent_family_count": consistent_families,
        "seed_consistency": consistency,
        "aggregate_auc_non_degradation": aggregate_floor,
        "passes_dg_independent_robustness_gate": passed,
    }
    return auc_rows, deltas, audit


def mechanism_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), int(row["seed"]), str(row["scenario"]))].append(row)
    result = []
    for (experiment, seed, scenario), values in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "experiment": experiment, "seed": seed, "scenario": scenario,
            "family": values[0]["family"], "severity": values[0]["severity"],
        }
        for key in MECHANISMS:
            valid = np.asarray([float(row[key]) for row in values if row.get(key, "") not in ("", None)], dtype=np.float64)
            out[f"{key}_count"] = int(valid.size)
            for label, q in (("mean", None), ("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
                out[f"{key}_{label}"] = "" if not valid.size else float(np.mean(valid) if q is None else np.quantile(valid, q))
        result.append(out)
    return result


def matched_scenario_deltas(metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = {(row["experiment"], int(row["seed"]), row["scenario"]): row for row in metrics}
    scenarios = sorted({str(row["scenario"]) for row in metrics})
    result = []
    for scenario in scenarios:
        sample = keyed[("A1", 42, scenario)]
        row: Dict[str, Any] = {
            "control": "A1", "treatment": "A3", "scenario": scenario,
            "family": sample["family"], "severity": sample["severity"],
        }
        for metric in METRICS:
            values = [float(keyed[("A3", seed, scenario)][metric]) - float(keyed[("A1", seed, scenario)][metric]) for seed in SEEDS]
            row[f"delta_{metric}_mean"] = float(np.mean(values)); row[f"delta_{metric}_std"] = sample_std(values)
            row[f"{metric}_positive_seeds"] = int(sum(value > 0 for value in values))
        result.append(row)
    return result


def metadata_map(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = read_csv(path)
    return {(row["Dialogue_ID"], row["Utterance_ID"]): row for row in rows}


def error_cases(
    predictions: Sequence[Mapping[str, Any]], mechanisms: Sequence[Mapping[str, Any]], metadata_path: Path,
) -> List[Dict[str, Any]]:
    meta = metadata_map(metadata_path)
    pred = {(row["experiment"], int(row["seed"]), row["scenario"], int(row["sample_index"])): row for row in predictions}
    mech = {(row["experiment"], int(row["seed"]), row["scenario"], int(row["sample_index"])): row for row in mechanisms}
    candidates: MutableMapping[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    irony = re.compile(r"\b(yeah right|sure|great|wonderful|oh really|seriously|as if|nice)\b", re.I)
    for idx in range(2593):
        clean = pred[("A3", 42, "video_missing_r0", idx)]
        noise = pred[("A3", 42, "video_noise_s100", idx)]
        block = pred[("A3", 42, "video_block_b050", idx)]
        missing = pred[("A3", 42, "video_missing_r100", idx)]
        mechanism = mech[("A3", 42, "video_missing_r0", idx)]
        metadata = meta.get((str(clean["dialogue_id"]), str(clean["utterance_id"])), {})
        utterance = metadata.get("Utterance", "")
        words = re.findall(r"[A-Za-z']+", utterance)
        js = float(mechanism["js_divergence"]) if mechanism.get("js_divergence", "") not in ("", None) else 0.0
        text_video_conflict = clean.get("text_prediction", "") != "" and clean.get("video_prediction", "") != "" and clean["text_prediction"] != clean["video_prediction"]
        base = {
            "selection_experiment": "A3", "selection_seed": 42, "sample_index": idx,
            "dialogue_id": clean["dialogue_id"], "utterance_id": clean["utterance_id"],
            "speaker": clean["speaker"], "utterance": utterance, "gold_label": clean["gold_label"],
            "clean_prediction": clean["prediction_label"], "noise_s100_prediction": noise["prediction_label"],
            "block_b050_prediction": block["prediction_label"], "missing_r100_prediction": missing["prediction_label"],
            "text_head_prediction": (LABELS[int(clean["text_prediction"])] if clean.get("text_prediction", "") != "" else ""),
            "video_head_prediction": (LABELS[int(clean["video_prediction"])] if clean.get("video_prediction", "") != "" else ""),
            "clean_js": js, "clean_disagreement_gate": mechanism.get("disagreement_gate", ""),
            "human_validated": False,
        }
        if text_video_conflict:
            candidates["text_video_modality_conflict"].append((js, {**base, "selection_rule": "unimodal predictions disagree; ranked by clean JS"}))
        if clean["prediction"] == clean["gold"] and (noise["prediction"] != noise["gold"] or block["prediction"] != block["gold"]):
            score = float(clean[f"prob_{clean['gold_label']}"]) - min(float(noise[f"prob_{clean['gold_label']}"]), float(block[f"prob_{clean['gold_label']}"]))
            candidates["camera_or_video_quality_noise_proxy"].append((score, {**base, "selection_rule": "clean correct and high video corruption causes error; no raw-frame validation"}))
        if len(words) <= 4 and (noise["prediction"] != clean["prediction"] or missing["prediction"] != clean["prediction"]):
            candidates["weak_expression_candidate"].append((js + 1.0 / max(len(words), 1), {**base, "selection_rule": "short utterance with corruption-sensitive prediction; weak-expression status unverified"}))
        if len(words) <= 6 or "..." in utterance or utterance.strip().endswith(("?", "--")):
            if missing["prediction"] != clean["prediction"] or text_video_conflict:
                candidates["ellipsis_or_context_dependence_candidate"].append((js + 1.0 / max(len(words), 1), {**base, "selection_rule": "short/fragment/question form plus conflict or missing-video flip"}))
        if irony.search(utterance) and (text_video_conflict or clean["prediction"] != clean["gold"]):
            candidates["irony_or_pragmatic_ambiguity_candidate"].append((js, {**base, "selection_rule": "irony lexical cue plus conflict/error; pragmatic reading requires human review"}))
    result: List[Dict[str, Any]] = []
    for category, rows in sorted(candidates.items()):
        for rank, (_, row) in enumerate(sorted(rows, key=lambda item: (-item[0], item[1]["sample_index"]))[:5], 1):
            result.append({"category": category, "rank": rank, **row})
    return result


def plot_curves(curves: Sequence[Mapping[str, Any]], out: Path) -> None:
    plot_dir = out / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    families = {
        "video_missing": ["video_missing_r0", "video_missing_r025", "video_missing_r050", "video_missing_r075", "video_missing_r100"],
        "gaussian_noise": ["video_missing_r0", "video_noise_s025", "video_noise_s050", "video_noise_s100"],
        "block_occlusion": ["video_missing_r0", "video_block_b025", "video_block_b050"],
    }
    keyed = {(row["experiment"], row["scenario"]): row for row in curves}
    colors = {"A1": "#1f77b4", "A3": "#d62728", "A6": "#2ca02c", "A7": "#9467bd"}
    for family, scenarios in families.items():
        width, height = 1000, 430
        panel_width, panel_height = 410, 290
        panel_origins = [(75, 80), (570, 80)]
        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.series{fill:none;stroke-width:2}</style>',
            f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">Step 29 frozen robustness: {html.escape(family.replace("_", " "))}</text>',
        ]
        for (origin_x, origin_y), metric, title in zip(panel_origins, ("macro_f1", "minority_f1"), ("Macro F1", "Minority F1")):
            all_values = []
            for experiment in PRIMARY_ROWS:
                for scenario in scenarios:
                    row = keyed[(experiment, scenario)]
                    mean = float(row[f"{metric}_mean"]); std = float(row[f"{metric}_std"])
                    all_values.extend([mean - std, mean + std])
            y_min = max(0.0, min(all_values) - 0.02)
            y_max = min(1.0, max(all_values) + 0.02)
            if y_max <= y_min:
                y_max = y_min + 0.1
            def sx(value: float) -> float:
                return origin_x + value * panel_width
            def sy(value: float) -> float:
                return origin_y + panel_height - (value - y_min) / (y_max - y_min) * panel_height
            elements += [
                f'<text x="{origin_x + panel_width/2}" y="{origin_y - 18}" text-anchor="middle" font-size="15">{title}</text>',
                f'<line class="axis" x1="{origin_x}" y1="{origin_y}" x2="{origin_x}" y2="{origin_y + panel_height}"/>',
                f'<line class="axis" x1="{origin_x}" y1="{origin_y + panel_height}" x2="{origin_x + panel_width}" y2="{origin_y + panel_height}"/>',
            ]
            for tick in range(6):
                x_value = tick / 5
                x_pos = sx(x_value)
                elements += [
                    f'<line class="grid" x1="{x_pos:.2f}" y1="{origin_y}" x2="{x_pos:.2f}" y2="{origin_y + panel_height}"/>',
                    f'<text x="{x_pos:.2f}" y="{origin_y + panel_height + 20}" text-anchor="middle" font-size="11">{x_value:.1f}</text>',
                ]
                y_value = y_min + tick * (y_max - y_min) / 5
                y_pos = sy(y_value)
                elements += [
                    f'<line class="grid" x1="{origin_x}" y1="{y_pos:.2f}" x2="{origin_x + panel_width}" y2="{y_pos:.2f}"/>',
                    f'<text x="{origin_x - 8}" y="{y_pos + 4:.2f}" text-anchor="end" font-size="11">{y_value:.3f}</text>',
                ]
            elements.append(f'<text x="{origin_x + panel_width/2}" y="{origin_y + panel_height + 44}" text-anchor="middle" font-size="12">Normalized severity</text>')
            for experiment in PRIMARY_ROWS:
                points = []
                for scenario in scenarios:
                    row = keyed[(experiment, scenario)]
                    severity = 0.0 if scenario == "video_missing_r0" else float(row["severity"])
                    mean = float(row[f"{metric}_mean"]); std = float(row[f"{metric}_std"])
                    x_pos, y_pos = sx(severity), sy(mean)
                    points.append(f"{x_pos:.2f},{y_pos:.2f}")
                    top, bottom = sy(min(y_max, mean + std)), sy(max(y_min, mean - std))
                    elements += [
                        f'<line x1="{x_pos:.2f}" y1="{top:.2f}" x2="{x_pos:.2f}" y2="{bottom:.2f}" stroke="{colors[experiment]}"/>',
                        f'<line x1="{x_pos-4:.2f}" y1="{top:.2f}" x2="{x_pos+4:.2f}" y2="{top:.2f}" stroke="{colors[experiment]}"/>',
                        f'<line x1="{x_pos-4:.2f}" y1="{bottom:.2f}" x2="{x_pos+4:.2f}" y2="{bottom:.2f}" stroke="{colors[experiment]}"/>',
                        f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="3.5" fill="{colors[experiment]}"/>',
                    ]
                elements.append(f'<polyline class="series" stroke="{colors[experiment]}" points="{" ".join(points)}"/>')
        legend_y = 412
        for index, experiment in enumerate(PRIMARY_ROWS):
            x_pos = 330 + index * 100
            elements += [
                f'<line x1="{x_pos}" y1="{legend_y-4}" x2="{x_pos+22}" y2="{legend_y-4}" stroke="{colors[experiment]}" stroke-width="2"/>',
                f'<text x="{x_pos+28}" y="{legend_y}" font-size="12">{experiment}</text>',
            ]
        elements.append("</svg>")
        (plot_dir / f"{family}_curves.svg").write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_result_notes(out: Path, audit: Mapping[str, Any], auc_deltas: Sequence[Mapping[str, Any]]) -> None:
    passed = bool(audit["passes_dg_independent_robustness_gate"])
    decision = "DG_ROBUSTNESS_SUPPORTED" if passed else "NO_GO_DG_INDEPENDENT_ROBUSTNESS_CLAIM"
    lines = [
        "# Step 29 负结果与主张审计", "", f"DG matched robustness decision: `{decision}`.", "",
        "A1/A3 是唯一主要 matched masking 对照；A6/A7 不用于替代 DG 独立归因。下表保留全部 family AUC 差值。", "",
        "| Family | ΔAcc AUC | ΔWF1 AUC | ΔMacro AUC | ΔMinority AUC |", "|---|---:|---:|---:|---:|",
    ]
    for row in auc_deltas:
        lines.append(
            f"| {row['family']} | {row['delta_accuracy_auc_mean']:+.4f} | {row['delta_weighted_f1_auc_mean']:+.4f} | "
            f"{row['delta_macro_f1_auc_mean']:+.4f} | {row['delta_minority_f1_auc_mean']:+.4f} |"
        )
    lines += [
        "", f"- all-family primary positive: `{audit['all_family_primary_auc_positive']}`",
        f"- seed-consistent family count: `{audit['seed_consistent_family_count']}` (required >= 2)",
        f"- aggregate AUC non-degradation: `{audit['aggregate_auc_non_degradation']}`",
        "- 无论 gate 结果如何，都不得根据这些 test 曲线重训练或修改 Step 28 候选。",
    ]
    (out / "negative_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    out: Path, runs: Sequence[Mapping[str, Any]], metrics: List[Dict[str, Any]], classes: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]], mechanisms: Sequence[Mapping[str, Any]], reproduction: Sequence[Mapping[str, Any]],
    manifest_summary: Mapping[str, Any], args: argparse.Namespace,
) -> Dict[str, Any]:
    add_clean_drops(metrics)
    curves = aggregate_curves(metrics)
    matched = matched_scenario_deltas(metrics)
    auc_rows, auc_deltas, audit = compute_family_auc(metrics)
    mech_summary = mechanism_summary(mechanisms)
    cases = error_cases(predictions, mechanisms, resolve(args.metadata_csv))
    metric_fields = ["experiment", "seed", "split", "scenario", "family", "severity", *METRICS]
    for metric in METRICS:
        metric_fields += [f"absolute_drop_{metric}", f"relative_drop_{metric}"]
    write_csv(out / "scenario_metrics.csv", metrics, metric_fields)
    write_csv(out / "per_class.csv", classes, list(classes[0]))
    write_csv(out / "predictions.csv", predictions, list(predictions[0]))
    write_csv(out / "mechanism_distributions.csv", mechanisms, list(mechanisms[0]))
    write_csv(out / "mechanism_summary.csv", mech_summary, list(mech_summary[0]))
    write_csv(out / "curve_source.csv", curves, list(curves[0]))
    write_csv(out / "matched_a3_minus_a1.csv", matched, list(matched[0]))
    write_csv(out / "family_auc_by_seed.csv", auc_rows, list(auc_rows[0]))
    write_csv(out / "family_auc_matched_deltas.csv", auc_deltas, list(auc_deltas[0]))
    write_csv(out / "clean_reproduction_check.csv", reproduction, list(reproduction[0]))
    if cases:
        write_csv(out / "error_case_candidates.csv", cases, list(cases[0]))
    else:
        write_csv(out / "error_case_candidates.csv", [], ["category", "human_validated"])
    plot_curves(curves, out)
    write_result_notes(out, audit, auc_deltas)
    passed = bool(audit["passes_dg_independent_robustness_gate"])
    result_gate = {
        "step": 29, "status": "formal_step29_results_ready", "training_performed": False,
        "test_evaluated": True, "test_curve_tuning_performed": False,
        "candidate_reselected": False, "clean_reproduction_passed": all(row["passes"] for row in reproduction),
        "primary_pair": ["A1", "A3"], "additional_rows": ["A6", "A7"],
        "dg_independent_robustness": audit,
        "claim_decision": "DG_ROBUSTNESS_SUPPORTED" if passed else "NO_GO_DG_INDEPENDENT_ROBUSTNESS_CLAIM",
        "decision": "GO_STEP30_DG_ROBUSTNESS_SUPPORTED" if passed else "GO_STEP30_WITH_DG_DOWNGRADED",
        "step30_unlocked": True,
        "error_cases_human_validated": False,
        "risk": "case categories are heuristic candidates because raw video/frame-quality annotations are unavailable",
    }
    write_json(out / "result_gate.json", result_gate)
    write_json(out / "gate.json", result_gate)
    runtime = {
        "step": 29, "status": "formal_evidence_written", "runner": rel(Path(__file__)),
        "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(out / "config.json"),
        "protocol_sha256": sha256(out / "protocol.md"), "step28_runs": [{
            "experiment": row["experiment"], "seed": row["seed"],
            "checkpoint": rel(row["checkpoint"]), "checkpoint_sha256": row["checkpoint_sha256"],
            "config": rel(row["config"]), "config_sha256": row["config_sha256"],
        } for row in runs], "corruption_manifest": manifest_summary,
        "row_counts": {
            "scenario_metrics": len(metrics), "per_class": len(classes), "predictions": len(predictions),
            "mechanism_distributions": len(mechanisms), "error_case_candidates": len(cases),
        },
    }
    write_json(out / "runtime_manifest.json", runtime)
    return result_gate


def smoke_test(out: Path) -> None:
    n, d = 16, 8
    temp_feature = out / "smoke_features"
    temp_feature.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    np.save(temp_feature / "train_video_features.npy", rng.normal(size=(32, d)).astype(np.float32))
    manifests, train_std, summary = build_manifests(out, temp_feature, (42,), n, d)
    video = torch.as_tensor(rng.normal(size=(n, d)).astype(np.float32))
    text = torch.as_tensor(rng.normal(size=(n, d)).astype(np.float32))
    noise = torch.as_tensor(noise_for_seed(42, n, d))
    indices = np.arange(n)
    missing_specs = [
        {"family": "video_missing", "severity": value, "name": f"r{value}"}
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    counts = []
    masks = []
    for spec in missing_specs:
        corrupted, _, row_mask, _ = apply_corruption(
            video, text, indices, spec, manifests[42], torch.as_tensor(train_std), noise,
        )
        counts.append(int(row_mask.sum()))
        masks.append(row_mask.numpy())
        assert torch.isfinite(corrupted).all()
    assert counts == [0, 4, 8, 12, 16]
    assert all(np.all(masks[i] <= masks[i + 1]) for i in range(len(masks) - 1))
    block, _, _, fraction = apply_corruption(
        video, text, indices, {"family": "block_occlusion", "severity": 0.5, "name": "block"},
        manifests[42], torch.as_tensor(train_std), noise,
    )
    assert torch.allclose((block == 0).float().mean(dim=1), torch.full((n,), 0.5))
    assert torch.allclose(fraction, torch.full((n,), 0.5))
    cfg = SimpleNamespace(
        dataset="meld", feat_dim=d, hidden_dim=8, num_classes=7, fusion="tv_disagreement", dropout=0.1,
        num_heads=2, with_style=False, rag_mode="none", rag_top_k=5, structure_mode="none", context_len=0,
        context_max_distance=20, structure_gate_scale=0.5, relation_dropout=0.0,
        relation_embedding_init_std=0.01, speaker_memory_mode="none", speaker_memory_slots=4,
        memory_gate_scale=0.5, speaker_prototype_path=None, disagreement_gate_min=0.1,
        disagreement_gate_temperature=1.5, disagreement_gate_bias_init=1.0,
    )
    model = build_model(cfg).eval()
    output = model(video_feat=video, text_feat=text, return_attention=True)
    assert output["logits"].shape == (n, 7)
    assert output["js_divergence"].shape[0] == n and output["disagreement_gate"].shape[0] == n
    write_json(out / "smoke_result.json", {
        "status": "PASS", "test_data_read": False, "training_performed": False,
        "missing_counts": counts, "manifest": summary,
    })
    print("smoke_test=PASS")


def run(args: argparse.Namespace) -> None:
    out = resolve(args.output_dir)
    step28 = resolve(args.step28_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_log = out / "run.log"
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    validate_pre_run_manifest(out)
    validate_protocol_gate(out, step28, config)
    runs = load_frozen_runs(step28)
    feature_dir = resolve(args.feature_dir)
    test_video = np.load(feature_dir / "test_video_features.npy", mmap_mode="r")
    if tuple(test_video.shape) != (2593, 3584):
        raise ValueError(f"unexpected frozen test video shape: {test_video.shape}")
    manifests, train_std, manifest_summary = build_manifests(out, feature_dir, SEEDS, 2593, 3584)
    clean_predictions = clean_reference(step28)
    expected_clean = len(PRIMARY_ROWS) * len(SEEDS) * 2593
    if len(clean_predictions) != expected_clean:
        raise ValueError(f"Step 28 clean prediction count mismatch: {len(clean_predictions)} != {expected_clean}")
    specs = scenario_specs(config)
    all_metrics: List[Dict[str, Any]] = []
    all_classes: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    all_mechanisms: List[Dict[str, Any]] = []
    reproduction: List[Dict[str, Any]] = []
    log(run_log, "Step 29 frozen evaluation starts; no training or candidate reselection")
    for item in runs:
        log(run_log, f"evaluate {item['experiment']} seed{item['seed']} across {len(specs)} frozen scenarios")
        metrics, classes, predictions, mechanisms, check = evaluate_run(
            item, args, step28, specs, manifests, train_std, clean_predictions,
        )
        all_metrics.extend(metrics); all_classes.extend(classes)
        all_predictions.extend(predictions); all_mechanisms.extend(mechanisms); reproduction.append(check)
    gate = save_outputs(
        out, runs, all_metrics, all_classes, all_predictions, all_mechanisms,
        reproduction, manifest_summary, args,
    )
    log(run_log, f"Step 29 evaluation complete; decision={gate['decision']}; Step 30 requires attribution review")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--step28_dir", default=str(STEP28_REL))
    value.add_argument("--feature_dir", default="datasets/MELD/features_v2")
    value.add_argument("--metadata_csv", default="datasets/MELD/test_sent_emo.csv")
    value.add_argument("--output_dir", default=str(STEP29_REL))
    value.add_argument("--batch_size", type=int, default=128)
    value.add_argument("--smoke_test", action="store_true", help="exercise corruption/model paths on synthetic data only")
    return value


def main() -> None:
    args = parser().parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        smoke_test(out)
    else:
        run(args)


if __name__ == "__main__":
    main()
