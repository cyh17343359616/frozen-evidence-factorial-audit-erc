#!/usr/bin/env python3
"""Phase III Step 30 frozen speaker-identity and prototype stress audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from data.feature_extractor_v2 import load_cached_features
from run_phase3_step28_factorial import eval_args
from training.train import DATASET_LABELS, MELDFeatureDataset, build_model, load_speaker_prototypes

LABELS = DATASET_LABELS["meld"]
MINORITY = {"fear", "disgust", "sadness"}
CONDITIONS = ("A0_memory_off", "A4_speaker_specific", "A4_global_only", "A4_id_permuted")
METRICS = ("accuracy", "weighted_f1", "macro_f1", "minority_f1")
SEEDS = (42, 43, 44)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(ROOT.resolve()))
    except ValueError:
        return str(value)


def resolve_frozen_prototype(path: str | Path) -> Path:
    """Resolve a frozen config's server path to the mirrored local Step 28 artifact."""
    value = Path(path)
    if value.is_file():
        return value
    local = ROOT / "outputs/phase3_ieee_access/step28_factorial_ablation/speaker_prototypes" / value.name
    if local.is_file():
        return local
    raise FileNotFoundError(value)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_index(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    selected: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_csv(path):
        if row["experiment"] not in {"A0", "A4"} or int(row["seed"]) not in SEEDS:
            continue
        row["seed"] = int(row["seed"])
        for kind in ("config", "checkpoint"):
            artifact = resolve(row[kind])
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            expected = row[f"{kind}_sha256"]
            if sha256(artifact) != expected:
                raise ValueError(f"{kind} hash mismatch for {row['experiment']} seed {row['seed']}")
        selected[(row["experiment"], row["seed"])] = row
    expected = {(row, seed) for row in ("A0", "A4") for seed in SEEDS}
    if set(selected) != expected:
        raise ValueError(f"checkpoint index mismatch: {sorted(selected)}")
    return selected


def validate_upstream(args: argparse.Namespace, allow_test: bool) -> Tuple[Dict[str, Any], Dict[Tuple[str, int], Dict[str, Any]]]:
    out = resolve(args.output_dir)
    config = read_json(out / "config.json")
    gate = read_json(out / ("dev_gate.json" if allow_test else "gate.json"))
    step29 = read_json(resolve(config["upstream_step29_gate"]))
    if step29.get("decision") != "GO_STEP30_WITH_DG_DOWNGRADED" or step29.get("step30_unlocked") is not True:
        raise ValueError("Step 29 does not unlock Step 30")
    if allow_test:
        if gate.get("test_evaluation_unlocked") is not True or gate.get("test_evaluated") is not False:
            raise ValueError("frozen test evaluation is not unlocked")
    else:
        if gate.get("decision") != "GO_DEV_FROZEN_EVALUATION" or gate.get("test_evaluation_unlocked") is not False:
            raise ValueError("initial Step 30 gate is invalid")
    index = load_index(resolve(config["upstream_checkpoint_index"]))
    return config, index


def speaker_arrays(feature_dir: Path) -> Dict[str, np.ndarray]:
    return {
        split: np.asarray(np.load(feature_dir / f"{split}_speakers.npy", allow_pickle=True), dtype=object).astype(str)
        for split in ("train", "dev", "test")
    }


def bucket_name(count: int) -> str:
    if count == 0:
        return "unseen"
    if count <= 4:
        return "rare"
    if count <= 19:
        return "low"
    if count <= 99:
        return "medium"
    return "high"


def derangement(speakers: Sequence[str], seed: int) -> Dict[str, str]:
    speakers = list(sorted(speakers))
    rng = np.random.default_rng(100000 + seed)
    for _ in range(10000):
        permuted = list(rng.permutation(speakers))
        if all(left != right for left, right in zip(speakers, permuted)):
            return dict(zip(speakers, permuted))
    raise RuntimeError("could not create fixed-point-free speaker permutation")


def audit(args: argparse.Namespace) -> None:
    config, index = validate_upstream(args, allow_test=False)
    out = resolve(args.output_dir)
    feature_dir = resolve(config["feature_dir"])
    arrays = speaker_arrays(feature_dir)
    counts = {split: Counter(values.tolist()) for split, values in arrays.items()}
    train_speakers = sorted(counts["train"])
    rows: List[Dict[str, Any]] = []
    all_speakers = sorted(set().union(*(set(counts[split]) for split in counts)))
    for speaker in all_speakers:
        train_count = counts["train"][speaker]
        rows.append({
            "speaker": speaker,
            "train_samples": train_count,
            "dev_samples": counts["dev"][speaker],
            "test_samples": counts["test"][speaker],
            "seen_in_train": train_count > 0,
            "train_frequency_bucket": bucket_name(train_count),
            "dev_uses_global_fallback": train_count == 0 and counts["dev"][speaker] > 0,
            "test_uses_global_fallback": train_count == 0 and counts["test"][speaker] > 0,
        })
    write_csv(out / "speaker_manifest.csv", rows, list(rows[0]))

    summary = {
        "step": 30,
        "source": "features_v2 speaker arrays; no labels used",
        "split_statistics": {
            split: {
                "samples": len(arrays[split]),
                "unique_speakers": len(counts[split]),
                "min_frequency": min(counts[split].values()),
                "median_frequency": float(np.median(list(counts[split].values()))),
                "max_frequency": max(counts[split].values()),
            }
            for split in counts
        },
        "unique_intersections": {
            "train_dev": len(set(counts["train"]) & set(counts["dev"])),
            "train_test": len(set(counts["train"]) & set(counts["test"])),
            "dev_test": len(set(counts["dev"]) & set(counts["test"])),
            "all_three": len(set(counts["train"]) & set(counts["dev"]) & set(counts["test"])),
        },
        "coverage": {
            split: {
                "direct_samples": sum(value for speaker, value in counts[split].items() if speaker in counts["train"]),
                "fallback_samples": sum(value for speaker, value in counts[split].items() if speaker not in counts["train"]),
                "fallback_rate": float(sum(value for speaker, value in counts[split].items() if speaker not in counts["train"]) / len(arrays[split])),
                "direct_unique_speakers": len(set(counts[split]) & set(counts["train"])),
                "fallback_unique_speakers": len(set(counts[split]) - set(counts["train"])),
            }
            for split in ("dev", "test")
        },
        "frequency_bucket_samples": {
            split: dict(Counter(bucket_name(counts["train"].get(speaker, 0)) for speaker in arrays[split]))
            for split in ("dev", "test")
        },
        "test_not_used_for_configuration": True,
    }
    write_json(out / "speaker_overlap_summary.json", summary)

    permutation_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        mapping = derangement(train_speakers, seed)
        for speaker in train_speakers:
            permutation_rows.append({"seed": seed, "source_speaker": speaker, "assigned_prototype_speaker": mapping[speaker], "fixed_point": False})
    write_csv(out / "speaker_permutation_manifest.csv", permutation_rows, list(permutation_rows[0]))

    frozen = {
        "created_before_dev_evaluation": True,
        "protocol_sha256": sha256(out / "protocol.md"),
        "config_sha256": sha256(out / "config.json"),
        "initial_gate_sha256": sha256(out / "gate.json"),
        "upstream": {
            config["upstream_checkpoint_index"]: sha256(resolve(config["upstream_checkpoint_index"])),
            config["upstream_attribution_gate"]: sha256(resolve(config["upstream_attribution_gate"])),
            config["upstream_step29_gate"]: sha256(resolve(config["upstream_step29_gate"])),
            config["prototype_audit"]: sha256(resolve(config["prototype_audit"])),
        },
        "feature_inputs": {
            relative(feature_dir / f"{split}_{kind}.npy"): sha256(feature_dir / f"{split}_{kind}.npy")
            for split in ("train", "dev", "test") for kind in ("speakers", "dialogue_ids", "utterance_ids", "video_features", "text_features", "labels")
        },
        "checkpoint_inputs": {
            relative(resolve(row["checkpoint"])): row["checkpoint_sha256"] for row in index.values()
        },
        "permutation_manifest_sha256": sha256(out / "speaker_permutation_manifest.csv"),
        "speaker_manifest_sha256": sha256(out / "speaker_manifest.csv"),
        "test_evaluated": False,
    }
    write_json(out / "pre_run_manifest.json", frozen)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def condition_prototypes(condition: str, base: Dict[str, Dict[str, np.ndarray]], mapping: Dict[str, str]) -> Dict[str, Dict[str, np.ndarray]] | None:
    if condition == "A0_memory_off":
        return None
    if condition == "A4_speaker_specific":
        return base
    if condition == "A4_global_only":
        return {"__GLOBAL__": base["__GLOBAL__"]}
    if condition == "A4_id_permuted":
        return {**{speaker: base[target] for speaker, target in mapping.items()}, "__GLOBAL__": base["__GLOBAL__"]}
    raise ValueError(condition)


def build_dataset(feature_dir: Path, split: str, cfg: Any, prototypes: Dict[str, Dict[str, np.ndarray]] | None) -> MELDFeatureDataset:
    data = load_cached_features(str(feature_dir), split)
    speakers = np.load(feature_dir / f"{split}_speakers.npy", allow_pickle=True)
    return MELDFeatureDataset(
        data["video_features"], data["text_features"], data["labels"],
        data.get("dialogue_ids"), data.get("utterance_ids"), speaker_ids=speakers,
        context_len=cfg.context_len, context_max_distance=cfg.context_max_distance,
        speaker_prototypes=prototypes, speaker_memory_slots=cfg.speaker_memory_slots,
    )


def metric_rows(gold: Sequence[int], pred: Sequence[int]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=range(7), zero_division=0)
    classes = [{
        "label": label, "precision": float(precision[i]), "recall": float(recall[i]),
        "f1": float(f1[i]), "support": int(support[i]),
    } for i, label in enumerate(LABELS)]
    metric = {
        "accuracy": float(accuracy_score(gold, pred)),
        "weighted_f1": float(f1_score(gold, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "minority_f1": float(np.mean([row["f1"] for row in classes if row["label"] in MINORITY])),
    }
    return metric, classes


@torch.no_grad()
def evaluate_one(
    condition: str, seed: int, run: Mapping[str, Any], feature_dir: Path, split: str,
    mapping: Dict[str, str], batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = eval_args(resolve(run["config"]))
    proto_path = resolve(run["config"])
    cfg_payload = read_json(proto_path)
    base = load_speaker_prototypes(resolve_frozen_prototype(cfg_payload["speaker_prototype_path"])) if condition != "A0_memory_off" else None
    prototypes = condition_prototypes(condition, base, mapping) if base is not None else None
    data = build_dataset(feature_dir, split, cfg, prototypes)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(resolve(run["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    gold: List[int] = []
    pred: List[int] = []
    predictions: List[Dict[str, Any]] = []
    train_counts = Counter(np.load(feature_dir / "train_speakers.npy", allow_pickle=True).astype(str).tolist())
    for batch in loader:
        output = model(
            video_feat=batch["video_feat"].to(device), text_feat=batch["text_feat"].to(device), return_attention=True,
            context_video_feat=batch["context_video_feat"].to(device),
            context_text_feat=batch["context_text_feat"].to(device), context_mask=batch["context_mask"].to(device),
            context_same_speaker=batch["context_same_speaker"].to(device),
            context_turn_distance=batch["context_turn_distance"].to(device),
            speaker_memory_video_feat=batch["speaker_memory_video_feat"].to(device),
            speaker_memory_text_feat=batch["speaker_memory_text_feat"].to(device),
            speaker_memory_mask=batch["speaker_memory_mask"].to(device),
        )
        probs = torch.softmax(output["logits"], dim=-1).cpu().numpy()
        batch_pred = probs.argmax(1).astype(int)
        batch_gold = batch["label"].numpy().astype(int)
        indices = batch["idx"].numpy().astype(int)
        memory_gate = output.get("memory_gate")
        memory_slots = output.get("memory_valid_slots")
        for pos, idx in enumerate(indices):
            speaker = str(data.speaker_ids[idx])
            count = train_counts.get(speaker, 0)
            predictions.append({
                "condition": condition, "seed": seed, "split": split, "sample_index": int(idx),
                "dialogue_id": str(data.dialogue_ids[idx]), "utterance_id": str(data.utterance_ids[idx]),
                "speaker": speaker, "seen_in_train": count > 0, "train_speaker_frequency": count,
                "frequency_bucket": bucket_name(count), "gold": int(batch_gold[pos]), "prediction": int(batch_pred[pos]),
                "gold_label": LABELS[int(batch_gold[pos])], "prediction_label": LABELS[int(batch_pred[pos])],
                "memory_gate": float(memory_gate[pos].item()) if memory_gate is not None else 0.0,
                "memory_valid_slots": float(memory_slots[pos].item()) if memory_slots is not None else 0.0,
                **{f"prob_{label}": float(probs[pos, j]) for j, label in enumerate(LABELS)},
            })
        gold.extend(batch_gold.tolist())
        pred.extend(batch_pred.tolist())
    metric, classes = metric_rows(gold, pred)
    metric.update({"condition": condition, "seed": seed, "split": split, "n_samples": len(gold)})
    for row in classes:
        row.update({"condition": condition, "seed": seed, "split": split})
    return metric, classes, predictions


def bucket_metrics(predictions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = sorted({(row["condition"], int(row["seed"]), row["split"]) for row in predictions})
    for condition, seed, split in keys:
        base = [row for row in predictions if row["condition"] == condition and int(row["seed"]) == seed and row["split"] == split]
        groups = [("seen_status", "seen", lambda row: bool(row["seen_in_train"])), ("seen_status", "unseen", lambda row: not bool(row["seen_in_train"]))]
        groups += [("frequency", bucket, lambda row, bucket=bucket: row["frequency_bucket"] == bucket) for bucket in ("unseen", "rare", "low", "medium", "high")]
        for bucket_type, bucket, predicate in groups:
            subset = [row for row in base if predicate(row)]
            if not subset:
                continue
            metric, _ = metric_rows([int(row["gold"]) for row in subset], [int(row["prediction"]) for row in subset])
            labels_present = len({int(row["gold"]) for row in subset})
            rows.append({
                "condition": condition, "seed": seed, "split": split, "bucket_type": bucket_type, "bucket": bucket,
                "n_samples": len(subset), "n_speakers": len({row["speaker"] for row in subset}),
                "labels_present": labels_present, "inferential": len(subset) >= 20 and labels_present == 7, **metric,
            })
    return rows


def paired_deltas(metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = {(row["condition"], int(row["seed"]), row["split"]): row for row in metrics}
    rows: List[Dict[str, Any]] = []
    comparisons = (
        ("memory_matched", "A0_memory_off", "A4_speaker_specific"),
        ("specific_vs_global", "A4_global_only", "A4_speaker_specific"),
        ("specific_vs_permuted", "A4_id_permuted", "A4_speaker_specific"),
    )
    for split in sorted({row["split"] for row in metrics}):
        for seed in SEEDS:
            for name, control, treatment in comparisons:
                c, t = keyed[(control, seed, split)], keyed[(treatment, seed, split)]
                rows.append({
                    "comparison": name, "control": control, "treatment": treatment, "seed": seed, "split": split,
                    **{f"delta_{metric}": float(t[metric]) - float(c[metric]) for metric in METRICS},
                })
    return rows


def save_split(out: Path, split: str, metrics: List[Dict[str, Any]], classes: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> None:
    write_csv(out / f"{split}_scenario_metrics.csv", metrics, ["condition", "seed", "split", "n_samples", *METRICS])
    write_csv(out / f"{split}_per_class.csv", classes, ["condition", "seed", "split", "label", "precision", "recall", "f1", "support"])
    pred_fields = [
        "condition", "seed", "split", "sample_index", "dialogue_id", "utterance_id", "speaker", "seen_in_train",
        "train_speaker_frequency", "frequency_bucket", "gold", "prediction", "gold_label", "prediction_label",
        "memory_gate", "memory_valid_slots", *[f"prob_{label}" for label in LABELS],
    ]
    write_csv(out / f"{split}_predictions.csv", predictions, pred_fields)
    buckets = bucket_metrics(predictions)
    write_csv(out / f"{split}_bucket_metrics.csv", buckets, [
        "condition", "seed", "split", "bucket_type", "bucket", "n_samples", "n_speakers", "labels_present", "inferential", *METRICS,
    ])
    deltas = paired_deltas(metrics)
    write_csv(out / f"{split}_paired_deltas.csv", deltas, ["comparison", "control", "treatment", "seed", "split", *[f"delta_{metric}" for metric in METRICS]])


def load_permutations(path: Path) -> Dict[int, Dict[str, str]]:
    mappings: Dict[int, Dict[str, str]] = {seed: {} for seed in SEEDS}
    for row in read_csv(path):
        mappings[int(row["seed"])][row["source_speaker"]] = row["assigned_prototype_speaker"]
    for seed, mapping in mappings.items():
        if not mapping or any(source == target for source, target in mapping.items()):
            raise ValueError(f"invalid permutation for seed {seed}")
    return mappings


def run_split(args: argparse.Namespace, split: str) -> None:
    config, index = validate_upstream(args, allow_test=split == "test")
    out = resolve(args.output_dir)
    pre = read_json(out / "pre_run_manifest.json")
    for path, expected in pre["upstream"].items():
        if sha256(resolve(path)) != expected:
            raise ValueError(f"upstream hash mismatch: {path}")
    if sha256(out / "speaker_permutation_manifest.csv") != pre["permutation_manifest_sha256"]:
        raise ValueError("permutation manifest changed")
    mappings = load_permutations(out / "speaker_permutation_manifest.csv")
    metrics: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    run_log = out / "run.log"
    for seed in SEEDS:
        for condition in CONDITIONS:
            row_id = "A0" if condition == "A0_memory_off" else "A4"
            log(run_log, f"evaluate split={split} condition={condition} seed={seed}")
            metric, per_class, pred = evaluate_one(condition, seed, index[(row_id, seed)], resolve(config["feature_dir"]), split, mappings[seed], args.batch_size)
            metrics.append(metric); classes.extend(per_class); predictions.extend(pred)
    expected_samples = 1103 if split == "dev" else 2593
    for condition in CONDITIONS:
        for seed in SEEDS:
            subset = [row for row in predictions if row["condition"] == condition and int(row["seed"]) == seed]
            if len(subset) != expected_samples or len({int(row["sample_index"]) for row in subset}) != expected_samples:
                raise ValueError(f"sample invariant failed: {condition} seed {seed} split {split}")
    save_split(out, split, metrics, classes, predictions)
    if split == "dev":
        deltas = [row for row in paired_deltas(metrics) if row["comparison"] == "memory_matched"]
        mean = {metric: float(np.mean([row[f"delta_{metric}"] for row in deltas])) for metric in METRICS}
        positive_seeds = sum(row["delta_macro_f1"] > 0 or row["delta_minority_f1"] > 0 for row in deltas)
        passed = (
            (mean["macro_f1"] > 0 or mean["minority_f1"] > 0)
            and mean["accuracy"] >= -0.01 and mean["weighted_f1"] >= -0.01 and positive_seeds >= 2
        )
        gate = {
            "step": 30, "status": "three_seed_dev_evaluation_complete",
            "decision": "GO_FROZEN_TEST_CONFIRMATION" if passed else "NO_GO_TEST_CONFIRMATION",
            "training_performed": False, "test_evaluated": False, "test_evaluation_unlocked": passed,
            "candidate_not_reselected": True, "matched_mean_deltas": mean,
            "matched_positive_seed_count": positive_seeds, "dev_sample_invariants_passed": True,
            "step31_unlocked": False,
        }
        write_json(out / "dev_gate.json", gate)
        write_json(out / "gate.json", gate)
    else:
        finalize(out)


def mean_delta(rows: Sequence[Mapping[str, Any]], comparison: str, split: str) -> Dict[str, float]:
    subset = [row for row in rows if row["comparison"] == comparison and row["split"] == split]
    return {metric: float(np.mean([float(row[f"delta_{metric}"]) for row in subset])) for metric in METRICS}


def positive_seed_count(rows: Sequence[Mapping[str, Any]], split: str) -> int:
    subset = [row for row in rows if row["comparison"] == "memory_matched" and row["split"] == split]
    return sum(float(row["delta_macro_f1"]) > 0 or float(row["delta_minority_f1"]) > 0 for row in subset)


def bucket_delta(out: Path, split: str, bucket_type: str, bucket: str) -> Dict[str, float] | None:
    rows = read_csv(out / f"{split}_bucket_metrics.csv")
    keyed = {(row["condition"], int(row["seed"]), row["bucket_type"], row["bucket"]): row for row in rows}
    values: Dict[str, List[float]] = {metric: [] for metric in METRICS}
    for seed in SEEDS:
        key0 = ("A0_memory_off", seed, bucket_type, bucket)
        key4 = ("A4_speaker_specific", seed, bucket_type, bucket)
        if key0 not in keyed or key4 not in keyed:
            continue
        if keyed[key4]["inferential"].lower() != "true":
            continue
        for metric in METRICS:
            values[metric].append(float(keyed[key4][metric]) - float(keyed[key0][metric]))
    if not values["macro_f1"]:
        return None
    return {metric: float(np.mean(items)) for metric, items in values.items()}


def write_paired_outputs(out: Path) -> None:
    paired_rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    fallback_failures: List[Dict[str, Any]] = []
    for split in ("dev", "test"):
        predictions = read_csv(out / f"{split}_predictions.csv")
        keyed: Dict[Tuple[int, int], Dict[str, Dict[str, str]]] = {}
        for row in predictions:
            keyed.setdefault((int(row["seed"]), int(row["sample_index"])), {})[row["condition"]] = row
        for (seed, sample_index), conditions in sorted(keyed.items()):
            if set(conditions) != set(CONDITIONS):
                raise ValueError(f"unpaired prediction group: {split} seed={seed} sample={sample_index}")
            base = conditions["A0_memory_off"]
            row: Dict[str, Any] = {
                "split": split, "seed": seed, "sample_index": sample_index,
                "dialogue_id": base["dialogue_id"], "utterance_id": base["utterance_id"], "speaker": base["speaker"],
                "seen_in_train": base["seen_in_train"], "train_speaker_frequency": base["train_speaker_frequency"],
                "frequency_bucket": base["frequency_bucket"], "gold": base["gold"], "gold_label": base["gold_label"],
            }
            for condition in CONDITIONS:
                pred = conditions[condition]
                row[f"prediction_{condition}"] = pred["prediction"]
                row[f"prediction_label_{condition}"] = pred["prediction_label"]
                row[f"correct_{condition}"] = int(pred["prediction"]) == int(pred["gold"])
            paired_rows.append(row)
            if row["seen_in_train"].lower() == "false" and not row["correct_A4_speaker_specific"]:
                fallback_failures.append({
                    **row,
                    "failure_type": "A0_correct_A4_wrong" if row["correct_A0_memory_off"] else "A0_and_A4_wrong",
                })

        buckets = read_csv(out / f"{split}_bucket_metrics.csv")
        by_key = {(row["condition"], int(row["seed"]), row["bucket_type"], row["bucket"]): row for row in buckets}
        for seed in SEEDS:
            for bucket_type, bucket in sorted({(row["bucket_type"], row["bucket"]) for row in buckets}):
                c = by_key.get(("A0_memory_off", seed, bucket_type, bucket))
                t = by_key.get(("A4_speaker_specific", seed, bucket_type, bucket))
                if c is None or t is None:
                    continue
                bucket_rows.append({
                    "split": split, "seed": seed, "bucket_type": bucket_type, "bucket": bucket,
                    "n_samples": t["n_samples"], "n_speakers": t["n_speakers"],
                    "labels_present": t["labels_present"], "inferential": t["inferential"],
                    **{f"delta_{metric}": float(t[metric]) - float(c[metric]) for metric in METRICS},
                })
    paired_fields = [
        "split", "seed", "sample_index", "dialogue_id", "utterance_id", "speaker", "seen_in_train",
        "train_speaker_frequency", "frequency_bucket", "gold", "gold_label",
    ]
    for condition in CONDITIONS:
        paired_fields += [f"prediction_{condition}", f"prediction_label_{condition}", f"correct_{condition}"]
    write_csv(out / "paired_predictions.csv", paired_rows, paired_fields)
    write_csv(out / "fallback_failure_cases.csv", fallback_failures, [*paired_fields, "failure_type"])
    write_csv(out / "bucket_paired_deltas.csv", bucket_rows, [
        "split", "seed", "bucket_type", "bucket", "n_samples", "n_speakers", "labels_present", "inferential",
        *[f"delta_{metric}" for metric in METRICS],
    ])


def write_prototype_coverage(out: Path) -> None:
    config = read_json(out / "config.json")
    index = load_index(resolve(config["upstream_checkpoint_index"]))
    overlap = read_json(out / "speaker_overlap_summary.json")
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        cfg = read_json(resolve(index[("A4", seed)]["config"]))
        path = resolve_frozen_prototype(cfg["speaker_prototype_path"])
        prototypes = load_speaker_prototypes(path)
        speakers = [speaker for speaker in prototypes if speaker != "__GLOBAL__"]
        active = [int(np.asarray(prototypes[speaker]["mask"]).sum()) for speaker in speakers]
        distribution = Counter(active)
        rows.append({
            "seed": seed, "prototype_path": relative(path), "prototype_sha256": sha256(path),
            "stored_entries_including_global": len(prototypes), "speaker_entries": len(speakers),
            "active_slots_non_global": sum(active),
            "speakers_with_1_slot": distribution.get(1, 0), "speakers_with_2_slots": distribution.get(2, 0),
            "speakers_with_3_slots": distribution.get(3, 0), "speakers_with_4_slots": distribution.get(4, 0),
            "dev_direct_samples": overlap["coverage"]["dev"]["direct_samples"],
            "dev_fallback_samples": overlap["coverage"]["dev"]["fallback_samples"],
            "test_direct_samples": overlap["coverage"]["test"]["direct_samples"],
            "test_fallback_samples": overlap["coverage"]["test"]["fallback_samples"],
        })
    write_csv(out / "prototype_coverage_by_seed.csv", rows, list(rows[0]))


def finalize(out: Path) -> None:
    write_paired_outputs(out)
    write_prototype_coverage(out)
    dev_metrics = read_csv(out / "dev_scenario_metrics.csv")
    test_metrics = read_csv(out / "test_scenario_metrics.csv")
    deltas = paired_deltas([{**row, **{metric: float(row[metric]) for metric in METRICS}, "seed": int(row["seed"])} for row in dev_metrics + test_metrics])
    write_csv(out / "paired_deltas.csv", deltas, ["comparison", "control", "treatment", "seed", "split", *[f"delta_{metric}" for metric in METRICS]])
    matched = {split: mean_delta(deltas, "memory_matched", split) for split in ("dev", "test")}
    specific_global = {split: mean_delta(deltas, "specific_vs_global", split) for split in ("dev", "test")}
    specific_permuted = {split: mean_delta(deltas, "specific_vs_permuted", split) for split in ("dev", "test")}
    unseen = {split: bucket_delta(out, split, "seen_status", "unseen") for split in ("dev", "test")}
    low_frequency = {
        split: {bucket: bucket_delta(out, split, "frequency", bucket) for bucket in ("rare", "low")}
        for split in ("dev", "test")
    }
    matched_pass = all(
        (matched[split]["macro_f1"] > 0 or matched[split]["minority_f1"] > 0)
        and matched[split]["accuracy"] >= -0.01 and matched[split]["weighted_f1"] >= -0.01
        and positive_seed_count(deltas, split) >= 2
        for split in ("dev", "test")
    )
    unseen_pass = all(
        unseen[split] is not None and not (unseen[split]["macro_f1"] < 0 and unseen[split]["minority_f1"] < 0)
        for split in ("dev", "test")
    )
    stress_pass = all(
        specific_global[split]["macro_f1"] <= 0.01 and specific_permuted[split]["macro_f1"] <= 0.01
        for split in ("dev", "test")
    )
    low_pass = True
    for split in ("dev", "test"):
        valid = [value for value in low_frequency[split].values() if value is not None]
        if valid and all(value["macro_f1"] < 0 and value["minority_f1"] < 0 for value in valid):
            low_pass = False
    general_go = matched_pass and unseen_pass and stress_pass and low_pass
    decision = "GO_GENERAL_SPEAKER_MODELING" if general_go else "NO_GO_GENERAL_SPEAKER_MODELING_DATASET_SPECIFIC_MEMORY"
    gate = {
        "step": 30, "status": "complete", "decision": decision, "training_performed": False,
        "test_evaluated": True, "test_used_for_selection": False, "candidate_reselected": False,
        "matched_mean_deltas": matched, "matched_positive_seed_count": {split: positive_seed_count(deltas, split) for split in ("dev", "test")},
        "speaker_specific_minus_global_mean_deltas": specific_global,
        "speaker_specific_minus_permuted_mean_deltas": specific_permuted,
        "unseen_matched_mean_deltas": unseen, "low_frequency_matched_mean_deltas": low_frequency,
        "criteria": {"matched_pass": matched_pass, "unseen_pass": unseen_pass, "stress_pass": stress_pass, "low_frequency_pass": low_pass},
        "true_speaker_independent_protocol_completed": False,
        "held_out_dev_protocol": "prototype-held-out evaluation only; no speaker-excluded retraining",
        "claim_policy": "general/unseen-speaker modeling claim allowed" if general_go else "memory must be described as MELD-specific experimental component; no general or speaker-independent claim",
        "step31_unlocked": True, "later_steps_unlocked": False,
    }
    write_json(out / "final_gate.json", gate)
    write_json(out / "gate.json", gate)
    summary = read_json(out / "speaker_overlap_summary.json")
    report = [
        "# Phase III Step 30 Speaker Generalization Results", "",
        "## Execution", "",
        "- Evaluation-only on frozen Step 28 A0/A4 checkpoints, seeds 42/43/44; no training or candidate reselection.",
        "- Dev was completed and gated before the frozen test evaluation.",
        "- Prototype lookup alone was changed; context same-speaker relations were held fixed.", "",
        "## Speaker overlap and fallback", "",
        f"- Unique train/dev/test speakers: {summary['split_statistics']['train']['unique_speakers']}/{summary['split_statistics']['dev']['unique_speakers']}/{summary['split_statistics']['test']['unique_speakers']}.",
        f"- Train-dev/train-test unique overlap: {summary['unique_intersections']['train_dev']}/{summary['unique_intersections']['train_test']}.",
        f"- Dev direct/fallback samples: {summary['coverage']['dev']['direct_samples']}/{summary['coverage']['dev']['fallback_samples']}; test: {summary['coverage']['test']['direct_samples']}/{summary['coverage']['test']['fallback_samples']}.", "",
        "## Three-seed mean deltas", "",
        "Matched A4 speaker-specific minus A0:",
        f"- Dev: {matched['dev']}", f"- Test: {matched['test']}", "",
        "A4 speaker-specific minus global-only:",
        f"- Dev: {specific_global['dev']}", f"- Test: {specific_global['test']}", "",
        "A4 speaker-specific minus ID-permuted:",
        f"- Dev: {specific_permuted['dev']}", f"- Test: {specific_permuted['test']}", "",
        "## Gate and limitations", "",
        f"- Final decision: `{decision}`.",
        f"- Criteria: {gate['criteria']}.",
        "- Natural unseen buckets are small and use the single train-global fallback; bucket metrics are descriptive where support/label coverage is insufficient.",
        "- The prototype-held-out dev diagnostic does not exclude speakers from network training and is not a speaker-independent protocol.",
        "- The initial pre-run manifest binds upstream artifacts, feature inputs, and frozen permutations but not the newly created runner; the final manifest binds the runner hash.",
        "- No result in this step changes the Step 28 memory instability finding or the Step 29 DG negative result.",
    ]
    (out / "step30_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest_files = [
        "protocol.md", "config.json", "pre_run_manifest.json", "speaker_manifest.csv", "speaker_overlap_summary.json",
        "prototype_coverage_by_seed.csv",
        "speaker_permutation_manifest.csv", "dev_gate.json", "dev_scenario_metrics.csv", "dev_per_class.csv",
        "dev_predictions.csv", "dev_bucket_metrics.csv", "dev_paired_deltas.csv", "test_scenario_metrics.csv",
        "test_per_class.csv", "test_predictions.csv", "test_bucket_metrics.csv", "test_paired_deltas.csv",
        "paired_deltas.csv", "paired_predictions.csv", "bucket_paired_deltas.csv", "fallback_failure_cases.csv",
        "step30_results.md", "final_gate.json", "run.log",
    ]
    manifest = {
        "step": 30, "status": "complete", "decision": decision,
        "row_counts": {name: sum(1 for _ in (out / name).open(encoding="utf-8")) - 1 for name in manifest_files if name.endswith(".csv")},
        "sha256": {name: sha256(out / name) for name in manifest_files},
        "execution": {"device": "cuda" if torch.cuda.is_available() else "cpu", "training_performed": False, "test_used_for_selection": False},
        "limitations": ["pre_run_manifest.json did not bind the Step 30 runner hash; final_manifest.json binds it after evaluation"],
    }
    manifest["sha256"]["scripts/run_phase3_step30_speaker_generalization.py"] = sha256(ROOT / "scripts/run_phase3_step30_speaker_generalization.py")
    write_json(out / "final_manifest.json", manifest)


def smoke_test(args: argparse.Namespace) -> None:
    speakers = ["A", "B", "C", "D"]
    mapping = derangement(speakers, 42)
    if set(mapping) != set(speakers) or set(mapping.values()) != set(speakers) or any(key == value for key, value in mapping.items()):
        raise AssertionError("derangement smoke failed")
    gold = [0, 1, 2, 3, 4, 5, 6]
    metric, classes = metric_rows(gold, gold)
    if metric["accuracy"] != 1.0 or len(classes) != 7:
        raise AssertionError("metric smoke failed")
    print(json.dumps({"status": "PASS", "test_evaluated": False, "mapping": mapping}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase III Step 30 frozen speaker generalization audit")
    parser.add_argument("--stage", choices=("audit", "dev", "test", "finalize"), default="audit")
    parser.add_argument("--output_dir", default="outputs/phase3_ieee_access/step30_speaker_generalization")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test(args)
        return
    if args.stage == "audit":
        audit(args)
    elif args.stage == "finalize":
        finalize(resolve(args.output_dir))
    else:
        run_split(args, args.stage)


if __name__ == "__main__":
    main()
