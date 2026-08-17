#!/usr/bin/env python3
"""Phase III Step 28 orthogonal A0-A7 ablation with hard split gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from data.feature_extractor_v2 import load_cached_features
from run_main_table import read_structure_config, run_training
from training.train import DATASET_LABELS, MELDFeatureDataset, build_model, load_speaker_prototypes

LABELS = DATASET_LABELS["meld"]
MINORITY = {"fear", "disgust", "sadness"}
SCENARIOS = ("full", "missing_video", "random_missing")
PAIRS = (
    ("masking", "A0", "A1"),
    ("dg_without_masking", "A0", "A2"),
    ("dg_matched_masking", "A1", "A3"),
    ("memory_independent", "A0", "A4"),
    ("s05_memory_path", "A4", "A5"),
    ("memory_dg_path", "A3", "A6"),
    ("s05_full_path", "A6", "A7"),
)
MATRIX = {
    "A0": dict(memory=False, dg=False, masking=False, s05=False),
    "A1": dict(memory=False, dg=False, masking=True, s05=False),
    "A2": dict(memory=False, dg=True, masking=False, s05=False),
    "A3": dict(memory=False, dg=True, masking=True, s05=False),
    "A4": dict(memory=True, dg=False, masking=False, s05=False),
    "A5": dict(memory=True, dg=False, masking=False, s05=True),
    "A6": dict(memory=True, dg=True, masking=True, s05=False),
    "A7": dict(memory=True, dg=True, masking=True, s05=True),
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def rel(path: str | Path) -> str:
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def spec(row_id: str, bundle: Path) -> Dict[str, Any]:
    bits = MATRIX[row_id]
    tokens: List[str] = ["--fusion", "tv_disagreement" if bits["dg"] else "gated", "{structure_args}"]
    if bits["dg"]:
        tokens += [
            "--projection_aux_weight", "0.3", "--disagreement_gate_min", "0.1",
            "--disagreement_gate_temperature", "1.5", "--disagreement_gate_bias_init", "1.0",
        ]
    if bits["masking"]:
        tokens += ["--train_video_mask_prob", "0.5"]
        if not bits["dg"]:
            tokens += ["--train_video_mask_apply_all_fusions"]
    if bits["memory"]:
        tokens += ["{memory_args}"]
    if bits["s05"]:
        tokens += ["--augmentation_bundle", str(bundle)]
    return {
        "name": row_id,
        "display_name": row_id,
        "description": json.dumps(bits, sort_keys=True),
        "args": tokens,
    }


def structure_args(path: Path) -> List[str]:
    cfg = read_structure_config(path)
    return [
        "--structure_mode", cfg["structure_mode"], "--context_len", str(cfg["context_len"]),
        "--context_max_distance", str(cfg["context_max_distance"]),
        "--structure_gate_scale", str(cfg["structure_gate_scale"]),
        "--relation_dropout", str(cfg["relation_dropout"]),
        "--relation_embedding_init_std", str(cfg["relation_embedding_init_std"]),
    ]


def common_args(args: argparse.Namespace, feature_dir: Path) -> List[str]:
    values = [
        "--dataset", "meld", "--feature_dir", str(feature_dir), "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size), "--hidden_dim", str(args.hidden_dim),
        "--num_heads", str(args.num_heads), "--dropout", str(args.dropout), "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay), "--patience", str(args.patience),
        "--loss_type", "ce", "--class_weight_mode", "none",
    ]
    if args.smoke_test:
        values += ["--smoke_test", "--feat_dim", "64"]
    return values


def train_row(row_id: str, seed: int, args: argparse.Namespace, out: Path) -> Dict[str, Any]:
    feature_dir = resolve(args.feature_dir)
    bundle = resolve(args.augmentation_bundle)
    proto = out / "speaker_prototypes" / f"{row_id}_seed{seed}_slots4.npz"
    return run_training(
        spec(row_id, bundle), seed, out, out / "logs", common_args(args, feature_dir),
        structure_args(resolve(args.structure_config)), proto,
    )


def eval_args(config_path: Path) -> SimpleNamespace:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return SimpleNamespace(
        dataset="meld", feat_dim=int(cfg.get("feat_dim", 3584)), hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_classes=7, fusion=cfg.get("fusion", "gated"), dropout=float(cfg.get("dropout", 0.3)),
        num_heads=int(cfg.get("num_heads", 4)), with_style=False, rag_mode="none", rag_top_k=5,
        structure_mode=cfg.get("structure_mode", "speaker_distance"), context_len=int(cfg.get("context_len", 5)),
        context_max_distance=int(cfg.get("context_max_distance", 20)),
        structure_gate_scale=float(cfg.get("structure_gate_scale", 0.5)),
        relation_dropout=float(cfg.get("relation_dropout", 0.2)),
        relation_embedding_init_std=float(cfg.get("relation_embedding_init_std", 0.01)),
        speaker_memory_mode=cfg.get("speaker_memory_mode", "none"),
        speaker_memory_slots=int(cfg.get("speaker_memory_slots", 4)),
        memory_gate_scale=float(cfg.get("memory_gate_scale", 0.5)),
        speaker_prototype_path=cfg.get("speaker_prototype_path"),
        disagreement_gate_min=float(cfg.get("disagreement_gate_min", 0.0)),
        disagreement_gate_temperature=float(cfg.get("disagreement_gate_temperature", 1.0)),
        disagreement_gate_bias_init=float(cfg.get("disagreement_gate_bias_init", 0.0)),
    )


def dataset(feature_dir: Path, split: str, cfg: SimpleNamespace) -> MELDFeatureDataset:
    data = load_cached_features(str(feature_dir), split)
    speaker_path = feature_dir / f"{split}_speakers.npy"
    if speaker_path.is_file():
        data["speakers"] = np.load(speaker_path, allow_pickle=True)
    prototypes = None
    if cfg.speaker_memory_mode == "prototype":
        prototypes = load_speaker_prototypes(resolve(cfg.speaker_prototype_path))
    return MELDFeatureDataset(
        data["video_features"], data["text_features"], data["labels"],
        data.get("dialogue_ids"), data.get("utterance_ids"), speaker_ids=data.get("speakers"),
        context_len=cfg.context_len, context_max_distance=cfg.context_max_distance,
        speaker_prototypes=prototypes, speaker_memory_slots=cfg.speaker_memory_slots,
    )


def mask_tensor(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.reshape([len(mask)] + [1] * (value.ndim - 1)), torch.zeros_like(value), value)


@torch.no_grad()
def evaluate(run: Mapping[str, Any], args: argparse.Namespace, split: str, scenario: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = eval_args(resolve(run["config"]))
    data = dataset(resolve(args.feature_dir), split, cfg)
    loader = DataLoader(data, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(resolve(run["checkpoint"]), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    gold: List[int] = []
    pred: List[int] = []
    samples: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        if scenario == "full":
            row_mask = torch.zeros(len(batch["label"]), dtype=torch.bool, device=device)
        elif scenario == "missing_video":
            row_mask = torch.ones(len(batch["label"]), dtype=torch.bool, device=device)
        else:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(run["seed"]) + batch_idx + (1_000_000 if split == "test" else 0))
            row_mask = torch.rand(len(batch["label"]), generator=gen, device=device) < 0.5
        output = model(
            video_feat=mask_tensor(batch["video_feat"].to(device), row_mask),
            text_feat=batch["text_feat"].to(device), return_attention=True,
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
        for pos, idx in enumerate(indices):
            samples.append({
                "experiment": run["experiment"], "seed": int(run["seed"]), "split": split,
                "scenario": scenario, "sample_index": int(idx),
                "dialogue_id": str(data.dialogue_ids[idx]) if data.dialogue_ids is not None else "",
                "utterance_id": str(data.utterance_ids[idx]) if data.utterance_ids is not None else "",
                "speaker": str(data.speaker_ids[idx]) if data.speaker_ids is not None else "",
                "gold": int(batch_gold[pos]), "prediction": int(batch_pred[pos]),
                "gold_label": LABELS[int(batch_gold[pos])], "prediction_label": LABELS[int(batch_pred[pos])],
                "video_masked": bool(row_mask[pos].item()),
                **{f"prob_{label}": float(probs[pos, j]) for j, label in enumerate(LABELS)},
            })
        gold.extend(batch_gold.tolist())
        pred.extend(batch_pred.tolist())
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=range(7), zero_division=0)
    per_class = [{
        "experiment": run["experiment"], "seed": int(run["seed"]), "split": split, "scenario": scenario,
        "label": label, "precision": float(precision[i]), "recall": float(recall[i]),
        "f1": float(f1[i]), "support": int(support[i]),
    } for i, label in enumerate(LABELS)]
    metric = {
        "experiment": run["experiment"], "seed": int(run["seed"]), "split": split, "scenario": scenario,
        "accuracy": float(accuracy_score(gold, pred)),
        "weighted_f1": float(f1_score(gold, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "minority_f1": float(np.mean([row["f1"] for row in per_class if row["label"] in MINORITY])),
    }
    return metric, per_class, samples


def pair_deltas(metrics: Sequence[Mapping[str, Any]], seed: int) -> List[Dict[str, Any]]:
    keyed = {(row["experiment"], row["scenario"]): row for row in metrics if int(row["seed"]) == seed and row["split"] == "dev"}
    rows: List[Dict[str, Any]] = []
    for factor, control, treatment in PAIRS:
        full_c, full_t = keyed[(control, "full")], keyed[(treatment, "full")]
        row: Dict[str, Any] = {"factor": factor, "control": control, "treatment": treatment, "seed": seed}
        for scenario in SCENARIOS:
            c, t = keyed[(control, scenario)], keyed[(treatment, scenario)]
            for metric in ("accuracy", "weighted_f1", "macro_f1", "minority_f1"):
                row[f"delta_{scenario}_{metric}"] = float(t[metric]) - float(c[metric])
        row["passes_full_gate"] = bool(
            (row["delta_full_macro_f1"] > 0 or row["delta_full_minority_f1"] > 0)
            and row["delta_full_accuracy"] >= -0.01 and row["delta_full_weighted_f1"] >= -0.01
        )
        row["robustness_positive"] = bool(
            row["delta_missing_video_macro_f1"] > 0 or row["delta_random_missing_macro_f1"] > 0
        )
        rows.append(row)
    return rows


def update_run_hashes(runs: List[Dict[str, Any]]) -> None:
    for run in runs:
        if run.get("success"):
            run["config_sha256"] = sha256(resolve(run["config"]))
            run["checkpoint_sha256"] = sha256(resolve(run["checkpoint"]))


def save_evidence(out: Path, runs: List[Dict[str, Any]], metrics: List[Dict[str, Any]], classes: List[Dict[str, Any]], samples: List[Dict[str, Any]], prefix: str) -> None:
    update_run_hashes(runs)
    write_csv(out / "checkpoint_index.csv", runs, [
        "experiment", "seed", "success", "config", "config_sha256", "checkpoint", "checkpoint_sha256",
        "log", "train_seconds", "command",
    ])
    write_csv(out / f"{prefix}_metrics.csv", metrics, ["experiment", "seed", "split", "scenario", "accuracy", "weighted_f1", "macro_f1", "minority_f1"])
    write_csv(out / f"{prefix}_per_class.csv", classes, ["experiment", "seed", "split", "scenario", "label", "precision", "recall", "f1", "support"])
    sample_fields = ["experiment", "seed", "split", "scenario", "sample_index", "dialogue_id", "utterance_id", "speaker", "gold", "prediction", "gold_label", "prediction_label", "video_masked"] + [f"prob_{label}" for label in LABELS]
    write_csv(out / f"{prefix}_predictions.csv", samples, sample_fields)


def validate_initial_gate(out: Path) -> None:
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    if gate.get("decision") != "GO_SEED42_DEV_SCREEN" or gate.get("training_unlocked") is not True:
        raise ValueError("Step 28 protocol gate does not unlock seed-42 dev screen")
    if gate.get("test_evaluation_unlocked") is not False or gate.get("test_evaluated") is not False:
        raise ValueError("test must be locked before screen")
    manifest = json.loads((out / "pre_run_manifest.json").read_text(encoding="utf-8"))
    # Deliberately do not open/hash test inputs during screen.
    for section in ("training_inputs", "code"):
        for path, expected in manifest[section].items():
            actual = sha256(resolve(path))
            if actual != expected:
                raise ValueError(f"pre-run hash mismatch for {path}: {actual} != {expected}")


def screen(args: argparse.Namespace, out: Path, run_log: Path) -> None:
    if not args.smoke_test:
        validate_initial_gate(out)
    if args.screen_seed != 42:
        raise ValueError("screen seed is frozen to 42")
    rows = list(MATRIX)[:2] if args.smoke_test else list(MATRIX)
    runs: List[Dict[str, Any]] = []
    metrics: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    for row_id in rows:
        log(run_log, f"train {row_id} seed42; train/dev only")
        run = train_row(row_id, 42, args, out)
        runs.append(run)
        if not run.get("success"):
            raise RuntimeError(f"training failed: {run}")
        if args.smoke_test:
            continue
        for scenario in SCENARIOS:
            metric, per_class, predictions = evaluate(run, args, "dev", scenario)
            metrics.append(metric); classes.extend(per_class); samples.extend(predictions)
    if args.smoke_test:
        update_run_hashes(runs)
        write_json(out / "smoke_result.json", {"status": "PASS", "test_evaluated": False, "runs": runs})
        return
    deltas = pair_deltas(metrics, 42)
    frozen = sorted({row for delta in deltas if delta["passes_full_gate"] for row in (delta["control"], delta["treatment"])})
    save_evidence(out, runs, metrics, classes, samples, "screen_dev")
    write_csv(out / "paired_deltas_seed42_dev.csv", deltas, list(deltas[0]))
    gate = {
        "step": 28, "status": "seed42_dev_screen_complete", "decision": "GO_CONFIRM_DEV" if frozen else "NO_GO_CONFIRMATION",
        "screen_seed": 42, "test_evaluated": False, "confirmation_unlocked": bool(frozen),
        "test_evaluation_unlocked": False, "frozen_rows": frozen, "pair_decisions": deltas,
    }
    write_json(out / "dev_selection_gate.json", gate)
    write_json(out / "gate.json", gate)
    log(run_log, f"screen complete; frozen rows={frozen}; test remains locked")


def load_runs(out: Path) -> List[Dict[str, Any]]:
    rows = read_csv(out / "checkpoint_index.csv")
    for row in rows:
        row["seed"] = int(row["seed"])
        row["success"] = row["success"].lower() == "true"
    return rows


def confirm_dev(args: argparse.Namespace, out: Path, run_log: Path) -> None:
    gate = json.loads((out / "dev_selection_gate.json").read_text(encoding="utf-8"))
    if gate.get("confirmation_unlocked") is not True or gate.get("test_evaluated") is not False:
        raise ValueError("dev confirmation is not unlocked")
    frozen = gate["frozen_rows"]
    runs = load_runs(out)
    for row_id in frozen:
        for seed in (43, 44):
            log(run_log, f"confirm-dev train {row_id} seed{seed}")
            run = train_row(row_id, seed, args, out)
            if not run.get("success"):
                raise RuntimeError(f"training failed: {run}")
            runs.append(run)
    metrics: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    for run in runs:
        if run["experiment"] not in frozen:
            continue
        for scenario in SCENARIOS:
            metric, per_class, predictions = evaluate(run, args, "dev", scenario)
            metrics.append(metric); classes.extend(per_class); samples.extend(predictions)
    save_evidence(out, runs, metrics, classes, samples, "confirmation_dev")
    means: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row_id in frozen:
        for scenario in SCENARIOS:
            subset = [row for row in metrics if row["experiment"] == row_id and row["scenario"] == scenario]
            means[(row_id, scenario)] = {metric: float(np.mean([row[metric] for row in subset])) for metric in ("accuracy", "weighted_f1", "macro_f1", "minority_f1")}
    pair_results = []
    eligible_pairs = {
        (row["factor"], row["control"], row["treatment"])
        for row in gate.get("pair_decisions", [])
        if row.get("passes_full_gate") is True
    }
    for factor, control, treatment in PAIRS:
        if (factor, control, treatment) not in eligible_pairs:
            continue
        if control not in frozen or treatment not in frozen:
            continue
        c, t = means[(control, "full")], means[(treatment, "full")]
        deltas = {metric: t[metric] - c[metric] for metric in c}
        passed = (deltas["macro_f1"] > 0 or deltas["minority_f1"] > 0) and deltas["accuracy"] >= -0.01 and deltas["weighted_f1"] >= -0.01
        pair_results.append({"factor": factor, "control": control, "treatment": treatment, **{f"delta_{k}": v for k, v in deltas.items()}, "passes_three_seed_dev_gate": passed})
    test_rows = sorted({row for pair in pair_results if pair["passes_three_seed_dev_gate"] for row in (pair["control"], pair["treatment"])})
    confirm_gate = {
        "step": 28, "status": "three_seed_dev_confirmation_complete", "test_evaluated": False,
        "test_evaluation_unlocked": bool(test_rows), "frozen_test_rows": test_rows,
        "candidate_not_reselected": True, "pair_results": pair_results,
        "decision": "GO_FROZEN_TEST_CONFIRMATION" if test_rows else "NO_GO_TEST",
    }
    write_json(out / "confirmation_gate.json", confirm_gate)
    write_json(out / "gate.json", confirm_gate)
    log(run_log, f"confirm-dev complete; test rows={test_rows}")


def confirm_test(args: argparse.Namespace, out: Path, run_log: Path) -> None:
    gate = json.loads((out / "confirmation_gate.json").read_text(encoding="utf-8"))
    if gate.get("test_evaluation_unlocked") is not True or gate.get("test_evaluated") is not False:
        raise ValueError("frozen test confirmation is not unlocked")
    manifest = json.loads((out / "pre_run_manifest.json").read_text(encoding="utf-8"))
    for path, expected_hash in manifest["test_inputs_recorded_but_locked"].items():
        actual_hash = sha256(resolve(path))
        if actual_hash != expected_hash:
            raise ValueError(f"frozen test hash mismatch for {path}: {actual_hash} != {expected_hash}")
    rows = gate["frozen_test_rows"]
    runs = [run for run in load_runs(out) if run["experiment"] in rows and run["seed"] in (42, 43, 44)]
    expected = {(row, seed) for row in rows for seed in (42, 43, 44)}
    actual = {(run["experiment"], run["seed"]) for run in runs}
    if actual != expected:
        raise ValueError(f"checkpoint set mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    metrics: List[Dict[str, Any]] = []
    classes: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    log(run_log, "frozen rows verified; one-shot test confirmation starts")
    for run in runs:
        for scenario in SCENARIOS:
            metric, per_class, predictions = evaluate(run, args, "test", scenario)
            metrics.append(metric); classes.extend(per_class); samples.extend(predictions)
    save_evidence(out, load_runs(out), metrics, classes, samples, "confirmation_test")
    final_gate = {
        **gate, "status": "formal_step28_results_ready", "test_evaluated": True,
        "test_evaluated_after_dev_freeze": True, "test_evaluation_unlocked": False,
        "decision": "STEP28_RESULTS_READY_FOR_ATTRIBUTION_REVIEW",
    }
    write_json(out / "contribution_gate.json", final_gate)
    write_json(out / "gate.json", final_gate)
    log(run_log, "test confirmation complete; attribution review required before Step 29")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "confirm-dev", "confirm-test"), required=True)
    parser.add_argument("--feature_dir", default="datasets/MELD/features_v2")
    parser.add_argument("--structure_config", default="configs/main_structure_context.json")
    parser.add_argument("--augmentation_bundle", default="server_results/phase2_minority_augmentation/bundles/S05.npz")
    parser.add_argument("--output_dir", default="outputs/phase3_ieee_access/step28_factorial_ablation")
    parser.add_argument("--screen_seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test and args.stage != "screen":
        parser.error("--smoke_test is only valid for screen")
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "speaker_prototypes").mkdir(exist_ok=True)
    run_log = out / "run.log"
    if args.stage == "screen" and not args.smoke_test:
        if (out / "dev_selection_gate.json").exists():
            parser.error("formal screen already exists")
        run_log.write_text("", encoding="utf-8")
    if args.stage == "screen":
        screen(args, out, run_log)
    elif args.stage == "confirm-dev":
        confirm_dev(args, out, run_log)
    else:
        confirm_test(args, out, run_log)


if __name__ == "__main__":
    main()
