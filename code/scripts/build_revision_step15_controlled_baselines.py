#!/usr/bin/env python3
"""Build the Step 15 controlled-baseline table from frozen Step 1 CSV evidence.

This script performs no training or inference and never reads the MELD feature or
test arrays. It verifies the downloaded Step 1 final manifest before aggregating
the already-frozen test_metrics.csv rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


MODELS = [
    "text_only_mlp",
    "video_only_mlp",
    "concat_mlp",
    "turn_crossattention",
]
MODEL_DISPLAY = {
    "text_only_mlp": "Text-only MLP",
    "video_only_mlp": "Video-only MLP",
    "concat_mlp": "Concat+MLP",
    "turn_crossattention": "Turn CrossAttention",
}
SEEDS = [42, 43, 44, 45, 46]
SCENARIOS = ["full", "missing_video"]
SCENARIO_DISPLAY = {"full": "Full", "missing_video": "Missing video"}
METRICS = ["accuracy", "weighted_f1", "macro_f1", "f1_fds"]
SERVER_PREFIX = "outputs/ieee_revision/step1_baselines/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_final_manifest(server_root: Path, manifest: dict[str, Any]) -> int:
    failures: list[str] = []
    entries = manifest.get("files", [])
    if len(entries) != 111:
        failures.append(f"expected 111 final-manifest entries, found {len(entries)}")
    for entry in entries:
        recorded = str(entry["path"])
        relative = recorded[len(SERVER_PREFIX):] if recorded.startswith(SERVER_PREFIX) else recorded
        path = server_root / relative
        if not path.is_file():
            failures.append(f"missing manifest file: {recorded}")
            continue
        if path.stat().st_size != int(entry["bytes"]):
            failures.append(f"size mismatch: {recorded}")
        if sha256(path) != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {recorded}")
    if failures:
        raise RuntimeError("final-manifest verification failed:\n" + "\n".join(failures))
    return len(entries)


def verify_contract(
    step1_gate: dict[str, Any],
    config: dict[str, Any],
    result_gate: dict[str, Any],
    final_manifest: dict[str, Any],
) -> None:
    for name, obj in (
        ("Step 1 gate", step1_gate),
        ("config", config),
        ("result gate", result_gate),
        ("final manifest", final_manifest),
    ):
        if obj.get("models") != MODELS:
            raise RuntimeError(f"{name} model set/order differs from the preregistered contract")
        if obj.get("seeds") != SEEDS:
            raise RuntimeError(f"{name} seed set/order differs from seeds 42--46")
    if config.get("test_scenarios") != SCENARIOS or result_gate.get("scenarios") != SCENARIOS:
        raise RuntimeError("scenario set/order differs from full and missing_video")
    if config.get("checkpoint_selection_split") != "dev":
        raise RuntimeError("checkpoint selection split is not dev")
    if config.get("checkpoint_selection_metric") != "weighted_f1":
        raise RuntimeError("checkpoint selection metric is not weighted_f1")
    if result_gate.get("test_used_for_selection") is not False:
        raise RuntimeError("result gate does not explicitly prohibit test-based selection")
    cross = step1_gate.get("crossattention_decision", {})
    if not (
        cross.get("padding_mask_used") is True
        and cross.get("bidirectional") is True
        and cross.get("legacy_single_token_attention_used") is False
    ):
        raise RuntimeError("CrossAttention contract is not the legal masked dialogue-turn version")


def verify_metrics(rows: list[dict[str, str]]) -> None:
    if len(rows) != len(MODELS) * len(SEEDS) * len(SCENARIOS):
        raise RuntimeError(f"expected 40 test metric rows, found {len(rows)}")
    observed: set[tuple[str, int, str]] = set()
    for row in rows:
        if row["split"] != "test":
            raise RuntimeError("test_metrics.csv contains a non-test row")
        key = (row["model"], int(row["seed"]), row["scenario"])
        if key in observed:
            raise RuntimeError(f"duplicate test metric group: {key}")
        observed.add(key)
        for metric in METRICS:
            value = float(row[metric])
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"out-of-range {metric} in {key}: {value}")
    expected = {(model, seed, scenario) for model in MODELS for seed in SEEDS for scenario in SCENARIOS}
    if observed != expected:
        raise RuntimeError(f"metric group mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")


def verify_per_class(rows: list[dict[str, str]]) -> None:
    expected_count = len(MODELS) * len(SEEDS) * len(SCENARIOS) * 7
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} per-class rows, found {len(rows)}")
    counts: dict[tuple[str, int, str], int] = {}
    for row in rows:
        if row["split"] != "test":
            raise RuntimeError("test_per_class.csv contains a non-test row")
        key = (row["model"], int(row["seed"]), row["scenario"])
        counts[key] = counts.get(key, 0) + 1
    if set(counts.values()) != {7}:
        raise RuntimeError("not every model/seed/scenario group has seven class rows")


def verify_checkpoints(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    if len(rows) != len(MODELS) * len(SEEDS):
        raise RuntimeError(f"expected 20 checkpoint rows, found {len(rows)}")
    seen: set[tuple[str, int]] = set()
    training: dict[str, list[float]] = {model: [] for model in MODELS}
    for row in rows:
        key = (row["model"], int(row["seed"]))
        if key in seen:
            raise RuntimeError(f"duplicate checkpoint row: {key}")
        seen.add(key)
        if row["success"].lower() != "true":
            raise RuntimeError(f"unsuccessful formal checkpoint row: {key}")
        training[row["model"]].append(float(row["train_seconds"]))
    expected = {(model, seed) for model in MODELS for seed in SEEDS}
    if seen != expected:
        raise RuntimeError(f"checkpoint group mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    return {
        model: {
            "mean_seconds": statistics.mean(values),
            "sample_sd_seconds": statistics.stdev(values),
        }
        for model, values in training.items()
    }


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for scenario in SCENARIOS:
            group = [r for r in rows if r["model"] == model and r["scenario"] == scenario]
            seeds = sorted(int(r["seed"]) for r in group)
            if seeds != SEEDS:
                raise RuntimeError(f"seed mismatch for {model}/{scenario}: {seeds}")
            record: dict[str, Any] = {"model": model, "scenario": scenario, "n": len(group)}
            for metric in METRICS:
                values = [float(r[metric]) for r in group]
                record[metric] = {
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                }
            output.append(record)
    return output


def metric_tex(record: dict[str, Any], metric: str) -> str:
    item = record[metric]
    return f"{item['mean']:.4f} $\\pm$ {item['sample_sd']:.4f}"


def render_tex(records: list[dict[str, Any]], test_metrics_hash: str) -> str:
    lines = [
        "% AUTO-GENERATED by scripts/build_revision_step15_controlled_baselines.py.",
        f"% Frozen source test_metrics.csv SHA-256: {test_metrics_hash}",
        "% Values are mean plus/minus sample SD over the common seeds 42--46; do not edit manually.",
    ]
    for record in records:
        lines.append(
            f"{MODEL_DISPLAY[record['model']]} & {SCENARIO_DISPLAY[record['scenario']]} & {record['n']} & "
            f"{metric_tex(record, 'accuracy')} & {metric_tex(record, 'weighted_f1')} & "
            f"{metric_tex(record, 'macro_f1')} & {metric_tex(record, 'f1_fds')} \\\\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/submission/source/generated/controlled_baseline_rows.tex"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("outputs/ieee_revision/step15_baseline_presentation/table_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    server = root / "server_results/ieee_revision/step1_baselines"
    output = args.output if args.output.is_absolute() else root / args.output
    manifest_output = args.manifest_output if args.manifest_output.is_absolute() else root / args.manifest_output

    input_paths = {
        "step1_gate": root / "outputs/ieee_revision/step1_baselines/gate.json",
        "step1_handoff": root / "outputs/ieee_revision/step1_baselines/handoff.md",
        "final_manifest": server / "final_manifest.json",
        "test_metrics": server / "test_metrics.csv",
        "test_per_class": server / "test_per_class.csv",
        "checkpoint_index": server / "checkpoint_index.csv",
        "config": server / "config.json",
        "result_gate": server / "result_gate.json",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Step 15 inputs:\n" + "\n".join(missing))

    step1_gate = read_json(input_paths["step1_gate"])
    config = read_json(input_paths["config"])
    result_gate = read_json(input_paths["result_gate"])
    final_manifest = read_json(input_paths["final_manifest"])
    verify_contract(step1_gate, config, result_gate, final_manifest)
    verified_entries = verify_final_manifest(server, final_manifest)

    metric_rows = read_csv(input_paths["test_metrics"])
    per_class_rows = read_csv(input_paths["test_per_class"])
    checkpoint_rows = read_csv(input_paths["checkpoint_index"])
    verify_metrics(metric_rows)
    verify_per_class(per_class_rows)
    training_seconds = verify_checkpoints(checkpoint_rows)
    records = aggregate(metric_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_tex(records, sha256(input_paths["test_metrics"])), encoding="utf-8")

    parameter_audit = step1_gate["parameter_audit"]
    parameters = {
        model: int(parameter_audit[model]["parameters"])
        for model in MODELS
    }
    manifest = {
        "schema": "ieee_revision_step15_controlled_baseline_table_v1",
        "step": 15,
        "machine_result_verified": True,
        "final_manifest_entries_verified": verified_entries,
        "final_manifest_mismatches": 0,
        "models": MODELS,
        "seeds": SEEDS,
        "scenarios": SCENARIOS,
        "metrics": METRICS,
        "aggregation": "mean and sample standard deviation (ddof=1) across the five common seeds",
        "selection_split": "dev",
        "selection_metric": "weighted_f1",
        "test_used_for_selection": False,
        "crossattention": {
            "dialogue_turn_sequence": True,
            "padding_mask": True,
            "bidirectional": True,
            "single_token_attention": False,
        },
        "parameters_verified_but_not_shown_in_main_table": parameters,
        "training_seconds_verified_but_not_shown_in_main_table": training_seconds,
        "aggregates": records,
        "inputs": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in input_paths.values()
        ],
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256(Path(__file__)),
        },
        "generated_table_source": {
            "path": str(output.relative_to(root)),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "manual_numeric_edits": False,
        "dgf_rerun_on_same_five_seeds": False,
        "dgf_superiority_claim_allowed": False,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"verified {verified_entries}/111 Step 1 manifest entries")
    print(f"wrote {output.relative_to(root)}")
    print(f"wrote {manifest_output.relative_to(root)}")


if __name__ == "__main__":
    main()
