#!/usr/bin/env python3
"""Step 5 audited-failure S05 filtering with dev/test hard separation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_phase3_step28_factorial as step28  # noqa: E402

SEEDS = tuple(range(42, 50))
SCENARIOS = ("full", "missing_video", "random_missing")
EXPERIMENTS = ("no_S05", "all_S05", "filtered_S05")
METRICS = ("accuracy", "weighted_f1", "macro_f1", "f1_fds")
COMPARE = (("all_vs_none", "no_S05", "all_S05"), ("filtered_vs_none", "no_S05", "filtered_S05"), ("filtered_vs_all", "all_S05", "filtered_S05"))


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
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
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


def build_filtered_bundle(cfg: Mapping[str, Any], out: Path) -> Path:
    source = resolve(cfg["original_s05_bundle"])
    rule = read_json(resolve(cfg["filter_manifest"]))
    if sha256(source) != rule["original_s05_sha256"] or rule["status"] != "FROZEN_BEFORE_STEP5_TEST":
        raise ValueError("frozen S05 or filter rule changed")
    data = np.load(source, allow_pickle=False)
    text = np.asarray(data["text_features"])
    indices = np.asarray(data["source_indices"])
    if text.shape != (608, 3584) or text.dtype != np.float32 or indices.shape != (608,):
        raise ValueError("original S05 contract changed")
    excluded = {int(row["s05_row_position_zero_based"]): row for row in rule["excluded"]}
    if len(excluded) != 5:
        raise ValueError("filter must exclude exactly five directly audited failures")
    for position, row in excluded.items():
        row_hash = hashlib.sha256(np.ascontiguousarray(text[position]).tobytes()).hexdigest()
        if int(indices[position]) != int(row["source_index"]) or row_hash != row["feature_row_sha256"]:
            raise ValueError(f"frozen exclusion identity mismatch at S05 row {position}")
    keep = np.asarray([i not in excluded for i in range(len(text))], dtype=bool)
    target = out / "filtered_S05_audited_failures_v1.npz"
    np.savez_compressed(target, text_features=text[keep], source_indices=indices[keep])
    check = np.load(target, allow_pickle=False)
    if check["text_features"].shape != (603, 3584) or check["source_indices"].shape != (603,):
        raise RuntimeError("filtered bundle shape check failed")
    write_json(out / "filtered_bundle_manifest.json", {
        "status": "complete", "experimental_object": "filtered_S05_audited_failures_v1",
        "source_bundle": relative(source), "source_sha256": sha256(source),
        "filter_manifest": relative(resolve(cfg["filter_manifest"])),
        "filter_manifest_sha256": sha256(resolve(cfg["filter_manifest"])),
        "rows_before": 608, "rows_excluded": 5, "rows_after": 603,
        "shape": [603, 3584], "dtype": "float32", "sha256": sha256(target),
        "test_used_to_define_filter": False,
    })
    return target


def train_args(cfg: Mapping[str, Any], bundle: Path, smoke: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        feature_dir=cfg["feature_dir"], structure_config=cfg["structure_config"],
        augmentation_bundle=relative(bundle), epochs=1 if smoke else int(cfg["epochs"]),
        batch_size=min(8, int(cfg["batch_size"])) if smoke else int(cfg["batch_size"]),
        hidden_dim=64 if smoke else int(cfg["hidden_dim"]), num_heads=int(cfg["num_heads"]),
        dropout=float(cfg["dropout"]), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]), patience=1 if smoke else int(cfg["patience"]),
        smoke_test=smoke, screen_seed=42,
    )


def source_prototype(cfg: Mapping[str, Any], seed: int) -> Path:
    return resolve(cfg["a6_prototypes"][str(seed)])


def train_filtered(cfg: Mapping[str, Any], out: Path, bundle: Path, seed: int) -> dict[str, Any]:
    source = source_prototype(cfg, seed)
    expected = cfg["a6_prototype_hashes"][str(seed)]
    if sha256(source) != expected:
        raise ValueError(f"A6 prototype changed for seed {seed}")
    target = out / "speaker_prototypes" / f"A7_seed{seed}_slots4.npz"
    shutil.copyfile(source, target)
    if sha256(target) != expected:
        raise RuntimeError(f"prototype copy failed for seed {seed}")
    run = step28.train_row("A7", seed, train_args(cfg, bundle), out)
    run["experiment"] = "filtered_S05"
    run["source"] = "step5_new"
    if not run.get("success"):
        raise RuntimeError(f"filtered-S05 training failed for seed {seed}")
    if sha256(target) != expected:
        raise RuntimeError(f"training modified frozen prototype for seed {seed}")
    run["config_sha256"] = sha256(resolve(run["config"]))
    run["checkpoint_sha256"] = sha256(resolve(run["checkpoint"]))
    run["prototype"] = relative(target)
    run["prototype_sha256"] = sha256(target)
    return run


def normalize_metric(row: Mapping[str, Any], experiment: str | None = None) -> dict[str, Any]:
    result = dict(row)
    result["experiment"] = experiment or str(result["experiment"])
    result["seed"] = int(result["seed"])
    if "minority_f1" in result and "f1_fds" not in result:
        result["f1_fds"] = result.pop("minority_f1")
    for metric in METRICS:
        result[metric] = float(result[metric])
    return result


def normalize_class(row: Mapping[str, Any], experiment: str | None = None) -> dict[str, Any]:
    result = dict(row); result["experiment"] = experiment or str(result["experiment"]); result["seed"] = int(result["seed"])
    result["support"] = int(result["support"])
    for field in ("precision", "recall", "f1"):
        result[field] = float(result[field])
    return result


def normalize_prediction(row: Mapping[str, Any], experiment: str | None = None) -> dict[str, Any]:
    result = dict(row); result["experiment"] = experiment or str(result["experiment"]); result["seed"] = int(result["seed"])
    result["sample_index"] = int(result["sample_index"]); result["gold"] = int(result["gold"]); result["prediction"] = int(result["prediction"])
    result["video_masked"] = str(result["video_masked"]).lower() in ("true", "1")
    for label in step28.LABELS:
        result[f"prob_{label}"] = float(result[f"prob_{label}"])
    return result


def imported_rows(cfg: Mapping[str, Any], split: str, kind: str) -> list[dict[str, Any]]:
    base = resolve(cfg["step2_results_dir"])
    filename = f"{split}_{kind}.csv"
    rows = read_csv(base / filename)
    output = []
    for row in rows:
        if row["experiment"] not in ("A6", "A7"):
            continue
        name = "no_S05" if row["experiment"] == "A6" else "all_S05"
        if kind == "metrics": output.append(normalize_metric(row, name))
        elif kind == "per_class": output.append(normalize_class(row, name))
        else: output.append(normalize_prediction(row, name))
    return output


def evaluate(run: Mapping[str, Any], cfg: Mapping[str, Any], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []; classes: list[dict[str, Any]] = []; predictions: list[dict[str, Any]] = []
    args = train_args(cfg, resolve(cfg["filtered_bundle_runtime_path"]))
    for scenario in SCENARIOS:
        metric, per_class, samples = step28.evaluate(run, args, split, scenario)
        metrics.append(normalize_metric(metric, "filtered_S05"))
        classes.extend(normalize_class(row, "filtered_S05") for row in per_class)
        predictions.extend(normalize_prediction(row, "filtered_S05") for row in samples)
    return metrics, classes, predictions


def run_fields() -> list[str]:
    return ["experiment", "seed", "source", "success", "config", "config_sha256", "checkpoint", "checkpoint_sha256", "prototype", "prototype_sha256", "log", "train_seconds", "command"]


def metric_fields() -> list[str]:
    return ["experiment", "seed", "split", "scenario", *METRICS]


def class_fields() -> list[str]:
    return ["experiment", "seed", "split", "scenario", "label", "precision", "recall", "f1", "support"]


def prediction_fields() -> list[str]:
    return ["experiment", "seed", "split", "scenario", "sample_index", "dialogue_id", "utterance_id", "speaker", "gold", "prediction", "gold_label", "prediction_label", "video_masked", *[f"prob_{label}" for label in step28.LABELS]]


def validate_predictions(rows: Sequence[Mapping[str, Any]], split: str, cfg: Mapping[str, Any]) -> bool:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: groups[(str(row["experiment"]), int(row["seed"]), str(row["scenario"]))].append(row)
    expected = {(e, s, c) for e in EXPERIMENTS for s in SEEDS for c in SCENARIOS}
    if set(groups) != expected: return False
    n = int(cfg["split_rows"][split]); reference = None
    for key in sorted(groups):
        group = groups[key]
        ids = {(int(r["sample_index"]), str(r["dialogue_id"]), str(r["utterance_id"]), str(r["speaker"]), int(r["gold"])) for r in group}
        if len(group) != n or len(ids) != n: return False
        for row in group:
            probs = np.asarray([float(row[f"prob_{label}"]) for label in step28.LABELS])
            if not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1, atol=2e-6): return False
        if reference is None: reference = ids
        elif ids != reference: return False
    return True


def save_runs(out: Path, runs: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]) -> None:
    write_csv(out / "checkpoint_index.csv", runs, run_fields())
    write_csv(out / "failure_index.csv", failures, ["time", "experiment", "seed", "stage", "error_type", "message"])


def dev_stage(cfg: Mapping[str, Any], out: Path) -> None:
    gate = read_json(out / "initial_gate.json")
    if gate.get("decision") != "GO_STEP5_FILTERED_S05_DEV" or gate.get("test_evaluation_unlocked") is not False:
        raise ValueError("initial gate does not authorize Step 5 dev")
    if not torch.cuda.is_available(): raise RuntimeError("formal Step 5 training requires Slurm CUDA")
    verify_manifest(read_json(out / "pre_run_manifest.json"), ("train_dev_inputs", "code_and_protocol"))
    if (out / "checkpoint_index.csv").exists(): raise RuntimeError("partial Step 5 runs exist; preserve and audit before rerun")
    bundle = build_filtered_bundle(cfg, out)
    cfg = dict(cfg); cfg["filtered_bundle_runtime_path"] = relative(bundle)
    runs: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    save_runs(out, runs, failures)
    for seed in SEEDS:
        try:
            runs.append(train_filtered(cfg, out, bundle, seed)); save_runs(out, runs, failures)
        except Exception as exc:
            failures.append({"time": utc_now(), "experiment": "filtered_S05", "seed": seed, "stage": "dev_train", "error_type": type(exc).__name__, "message": str(exc)})
            save_runs(out, runs, failures); raise
    metrics = imported_rows(cfg, "dev", "metrics"); classes = imported_rows(cfg, "dev", "per_class"); predictions = imported_rows(cfg, "dev", "predictions")
    for run in runs:
        m, c, p = evaluate(run, cfg, "dev"); metrics.extend(m); classes.extend(c); predictions.extend(p)
    write_csv(out / "dev_metrics.csv", metrics, metric_fields()); write_csv(out / "dev_per_class.csv", classes, class_fields()); write_csv(out / "dev_predictions.csv", predictions, prediction_fields())
    complete = len(runs) == 8 and validate_predictions(predictions, "dev", cfg)
    dev_gate = {
        "step": 5, "stage": "dev", "status": "complete", "decision": "GO_FROZEN_STEP5_TEST" if complete else "NO_GO_STEP5_TEST",
        "filter_rule_frozen_before_test": True, "candidate_reselection_allowed": False,
        "seeds": list(SEEDS), "selection_split": "dev", "selection_metric": "weighted_f1",
        "checkpoint_set_complete": len(runs) == 8, "dev_prediction_groups_complete": validate_predictions(predictions, "dev", cfg),
        "test_used_for_selection": False, "test_evaluated": False, "test_evaluation_unlocked": complete,
        "failed_attempts_recorded": len(failures), "created_at": utc_now(),
    }
    write_json(out / "dev_selection_gate.json", dev_gate); print(json.dumps(dev_gate, indent=2))


def load_runs(out: Path) -> list[dict[str, Any]]:
    rows=[]
    for row in read_csv(out / "checkpoint_index.csv"):
        item=dict(row); item["seed"]=int(item["seed"]); item["success"]=str(item["success"]).lower()=="true"; rows.append(item)
    return rows


def summarize(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str,str,str],list[Mapping[str,Any]]] = defaultdict(list)
    for row in metrics: groups[(str(row["experiment"]),str(row["split"]),str(row["scenario"]))].append(row)
    result=[]
    for key, rows in sorted(groups.items()):
        item={"experiment":key[0],"split":key[1],"scenario":key[2],"seed_count":len(rows)}
        for metric in METRICS:
            values=np.asarray([float(r[metric]) for r in rows]); item.update({f"{metric}_mean":float(values.mean()),f"{metric}_sd":float(values.std(ddof=1)),f"{metric}_median":float(np.median(values)),f"{metric}_min":float(values.min()),f"{metric}_max":float(values.max())})
        result.append(item)
    return result


def differences(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keyed={(str(r["experiment"]),int(r["seed"]),str(r["split"]),str(r["scenario"])):r for r in metrics}; result=[]
    for name, control, treatment in COMPARE:
        for split in ("dev","test"):
            for scenario in SCENARIOS:
                for seed in SEEDS:
                    c=keyed[(control,seed,split,scenario)]; t=keyed[(treatment,seed,split,scenario)]
                    result.append({"contrast":name,"control":control,"treatment":treatment,"seed":seed,"split":split,"scenario":scenario,**{f"delta_{m}":float(t[m])-float(c[m]) for m in METRICS}})
    return result


def test_stage(cfg: Mapping[str, Any], out: Path) -> None:
    if (out / "test_predictions.csv").exists(): raise RuntimeError("repeated Step 5 test inference is forbidden")
    gate=read_json(out / "dev_selection_gate.json")
    if gate.get("decision")!="GO_FROZEN_STEP5_TEST" or gate.get("test_evaluation_unlocked") is not True or gate.get("test_evaluated") is not False: raise ValueError("test locked")
    if not torch.cuda.is_available(): raise RuntimeError("formal Step 5 test requires Slurm CUDA")
    verify_manifest(read_json(out / "pre_run_manifest.json"), ("test_inputs", "code_and_protocol"))
    runs=load_runs(out)
    if {int(r["seed"]) for r in runs} != set(SEEDS): raise ValueError("frozen checkpoint set incomplete")
    for run in runs:
        if sha256(resolve(run["config"]))!=run["config_sha256"] or sha256(resolve(run["checkpoint"]))!=run["checkpoint_sha256"] or sha256(resolve(run["prototype"]))!=run["prototype_sha256"]: raise ValueError(f"frozen run changed: seed {run['seed']}")
    cfg=dict(cfg); cfg["filtered_bundle_runtime_path"]=relative(out / "filtered_S05_audited_failures_v1.npz")
    test_metrics=imported_rows(cfg,"test","metrics"); test_classes=imported_rows(cfg,"test","per_class"); test_predictions=imported_rows(cfg,"test","predictions")
    for run in runs:
        m,c,p=evaluate(run,cfg,"test"); test_metrics.extend(m); test_classes.extend(c); test_predictions.extend(p)
    if not validate_predictions(test_predictions,"test",cfg): raise ValueError("test prediction completeness failed")
    write_csv(out/"test_metrics.csv",test_metrics,metric_fields()); write_csv(out/"test_per_class.csv",test_classes,class_fields()); write_csv(out/"test_predictions.csv",test_predictions,prediction_fields())
    dev_metrics=[normalize_metric(r) for r in read_csv(out/"dev_metrics.csv")]; all_metrics=dev_metrics+test_metrics
    write_csv(out/"per_seed_metrics.csv",all_metrics,metric_fields())
    summary=summarize(all_metrics); write_csv(out/"summary_metrics.csv",summary,["experiment","split","scenario","seed_count",*[f"{m}_{s}" for m in METRICS for s in ("mean","sd","median","min","max")]])
    diff=differences(all_metrics); write_csv(out/"per_seed_differences.csv",diff,["contrast","control","treatment","seed","split","scenario",*[f"delta_{m}" for m in METRICS]])
    write_json(out/"result_gate.json",{"step":5,"stage":"test","status":"complete","decision":"STEP5_FILTER_RESULTS_READY_FOR_LOCAL_AUDIT","seeds":list(SEEDS),"filter_rule_changed":False,"test_used_for_selection":False,"new_test_inference_passes_per_checkpoint_scenario":1,"created_at":utc_now()})
    finalize(out)


def finalize(out: Path) -> None:
    files=[]
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name!="final_manifest.json": files.append({"path":relative(path),"bytes":path.stat().st_size,"sha256":sha256(path)})
    write_json(out/"final_manifest.json",{"step":5,"created_at":utc_now(),"manual_numeric_edits":False,"files":files})


def smoke_stage(cfg: Mapping[str, Any], out: Path) -> None:
    bundle=build_filtered_bundle(cfg,out); data=np.load(bundle,allow_pickle=False)
    assert data["text_features"].shape==(603,3584) and np.isfinite(data["text_features"]).all()
    assert tuple(cfg["seeds"])==SEEDS and cfg["selection_metric"]=="weighted_f1"
    print(json.dumps({"status":"PASS","synthetic_or_bundle_contract_only":True,"formal_training":False,"test_opened":False,"filtered_shape":[603,3584],"prediction_schema":prediction_fields(),"metric_schema":metric_fields()},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--stage",choices=("smoke","dev","test","finalize"),required=True); parser.add_argument("--config",default="outputs/ieee_revision/step5_s05/config.json"); parser.add_argument("--output-dir",default="outputs/ieee_revision/step5_s05"); args=parser.parse_args()
    cfg=read_json(resolve(args.config)); out=resolve(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    for child in ("logs","runs","runtime","slurm","speaker_prototypes"): (out/child).mkdir(parents=True,exist_ok=True)
    if tuple(cfg["seeds"])!=SEEDS or cfg["selection_metric"]!="weighted_f1" or cfg["test_policy"]!="one frozen inference per new checkpoint and scenario after dev gate": raise ValueError("Step 5 frozen protocol changed")
    if args.stage=="smoke": smoke_stage(cfg,out)
    elif args.stage=="dev": dev_stage(cfg,out)
    elif args.stage=="test": test_stage(cfg,out)
    else: finalize(out)


if __name__=="__main__": main()
