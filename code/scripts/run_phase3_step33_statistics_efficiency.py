#!/usr/bin/env python3
"""Step 33 frozen paired-bootstrap, efficiency, and reproducibility audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from data.feature_extractor_v2 import load_cached_features
from run_phase3_step28_factorial import eval_args
from training.train import MELDFeatureDataset, build_model, load_speaker_prototypes

LABELS = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")
MINORITY_IDX = (2, 3, 5)
METRICS = ("accuracy", "weighted_f1", "macro_f1", "minority_f1")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def log(out: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (out / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def config(out: Path) -> Dict[str, Any]:
    return read_json(out / "config.json")


def audit(out: Path) -> None:
    cfg = config(out)
    gate = read_json(out / "gate.json")
    if gate.get("decision") != "GO_FROZEN_STATISTICS_EFFICIENCY_REPRO_ONLY":
        raise ValueError("invalid Step 33 initial gate")
    step32 = read_json(resolve(cfg["step32_final_gate"]))
    phase2 = read_json(resolve(cfg["phase2_final_gate"]))
    if step32.get("decision") != "NO_GO_NO_NEW_MODEL" or step32.get("step33_unlocked") is not True:
        raise ValueError("Step 32 does not unlock Step 33")
    if phase2.get("decision") != "GO_PAPER_FREEZE":
        raise ValueError("Phase II final is not frozen")
    paths = [
        out / "protocol.md", out / "config.json", out / "gate.json",
        resolve(cfg["prediction_source"]), resolve(cfg["phase2_checkpoint_index"]), resolve(cfg["phase2_final_gate"]),
        resolve(cfg["step27_feature_provenance"]), resolve(cfg["step27_prototype_manifest"]),
        resolve(cfg["step27_augmentation_manifest"]), resolve(cfg["step32_final_gate"]),
        ROOT / "scripts/run_phase3_step33_statistics_efficiency.py",
        ROOT / "scripts/rebuild_step33_tables_figures.py",
        ROOT / "src/data/feature_extractor_v2.py", ROOT / "src/training/train.py", ROOT / "src/models/fusion_model.py",
        ROOT / "server_results/phase2_sensitivity/logs/phase2_step20b_confirm_3029175.out",
        ROOT / "server_results/phase2_sensitivity/logs/SM_s4_g0p5_seed42.log",
        ROOT / "server_results/phase2_sensitivity/logs/SM_s4_g0p5_seed43.log",
        ROOT / "server_results/phase2_sensitivity/logs/SM_s4_g0p5_seed44.log",
    ]
    feature = read_json(resolve(cfg["step27_feature_provenance"]))
    prototype = read_json(resolve(cfg["step27_prototype_manifest"]))
    augmentation = read_json(resolve(cfg["step27_augmentation_manifest"]))
    paths.extend(ROOT / "datasets/MELD/features_v2" / row["file"] for row in feature["frozen_arrays"])
    for row in read_csv(resolve(cfg["phase2_checkpoint_index"])):
        paths.extend([resolve(row["checkpoint"]), resolve(row["config"])])
    paths.extend(resolve(row["path"]) for row in prototype["frozen_anchor"]["files"])
    paths.append(resolve(augmentation["bundle"]["path"]))
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    manifest = {
        "step": 33, "created_before_execution": True,
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "training_allowed": False, "model_selection_allowed": False,
    }
    write_json(out / "pre_run_manifest.json", manifest)
    log(out, f"audit complete; frozen files={len(paths)}")


def validate_pre(out: Path) -> Dict[str, Any]:
    cfg = config(out)
    pre = read_json(out / "pre_run_manifest.json")
    for path, expected in pre["files"].items():
        if sha256(resolve(path)) != expected:
            raise ValueError(f"pre-run hash mismatch: {path}")
    return cfg


def metrics_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return [...,4] global metrics and [...,7] class F1."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    denom = 2 * tp + (predicted - tp) + (support - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    n = cm.sum(axis=(-2, -1))
    accuracy = np.divide(tp.sum(axis=-1), n, out=np.zeros_like(n, dtype=float), where=n != 0)
    weighted = np.divide((f1 * support).sum(axis=-1), n, out=np.zeros_like(n, dtype=float), where=n != 0)
    macro = f1.mean(axis=-1)
    minority = f1[..., list(MINORITY_IDX)].mean(axis=-1)
    return np.stack([accuracy, weighted, macro, minority], axis=-1), f1


def confusion(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(gold * 7 + pred, minlength=49).reshape(7, 7)


def bootstrap_pair(seed_arrays: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], resamples: int, rng: np.random.Generator, chunk: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    n = len(seed_arrays[0][0])
    global_dist = np.empty((resamples, 4), dtype=np.float32)
    class_dist = np.empty((resamples, 7), dtype=np.float32)
    cursor = 0
    while cursor < resamples:
        size = min(chunk, resamples - cursor)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        global_sum = np.zeros((size, 4), dtype=float)
        class_sum = np.zeros((size, 7), dtype=float)
        offsets = np.arange(size, dtype=np.int64)[:, None] * 49
        for gold, control, treatment in seed_arrays:
            control_codes = gold * 7 + control
            treatment_codes = gold * 7 + treatment
            ccm = np.bincount((control_codes[indices] + offsets).ravel(), minlength=size * 49).reshape(size, 7, 7)
            tcm = np.bincount((treatment_codes[indices] + offsets).ravel(), minlength=size * 49).reshape(size, 7, 7)
            cg, cf = metrics_from_cm(ccm)
            tg, tf = metrics_from_cm(tcm)
            global_sum += tg - cg
            class_sum += tf - cf
        global_dist[cursor:cursor + size] = global_sum / len(seed_arrays)
        class_dist[cursor:cursor + size] = class_sum / len(seed_arrays)
        cursor += size
    return global_dist, class_dist


def sign_p(values: np.ndarray) -> float:
    b = len(values)
    lower = (1 + int(np.count_nonzero(values <= 0))) / (b + 1)
    upper = (1 + int(np.count_nonzero(values >= 0))) / (b + 1)
    return min(1.0, 2.0 * min(lower, upper))


def holm(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_prediction_groups(path: Path) -> Dict[Tuple[str, int, str], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, str]]] = {}
    for row in read_csv(path):
        key = (row["experiment"], int(row["seed"]), row["scenario"])
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["sample_index"]))
    return groups


def statistics(out: Path) -> None:
    cfg = validate_pre(out)
    groups = load_prediction_groups(resolve(cfg["prediction_source"]))
    rng = np.random.Generator(np.random.PCG64(cfg["bootstrap"]["seed"]))
    resamples = int(cfg["bootstrap"]["resamples"])
    summary_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    pairing: List[Dict[str, Any]] = []
    distributions: Dict[str, np.ndarray] = {}
    for comparison in cfg["comparisons"]:
        cid, control, treatment = comparison["id"], comparison["control"], comparison["treatment"]
        for scenario in cfg["scenarios"]:
            arrays = []
            observed_global = []
            observed_class = []
            reference_keys = None
            for seed in (42, 43, 44):
                crows = groups[(control, seed, scenario)]
                trows = groups[(treatment, seed, scenario)]
                ckeys = [(r["sample_index"], r["dialogue_id"], r["utterance_id"], r["gold"]) for r in crows]
                tkeys = [(r["sample_index"], r["dialogue_id"], r["utterance_id"], r["gold"]) for r in trows]
                if ckeys != tkeys or len(ckeys) != 2593 or (reference_keys is not None and ckeys != reference_keys):
                    raise ValueError(f"pairing mismatch {cid} {scenario} seed={seed}")
                reference_keys = ckeys
                gold = np.asarray([int(r["gold"]) for r in crows], dtype=np.int16)
                cpred = np.asarray([int(r["prediction"]) for r in crows], dtype=np.int16)
                tpred = np.asarray([int(r["prediction"]) for r in trows], dtype=np.int16)
                arrays.append((gold, cpred, tpred))
                cg, cf = metrics_from_cm(confusion(gold, cpred))
                tg, tf = metrics_from_cm(confusion(gold, tpred))
                gd, fd = tg - cg, tf - cf
                observed_global.append(gd); observed_class.append(fd)
                for i, metric in enumerate(METRICS):
                    seed_rows.append({"comparison": cid, "control": control, "treatment": treatment, "scenario": scenario, "seed": seed, "metric": metric, "delta": float(gd[i])})
            pairing.append({"comparison": cid, "scenario": scenario, "seeds": 3, "samples_per_seed": 2593, "pair_key": "sample_index+dialogue_id+utterance_id+gold", "pairing_passed": True})
            observed_global_mean = np.mean(observed_global, axis=0)
            observed_class_mean = np.mean(observed_class, axis=0)
            global_dist, class_dist = bootstrap_pair(arrays, resamples, rng)
            distributions[f"{cid}__{scenario}__global"] = global_dist
            distributions[f"{cid}__{scenario}__class_f1"] = class_dist
            for i, metric in enumerate(METRICS):
                low, high = np.quantile(global_dist[:, i], [0.025, 0.975])
                p = sign_p(global_dist[:, i])
                decision = "significant_positive" if low > 0 and p < 0.05 else "significant_negative" if high < 0 and p < 0.05 else "not_significant"
                summary_rows.append({
                    "comparison": cid, "claim_role": comparison["claim_role"], "control": control, "treatment": treatment,
                    "scenario": scenario, "metric": metric, "observed_delta": float(observed_global_mean[i]),
                    "ci95_low": float(low), "ci95_high": float(high), "p_two_sided": p,
                    "multiplicity": "four_prespecified_global_outcomes_no_adjustment", "decision": decision,
                    "resamples": resamples, "bootstrap_seed": cfg["bootstrap"]["seed"], "unit": "test_utterance",
                })
            raw_p = [sign_p(class_dist[:, i]) for i in range(7)]
            adjusted = holm(raw_p)
            for i, label in enumerate(LABELS):
                low, high = np.quantile(class_dist[:, i], [0.025, 0.975])
                decision = "significant_positive" if low > 0 and adjusted[i] < 0.05 else "significant_negative" if high < 0 and adjusted[i] < 0.05 else "not_significant"
                class_rows.append({
                    "comparison": cid, "claim_role": comparison["claim_role"], "control": control, "treatment": treatment,
                    "scenario": scenario, "label": label, "observed_delta_f1": float(observed_class_mean[i]),
                    "ci95_low": float(low), "ci95_high": float(high), "p_two_sided_raw": raw_p[i],
                    "p_holm_7_classes": adjusted[i], "decision_holm": decision,
                    "resamples": resamples, "bootstrap_seed": cfg["bootstrap"]["seed"],
                })
            log(out, f"bootstrap complete comparison={cid} scenario={scenario}")
    write_csv(out / "paired_bootstrap_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(out / "paired_bootstrap_per_class.csv", class_rows, list(class_rows[0]))
    write_csv(out / "paired_observed_seed_deltas.csv", seed_rows, list(seed_rows[0]))
    write_csv(out / "pairing_audit.csv", pairing, list(pairing[0]))
    np.savez_compressed(out / "bootstrap_distributions.npz", **distributions)
    write_json(out / "statistics_gate.json", {
        "step": 33, "status": "complete", "pairing_passed": True, "comparisons": len(cfg["comparisons"]),
        "scenarios": len(cfg["scenarios"]), "global_rows": len(summary_rows), "per_class_rows": len(class_rows),
        "resamples": resamples, "bootstrap_seed": cfg["bootstrap"]["seed"],
        "test_performance_re_evaluated": False, "model_selected": False,
    })


def phase2_seed42(cfg: Mapping[str, Any]) -> Dict[str, str]:
    for row in read_csv(resolve(cfg["phase2_checkpoint_index"])):
        if int(row["seed"]) == int(cfg["efficiency"]["seed"]):
            return row
    raise ValueError("missing seed42 final checkpoint")


def local_prototype(seed: int) -> Path:
    return ROOT / f"server_results/phase2_sensitivity/speaker_prototypes/SM_s4_g0p5_seed{seed}_slots4.npz"


def build_final_dataset(cfg: Mapping[str, Any], split: str, model_cfg: Any, seed: int) -> MELDFeatureDataset:
    feature_dir = ROOT / "datasets/MELD/features_v2"
    data = load_cached_features(str(feature_dir), split)
    speakers = np.load(feature_dir / f"{split}_speakers.npy", allow_pickle=True)
    prototypes = load_speaker_prototypes(local_prototype(seed))
    return MELDFeatureDataset(
        data["video_features"], data["text_features"], data["labels"], data.get("dialogue_ids"), data.get("utterance_ids"),
        speaker_ids=speakers, context_len=model_cfg.context_len, context_max_distance=model_cfg.context_max_distance,
        speaker_prototypes=prototypes, speaker_memory_slots=model_cfg.speaker_memory_slots,
    )


def forward(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> None:
    model(
        video_feat=batch["video_feat"], text_feat=batch["text_feat"],
        context_video_feat=batch["context_video_feat"], context_text_feat=batch["context_text_feat"],
        context_mask=batch["context_mask"], context_same_speaker=batch["context_same_speaker"],
        context_turn_distance=batch["context_turn_distance"],
        speaker_memory_video_feat=batch["speaker_memory_video_feat"], speaker_memory_text_feat=batch["speaker_memory_text_feat"],
        speaker_memory_mask=batch["speaker_memory_mask"],
    )


def benchmark(out: Path) -> None:
    cfg = validate_pre(out)
    if torch.cuda.is_available():
        raise ValueError("protocol freezes this benchmark to CPU, but CUDA is available")
    row = phase2_seed42(cfg)
    model_cfg = eval_args(resolve(row["config"]))
    dataset = build_final_dataset(cfg, cfg["efficiency"]["split"], model_cfg, int(row["seed"]))
    loader = DataLoader(dataset, batch_size=cfg["efficiency"]["batch_size"], shuffle=False, num_workers=0)
    model = build_model(model_cfg).cpu()
    state = torch.load(resolve(row["checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    iterator = iter(loader)
    with torch.inference_mode():
        for _ in range(cfg["efficiency"]["warmup_batches"]):
            forward(model, next(iterator))
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak = [baseline_rss]
    stop = threading.Event()
    def monitor() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.002)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    pass_seconds: List[float] = []
    batch_ms: List[float] = []
    with torch.inference_mode():
        for repetition in range(cfg["efficiency"]["measured_full_passes"]):
            start_pass = time.perf_counter()
            for batch in loader:
                start_batch = time.perf_counter()
                forward(model, batch)
                batch_ms.append((time.perf_counter() - start_batch) * 1000.0)
            pass_seconds.append(time.perf_counter() - start_pass)
            log(out, f"benchmark pass={repetition + 1} seconds={pass_seconds[-1]:.6f}")
    stop.set(); watcher.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    n = len(dataset)
    mean_seconds = float(np.mean(pass_seconds))
    result = {
        "scope": cfg["efficiency"]["scope"], "model": "Phase II final SM_s4_g0p5 seed42",
        "device": "cpu", "hardware_chip": "Apple M5", "cpu_cores": 10, "system_memory_bytes": 16 * 1024 ** 3,
        "batch_size": cfg["efficiency"]["batch_size"], "num_workers": 0,
        "warmup_batches": cfg["efficiency"]["warmup_batches"], "measured_full_passes": len(pass_seconds),
        "samples_per_pass": n, "pass_seconds": pass_seconds, "mean_pass_seconds": mean_seconds,
        "std_pass_seconds": float(np.std(pass_seconds, ddof=0)), "throughput_utterances_per_second": n / mean_seconds,
        "latency_ms_per_utterance": mean_seconds * 1000.0 / n,
        "batch_latency_ms_p50": float(np.quantile(batch_ms, 0.50)), "batch_latency_ms_p95": float(np.quantile(batch_ms, 0.95)),
        "process_rss_before_measured_bytes": baseline_rss, "process_peak_rss_observed_bytes": peak[0],
        "process_peak_rss_increment_bytes": peak[0] - baseline_rss,
        "peak_gpu_memory": "not_applicable_no_gpu", "total_parameters": total_params, "trainable_parameters": trainable_params,
        "strict_checkpoint_load": True, "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "raw_video_or_qwen_extraction_included": False, "test_metrics_computed": False,
    }
    write_json(out / "inference_benchmark.json", result)


def sanitized_hardware() -> Dict[str, Any]:
    result: Dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        text = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, check=True).stdout
        allowed = ("Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory")
        for line in text.splitlines():
            stripped = line.strip()
            if any(stripped.startswith(name + ":") for name in allowed):
                key, value = stripped.split(":", 1)
                result[key.lower().replace(" ", "_")] = value.strip()
    except Exception:
        result["system_profiler"] = "unavailable"
    return result


def reproducibility(out: Path, supplementary: Path) -> None:
    cfg = validate_pre(out)
    supplementary.mkdir(parents=True, exist_ok=True)
    feature = read_json(resolve(cfg["step27_feature_provenance"]))
    prototype = read_json(resolve(cfg["step27_prototype_manifest"]))
    augmentation = read_json(resolve(cfg["step27_augmentation_manifest"]))
    array_rows = []
    for row in feature["frozen_arrays"]:
        path = ROOT / "datasets/MELD/features_v2" / row["file"]
        array = np.load(path, mmap_mode="r", allow_pickle=True)
        actual_shape = list(array.shape)
        actual_dtype = str(array.dtype)
        actual_hash = sha256(path)
        if actual_shape != row["shape"] or actual_dtype != row["dtype"] or actual_hash != row["sha256"]:
            raise ValueError(f"frozen array verification failed: {path}")
        array_rows.append({
            "file": row["file"], "shape": "x".join(map(str, actual_shape)), "dtype": actual_dtype,
            "sha256": actual_hash, "shape_match_step27": True, "dtype_match_step27": True,
            "sha256_match_step27": True, "status": "VERIFIED_ARTIFACT_RECHECKED_STEP33",
        })
    write_csv(supplementary / "frozen_arrays.csv", array_rows, list(array_rows[0]))
    checkpoint_rows = read_csv(resolve(cfg["phase2_checkpoint_index"]))
    artifact_rows: List[Dict[str, Any]] = []
    for row in checkpoint_rows:
        for kind in ("checkpoint", "config"):
            path = resolve(row[kind])
            artifact_rows.append({"artifact_type": kind, "seed": row["seed"], "path": row[kind], "bytes": path.stat().st_size, "sha256": sha256(path), "status": "VERIFIED_ARTIFACT"})
        proto = local_prototype(int(row["seed"]))
        artifact_rows.append({"artifact_type": "speaker_prototype", "seed": row["seed"], "path": str(proto.relative_to(ROOT)), "bytes": proto.stat().st_size, "sha256": sha256(proto), "status": "VERIFIED_ARTIFACT"})
    s05 = resolve(augmentation["bundle"]["path"])
    artifact_rows.append({"artifact_type": "S05_bundle", "seed": "", "path": augmentation["bundle"]["path"], "bytes": s05.stat().st_size, "sha256": sha256(s05), "status": "VERIFIED_ARTIFACT"})
    write_csv(supplementary / "artifact_inventory.csv", artifact_rows, list(artifact_rows[0]))
    extractor_path = resolve(feature["extractor_code"]["path"])
    backbone = {
        "model_family": feature["model_family"], "extractor_current_code_path": feature["extractor_code"],
        "current_extractor_sha256_rechecked": sha256(extractor_path),
        "last_token_formula": "hidden_states[-1][0, -1, :]",
        "text_prompt": feature["extractor_code"]["text_prompt"], "video_prompt": feature["extractor_code"]["video_prompt"],
        "requested_video_max_frames": 16, "feature_dimension": 3584, "cache_dtype": "float32",
        "normalization_in_classifier_loader": "row-wise L2 normalization of text and video separately",
        "historical_binding": feature["extractor_code"]["historical_run_binding"],
        "historical_extraction": feature["historical_extraction"], "frame_sampling": feature["frame_sampling"],
    }
    write_json(supplementary / "feature_backbone_manifest.json", backbone)
    current_hashes = {
        "src/data/feature_extractor_v2.py": sha256(ROOT / "src/data/feature_extractor_v2.py"),
        "src/training/train.py": sha256(ROOT / "src/training/train.py"),
        "src/models/fusion_model.py": sha256(ROOT / "src/models/fusion_model.py"),
    }
    training_rows = [{"seed": row["seed"], "train_seconds": float(row["train_seconds"]), "source": cfg["phase2_checkpoint_index"], "historical_device": "cuda", "historical_gpu": "Tesla V100-SXM2-16GB", "gpu_evidence": "server_results/phase2_sensitivity/logs/phase2_step20b_confirm_3029175.out"} for row in checkpoint_rows]
    write_csv(out / "training_time.csv", training_rows, list(training_rows[0]))
    environment = {
        "step33_measurement": {
            "captured_at": datetime.now().isoformat(timespec="seconds"), "hardware": sanitized_hardware(),
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "psutil": psutil.__version__,
            "device": "cpu", "cuda_available": torch.cuda.is_available(),
        },
        "phase2_final_training": {
            "python": "3.10.20", "device": "cuda", "gpu": "Tesla V100-SXM2-16GB", "driver": "575.57.08", "reported_cuda": "12.9",
            "node": "g0118", "evidence": "server_results/phase2_sensitivity/logs/phase2_step20b_confirm_3029175.out",
            "torch_version": "unavailable_in_archived_final_run_evidence", "peak_gpu_memory": "unavailable",
            "conda_lockfile": "unavailable", "epochs_cap": 20, "early_stopping_patience": 5,
            "note": "This is the archived final run environment, not the prior thesis AutoDL/50-epoch environment."
        },
        "current_source_hashes": current_hashes,
        "step27_recorded_source_hashes": read_json(ROOT / "outputs/phase3_ieee_access/step27_reproducibility_audit/manifest.json")["code_hashes"],
        "source_binding_note": "Current train.py differs from the Step 27 recorded hash; checkpoints strict-load, but byte-identical historical training source remains unverified."
    }
    write_json(supplementary / "environment_manifest.json", environment)
    packages = sorted((dist.metadata.get("Name", dist.name), dist.version) for dist in importlib.metadata.distributions())
    (supplementary / "requirements_frozen.txt").write_text("\n".join(f"{name}=={version}" for name, version in packages) + "\n", encoding="utf-8")
    unknowns = [
        ("historical_qwen_revision", "unknown", "No immutable extraction checkpoint/revision was archived."),
        ("historical_processor_tokenizer_revision", "unknown", "No extraction-time manifest."),
        ("historical_frame_indices_timestamps", "unknown", "No per-clip trace."),
        ("historical_decoder_runtime", "unknown", "No decoder/package lockfile."),
        ("historical_extraction_gpu_dtype_command", "unknown", "Current code path is not a historical execution binding."),
        ("byte_identical_final_training_source", "unknown", "Checkpoint manifest did not bind training sources; current train.py hash differs from Step 27 record."),
        ("phase2_final_training_torch_version", "unavailable", "Not present in archived final run evidence."),
        ("phase2_final_peak_gpu_memory", "unavailable", "Not measured in archived final run."),
        ("step33_peak_gpu_memory", "not_applicable", "Step 33 benchmark ran on CPU-only hardware."),
    ]
    write_csv(supplementary / "unknowns.csv", [{"field": a, "status": b, "reason": c} for a, b, c in unknowns], ["field", "status", "reason"])
    commands = """# Step 33 reproduction commands

Run from the repository root with the environment captured in `environment_manifest.json`:

```bash
python3 scripts/run_phase3_step33_statistics_efficiency.py --stage audit
python3 scripts/run_phase3_step33_statistics_efficiency.py --stage statistics
python3 scripts/run_phase3_step33_statistics_efficiency.py --stage benchmark
python3 scripts/run_phase3_step33_statistics_efficiency.py --stage reproducibility
python3 scripts/rebuild_step33_tables_figures.py
python3 scripts/run_phase3_step33_statistics_efficiency.py --stage finalize
```

The bootstrap is deterministic. Efficiency timing is hardware/runtime specific and must be rerun, not copied across machines.
"""
    (supplementary / "commands.md").write_text(commands, encoding="utf-8")
    readme = """# Supplementary reproducibility package

This package rebuilds Step 33 from frozen artifacts. `frozen_arrays.csv` records all 18 arrays; `feature_backbone_manifest.json` records prompts, current extractor path, Last Token indexing and historical unknowns; `artifact_inventory.csv` records checkpoint/prototype/S05 size and hashes; `environment_manifest.json` separates the archived Phase II final training environment from the current CPU benchmark environment. No prior-thesis AutoDL or 50-epoch configuration is used.

Use `commands.md` for the exact sequence. Tables and figures are generated only by `scripts/rebuild_step33_tables_figures.py`.
"""
    (supplementary / "README.md").write_text(readme, encoding="utf-8")
    write_json(out / "reproducibility_gate.json", {"step": 33, "arrays": len(array_rows), "artifact_rows": len(artifact_rows), "unknowns": len(unknowns), "status": "complete"})


def finalize(out: Path, supplementary: Path) -> None:
    cfg = validate_pre(out)
    required = [
        out / "paired_bootstrap_summary.csv", out / "paired_bootstrap_per_class.csv", out / "pairing_audit.csv",
        out / "inference_benchmark.json", out / "training_time.csv", out / "statistics_gate.json",
        out / "reproducibility_gate.json", out / "paper_tables/statistical_comparisons.csv",
        out / "paper_tables/efficiency_table.csv", out / "figures/bootstrap_ci.png",
        supplementary / "frozen_arrays.csv", supplementary / "artifact_inventory.csv", supplementary / "environment_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"missing required outputs: {missing}")
    stats = read_json(out / "statistics_gate.json")
    benchmark_result = read_json(out / "inference_benchmark.json")
    gate = {
        "step": 33, "status": "complete", "decision": "GO_STEP34_PROTOCOL_ONLY_WITH_STATISTICAL_LIMITS",
        "paired_bootstrap_complete": stats["pairing_passed"], "bootstrap_resamples": stats["resamples"],
        "efficiency_benchmark_complete": True, "benchmark_device": benchmark_result["device"],
        "reproducibility_package_complete": True, "training_performed": False, "model_selected": False,
        "test_performance_re_evaluated": False, "phase2_final_modified": False,
        "step34_protocol_unlocked": True, "step34_training_unlocked": False, "later_steps_unlocked": False,
        "claim_policy": "CI/significance must accompany frozen comparisons; non-significant and negative results remain explicit; efficiency is cached-feature CPU only",
    }
    write_json(out / "final_gate.json", gate)
    rows = read_csv(out / "paired_bootstrap_summary.csv")
    full = [r for r in rows if r["scenario"] == "full" and r["metric"] in ("macro_f1", "minority_f1")]
    lines = ["# Phase III Step 33 results", "", f"Final decision: `{gate['decision']}`.", "", "## Full-test paired bootstrap", ""]
    for row in full:
        lines.append(f"- {row['comparison']} {row['metric']}: delta={float(row['observed_delta']):+.4f}, 95% CI [{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}], p={float(row['p_two_sided']):.4g}, {row['decision']}.")
    lines += ["", "## Efficiency", "", f"- Parameters: {benchmark_result['total_parameters']:,} total / {benchmark_result['trainable_parameters']:,} trainable.", f"- Apple M5 CPU cached-feature throughput: {benchmark_result['throughput_utterances_per_second']:.2f} utterances/s; latency {benchmark_result['latency_ms_per_utterance']:.4f} ms/utterance.", f"- Observed process peak RSS: {benchmark_result['process_peak_rss_observed_bytes'] / 1024**2:.2f} MiB; GPU memory is not applicable.", "", "All tables/figures are script-generated. Historical extraction/source/environment unknowns remain explicit in `supplementary/reproducibility/unknowns.csv`."]
    (out / "step33_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_files = [p for p in out.rglob("*") if p.is_file() and p.name != "final_manifest.json"]
    supplementary_files = [p for p in supplementary.rglob("*") if p.is_file()]
    manifest = {
        "step": 33, "status": "complete", "decision": gate["decision"], "hash_algorithm": "sha256",
        "outputs": {str(path.relative_to(out)): sha256(path) for path in sorted(manifest_files)},
        "supplementary": {str(path.relative_to(supplementary)): sha256(path) for path in sorted(supplementary_files)},
        "execution": {"training_performed": False, "model_selection": False, "benchmark_device": "cpu", "server_used": False},
    }
    write_json(out / "final_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "statistics", "benchmark", "reproducibility", "finalize"), required=True)
    parser.add_argument("--output_dir", default="outputs/phase3_ieee_access/step33_statistics_efficiency")
    parser.add_argument("--supplementary_dir", default="supplementary/reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out, supplementary = resolve(args.output_dir), resolve(args.supplementary_dir)
    if args.stage == "audit": audit(out)
    elif args.stage == "statistics": statistics(out)
    elif args.stage == "benchmark": benchmark(out)
    elif args.stage == "reproducibility": reproducibility(out, supplementary)
    else: finalize(out, supplementary)


if __name__ == "__main__":
    main()
