#!/usr/bin/env python3
"""IEEE revision Step 1: frozen-feature controlled baselines.

Formal stages are deliberately separated:
  dev   - train on train, select checkpoint epochs on dev weighted F1, never open test.
  test  - require the frozen dev gate, verify checkpoints and test hashes, infer once.
  finalize - rebuild manifests/gates from already written files; performs no inference.
  smoke - synthetic mini-batch only; performs no dataset access.

The historical single-token CrossModalAttention is not used.  The sequence model
below attends over the previous five dialogue turns plus the current turn with an
explicit padding mask and is therefore a new experimental object.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.fusion_model import EmotionClassifier, RelationAwareContextEncoder  # noqa: E402
from training.train import MELDFeatureDataset  # noqa: E402


LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
FDS = (2, 5, 3)
MODELS = ("text_only_mlp", "video_only_mlp", "concat_mlp", "turn_crossattention")
PREDICTION_FIELDS = [
    "model", "seed", "split", "scenario", "sample_index", "dialogue_id",
    "utterance_id", "speaker", "gold", "prediction", "gold_label",
    "prediction_label", *[f"prob_{label}" for label in LABELS],
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def projection(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim), nn.ReLU(), nn.Dropout(dropout),
    )


def classifier(hidden_dim: int, dropout: float, num_classes: int = 7) -> nn.Sequential:
    """Two hidden-layer MLP, matching the inherited DGF classifier depth."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout * 0.5),
        nn.Linear(hidden_dim // 4, num_classes),
    )


def init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ProjectedConcatModel(nn.Module):
    """Separate normalization/projection, concatenation, then a two-hidden-layer head."""

    def __init__(self, feat_dim: int, hidden_dim: int, dropout: float, max_distance: int,
                 structure_gate_scale: float, relation_dropout: float,
                 relation_embedding_init_std: float):
        super().__init__()
        self.video_projection = projection(feat_dim, hidden_dim, dropout * 0.5)
        self.text_projection = projection(feat_dim, hidden_dim, dropout * 0.5)
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.context_encoder = RelationAwareContextEncoder(
            hidden_dim=hidden_dim, max_distance=max_distance,
            use_relation_embeddings=True, use_validated_relations=False,
            dropout=dropout, structure_gate_scale=structure_gate_scale,
            relation_dropout=relation_dropout,
            relation_embedding_init_std=relation_embedding_init_std,
        )
        self.classifier = classifier(hidden_dim, dropout)
        self.apply(init_module)

    def encode(self, video: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat([self.video_projection(video), self.text_projection(text)], dim=-1))

    def forward(self, video_feat: torch.Tensor, text_feat: torch.Tensor,
                context_video_feat: torch.Tensor, context_text_feat: torch.Tensor,
                context_mask: torch.Tensor, context_same_speaker: torch.Tensor,
                context_turn_distance: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        current = self.encode(video_feat, text_feat)
        batch, turns = context_mask.shape
        context = self.encode(
            context_video_feat.reshape(batch * turns, -1),
            context_text_feat.reshape(batch * turns, -1),
        ).reshape(batch, turns, -1)
        fused, weights = self.context_encoder(
            current, context, context_mask, context_same_speaker, context_turn_distance,
        )
        logits = self.classifier(fused)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1), "attention_weights": weights}


class DialogueTurnCrossAttention(nn.Module):
    """Bidirectional cross-modal attention over valid previous turns + current turn."""

    def __init__(self, feat_dim: int, hidden_dim: int, num_heads: int, dropout: float,
                 max_distance: int, relation_dropout: float,
                 relation_embedding_init_std: float):
        super().__init__()
        self.video_projection = projection(feat_dim, hidden_dim, dropout * 0.5)
        self.text_projection = projection(feat_dim, hidden_dim, dropout * 0.5)
        self.same_speaker_embedding = nn.Embedding(2, hidden_dim)
        self.distance_embedding = nn.Embedding(max_distance + 1, hidden_dim)
        self.relation_norm_video = nn.LayerNorm(hidden_dim)
        self.relation_norm_text = nn.LayerNorm(hidden_dim)
        self.relation_dropout = nn.Dropout(relation_dropout)
        self.t2v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout * 0.5, batch_first=True)
        self.v2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout * 0.5, batch_first=True)
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = classifier(hidden_dim, dropout)
        self.max_distance = max_distance
        self.apply(init_module)
        nn.init.normal_(self.same_speaker_embedding.weight, mean=0.0, std=relation_embedding_init_std)
        nn.init.normal_(self.distance_embedding.weight, mean=0.0, std=relation_embedding_init_std)

    def forward(self, video_feat: torch.Tensor, text_feat: torch.Tensor,
                context_video_feat: torch.Tensor, context_text_feat: torch.Tensor,
                context_mask: torch.Tensor, context_same_speaker: torch.Tensor,
                context_turn_distance: torch.Tensor, return_attention: bool = False,
                **_: Any) -> Dict[str, torch.Tensor]:
        video_sequence = torch.cat([context_video_feat, video_feat.unsqueeze(1)], dim=1)
        text_sequence = torch.cat([context_text_feat, text_feat.unsqueeze(1)], dim=1)
        current_valid = torch.ones(context_mask.size(0), 1, device=context_mask.device, dtype=context_mask.dtype)
        valid = torch.cat([context_mask, current_valid], dim=1) > 0
        same = torch.cat([
            context_same_speaker.long().clamp(0, 1),
            torch.ones(context_mask.size(0), 1, device=context_mask.device, dtype=torch.long),
        ], dim=1)
        distance = torch.cat([
            context_turn_distance.long().clamp(0, self.max_distance),
            torch.zeros(context_mask.size(0), 1, device=context_mask.device, dtype=torch.long),
        ], dim=1)
        relation = self.relation_dropout(
            self.same_speaker_embedding(same) + self.distance_embedding(distance)
        )
        video_hidden = self.relation_norm_video(self.video_projection(video_sequence) + relation)
        text_hidden = self.relation_norm_text(self.text_projection(text_sequence) + relation)
        padding_mask = ~valid
        text_from_video, t2v_weights = self.t2v(
            text_hidden, video_hidden, video_hidden,
            key_padding_mask=padding_mask, need_weights=return_attention,
            average_attn_weights=False,
        )
        video_from_text, v2t_weights = self.v2t(
            video_hidden, text_hidden, text_hidden,
            key_padding_mask=padding_mask, need_weights=return_attention,
            average_attn_weights=False,
        )
        text_current = self.t_norm(text_hidden[:, -1] + text_from_video[:, -1])
        video_current = self.v_norm(video_hidden[:, -1] + video_from_text[:, -1])
        fused = self.merge(torch.cat([text_current, video_current], dim=-1))
        logits = self.classifier(fused)
        output = {"logits": logits, "probs": F.softmax(logits, dim=-1), "sequence_mask": valid}
        if return_attention:
            output["attention_weights"] = {"t2v": t2v_weights, "v2t": v2t_weights}
        return output


def build_model(name: str, cfg: Mapping[str, Any], feat_dim: int | None = None) -> nn.Module:
    dim = int(feat_dim or cfg["feature_dim"])
    common = dict(
        hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
        max_distance=int(cfg["context_max_distance"]),
        relation_dropout=float(cfg["relation_dropout"]),
        relation_embedding_init_std=float(cfg["relation_embedding_init_std"]),
    )
    if name in ("text_only_mlp", "video_only_mlp"):
        return EmotionClassifier(
            video_dim=dim, text_dim=dim, num_classes=7,
            fusion_type="text_only" if name == "text_only_mlp" else "video_only",
            num_heads=int(cfg["num_heads"]), structure_mode="speaker_distance",
            max_context_len=int(cfg["context_len"]),
            structure_gate_scale=float(cfg["structure_gate_scale"]), **common,
        )
    if name == "concat_mlp":
        return ProjectedConcatModel(
            feat_dim=dim, structure_gate_scale=float(cfg["structure_gate_scale"]), **common,
        )
    if name == "turn_crossattention":
        return DialogueTurnCrossAttention(
            feat_dim=dim, num_heads=int(cfg["num_heads"]), **common,
        )
    raise ValueError(f"unknown model: {name}")


def dgf_reference_parameters(cfg: Mapping[str, Any], feat_dim: int | None = None) -> int:
    model = EmotionClassifier(
        video_dim=int(feat_dim or cfg["feature_dim"]), text_dim=int(feat_dim or cfg["feature_dim"]),
        hidden_dim=int(cfg["hidden_dim"]), num_classes=7, fusion_type="gated",
        dropout=float(cfg["dropout"]), num_heads=int(cfg["num_heads"]),
        structure_mode="speaker_distance", max_context_len=int(cfg["context_len"]),
        max_distance=int(cfg["context_max_distance"]),
        structure_gate_scale=float(cfg["structure_gate_scale"]),
        relation_dropout=float(cfg["relation_dropout"]),
        relation_embedding_init_std=float(cfg["relation_embedding_init_std"]),
    )
    return sum(parameter.numel() for parameter in model.parameters())


def model_parameters(cfg: Mapping[str, Any], feat_dim: int | None = None) -> Dict[str, Dict[str, float]]:
    reference = dgf_reference_parameters(cfg, feat_dim)
    result: Dict[str, Dict[str, float]] = {}
    for name in MODELS:
        count = sum(parameter.numel() for parameter in build_model(name, cfg, feat_dim).parameters())
        result[name] = {"parameters": count, "dgf_reference_parameters": reference, "ratio_to_dgf": count / reference}
    return result


def load_split(feature_dir: Path, split: str, cfg: Mapping[str, Any]) -> MELDFeatureDataset:
    arrays = {}
    for key in ("video_features", "text_features", "labels", "dialogue_ids", "utterance_ids", "speakers"):
        path = feature_dir / f"{split}_{key}.npy"
        arrays[key] = np.load(path, allow_pickle=key == "speakers")
    dataset = MELDFeatureDataset(
        arrays["video_features"], arrays["text_features"], arrays["labels"],
        arrays["dialogue_ids"], arrays["utterance_ids"], speaker_ids=arrays["speakers"],
        context_len=int(cfg["context_len"]), context_max_distance=int(cfg["context_max_distance"]),
    )
    expected = int(cfg["split_rows"][split])
    if len(dataset) != expected:
        raise ValueError(f"{split} rows changed after loader: {len(dataset)} != {expected}")
    return dataset


def to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def forward_model(model: nn.Module, name: str, batch: Mapping[str, torch.Tensor],
                  scenario: str = "full", return_attention: bool = False) -> Dict[str, torch.Tensor]:
    video = batch["video_feat"]
    if scenario == "missing_video":
        video = torch.zeros_like(video)
    kwargs = {
        "video_feat": None if name == "text_only_mlp" else video,
        "text_feat": None if name == "video_only_mlp" else batch["text_feat"],
        "context_video_feat": batch["context_video_feat"],
        "context_text_feat": batch["context_text_feat"],
        "context_mask": batch["context_mask"],
        "context_same_speaker": batch["context_same_speaker"],
        "context_turn_distance": batch["context_turn_distance"],
        "context_relation_ids": batch["context_relation_ids"],
        "return_attention": return_attention,
    }
    return model(**kwargs)


def f1_values(gold: np.ndarray, prediction: np.ndarray) -> List[float]:
    values = []
    for label in range(7):
        tp = int(((gold == label) & (prediction == label)).sum())
        fp = int(((gold != label) & (prediction == label)).sum())
        fn = int(((gold == label) & (prediction != label)).sum())
        values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return values


def metrics(gold: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    per_class = f1_values(gold, prediction)
    support = np.bincount(gold, minlength=7)
    return {
        "accuracy": float((gold == prediction).mean()),
        "weighted_f1": float(np.average(per_class, weights=support)),
        "macro_f1": float(np.mean(per_class)),
        "f1_fds": float(np.mean([per_class[index] for index in FDS])),
    }


@torch.no_grad()
def evaluate(model: nn.Module, name: str, loader: DataLoader, dataset: MELDFeatureDataset,
             device: torch.device, split: str, scenario: str,
             seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    prediction_rows: List[Dict[str, Any]] = []
    gold_all: List[int] = []
    prediction_all: List[int] = []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        output = forward_model(model, name, batch, scenario)
        probabilities = output["probs"].detach().cpu().numpy()
        guesses = probabilities.argmax(axis=1)
        gold = batch["label"].detach().cpu().numpy()
        indices = batch["idx"].detach().cpu().numpy()
        gold_all.extend(gold.tolist())
        prediction_all.extend(guesses.tolist())
        for row_index, sample_index in enumerate(indices):
            g, pred = int(gold[row_index]), int(guesses[row_index])
            row: Dict[str, Any] = {
                "model": name, "seed": seed, "split": split, "scenario": scenario,
                "sample_index": int(sample_index),
                "dialogue_id": str(dataset.dialogue_ids[sample_index]),
                "utterance_id": str(dataset.utterance_ids[sample_index]),
                "speaker": str(dataset.speaker_ids[sample_index]),
                "gold": g, "prediction": pred,
                "gold_label": LABELS[g], "prediction_label": LABELS[pred],
            }
            for label, probability in zip(LABELS, probabilities[row_index]):
                row[f"prob_{label}"] = float(probability)
            prediction_rows.append(row)
    gold_np = np.asarray(gold_all, dtype=np.int64)
    pred_np = np.asarray(prediction_all, dtype=np.int64)
    summary = {"model": name, "seed": seed, "split": split, "scenario": scenario, **metrics(gold_np, pred_np)}
    per_class = []
    f1s = f1_values(gold_np, pred_np)
    for index, label in enumerate(LABELS):
        per_class.append({
            "model": name, "seed": seed, "split": split, "scenario": scenario,
            "label": label, "support": int((gold_np == index).sum()), "f1": f1s[index],
        })
    return summary, per_class, prediction_rows


def run_epoch(model: nn.Module, name: str, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    gold_all: List[int] = []
    pred_all: List[int] = []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = forward_model(model, name, batch)
            loss = F.cross_entropy(output["logits"], batch["label"])
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch["label"])
        gold_all.extend(batch["label"].detach().cpu().tolist())
        pred_all.extend(output["logits"].detach().argmax(dim=-1).cpu().tolist())
    result = metrics(np.asarray(gold_all), np.asarray(pred_all))
    result["loss"] = total_loss / len(gold_all)
    return result


def train_one(name: str, seed: int, cfg: Mapping[str, Any], train_dataset: MELDFeatureDataset,
              dev_dataset: MELDFeatureDataset, device: torch.device, out: Path) -> Dict[str, Any]:
    set_seed(seed)
    run_dir = out / "runs" / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "logs" / f"{name}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
                              generator=generator, num_workers=0, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=int(cfg["batch_size"]) * 2, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = build_model(name, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg["scheduler_eta_min"]),
    )
    best = -1.0
    best_epoch = 0
    patience_count = 0
    history = []
    checkpoint = run_dir / "best_model.pt"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee_out, tee_err = Tee(sys.stdout, log_handle), Tee(sys.stderr, log_handle)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(json.dumps({"event":"run_start","model":name,"seed":seed,"device":str(device),"time":utc_now()}))
            print(json.dumps(model_parameters(cfg)[name], sort_keys=True))
            for epoch in range(1, int(cfg["epochs"]) + 1):
                train_result = run_epoch(model, name, train_loader, device, optimizer)
                dev_result = run_epoch(model, name, dev_loader, device)
                scheduler.step()
                row = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()[0],
                       **{f"train_{key}": value for key, value in train_result.items()},
                       **{f"dev_{key}": value for key, value in dev_result.items()}}
                history.append(row)
                print(json.dumps(row, sort_keys=True))
                if dev_result["weighted_f1"] > best:
                    best = dev_result["weighted_f1"]
                    best_epoch = epoch
                    patience_count = 0
                    torch.save(model.state_dict(), checkpoint)
                else:
                    patience_count += 1
                    if patience_count >= int(cfg["patience"]):
                        break
            print(json.dumps({"event":"run_end","best_epoch":best_epoch,"best_dev_weighted_f1":best,"time":utc_now()}))
    elapsed = time.perf_counter() - started
    write_json(run_dir / "training_history.json", history)
    run_config = {
        "model": name, "seed": seed, "selection_split": "dev",
        "selection_metric": "weighted_f1", "best_epoch": best_epoch,
        "best_dev_weighted_f1": best, "train_seconds": elapsed,
        "shared_config": dict(cfg), "created_at": utc_now(),
    }
    write_json(run_dir / "config.json", run_config)
    return {
        "model": name, "seed": seed, "success": True,
        "config": relative(run_dir / "config.json"), "config_sha256": sha256(run_dir / "config.json"),
        "checkpoint": relative(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "history": relative(run_dir / "training_history.json"), "history_sha256": sha256(run_dir / "training_history.json"),
        "log": relative(log_path), "log_sha256": sha256(log_path),
        "best_epoch": best_epoch, "best_dev_weighted_f1": best, "train_seconds": elapsed,
    }


def verify_manifest_inputs(manifest: Mapping[str, Any], sections: Sequence[str]) -> None:
    for section in sections:
        for path, expected in manifest[section].items():
            actual = sha256(resolve(path))
            if actual != expected:
                raise ValueError(f"hash mismatch: {path}: {actual} != {expected}")


def environment_record(device: torch.device) -> Dict[str, Any]:
    return {
        "time": utc_now(), "hostname": platform.node(), "cwd": str(Path.cwd()),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def dev_stage(args: argparse.Namespace, cfg: Mapping[str, Any], out: Path) -> None:
    initial = read_json(out / "initial_gate.json")
    if initial.get("decision") != "GO_DEV_ONLY" or initial.get("test_evaluation_unlocked") is not False:
        raise ValueError("initial gate does not authorize dev-only execution")
    manifest = read_json(out / "pre_run_manifest.json")
    # Step 0 evidence is provenance-bound in the manifest but is not part of the
    # checklist-authorized Step 1 upload bundle. Verify only files required for
    # the formal server execution here.
    verify_manifest_inputs(manifest, ("train_dev_inputs", "code_and_protocol"))
    if not torch.cuda.is_available() and not args.allow_cpu_formal:
        raise RuntimeError("formal dev training requires CUDA; use --stage smoke locally")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(out / "runtime" / "dev_environment.json", environment_record(device))
    feature_dir = resolve(cfg["feature_dir"])
    train_dataset = load_split(feature_dir, "train", cfg)
    dev_dataset = load_split(feature_dir, "dev", cfg)
    runs = []
    for name in cfg["models"]:
        for seed in cfg["seeds"]:
            runs.append(train_one(name, int(seed), cfg, train_dataset, dev_dataset, device, out))
    write_csv(out / "checkpoint_index.csv", runs, list(runs[0]))
    dev_loader = DataLoader(dev_dataset, batch_size=int(cfg["batch_size"]) * 2, shuffle=False, num_workers=0)
    metric_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    for run in runs:
        model = build_model(run["model"], cfg).to(device)
        model.load_state_dict(load_state_dict(resolve(run["checkpoint"]), device), strict=True)
        summary, per_class, predictions = evaluate(
            model, run["model"], dev_loader, dev_dataset, device, "dev", "full", int(run["seed"]),
        )
        metric_rows.append(summary); per_class_rows.extend(per_class); prediction_rows.extend(predictions)
    write_csv(out / "dev_metrics.csv", metric_rows, list(metric_rows[0]))
    write_csv(out / "dev_per_class.csv", per_class_rows, list(per_class_rows[0]))
    write_csv(out / "dev_predictions.csv", prediction_rows, PREDICTION_FIELDS)
    expected = {(name, int(seed)) for name in cfg["models"] for seed in cfg["seeds"]}
    actual = {(row["model"], int(row["seed"])) for row in runs}
    identity_ok = len(prediction_rows) == len(expected) * int(cfg["split_rows"]["dev"])
    gate = {
        "step": 1, "stage": "dev", "status": "complete",
        "decision": "GO_FROZEN_TEST" if actual == expected and identity_ok else "NO_GO_TEST",
        "selection_split": "dev", "selection_metric": "weighted_f1",
        "models": list(cfg["models"]), "seeds": list(cfg["seeds"]),
        "checkpoint_set_complete": actual == expected, "dev_prediction_rows_complete": identity_ok,
        "test_evaluated": False, "test_evaluation_unlocked": actual == expected and identity_ok,
        "candidate_reselection_allowed": False, "created_at": utc_now(),
    }
    write_json(out / "dev_selection_gate.json", gate)
    print(json.dumps(gate, indent=2))


def test_stage(args: argparse.Namespace, cfg: Mapping[str, Any], out: Path) -> None:
    gate_path = out / "dev_selection_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("test remains locked: dev_selection_gate.json does not exist")
    gate = read_json(gate_path)
    if gate.get("decision") != "GO_FROZEN_TEST" or gate.get("test_evaluation_unlocked") is not True:
        raise ValueError("test remains locked by dev_selection_gate.json")
    manifest = read_json(out / "pre_run_manifest.json")
    verify_manifest_inputs(manifest, ("test_inputs", "code_and_protocol"))
    if not torch.cuda.is_available() and not args.allow_cpu_formal:
        raise RuntimeError("formal test inference requires CUDA; no local test execution allowed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(out / "runtime" / "test_environment.json", environment_record(device))
    runs = read_csv(out / "checkpoint_index.csv")
    expected = {(name, int(seed)) for name in cfg["models"] for seed in cfg["seeds"]}
    actual = {(row["model"], int(row["seed"])) for row in runs}
    if actual != expected:
        raise ValueError(f"frozen checkpoint set changed: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    for run in runs:
        if sha256(resolve(run["checkpoint"])) != run["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash changed: {run['checkpoint']}")
        if sha256(resolve(run["config"])) != run["config_sha256"]:
            raise ValueError(f"config hash changed: {run['config']}")
    test_dataset = load_split(resolve(cfg["feature_dir"]), "test", cfg)
    test_loader = DataLoader(test_dataset, batch_size=int(cfg["batch_size"]) * 2, shuffle=False, num_workers=0)
    metric_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    for run in runs:
        model = build_model(run["model"], cfg).to(device)
        model.load_state_dict(load_state_dict(resolve(run["checkpoint"]), device), strict=True)
        for scenario in cfg["test_scenarios"]:
            summary, per_class, predictions = evaluate(
                model, run["model"], test_loader, test_dataset, device,
                "test", scenario, int(run["seed"]),
            )
            metric_rows.append(summary); per_class_rows.extend(per_class); prediction_rows.extend(predictions)
    write_csv(out / "test_metrics.csv", metric_rows, list(metric_rows[0]))
    write_csv(out / "test_per_class.csv", per_class_rows, list(per_class_rows[0]))
    write_csv(out / "test_predictions.csv", prediction_rows, PREDICTION_FIELDS)
    final = {
        "step": 1, "stage": "test", "status": "complete",
        "decision": "FORMAL_RESULTS_READY_FOR_LOCAL_MIRROR_AUDIT",
        "models": list(cfg["models"]), "seeds": list(cfg["seeds"]),
        "scenarios": list(cfg["test_scenarios"]), "selection_split": "dev",
        "selection_metric": "weighted_f1", "test_used_for_selection": False,
        "test_inference_passes_per_checkpoint_scenario": 1,
        "test_prediction_rows": len(prediction_rows), "created_at": utc_now(),
    }
    write_json(out / "result_gate.json", final)
    finalize_stage(cfg, out)
    print(json.dumps(final, indent=2))


def finalize_stage(cfg: Mapping[str, Any], out: Path) -> None:
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "final_manifest.json":
            files.append({"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(out / "final_manifest.json", {
        "step": 1, "created_at": utc_now(), "manual_numeric_edits": False,
        "models": list(cfg["models"]), "seeds": list(cfg["seeds"]), "files": files,
    })


def smoke_stage(cfg: Mapping[str, Any]) -> None:
    set_seed(42)
    feat_dim = 64
    samples = 18
    video = np.random.standard_normal((samples, feat_dim)).astype(np.float32)
    text = np.random.standard_normal((samples, feat_dim)).astype(np.float32)
    labels = np.arange(samples, dtype=np.int64) % 7
    dialogue = np.repeat(np.arange(3), 6)
    utterance = np.tile(np.arange(6), 3)
    speakers = np.tile(np.asarray(["A", "B", "A", "C", "B", "A"]), 3)
    dataset = MELDFeatureDataset(
        video, text, labels, dialogue, utterance, speaker_ids=speakers,
        context_len=int(cfg["context_len"]), context_max_distance=int(cfg["context_max_distance"]),
    )
    loader = DataLoader(dataset, batch_size=6, shuffle=False)
    raw_batch = next(iter(loader))
    batch = to_device(raw_batch, torch.device("cpu"))
    checks = []
    parameters = model_parameters(cfg, feat_dim)
    for name in MODELS:
        model = build_model(name, cfg, feat_dim)
        output = forward_model(model, name, batch, return_attention=name == "turn_crossattention")
        loss = F.cross_entropy(output["logits"], batch["label"])
        loss.backward()
        assert output["logits"].shape == (6, 7)
        assert output["probs"].shape == (6, 7)
        assert torch.isfinite(output["logits"]).all()
        assert torch.allclose(output["probs"].sum(dim=-1), torch.ones(6), atol=1e-5)
        if name == "turn_crossattention":
            mask = output["sequence_mask"]
            assert mask.shape == (6, int(cfg["context_len"]) + 1)
            assert mask[:, -1].all()
            weights = output["attention_weights"]
            for direction in ("t2v", "v2t"):
                assert weights[direction].shape[-2:] == (int(cfg["context_len"]) + 1,) * 2
        checks.append({"model": name, "logits_shape": list(output["logits"].shape),
                       "loss": float(loss.detach()), **parameters[name]})
    identity = {
        "model": MODELS[0], "seed": 42, "split": "smoke", "scenario": "full",
        "sample_index": 0, "dialogue_id": 0, "utterance_id": 0, "speaker": "A",
        "gold": 0, "prediction": 0, "gold_label": LABELS[0], "prediction_label": LABELS[0],
        **{f"prob_{label}": 1.0 / 7.0 for label in LABELS},
    }
    assert set(identity) == set(PREDICTION_FIELDS)
    print(json.dumps({
        "status": "PASS", "synthetic_only": True, "dataset_files_opened": False,
        "batch_shape": [6, feat_dim], "context_shape": [6, int(cfg["context_len"]), feat_dim],
        "prediction_schema": PREDICTION_FIELDS, "checks": checks,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "dev", "test", "finalize"), required=True)
    parser.add_argument("--config", default="outputs/ieee_revision/step1_baselines/config.json")
    parser.add_argument("--output-dir", default="outputs/ieee_revision/step1_baselines")
    parser.add_argument("--allow-cpu-formal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = read_json(resolve(args.config))
    if tuple(cfg["models"]) != MODELS or tuple(cfg["seeds"]) != (42, 43, 44, 45, 46):
        raise ValueError("model/seed preregistration changed")
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        smoke_stage(cfg)
    elif args.stage == "dev":
        dev_stage(args, cfg, out)
    elif args.stage == "test":
        test_stage(args, cfg, out)
    else:
        finalize_stage(cfg, out)


if __name__ == "__main__":
    main()
