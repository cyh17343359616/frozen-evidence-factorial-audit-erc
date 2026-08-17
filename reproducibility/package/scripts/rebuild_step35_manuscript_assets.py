#!/usr/bin/env python3
"""Rebuild submission assets from an explicit frozen-derived copy contract.

This packaging entry point deliberately does not reopen row-level test predictions,
feature arrays, checkpoints, or prototypes. The historical Step 35 recomputation
path depended on those restricted artifacts and also opened undeclared
``outputs/ablation_v2/**/config.json`` files. Step 17 replaces that implicit path
with a self-contained package of non-row-level derived assets. Every file read by
the rebuild is listed and hashed in the emitted manifest.

``--prepare-package`` snapshots only already-generated TeX/figure assets and
environment metadata; it never reads the historical numerical inputs. Normal
rebuild mode verifies the contract, copies assets into a clean target tree, and
records the original generator/input provenance without claiming recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


GENERATED_TEX = [
    "all_array_rows.tex", "comparison_rows.tex", "confusion_matrix_rows.tex",
    "controlled_baseline_rows.tex", "dg_stress_rows.tex", "efficiency_rows.tex",
    "error_bucket_rows.tex", "factorial_rows.tex", "failure_case_rows.tex",
    "feature_cache_rows.tex", "final_robustness_rows.tex",
    "frozen_evidence_factorial_audit_algorithm.tex", "legacy_diagnostic_rows.tex",
    "main_configuration_rows.tex", "per_class_rows.tex",
    "s05_filtering_numbers.tex", "spm_step6_numbers.tex",
]

FIGURE_ASSETS = [
    "architecture.pdf", "artifact_sizes.png", "bootstrap_ci.png",
    "error_distribution_analysis/fig_confusion_heatmap_A.pdf",
    "factorial_design/fig_factorial_branches_B.pdf",
    "jsdg_workflow/fig_jsdg_core_flow.pdf",
    "jsdg_workflow/fig_jsdg_core_flow.svg",
    "overall_framework/fig_overall_framework_B.pdf",
    "robustness_evaluation/fig_jsdg_stress_auc_B.pdf",
]

BASE_STEP35_TEX = {
    "all_array_rows.tex", "comparison_rows.tex", "confusion_matrix_rows.tex",
    "dg_stress_rows.tex", "efficiency_rows.tex", "error_bucket_rows.tex",
    "factorial_rows.tex", "failure_case_rows.tex", "feature_cache_rows.tex",
    "final_robustness_rows.tex", "legacy_diagnostic_rows.tex",
    "main_configuration_rows.tex", "per_class_rows.tex",
}

ORIGIN_BY_TEX = {
    **{name: "historical_step35" for name in BASE_STEP35_TEX},
    "controlled_baseline_rows.tex": "step15_controlled_baselines",
    "frozen_evidence_factorial_audit_algorithm.tex": "step7_algorithm",
    "s05_filtering_numbers.tex": "step5_s05",
    "spm_step6_numbers.tex": "step6_spm",
}

RESTRICTED_LEGACY_INPUTS = {
    "outputs/phase3_ieee_access/step28_factorial_ablation/confirmation_test_predictions.csv",
    "outputs/phase3_ieee_access/step29_robustness_stress/predictions.csv",
    "outputs/phase3_ieee_access/step29_robustness_stress/mechanism_distributions.csv",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def hash_from_manifest(manifest: dict[str, Any], filename: str) -> str:
    values = manifest.get("sha256") or manifest.get("outputs", {})
    if filename not in values:
        raise KeyError(f"{filename} absent from archived manifest")
    return str(values[filename])


def legacy_dependency_registry(repo_root: Path) -> list[dict[str, Any]]:
    """Describe every historical Step 35 numerical read without reopening it."""
    manifests = {
        key: read_json(repo_root / path)
        for key, path in {
            "s28": "outputs/phase3_ieee_access/step28_factorial_ablation/final_manifest.json",
            "s29": "outputs/phase3_ieee_access/step29_robustness_stress/final_manifest.json",
            "s30": "outputs/phase3_ieee_access/step30_speaker_generalization/final_manifest.json",
            "s31": "outputs/phase3_ieee_access/step31_external_baselines/final_manifest.json",
            "s33": "outputs/phase3_ieee_access/step33_statistics_efficiency/final_manifest.json",
        }.items()
    }
    roots = {
        "s28": "outputs/phase3_ieee_access/step28_factorial_ablation",
        "s29": "outputs/phase3_ieee_access/step29_robustness_stress",
        "s30": "outputs/phase3_ieee_access/step30_speaker_generalization",
        "s31": "outputs/phase3_ieee_access/step31_external_baselines",
        "s33": "outputs/phase3_ieee_access/step33_statistics_efficiency",
    }
    specs = [
        ("s28", "confirmation_test_metrics.csv"),
        ("s33", "paired_bootstrap_summary.csv"),
        ("s28", "confirmation_test_per_class.csv"),
        ("s28", "confirmation_test_predictions.csv"),
        ("s30", "speaker_manifest.csv"),
        ("s29", "mechanism_distributions.csv"),
        ("s29", "predictions.csv"),
        ("s29", "family_auc_matched_deltas.csv"),
        ("s30", "test_paired_deltas.csv"),
        ("s31", "reported_results.csv"),
        ("s31", "proxy_results.csv"),
        ("s33", "paper_tables/efficiency_table.csv"),
        ("s33", "paper_tables/artifact_sizes.csv"),
    ]
    records: list[dict[str, Any]] = []
    for key, filename in specs:
        path = f"{roots[key]}/{filename}"
        records.append({
            "path": path,
            "sha256": hash_from_manifest(manifests[key], filename),
            "hash_source": f"{roots[key]}/final_manifest.json",
            "included_in_standalone": False,
            "restricted": path in RESTRICTED_LEGACY_INPUTS,
        })

    # This audit table was an actual direct input but was omitted from the
    # archived Step 28 final_manifest.json, so register its present hash rather
    # than pretending that the historical manifest covered it.
    legacy_audit = repo_root / roots["s28"] / "legacy_feature_evidence_audit.csv"
    records.append({
        **relative_record(legacy_audit, repo_root),
        "hash_source": "direct hash; absent from archived Step 28 final_manifest.json",
        "included_in_standalone": False,
        "restricted": False,
    })

    frozen_arrays = repo_root / "supplementary/reproducibility/frozen_arrays.csv"
    records.append({
        **relative_record(frozen_arrays, repo_root),
        "hash_source": "local metadata file",
        "included_in_standalone": False,
        "restricted": False,
    })
    # These five dynamic config paths were omitted from the historical manifest.
    config_paths = [
        "outputs/ablation_v2/text_only/text_only_20260312_185357/config.json",
        "outputs/ablation_v2/video_only/video_only_20260312_185426/config.json",
        "outputs/ablation_v2/concat/concat_20260312_185508/config.json",
        "outputs/ablation_v2/cross_attention/attention_20260312_185533/config.json",
        "outputs/ablation_v2/gated_fusion/gated_20260312_185617/config.json",
    ]
    for name in config_paths:
        path = repo_root / name
        records.append({
            **relative_record(path, repo_root),
            "hash_source": "direct hash; dynamically opened by legacy_feature_evidence_audit.csv",
            "included_in_standalone": False,
            "restricted": False,
        })
    return records


def provenance_contracts(repo_root: Path) -> dict[str, Any]:
    step15 = read_json(repo_root / "outputs/ieee_revision/step15_baseline_presentation/table_manifest.json")
    step5_gate = read_json(repo_root / "outputs/ieee_revision/step5_s05/gate.json")
    step6_gate = read_json(repo_root / "outputs/ieee_revision/step6_spm/gate.json")
    step7_gate = read_json(repo_root / "outputs/ieee_revision/step7_protocol/gate.json")
    return {
        "historical_step35": {
            "mode": "frozen_derived_copy",
            "historical_generator": {
                "path": "scripts/rebuild_step35_manuscript_assets.py",
                "sha256_before_step17_repair": "e4cb8a4852ab64c7dee96c52aeb598bcf0ec7b6b22d678bbe334d2673920ad2b",
            },
            "actual_historical_inputs": legacy_dependency_registry(repo_root),
            "recomputation_boundary": "row-level inputs are not reopened or bundled; Step 17 verifies frozen derived outputs",
        },
        "step15_controlled_baselines": {
            "mode": "frozen_derived_copy",
            "generator": step15["generator"],
            "original_inputs": step15["inputs"],
            "source_manifest": {
                "path": "outputs/ieee_revision/step15_baseline_presentation/table_manifest.json",
                "sha256": sha256(repo_root / "outputs/ieee_revision/step15_baseline_presentation/table_manifest.json"),
            },
        },
        "step5_s05": {
            "mode": "frozen_derived_copy",
            "generator": {
                "path": "scripts/audit_revision_step5_s05.py",
                "sha256": sha256(repo_root / "scripts/audit_revision_step5_s05.py"),
            },
            "original_inputs_by_hash": step5_gate["evidence"],
            "gate": {
                "path": "outputs/ieee_revision/step5_s05/gate.json",
                "sha256": sha256(repo_root / "outputs/ieee_revision/step5_s05/gate.json"),
            },
        },
        "step6_spm": {
            "mode": "frozen_derived_copy",
            "generator": {
                "path": "scripts/audit_revision_step6_spm.py",
                "sha256": sha256(repo_root / "scripts/audit_revision_step6_spm.py"),
            },
            "original_inputs_by_hash": step6_gate["machine_evidence"],
            "gate": {
                "path": "outputs/ieee_revision/step6_spm/gate.json",
                "sha256": sha256(repo_root / "outputs/ieee_revision/step6_spm/gate.json"),
            },
        },
        "step7_algorithm": {
            "mode": "frozen_derived_copy",
            "generator": step7_gate["generator"],
            "original_inputs_by_hash": step7_gate["inherited_gate_handoff_hashes"],
            "canonical_source": step7_gate["algorithm"],
        },
    }


def preparation_input_registry(repo_root: Path) -> list[dict[str, Any]]:
    """Register every repository file opened while preparing the package."""
    paths = [
        *(f"paper/submission/source/generated/{name}" for name in GENERATED_TEX),
        *(f"paper/submission/source/figures/{name}" for name in FIGURE_ASSETS),
        "supplementary/reproducibility/environment_manifest.json",
        "supplementary/reproducibility/requirements_frozen.txt",
        "supplementary/reproducibility/requirements_rebuild.txt",
        "paper/submission/supplementary/step33/figure_sources/artifact_size_source.csv",
        "paper/submission/supplementary/step33/figure_sources/bootstrap_ci_source.csv",
        "outputs/phase3_ieee_access/step28_factorial_ablation/final_manifest.json",
        "outputs/phase3_ieee_access/step29_robustness_stress/final_manifest.json",
        "outputs/phase3_ieee_access/step30_speaker_generalization/final_manifest.json",
        "outputs/phase3_ieee_access/step31_external_baselines/final_manifest.json",
        "outputs/phase3_ieee_access/step33_statistics_efficiency/final_manifest.json",
        "outputs/phase3_ieee_access/step28_factorial_ablation/legacy_feature_evidence_audit.csv",
        "supplementary/reproducibility/frozen_arrays.csv",
        "outputs/ablation_v2/text_only/text_only_20260312_185357/config.json",
        "outputs/ablation_v2/video_only/video_only_20260312_185426/config.json",
        "outputs/ablation_v2/concat/concat_20260312_185508/config.json",
        "outputs/ablation_v2/cross_attention/attention_20260312_185533/config.json",
        "outputs/ablation_v2/gated_fusion/gated_20260312_185617/config.json",
        "outputs/ieee_revision/step15_baseline_presentation/table_manifest.json",
        "outputs/ieee_revision/step5_s05/gate.json",
        "outputs/ieee_revision/step6_spm/gate.json",
        "outputs/ieee_revision/step7_protocol/gate.json",
        "scripts/audit_revision_step5_s05.py",
        "scripts/audit_revision_step6_spm.py",
    ]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate preparation input path")
    return [relative_record(repo_root / name, repo_root) for name in paths]


def prepare_package(repo_root: Path, package_root: Path) -> None:
    inputs = package_root / "rebuild_inputs"
    generated, figures, environment = inputs / "generated", inputs / "figures", inputs / "environment"
    for path in (generated, figures, environment):
        path.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    for name in GENERATED_TEX:
        source = repo_root / "paper/submission/source/generated" / name
        target = generated / name
        shutil.copyfile(source, target)
        assets.append({
            "id": f"generated/{name}",
            "source": str(target.relative_to(package_root)),
            "source_sha256": sha256(target),
            "destinations": [f"paper/generated/{name}", f"paper/submission/source/generated/{name}"],
            "origin_contract": ORIGIN_BY_TEX[name],
            "kind": "generated_tex",
            "rebuild_mode": "verified_copy",
        })

    for name in FIGURE_ASSETS:
        source = repo_root / "paper/submission/source/figures" / name
        target = figures / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assets.append({
            "id": f"figures/{name}",
            "source": str(target.relative_to(package_root)),
            "source_sha256": sha256(target),
            "destinations": [f"paper/submission/source/figures/{name}"],
            "origin_contract": "figure_assets",
            "kind": "figure_or_figure_source",
            "rebuild_mode": "verified_copy",
        })

    evidence_sources = [
        "supplementary/reproducibility/environment_manifest.json",
        "supplementary/reproducibility/requirements_frozen.txt",
        "supplementary/reproducibility/requirements_rebuild.txt",
    ]
    declared_evidence_inputs = []
    for name in evidence_sources:
        source = repo_root / name
        target = environment / source.name
        shutil.copyfile(source, target)
        declared_evidence_inputs.append({
            "path": str(target.relative_to(package_root)),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "purpose": "environment or dependency boundary",
        })
    for name in [
        "step33/figure_sources/artifact_size_source.csv",
        "step33/figure_sources/bootstrap_ci_source.csv",
    ]:
        path = package_root / name
        declared_evidence_inputs.append({
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "purpose": "archived numerical figure source; PNG is verified-copy for byte identity",
        })

    contract = {
        "schema": "ieee_revision_step17_rebuild_contract_v1",
        "scope": "non-row-level frozen-derived submission asset package",
        "training_performed": False,
        "inference_performed": False,
        "test_predictions_read": False,
        "assets": assets,
        "package_preparation_inputs": preparation_input_registry(repo_root),
        "declared_evidence_inputs": declared_evidence_inputs,
        "provenance_contracts": {
            **provenance_contracts(repo_root),
            "figure_assets": {
                "mode": "verified_copy",
                "note": "Figure outputs are copied byte-identically; archived source CSV/SVG files remain declared for provenance.",
            },
        },
        "restricted_input_policy": {
            "excluded": True,
            "excluded_categories": [
                "raw MELD media and transcripts",
                "row-level MELD identifiers or predictions",
                "MELD-derived feature arrays",
                "checkpoints and speaker prototype arrays",
                "S05 source-linked row-level artifacts",
                "third-party model weights",
            ],
            "standalone_recomputation_from_restricted_inputs": "NO_GO",
            "standalone_byte_identical_rebuild_from_frozen_derived_assets": "SUPPORTED",
        },
    }
    write_json(package_root / "rebuild_contract.json", contract)
    print(f"prepared non-restricted rebuild package at {package_root}")


def rebuild(package_root: Path, target_root: Path, manifest_output: Path | None) -> dict[str, Any]:
    observed_inputs: list[dict[str, Any]] = []
    contract_path = package_root / "rebuild_contract.json"
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    observed_inputs.append({"path": "rebuild_contract.json", "bytes": len(contract_bytes), "sha256": sha256_bytes(contract_bytes)})

    for record in contract["declared_evidence_inputs"]:
        path = package_root / record["path"]
        actual = relative_record(path, package_root)
        if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
            raise RuntimeError(f"declared evidence input mismatch: {record['path']}")
        observed_inputs.append(actual)

    output_records: list[dict[str, Any]] = []
    for asset in contract["assets"]:
        source = package_root / asset["source"]
        data = source.read_bytes()
        actual_hash = sha256_bytes(data)
        if actual_hash != asset["source_sha256"]:
            raise RuntimeError(f"source hash mismatch: {asset['source']}")
        observed_inputs.append({"path": asset["source"], "bytes": len(data), "sha256": actual_hash})
        for destination_name in asset["destinations"]:
            destination = target_root / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            output_records.append({
                "asset_id": asset["id"], "path": destination_name,
                "bytes": destination.stat().st_size, "sha256": sha256(destination),
                "origin_contract": asset["origin_contract"], "rebuild_mode": asset["rebuild_mode"],
            })

    expected_inputs = {"rebuild_contract.json"}
    expected_inputs.update(x["path"] for x in contract["declared_evidence_inputs"])
    expected_inputs.update(x["source"] for x in contract["assets"])
    observed_names = {x["path"] for x in observed_inputs}
    if observed_names != expected_inputs:
        raise RuntimeError(f"input registry mismatch: missing={sorted(expected_inputs-observed_names)} extra={sorted(observed_names-expected_inputs)}")

    generator_path = Path(__file__).resolve()
    manifest = {
        "schema": "ieee_revision_step17_asset_manifest_v1",
        "generator": {"path": "scripts/rebuild_step35_manuscript_assets.py", "bytes": generator_path.stat().st_size, "sha256": sha256(generator_path)},
        "contract": observed_inputs[0],
        "mode": "byte-identical verified copy from non-row-level frozen-derived assets",
        "model_inference_performed": False,
        "training_performed": False,
        "test_predictions_read": False,
        "all_read_paths_registered": True,
        "restricted_inputs_excluded": True,
        "standalone_recomputation_from_restricted_inputs": "NO_GO",
        "observed_inputs": observed_inputs,
        "outputs": output_records,
        "provenance_contracts": contract["provenance_contracts"],
        "restricted_input_policy": contract["restricted_input_policy"],
    }
    if manifest_output is None:
        manifest_output = target_root / "paper/generated/asset_manifest.json"
    write_json(manifest_output, manifest)
    print(f"rebuilt {len(output_records)} destination files; manifest={manifest_output}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--prepare-package", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    package_root = args.package_root.resolve() if args.package_root else repo_root / "paper/submission/supplementary"
    if args.prepare_package:
        prepare_package(repo_root, package_root)
        return
    target_root = args.target_root.resolve() if args.target_root else repo_root
    manifest_output = args.manifest_output.resolve() if args.manifest_output else None
    rebuild(package_root, target_root, manifest_output)


if __name__ == "__main__":
    main()
