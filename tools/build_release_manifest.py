#!/usr/bin/env python3
"""Build a deterministic inventory and SHA256SUMS for the data-safe candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
CHECKSUMS = ROOT / "SHA256SUMS"
EXCLUDED_OUTPUTS = {MANIFEST.name, CHECKSUMS.name}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or path.name == ".DS_Store"
        ):
            continue
        if path.parent == ROOT and path.name in EXCLUDED_OUTPUTS:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    files = candidate_files()
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    payload = {
        "schema": "ieee_access_data_safe_release_candidate_v1",
        "status": "PUBLIC_REPOSITORY_VERIFIED_ZENODO_DOI_RESERVED_READY_TO_TAG",
        "publication_authorized": True,
        "remote_repository_created": True,
        "remote_repository_url": "https://github.com/cyh17343359616/frozen-evidence-factorial-audit-erc",
        "remote_visibility": "public",
        "public_repository_verified_on": "2026-08-17",
        "tag_created": False,
        "archive_doi": "10.5281/zenodo.21981112",
        "archive_doi_status": "reserved_not_registered",
        "creators": [
            "YUHAN CHEN",
            "SENNING WANG",
            "YANGGE TIAN",
            "JIANGUANG TU",
            "XIAOLEI LI",
        ],
        "creator_order_confirmed": True,
        "license": {
            "candidate": "Mulan Permissive Software License Version 2",
            "spdx_identifier": "MulanPSL-2.0",
            "application_to_current_files_confirmed": True,
            "copyright_holders_confirmed": True,
            "copyright_holders": [
                "YUHAN CHEN",
                "SENNING WANG",
                "YANGGE TIAN",
                "JIANGUANG TU",
                "XIAOLEI LI",
            ],
        },
        "funding": {
            "support_relationship_confirmed": True,
            "programme": "武汉市2025年度重点研发计划",
            "project_number": "2025020602030097",
            "project_title": "基于Deepseek的神经内科AI辅助诊疗系统",
            "issuing_agency": "武汉市科技创新局",
            "responsible_institution": "湖北省人民医院（武汉大学人民医院）",
            "official_record_url": "https://kjj.wuhan.gov.cn/zwgk_8/fdzdnrgk/sjczzxzj/xmhzjap/202509/t20250917_2648782.shtml",
            "official_english_name_confirmed": False,
        },
        "manuscript_binding": "MANUSCRIPT_BINDING.json",
        "included_file_count_excluding_generated_inventory": len(records),
        "files": records,
        "mandatory_exclusions": [
            "MELD media transcripts annotations and source CSVs",
            "row-level utterance dialogue speaker label and sample records",
            "derived feature arrays and cached embeddings",
            "row-level predictions probabilities logits gates and JS values",
            "checkpoints optimizer states prototypes and lookup arrays",
            "S05 source-linked rows and bundles",
            "third-party model weights processors and caches",
            "server logs credentials personal paths and unrelated files",
        ],
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    checksum_paths = sorted(
        [*files, MANIFEST], key=lambda item: item.relative_to(ROOT).as_posix()
    )
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in checksum_paths
    ]
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(records)} source files")
    print(f"wrote {CHECKSUMS.relative_to(ROOT)} with {len(checksum_paths)} hashes")


if __name__ == "__main__":
    main()
