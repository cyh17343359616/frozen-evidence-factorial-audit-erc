# Frozen-Evidence Factorial Audit for Multimodal ERC

This directory is a **data-safe local release candidate** for the IEEE Access revision. It contains project code, frozen configurations, non-row-level aggregate baseline evidence, and a byte-identical reconstruction package for manuscript-generated assets.

It does not contain MELD media or transcripts, row-level identifiers or labels, frozen feature arrays, row-level predictions/logits, checkpoints, prototypes, S05 source-linked rows, or third-party model weights.

## Publication status

The five software creators and joint copyright holders have authorized this
data-safe inventory for public release under Mulan Permissive Software License
Version 2. `PUBLICATION_HOLD.md` retains the closed authorization record and the
boundary between authorization and objects that do not yet exist. The GitHub
repository is public and was verified on 2026-08-17. No tag, GitHub Release,
release date, or DOI is claimed until each is created and verified.

## Contents

- `code/`: current project implementation and revision runners relevant to the paper.
- `configs/`: frozen experimental configurations. Paths to restricted inputs are contracts only; the inputs are not distributed.
- `evidence/`: aggregate Step 1 metrics and the Step 15 controlled-baseline table contract. No per-utterance rows are included.
- `reproducibility/package/`: Step 17 non-row-level package that reconstructs 43 generated TeX/figure destinations byte-for-byte.
- `MANUSCRIPT_BINDING.json`: hashes binding this candidate to the audited manuscript source/PDF without distributing the manuscript or author photograph.
- `RELEASE_MANIFEST.json` and `SHA256SUMS`: machine-readable release inventory and checksums.
- `GITHUB_ZENODO_RELEASE_GUIDE.md`: exact publication procedure after the HOLD is closed.

## Local verification

Run from this directory:

```bash
python3 -m py_compile code/scripts/*.py code/src/data/*.py \
  code/src/models/*.py code/src/training/*.py \
  reproducibility/package/scripts/rebuild_step35_manuscript_assets.py

python3 tools/build_release_manifest.py

VERIFY_ROOT=$(mktemp -d)
python3 reproducibility/package/scripts/rebuild_step35_manuscript_assets.py \
  --repo-root reproducibility/package \
  --package-root reproducibility/package \
  --target-root "$VERIFY_ROOT"
```

The last command verifies/copies frozen derived assets. It does not recompute results from excluded MELD-derived inputs.

## Training and evaluation boundary

The training/evaluation runners document the tested contracts but require separately obtained, rights-controlled frozen features and other private inputs. Test selection remains prohibited: checkpoints must be selected on train/dev, frozen, and only then evaluated once on test under the declared scenario.

## License

The repository carries the existing Mulan Permissive Software License Version 2
text. All five confirmed copyright holders have authorized its application to
the project-authored files in this release. The license does not apply to MELD
or third-party artifacts that are not distributed here.
