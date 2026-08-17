# Release Candidate Audit

Audit date: 2026-08-17 (Asia/Shanghai)

## Outcome

The directory is a locally verified, data-safe release candidate. All five
software creators and joint copyright holders have authorized publication of
the approved inventory under Mulan Permissive Software License Version 2. The
repository, tag, release, and DOI must still be verified as separate external
objects before they are cited as existing.

The public GitHub repository was verified on 2026-08-17 at
`https://github.com/cyh17343359616/frozen-evidence-factorial-audit-erc`.
Zenodo DOI `10.5281/zenodo.21981112` was reserved but not registered at this
audit point. No release tag or Zenodo archive existed.

The authors confirmed support from 武汉市2025年度重点研发计划, project number
`2025020602030097`. `FUNDING.md` records the official Chinese project metadata,
government evidence URL, and the boundary against inventing official English
names.

## Offline checks completed

- Python syntax compilation passed for all included Python sources.
- The packaged manuscript-asset rebuild completed successfully and produced
  43 hash-verified destination files in a clean temporary directory.
- `SHA256SUMS` verification passed for every listed file.
- No symbolic links or files larger than 20 MiB are present.
- No model checkpoints, feature arrays, media archives, audio/video files, or
  other blocked binary types are present.
- No CSV file exposes sample, utterance, dialogue, speaker, gold-label, or
  prediction columns.
- No common private-key or access-token pattern was detected.
- No WHU server path, local user path, server account, or server hostname is
  embedded in the distributable sources.

## Scope boundary

Aggregate metrics and manuscript assets are included for provenance. MELD
content, source-linked audit rows, features, predictions, checkpoints,
prototypes, third-party weights, and server logs are excluded. Their absence
is intentional and must not be treated as a missing-file defect.

## Reproduction level

The included Step 17 rebuild is an offline hash-verified reconstruction of the
manuscript assets from the declared archive inputs. It is not a claim that a
third party can retrain models or regenerate restricted features without
obtaining the upstream data and model dependencies separately.
