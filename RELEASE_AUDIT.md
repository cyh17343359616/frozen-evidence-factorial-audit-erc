# Release Candidate Audit

Audit date: 2026-08-17 (Asia/Shanghai)

## Outcome

The directory is a locally verified, data-safe release candidate. All five
software creators and joint copyright holders have authorized publication of
the approved inventory under Mulan Permissive Software License Version 2. The
repository, tag, release, and DOI must still be verified as separate external
objects before they are cited as existing.

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
