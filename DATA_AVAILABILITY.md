# Data and artifact availability

This release candidate distributes no MELD data or row-level MELD-derived artifact.

## Included

- Project-authored source code, subject to the publication hold and license confirmation.
- Data-free experimental configuration contracts.
- Aggregate model/seed/scenario and per-class metrics for the controlled baselines.
- Generated TeX and figure assets that contain aggregate results only.
- Environment, dependency, provenance-limit, and hash records.

## Excluded

- MELD media, transcripts, annotations, and source CSV files.
- Utterance-, dialogue-, speaker-, label-, or sample-level identity records.
- Text, video, or audio feature arrays and cached embeddings.
- Row-level predictions, probabilities, logits, gates, Jensen--Shannon values, and joins.
- Checkpoints, optimizer states, speaker prototypes, and lookup arrays.
- S05 source-linked augmentation and audit rows or bundles.
- Qwen and all other third-party model weights/processors/caches.
- Server logs, credentials, personal paths, and unrelated user material.

The paths and hashes of some excluded artifacts appear in provenance contracts so that an authorized holder can verify a private copy. Such hashes do not redistribute the underlying content or grant access rights.

Users must obtain MELD and any third-party models from their respective authorized sources and comply with the original terms. No claim is made that historical Qwen feature extraction can be reproduced byte-for-byte: the immutable model/processor revisions, actual frame trace, token range, and complete historical extraction environment were not archived.
