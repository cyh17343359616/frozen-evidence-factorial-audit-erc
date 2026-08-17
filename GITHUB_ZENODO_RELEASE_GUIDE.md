# GitHub and Zenodo release guide

Complete these steps only after every item in `PUBLICATION_HOLD.md` is closed.

## 1. Final local review

Run from this `release/` directory:

```bash
python3 tools/build_release_manifest.py
python3 -m py_compile code/scripts/*.py code/src/data/*.py \
  code/src/models/*.py code/src/training/*.py \
  reproducibility/package/scripts/rebuild_step35_manuscript_assets.py

rg -n -i '/Users/|/project2/|swarm\.whu|lixiaolei|password|passwd|secret|api[_-]?key|access[_-]?token|private[_-]?key' .

find . -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.pt' \
  -o -name '*.pth' -o -name '*.ckpt' -o -name '*.pkl' \
  -o -name '*.mp4' -o -name '*.wav' \) -print
```

Expected: no restricted binary is found. Text matches must be reviewed; policy documents will intentionally mention prohibited terms.

Add an accurate `CITATION.cff` using only the confirmed creator names. ORCID fields may be omitted when unconfirmed. Do not invent funding, affiliations, author order, or identifiers. Update `NOTICE` with the confirmed copyright holder(s), then rerun the manifest builder.

## 2. Create the local Git history

```bash
git init -b main
git add --all
git status --short
git diff --cached --stat
git diff --cached
git commit -m "IEEE Access reproducibility release v1.0.0"
git rev-parse HEAD
```

Save the resulting full commit SHA. Do not use the parent project's old commit.

## 3. Create an empty GitHub repository

In GitHub:

1. Click **New repository**.
2. Choose the responsible owner account or organization.
3. Set a repository name.
4. Initially choose **Private** for the final online inspection.
5. Do not initialize README, `.gitignore`, or LICENSE.
6. Create the repository.

Connect and push, replacing the two shell variables with the real values:

```bash
GITHUB_ACCOUNT='confirmed-account'
GITHUB_REPOSITORY='confirmed-repository-name'

git remote add origin "https://github.com/${GITHUB_ACCOUNT}/${GITHUB_REPOSITORY}.git"
git push -u origin main
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

The two commit hashes must agree. Inspect every file on GitHub, then change repository visibility to **Public** in GitHub Settings. Re-open the public URL in a signed-out/private browser window.

## 4. Enable Zenodo before making the GitHub Release

1. Sign in to Zenodo.
2. Under **Linked accounts**, connect GitHub.
3. Open the Zenodo **GitHub** page and click **Sync now**.
4. Find this repository and enable its integration toggle.
5. Confirm that Zenodo shows it as connected.

## 5. Create and push the fixed tag

```bash
git tag -a v1.0.0 -m "IEEE Access reproducibility release v1.0.0"
git rev-list -n 1 v1.0.0
git push origin v1.0.0
git ls-remote origin refs/tags/v1.0.0 refs/tags/v1.0.0^{}
```

The peeled annotated tag must resolve to the same commit recorded above.

## 6. Create the GitHub Release

Using GitHub CLI:

```bash
gh release create v1.0.0 \
  --verify-tag \
  --title "IEEE Access reproducibility release v1.0.0" \
  --notes "Data-safe project code, frozen aggregate evidence, and non-row-level reconstruction materials. Restricted MELD- and model-derived artifacts are excluded."
```

Alternatively, use GitHub's Releases page and select the existing `v1.0.0` tag. Do not upload any extra archive from the parent project.

## 7. Obtain and verify the Zenodo DOI

Wait for Zenodo to process the GitHub Release. Open the generated record and verify:

- software title and creator list;
- version `v1.0.0`;
- Mulan PSL v2 license, only after its scope is confirmed;
- the GitHub repository/release relation;
- the archived files include `RELEASE_MANIFEST.json` and `SHA256SUMS`;
- the record has a version DOI beginning with `10.5281/zenodo.`.

Record the version DOI, the tagged commit SHA, the public repository/release URLs, and the real verification date:

```bash
date +%F
```

Do not use a DOI until its public Zenodo page resolves.

## 8. Code Availability text

Only after all checks pass, replace the internal statement with a factual sentence containing the real values:

```text
The project-authored code and non-restricted reproducibility materials are
available at the public GitHub repository and release verified by the authors,
at release v1.0.0 and its full commit SHA, under Mulan PSL v2. The same release
is archived under its resolving Zenodo version DOI. Accessed on the actual
verification date. MELD media/transcripts, row-level metadata, derived features,
predictions, checkpoints, prototypes, S05 source-linked artifacts, and
third-party model weights are not redistributed.
```

Insert the actual URLs, SHA, DOI, and date only after verification. Recompile the manuscript and update its source/PDF binding if the Code Availability edit changes either hash.

## Gitee alternative

The same local candidate may be pushed to a new Gitee repository and tagged `v1.0.0`. Because Zenodo's automatic integration is for GitHub, create the exact tagged ZIP manually:

```bash
git archive --format=zip --prefix=project-v1.0.0/ \
  -o project-v1.0.0.zip v1.0.0
shasum -a 256 project-v1.0.0.zip
```

Upload only that ZIP to a Zenodo **Software** record, enter the confirmed creators, license, version, and Gitee tag/commit relation, publish it, and verify the resolving DOI. Do not commit or upload the ZIP back into the repository.
