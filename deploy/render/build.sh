#!/usr/bin/env bash
# Render build step: install runtime deps and bake the prebuilt index into the image.
# The index is a PRIVATE GitHub release asset (licensed corpus — not public). It is
# fetched with a token (GH_TOKEN) via the GitHub API and extracted into rag/, so
# rag/chromadb and rag/meta_index.json are present when the app starts. Build-time
# files persist across restarts/wakes (only runtime writes are ephemeral).
set -euo pipefail

echo "[build] installing dependencies"
pip install -r deploy/render/requirements.txt

: "${INDEX_REPO:?set INDEX_REPO, e.g. JaspaNHS/abstract-lens-index}"
: "${INDEX_TAG:?set INDEX_TAG, e.g. index-v1}"
: "${GH_TOKEN:?set GH_TOKEN (read access to the private index repo) as a Render secret}"

echo "[build] locating private index asset in ${INDEX_REPO}@${INDEX_TAG}"
ASSET_URL=$(curl -sfL -H "Authorization: token ${GH_TOKEN}" \
  "https://api.github.com/repos/${INDEX_REPO}/releases/tags/${INDEX_TAG}" \
  | python -c "import sys,json; d=json.load(sys.stdin); print(next(a['url'] for a in d['assets'] if a['name']=='index.zip'))")

echo "[build] downloading private index"
curl -fL -H "Authorization: token ${GH_TOKEN}" -H "Accept: application/octet-stream" \
  "${ASSET_URL}" -o /tmp/index.zip

echo "[build] extracting index into rag/"
python -c "import zipfile; zipfile.ZipFile('/tmp/index.zip').extractall('rag')"
rm -f /tmp/index.zip

test -d rag/chromadb && echo "[build] chromadb present"
test -f rag/meta_index.json && echo "[build] meta_index.json present"
echo "[build] done"
