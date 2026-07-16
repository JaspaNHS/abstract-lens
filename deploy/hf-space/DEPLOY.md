# Deploying Abstract Lens to Hugging Face Spaces (free, stable URL)

This gives you a permanent URL like `https://<your-user>-abstract-lens.hf.space` that
does not depend on your PC. Free tier: 2 vCPU / 16 GB RAM (plenty). The Space sleeps
after ~48 h of inactivity and wakes on the next visit (~30 s cold start).

You only do steps 1, 3, 5, 6 by hand; the rest is one script.

---

## 1. Create the account and the Space
1. Sign up (free) at https://huggingface.co/join
2. New Space → https://huggingface.co/new-space
   - Owner: your user
   - Space name: `abstract-lens`
   - License: your choice
   - **Space SDK: Docker** → **Blank**
   - Visibility: Public (the app is still password-gated) or Private
3. Create the Space.

## 2. Assemble the files to upload (one command)
From the project root, on Windows PowerShell:

```powershell
powershell -File deploy\hf-space\prepare_space.ps1
```

This creates `deploy/hf-space/space-build/` containing the app, the prebuilt ChromaDB
index (~588 MB), `meta_index.json`, the Dockerfile, requirements, README and
`.gitattributes`.

## 3. Install Git LFS (once)
The index is large and binary, so it must go through Git LFS.

```powershell
winget install GitHub.GitLFS      # or: https://git-lfs.com
git lfs install
```

## 4. Push the files to your Space
Replace `<your-user>` with your Hugging Face username.

```powershell
cd deploy\hf-space\space-build
git init
git lfs install
git lfs track "*.sqlite3" "*.bin"
git add .gitattributes
git add .
git commit -m "Abstract Lens on HF Spaces"
git branch -M main
git remote add origin https://huggingface.co/spaces/<your-user>/abstract-lens
git push -u origin main
```

When git asks for a password, use a **Hugging Face access token** (create one at
https://huggingface.co/settings/tokens with `write` scope), not your account password.

## 5. Set the secrets (in the Space web UI)
Space → **Settings** → **Variables and secrets** → **New secret**:
- `ANTHROPIC_API_KEY` = your Anthropic API key
- `APP_PASSWORD`      = the shared password you give colleagues (e.g. a new one)

Add them as **Secrets** (not public variables). The Space rebuilds automatically.

## 6. Use it
- URL: `https://<your-user>-abstract-lens.hf.space`
- The browser will ask for a username (anything) and the `APP_PASSWORD`.
- Put this URL in the abstract instead of the temporary Cloudflare tunnel.

---

## Notes
- **Cost:** hosting is free; each question still calls the Anthropic API (~1–3 cents).
  Set a monthly spend limit in the Anthropic console. The app also rate-limits per IP.
- **Rebuilding the index:** if you re-run the pipeline locally, re-run
  `prepare_space.ps1` and push again to refresh the Space.
- **chromadb is version-pinned** (1.5.9) in requirements.txt so the shipped index reads
  correctly — do not change that pin without rebuilding the index.
