# Deploying Abstract Lens to Render (free, permanent URL, private index)

Free web service, 512 MB RAM (the app uses ~290 MB), stable URL
`https://abstract-lens.onrender.com`, no credit card. The service sleeps after 15 min
idle and wakes on the next request (~30–60 s). The licensed corpus is kept in a PRIVATE
GitHub repo and fetched at build time with a token — it is never made public.

You do steps 1, 2, 4, 5 by hand. Everything else is already in the repo (`render.yaml`,
`deploy/render/build.sh`).

---

## 1. Upload the index to your PRIVATE repo (you run this)
The private repo `JaspaNHS/abstract-lens-index` already exists. From the project folder
(where `index.zip` is), in PowerShell:

```powershell
gh release create index-v1 index.zip --repo JaspaNHS/abstract-lens-index --title "Prebuilt index v1" --notes "Private index for Render. Licensed corpus - do not redistribute."
```

(If you ever rebuild the index: re-zip and run `gh release upload index-v1 index.zip --repo JaspaNHS/abstract-lens-index --clobber`.)

## 2. Create a GitHub token for Render to read the private index
- Go to https://github.com/settings/tokens → Fine-grained tokens → Generate new token.
- Repository access: only `JaspaNHS/abstract-lens-index`.
- Permissions: Contents → Read-only.
- Generate and copy the token (starts with `github_pat_...`). You will paste it into
  Render as `GH_TOKEN`.

## 3. (already done) Deployment config is in the repo
`render.yaml` (build/start commands, env var names) and `deploy/render/build.sh`
(private index fetch) are committed to the public code repo — no corpus is in them.

## 4. Create the Render service
- Sign up (free, no card) at https://render.com → New → **Blueprint**.
- Connect your GitHub and select `JaspaNHS/abstract-lens`.
- Render reads `render.yaml` and proposes the `abstract-lens` web service → Apply.

## 5. Set the secrets (Render dashboard → the service → Environment)
Add these three (as secrets):
- `GH_TOKEN`          = the fine-grained token from step 2
- `ANTHROPIC_API_KEY` = your Anthropic API key
- `APP_PASSWORD`      = the shared password for colleagues

Save → Render builds (downloads the private index, ~1–2 min) and deploys.

## 6. Use it
- URL: `https://abstract-lens.onrender.com` (Render shows the exact URL).
- The browser asks for a username (anything) and `APP_PASSWORD`.
- Put this URL in the abstract instead of the temporary Cloudflare tunnel.

---

## Notes
- **Cost:** hosting is free; each question still calls the Anthropic API (~1–3 cents).
  Keep a monthly spend limit set in the Anthropic console. The app rate-limits per IP.
- **Cold start:** after 15 min idle the first request takes ~30–60 s while the service
  wakes; subsequent requests are fast.
- **chromadb is version-pinned** (1.5.9) so the shipped index reads correctly.
- **Licence:** the corpus lives only in your private repo and on your private Render
  instance; it is never published. Do not change the index repo to public.
