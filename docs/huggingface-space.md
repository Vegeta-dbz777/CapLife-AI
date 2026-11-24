# Hugging Face Space mirror

The main app stays on Streamlit Community Cloud. A Hugging Face Space can run the same code as a
second copy. After setup, every merge into `main` updates both: Streamlit Cloud redeploys itself,
and the GitHub Action `Mirror to Hugging Face Space` uploads the code to the Space, which rebuilds.

Files involved:

| File | Purpose |
| --- | --- |
| `Dockerfile` | Builds the Space: Python 3.11, CPU-only PyTorch, `requirements.txt`, port 7860 |
| `deploy/huggingface/start.sh` | Writes the Space secret `STREAMLIT_SECRETS` to `.streamlit/secrets.toml`, then starts Streamlit |
| `deploy/huggingface/space-header.md` | Space settings (title, emoji, `sdk: docker`) put in front of `README.md` on the Space only |
| `deploy/huggingface/sync.py` | Copies the repository (minus workflows and tests) to the Space in one commit |
| `.github/workflows/huggingface-space.yml` | Runs `sync.py` after every push to `main`; skipped until `HF_SPACE` is set |

## One-time setup

1. **Create the Space.** On huggingface.co: *New* → *Space*. Name it (for example `caplife-ai`),
   choose SDK **Docker** → *Blank*, hardware **CPU basic (free)**, visibility **Public**, then
   *Create Space*. Leave it empty; the first mirror run fills it.
2. **Create a token.** Hugging Face → *Settings* → *Access Tokens* → *Create new token*, type
   **Write** (or fine-grained with write access to that one Space). Copy it; it is shown once.
3. **Give GitHub the token and the Space name.** GitHub repository → *Settings* →
   *Secrets and variables* → *Actions*:
   - *Secrets* tab → *New repository secret*: name `HF_TOKEN`, value the token.
   - *Variables* tab → *New repository variable*: name `HF_SPACE`, value `your-username/caplife-ai`.
4. **Give the Space the app secrets (optional, for sign-in and saving).** Space → *Settings* →
   *Variables and secrets* → *New secret*: name `STREAMLIT_SECRETS`, value the same TOML as the
   Streamlit Cloud app's *Settings → Secrets*, with one change in `[auth]`:

   ```toml
   redirect_uri = "https://your-username-caplife-ai.hf.space/oauth2callback"
   ```

   Using the same `[storage]` URL means both copies share the same accounts and saved data.
   Without this secret the Space still works, but without Sign in.
5. **Allow the new address in Google.** Google Cloud console → *APIs & Services* → *Credentials* →
   the OAuth client → *Authorized redirect URIs* → add the `redirect_uri` from step 4 and save.
   Keep the Streamlit Cloud one.
6. **Run the mirror once.** GitHub → *Actions* → *Mirror to Hugging Face Space* → *Run workflow*.
   Later merges into `main` run it automatically. The Space's *Logs* tab shows the build; the first
   build takes several minutes (PyTorch download).

## Notes

- **Share the direct address** `https://your-username-caplife-ai.hf.space`. The page at
  `huggingface.co/spaces/...` wraps the app in Hugging Face's frame with its own header and like
  button; the direct address shows only the app. Sign in also works reliably only on the direct address.
- **Sleeping.** Free Spaces sleep after about 48 hours without visitors and wake on the next visit
  (about a minute).
- **Uploads.** `start.sh` turns off Streamlit's XSRF check, because the cookie it needs is blocked
  inside the huggingface.co frame and file uploads would fail there. Sign in still uses Google OAuth.
- **Never commit secrets.** They live only in GitHub Actions secrets and the Space's secrets.
- **Stop mirroring:** delete the `HF_SPACE` variable; the workflow then skips.
- **Manual sync** from a checkout: `HF_TOKEN=... HF_SPACE=user/caplife-ai python deploy/huggingface/sync.py`
  (`--dry-run` lists what would be uploaded).
