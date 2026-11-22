# Launching and sharing CapLife AI (beginner steps)

## Option A: Open the app on your Windows computer

1. Install Python 3.11 or 3.12. If asked, check **Add Python to PATH**.
2. Unzip `CapLife_AI_Starter.zip` to a folder.
3. Open that folder in File Explorer, click its address bar, type `powershell` and press Enter.
4. Enter `py -m pip install -r requirements.txt`.
5. Enter `py -m streamlit run app.py`.
6. A browser tab opens, usually at `http://localhost:8501`.
7. Click **Load public example** to explore the bundled NASA ES14 measurements, or choose
   **RUL prediction** in the sidebar and **Use simulated example capacitors**.
8. To stop the local server, return to PowerShell and press **Ctrl+C**.

## Option B: Share an online website

1. Create an empty GitHub repository.
2. Add the **contents** of `CapLife_AI_Starter` to the repository root. In particular:
   `app.py`, `requirements.txt`, `.streamlit/`, `assets/`, `caplife/`, `artifacts/`, `sample_data/`.
3. Check that `artifacts/gru_demo_state.pt`, `artifacts/metadata.json` and the
   `artifacts/six_rul/` folder (the six-parameter RUL model) were committed.
4. Go to https://share.streamlit.io and choose to deploy a new app.
5. Select your repository and branch. Enter `app.py` as the main file.
6. Follow deployment logs if any dependency issue appears.
7. Share your assigned `*.streamlit.app` URL with collaborators.

## Update the live app

Streamlit Community Cloud redeploys automatically when the deployed branch (`main`) changes.
Merge a tested branch into `main`, then open the app and check **Load public example** and
**RUL prediction → Use simulated example capacitors**. If the app stays on its startup screen
after dependencies are processed, reboot it once from **Manage app**.

The app is tested with Python 3.11 and Streamlit 1.67. It uses `width="stretch"` on widgets,
which needs a recent Streamlit; `requirements.txt` allows any Streamlit 1.x, and Cloud installs
the newest one. Changing `requirements.txt` makes Cloud reinstall every dependency.

## Check the app before merging

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/test_app_pages.py` opens every workspace headlessly and runs its main actions.

**Important:** This is a *public-facing demonstration*, not a real-world validated predictor. Users should not upload proprietary or private experimental datasets to an unreviewed public deployment. The app processes uploads on the host server.
