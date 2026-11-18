# Launching and sharing CapLife AI (beginner steps)

## Option A: Open the app on your Windows computer

1. Install Python 3.11 or 3.12. If asked, check **Add Python to PATH**.
2. Unzip `CapLife_AI_Starter.zip` to a folder.
3. Open that folder in File Explorer, click its address bar, type `powershell` and press Enter.
4. Enter `py -m pip install -r requirements.txt`.
5. Enter `py -m streamlit run app.py`.
6. A browser tab opens, usually at `http://localhost:8501`.
7. Click **Analyze capacitor**; the bundled simulated dataset will work with no upload.
8. To stop the local server, return to PowerShell and press **Ctrl+C**.

## Option B: Share an online website

1. Create an empty GitHub repository.
2. Add the **contents** of `CapLife_AI_Starter` to the repository root. In particular:
   `app.py`, `requirements.txt`, `caplife/`, `artifacts/`, `sample_data/`.
3. Check that `artifacts/gru_demo_state.pt` and `artifacts/metadata.json` were committed.
4. Go to https://share.streamlit.io and choose to deploy a new app.
5. Select your repository and branch. Enter `app.py` as the main file.
6. Follow deployment logs if any dependency issue appears.
7. Share your assigned `*.streamlit.app` URL with collaborators.

**Important:** This is a *public-facing demonstration*, not a real-world validated predictor. Users should not upload proprietary or private experimental datasets to an unreviewed public deployment. The app processes uploads on the host server.
