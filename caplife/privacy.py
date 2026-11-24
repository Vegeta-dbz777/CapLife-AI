"""Privacy policy page, served at ?page=privacy so it shares the app's domain."""
from __future__ import annotations

import streamlit as st

LAST_UPDATED = "22 November 2026"
REPOSITORY_URL = "https://github.com/Vegeta-dbz777/CapLife-AI"


def render(app_url: str) -> None:
    st.markdown(f"[← Back to CapLife AI]({app_url})")
    st.title("Privacy policy")
    st.caption(f"CapLife AI · last updated {LAST_UPDATED}")
    st.markdown(f"""
CapLife AI is a research prototype for monitoring aluminum electrolytic capacitor health and
predicting remaining useful life. This page explains what the app does with your information.

### Using the app without signing in

You can use every analysis page without an account. Files you upload are processed in the
server's memory to produce results and are discarded afterwards. They are not saved, and the app
does not use them for any other purpose.

### Signing in with Google

Signing in is optional and only needed to save datasets. When you sign in, Google shares your
**name**, **email address** and a **unique account ID** with the app. CapLife AI uses them only to
show who is signed in and to keep your saved datasets separate from everyone else's. It does not
request access to your Gmail, Drive, contacts or any other Google data.

### Saved datasets

When you choose **Save this dataset to your account**, the app stores the CSV files you saved, the
name you gave them, the time you saved them and your account ID in its database. Only you can list,
open, download or delete them. You can delete one dataset, or everything, at any time on the
**My Data** page; deletion takes effect immediately.

### Sharing

CapLife AI does not sell your information, show advertising or share your data with anyone except
the services that run the app: Streamlit Community Cloud (hosting), Google (sign-in) and the
database provider that stores saved datasets. Those services process data on the app's behalf.

### Research use

Results are produced by research models, including a neural network trained on simulated
capacitors. They are not validated for maintenance or safety decisions.

### Contact

Questions or deletion requests can be raised through the project's
[GitHub repository]({REPOSITORY_URL}). Changes to this policy will be posted on this page.
""")
