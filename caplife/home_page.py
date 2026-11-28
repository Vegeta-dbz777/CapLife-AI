"""Streamlit page: the app's home page, introducing CapLife AI and its three workspaces."""
from __future__ import annotations

from base64 import b64encode
from pathlib import Path
from typing import Callable

import streamlit as st

CARDS = (
    ("monitor", "monitor_heart", "Monitor and Predict",
     "Health of up to six parameters against your own failure limits and operating conditions, a neural "
     "network's remaining useful life with an 80% range, the predicted trend of every parameter, and a "
     "downloadable report.",
     "Needs: a measurement file (CSV or Excel)."),
    ("train", "neurology", "Train The AI",
     "Train a neural network on your own tested capacitors, check it on capacitors it never saw, and forecast "
     "the rest only if it passes.",
     "Needs: a measurement file and an outcome file."),
    ("data", "folder_open", "My Data",
     "The datasets and trained models you saved: open them again, train a model again with newer data, "
     "download or delete them. Only you can see them.",
     "Needs: signing in (top right)."),
)
STEPS = (
    ("Upload your readings", "Any of the six parameters, from a CSV or Excel file with your own column names."),
    ("See health and remaining life", "Every capacitor scored against your limits, with an honest 80% range."),
    ("Train on your failures", "Teach the AI your capacitor type, test it fairly, and use it everywhere."),
)


def render(hero_image: Path, open_workspace: Callable[[str], None], workspaces: dict[str, str]) -> None:
    hero = b64encode(hero_image.read_bytes()).decode("ascii")
    st.html(
        '<section class="caplife-hero">'
        f'<img src="data:image/webp;base64,{hero}" alt="">'
        '<div><div class="caplife-overline">Aluminum electrolytic capacitors</div>'
        '<h1>CapLife AI</h1>'
        '<p>Monitor the Health of Aluminum Electrolytic Capacitors and Predict Their RUL with a Neural Network</p>'
        '<div class="caplife-hero-facts"><span><b>6</b> health parameters</span>'
        '<span><b>80%</b> ranges, tested for honesty</span><span><b>Your own</b> trained AI</span></div>'
        '</div></section>'
    )
    st.html('<div class="caplife-section-title">How it works</div>'
            '<div class="caplife-section-lead">CapLife AI tracks capacitance, ESR, leakage current, case expansion, '
            'generated hydrogen and internal pressure. Use any of them you measure.</div>'
            '<div class="caplife-steps">'
            + "".join(f'<div class="caplife-step"><div class="caplife-step-number">{n}</div>'
                      f'<div><b>{title}</b><span>{text}</span></div></div>'
                      for n, (title, text) in enumerate(STEPS, start=1))
            + '</div><div class="caplife-section-title">Choose where to start</div>')
    columns = st.columns(3)
    for column, (key, icon, title, text, needs) in zip(columns, CARDS):
        with column, st.container(border=True, key=f"caplife_home_card_{key}"):
            st.markdown(f"#### :material/{icon}: {title}")
            st.write(text)
            st.caption(needs)
            st.button(f"Open {title} →", key=f"caplife_home_open_{key}", type="primary", width="stretch",
                      on_click=open_workspace, args=(workspaces[key],))
    st.caption("New here? Open **Monitor and Predict** and upload a measurement file; *Download CSV template* "
               "shows the columns it reads. The neural networks learned from simulated capacitors and were not "
               "reliable on real ones from public aging studies; train them on your own capacitors on Train The AI. "
               "Not a maintenance or safety tool.")
