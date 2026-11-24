#!/bin/sh
# Start CapLife AI on a Hugging Face Space.
# Streamlit reads secrets from .streamlit/secrets.toml; a Space stores secrets as environment variables.
# Paste the same TOML you use on Streamlit Cloud (with this Space's own redirect_uri) into a Space secret
# named STREAMLIT_SECRETS; it is written to the file here and never committed. Without it, sign-in and
# saving stay off and the app runs as a public demo.
set -e
if [ -n "$STREAMLIT_SECRETS" ]; then
    mkdir -p .streamlit
    printf '%s\n' "$STREAMLIT_SECRETS" > .streamlit/secrets.toml
fi
# The Space page shows the app inside a frame on huggingface.co, where the browser blocks the cookie that
# Streamlit's upload protection needs; it is switched off here so file uploads work in that frame.
exec streamlit run app.py \
    --server.port 7860 --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableXsrfProtection false
