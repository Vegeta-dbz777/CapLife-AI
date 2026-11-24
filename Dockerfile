# CapLife AI on a Hugging Face Space (Docker SDK). The app itself is unchanged; the main deployment
# stays on Streamlit Community Cloud. See docs/huggingface-space.md.
FROM python:3.11-slim

# Hugging Face runs Spaces as user 1000 with a writable home directory.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app

# CPU-only PyTorch is far smaller than the default build, then the app's own requirements.
COPY --chown=user requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["sh", "deploy/huggingface/start.sh"]
