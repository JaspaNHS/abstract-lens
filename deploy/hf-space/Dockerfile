# Hugging Face Space (Docker SDK) for Abstract Lens.
# HF Spaces route traffic to port 7860.
FROM python:3.11-slim

# Non-root user (HF Spaces requirement; gives a writable HOME for the ONNX model cache)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# App code + prebuilt index (chromadb/, meta_index.json) — added by prepare_space
COPY --chown=user . .

EXPOSE 7860

# One worker (the vector store + embedding model load once into memory); threads give
# light concurrency. gunicorn imports app:app, which loads ChromaDB at startup.
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--threads", "4", "-t", "180", "app:app"]
