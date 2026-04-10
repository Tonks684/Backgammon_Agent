# Oracle VM training image — CUDA 12.1 + PyTorch 2.3
# Build:  docker build -t backgammon-rl .
# Run:    docker compose up   (see docker-compose.yml)

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# --------------------------------------------------------------------------- #
# System dependencies
# --------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        gnubg \
        git \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Python dependencies
# --------------------------------------------------------------------------- #
WORKDIR /workspace

# Copy dependency manifests first so this layer is cached on code-only changes
COPY requirements.txt setup.py ./
COPY backgammon/__init__.py backgammon/__init__.py

# torch is already present in the base image; pip will skip it
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------- #
# Project source
# --------------------------------------------------------------------------- #
COPY . .

# Install the package in editable mode so imports resolve without PYTHONPATH
RUN pip install --no-cache-dir -e .

# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
# data/ is expected to be mounted from the host (checkpoints, evals, raw data)
VOLUME ["/workspace/data"]

# Default: launch the training entry point.
# Override with `docker compose run backgammon-rl python backgammon/main.py eval ...`
CMD ["python", "backgammon/main.py", "train"]
