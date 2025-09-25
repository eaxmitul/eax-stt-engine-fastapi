FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and curl for model download
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python deps
# COPY requirements.txt .

# # Always use the latest pip and install requirements
# RUN pip3 install --upgrade pip setuptools wheel
# RUN pip3 install -r requirements.txt

# # Download VAD model with correct filename (whisperx expects this exact name)
# RUN mkdir -p /root/.cache/whisperx/models \
#     && curl -L https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin \
#        -o /root/.cache/whisperx/models/whisperx-vad-segmentation.bin

# Copy requirements
COPY requirements.txt .

# 1. Upgrade PyTorch and Torchaudio to 2.3.1+cu121 (skip torchvision)
RUN pip3 install --upgrade \
      torch==2.3.1+cu121 \
      torchaudio==2.3.1+cu121 \
      --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade pip setuptools wheel \
    && pip3 install -r requirements.txt

# 2. Download VAD model
RUN mkdir -p /root/.cache/whisperx/models \
    && curl -L https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin \
       -o /root/.cache/whisperx/models/whisperx-vad-segmentation.bin

# 3. Permanently upgrade the WhisperX Lightning checkpoint
RUN python3 -m pytorch_lightning.utilities.upgrade_checkpoint \
      /usr/local/lib/python3.10/dist-packages/whisperx/assets/pytorch_model.bin

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
