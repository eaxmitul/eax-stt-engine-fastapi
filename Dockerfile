# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive

# # Install system dependencies and curl for model download
# RUN apt-get update && apt-get install -y \
#     git \
#     ffmpeg \
#     libsndfile1 \
#     python3-pip \
#     python3-dev \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy requirements and install python deps
# COPY requirements.txt .

# # Always use the latest pip and install requirements
# RUN pip3 install --upgrade pip setuptools wheel
# RUN pip3 install -r requirements.txt

# # Download VAD model with correct filename (whisperx expects this exact name)
# RUN mkdir -p /root/.cache/whisperx/models \
#     && curl -L https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin \
#        -o /root/.cache/whisperx/models/whisperx-vad-segmentation.bin

# COPY . .

# ENV PYTHONUNBUFFERED=1

# EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

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

COPY requirements.txt .

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

# Pre-download diarization VAD model with correct name
RUN mkdir -p /root/.cache/whisperx/models \
    && curl -L https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin \
       -o /root/.cache/whisperx/models/vad-segments.bin

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]