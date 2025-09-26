import os
import shutil
import logging
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header , Depends
import whisperx
from typing import Optional
import torch

API_KEY = os.getenv("API_KEY")


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("whisperx-api")


# Config
HF_TOKEN = os.environ.get("HF_TOKEN")
app = FastAPI(title="WhisperX API")

# Global variables
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16" if device == "cuda" else "int8"


# Model Loading
def load_models():
    """Load WhisperX and alignment models on startup."""
    global model
    try:
        logger.info(f"Loading WhisperX Large v3 model on {device.upper()} ({compute_type})...")
        model = whisperx.load_model(
            "large-v3",
            device,
            compute_type=compute_type,
            asr_options={
                "max_new_tokens": None,
                "multilingual": True,                       # or False as needed
                "clip_timestamps": False,                   # set as needed
                "hallucination_silence_threshold": 0.0,     # set as needed
                "hotwords": None                            # or a list of hotwords
            }
        )

        logger.info("WhisperX model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load WhisperX model")
        raise RuntimeError("Failed to load WhisperX model") from e

@app.on_event("startup")
async def startup_event():
    load_models()

async def verify_api_key(x_api_key: str = Header(...)):
    
    if x_api_key != API_KEY:
        logger.warning(f"Unauthorized Access: {x_api_key}. Expected: {API_KEY}")
        raise HTTPException(status_code=401, detail="Unauthorized Access")

# Transcription Endpoint
@app.post("/transcribe_audio")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = "en",
    diarize: Optional[bool] = False,
    _=Depends(verify_api_key)  # Enforce API key check

):
    """
    Transcribes an audio file with speaker diarization and word timestamps.
    """
    if not model:
        logger.error("Model not loaded at request time")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received transcription request for file={file.filename}, "
                f"language={language}, diarize={diarize}")

    temp_file_path = f"temp_{request_id}_{file.filename}"
    try:
        # Save file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.debug(f"[{request_id}] Saved file temporarily at {temp_file_path}")

        # Load audio
        audio = whisperx.load_audio(temp_file_path)

        # Transcribe
        logger.info(f"[{request_id}] Starting transcription...")
        result = model.transcribe(audio, batch_size=batch_size, language=language)
        logger.info(f"[{request_id}] Transcription complete.")

        # Align
        logger.info(f"[{request_id}] Aligning words...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        logger.info(f"[{request_id}] Alignment complete.")

        if diarize:
            if not HF_TOKEN:
                logger.error(f"[{request_id}] Hugging Face token missing for diarization")
                raise HTTPException(status_code=400, detail="Hugging Face token is required for diarization.")

            logger.info(f"[{request_id}] Performing diarization with optimized settings...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)

            diarize_model.model.embedding_batch_size = 8
            diarize_model.model.segmentation_batch_size = 4

            # Specify expected speaker count range (adjust as needed)
            diarize_segments = diarize_model(
                audio,
                min_speakers=2,
                max_speakers=4
            )

            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info(f"[{request_id}] Diarization complete (optimized).")

        logger.info(f"[{request_id}] Request completed successfully.")
        return {"request_id": request_id, "segments": result["segments"]}

    except Exception as e:
        logger.exception(f"[{request_id}] Error during transcription")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"[{request_id}] Cleaned up temporary file {temp_file_path}")