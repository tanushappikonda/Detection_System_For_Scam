import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import librosa
import io
import pickle
from tensorflow.keras.models import load_model
from typing import Optional
from pathlib import Path
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Scam Detection API",
    description="API for detecting scams in text, images, and audio",
    version="1.0.0"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_DIR = Path(__file__).parent / "models"
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png"]
ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/mpeg"]

# Load Models (with error handling)
def load_models():
    """Load all ML models at startup"""
    models = {}
    try:
        logger.info("Loading text models...")
        with open(MODEL_DIR / 'text_vectorizer.pkl', 'rb') as f:
            models['text_vectorizer'] = pickle.load(f)
        with open(MODEL_DIR / 'text_model.pkl', 'rb') as f:
            models['text_model'] = pickle.load(f)
        
        logger.info("Loading image model...")
        models['img_model'] = load_model(MODEL_DIR / 'image_model.keras')
        
        logger.info("Loading audio model...")
        models['audio_model'] = load_model(MODEL_DIR / 'audio_model.keras')
        
        logger.info("All models loaded successfully")
        return models
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

models = load_models()

# Processing Functions
async def process_text(text: str):
    """Process text input for scam detection"""
    try:
        features = models['text_vectorizer'].transform([text])
        proba = models['text_model'].predict_proba(features)[0][1]
        return {"is_scam": bool(proba > 0.5), "confidence": float(proba)}
    except Exception as e:
        logger.error(f"Text processing error: {str(e)}")
        raise HTTPException(400, detail=f"Text processing error: {str(e)}")

async def process_image(file: UploadFile):
    """Process image upload for scam detection"""
    try:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(400, detail="Invalid image format. Only JPEG/PNG allowed.")
        
        img = Image.open(io.BytesIO(await file.read()))
        img = img.resize((64, 64))  # Adjust based on your model's expected input
        arr = np.expand_dims(np.array(img)/255.0, axis=0)
        proba = models['img_model'].predict(arr)[0][0]
        return {"is_scam": bool(proba > 0.5), "confidence": float(proba)}
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(400, detail=f"Image processing error: {str(e)}")

async def process_audio(file: UploadFile):
    """Process audio upload for scam detection"""
    try:
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(400, detail="Invalid audio format. Only WAV/MP3 allowed.")
        
        contents = await file.read()
        y, sr = librosa.load(io.BytesIO(contents))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.expand_dims(mfcc.mean(axis=1), axis=0)
        proba = models['audio_model'].predict(features)[0][0]
        return {"is_scam": bool(proba > 0.5), "confidence": float(proba)}
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(400, detail=f"Audio processing error: {str(e)}")

# API Endpoints
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Scam Detection API",
        "version": app.version
    }

@app.post("/detect")
async def detect_all(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """
    Unified detection endpoint that handles:
    - Text scam detection
    - Image scam detection
    - Audio scam detection
    """
    results = {}
    
    if text:
        results["text"] = await process_text(text)
    if image:
        results["image"] = await process_image(image)
    if audio:
        results["audio"] = await process_audio(audio)
    
    if not results:
        raise HTTPException(400, detail="No valid inputs provided")
    
    return {
        "status": "success",
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),  # Render uses port 10000
        log_level="info"
    )