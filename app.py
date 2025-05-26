from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import librosa
import io
import pickle
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Scam Detection API")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
try:
    with open('models/text_vectorizer.pkl', 'rb') as f:
        text_vectorizer = pickle.load(f)
    with open('models/text_model.pkl', 'rb') as f:
        text_model = pickle.load(f)
    img_model = load_model('models/image_model.keras')
    audio_model = load_model('models/audio_model.keras')
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Processing Functions
async def process_text(text: str):
    features = text_vectorizer.transform([text])
    proba = text_model.predict_proba(features)[0][1]
    return {"is_scam": bool(proba > 0.5), "confidence": float(proba)}

async def process_image(file: UploadFile):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img = img.resize((64, 64))
        arr = np.expand_dims(np.array(img)/255.0, axis=0)
        proba = img_model.predict(arr)[0][0]
        return {"is_scam": bool(proba > 0.5), "confidence": float(proba)}
    except Exception as e:
        raise HTTPException(400, f"Image error: {str(e)}")

async def process_audio(file: UploadFile):
    try:
        contents = await file.read()
        y, sr = librosa.load(io.BytesIO(contents))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.expand_dims(mfcc.mean(axis=1), axis=0)
        proba = audio_model.predict(features)[0][0]
        return {"is_scam": bool(proba > 0.5), "confidence": float(proba)}
    except Exception as e:
        raise HTTPException(400, f"Audio error: {str(e)}")

# Unified Endpoint
@app.post("/detect")
async def detect_all(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    results = {}
    
    if text:
        results["text"] = await process_text(text)
    if image:
        results["image"] = await process_image(image)
    if audio:
        results["audio"] = await process_audio(audio)
    
    if not results:
        raise HTTPException(400, "No valid inputs provided")
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)