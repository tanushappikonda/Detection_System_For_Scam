import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import librosa

class ScamDetector:
    def __init__(self, model_dir="models"):
        """Load all trained models"""
        # Text models
        with open(f'{model_dir}/text_vectorizer.pkl', 'rb') as f:
            self.text_vectorizer = pickle.load(f)
        with open(f'{model_dir}/text_model.pkl', 'rb') as f:
            self.text_model = pickle.load(f)
        
        # Image model
        self.image_model = load_model(f'{model_dir}/image_model.keras')
        
        # Audio model
        self.audio_model = load_model(f'{model_dir}/audio_model.keras')

    def detect_text(self, text):
        """Analyze text for scams"""
        features = self.text_vectorizer.transform([text])
        proba = self.text_model.predict_proba(features)[0][1]
        return {
            'is_scam': bool(proba > 0.5),
            'confidence': float(proba),
            'type': 'text'
        }

    def detect_image(self, image_path):
        """Analyze image for scam indicators"""
        img = Image.open(image_path).resize((64, 64))
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)
        proba = self.image_model.predict(img_array)[0][0]
        return {
            'is_scam': bool(proba > 0.5),
            'confidence': float(proba),
            'type': 'image'
        }

    def detect_audio(self, audio_path):
        """Analyze audio for scam patterns"""
        y, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.expand_dims(mfcc.mean(axis=1), axis=0)
        proba = self.audio_model.predict(features)[0][0]
        return {
            'is_scam': bool(proba > 0.5),
            'confidence': float(proba),
            'type': 'audio'
        }

# Example Usage
if __name__ == "__main__":
    detector = ScamDetector()
    
    # Test text
    text_result = detector.detect_text("You won $1000! Click now!")
    print("Text Result:", text_result)
    
    # Test image
    image_result = detector.detect_image("./datasets/images/scam_26.png")
    print("Image Result:", image_result)
    
    # Test audio
    audio_result = detector.detect_audio("./datasets/audio/scam_40.mp3")
    print("Audio Result:", audio_result)