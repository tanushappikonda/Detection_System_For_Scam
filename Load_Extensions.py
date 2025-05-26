import pickle
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import librosa

class ModelLoader:
    def __init__(self, model_dir="models"):
        """Load all trained models from specified directory"""
        self.models = {
            'text': self._load_text_models(model_dir),
            'image': self._load_keras_model(f'{model_dir}/image_model.keras'),
            'audio': self._load_keras_model(f'{model_dir}/audio_model.keras')
        }
    
    def _load_text_models(self, model_dir):
        """Load text vectorizer and classifier"""
        with open(f'{model_dir}/text_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open(f'{model_dir}/text_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return {'vectorizer': vectorizer, 'model': model}
    
    def _load_keras_model(self, path):
        """Load a Keras model with safety checks"""
        try:
            return load_model(path, compile=False)
        except:
            print(f"Warning: Couldn't load {path}, falling back to compiled version")
            return load_model(path)  # Try with compilation

# Usage Example
if __name__ == "__main__":
    # Initialize loader (auto-loads all models)
    loader = ModelLoader()
    
    # Access individual models
    text_vectorizer = loader.models['text']['vectorizer']
    text_model = loader.models['text']['model']
    image_model = loader.models['image']
    audio_model = loader.models['audio']
    
    print("✅ Models loaded successfully!")
    print(f"Text model type: {type(text_model)}")
    print(f"Image model type: {type(image_model)}")
    print(f"Audio model type: {type(audio_model)}")
    
    # Example prediction (adapt to your needs)
    sample_text = "You won a prize!"
    text_features = text_vectorizer.transform([sample_text])
    prediction = text_model.predict_proba(text_features)[0][1]
    print(f"\nSample text prediction: {prediction:.2f} (1=scam)")