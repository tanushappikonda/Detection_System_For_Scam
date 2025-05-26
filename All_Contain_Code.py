import os
import random
import pandas as pd
from faker import Faker
from PIL import Image, ImageDraw
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models, Input
import librosa
import pickle
import pyttsx3
from sklearn.utils import class_weight

# Initialize
fake = Faker()
engine = pyttsx3.init()
os.makedirs("datasets", exist_ok=True)
os.makedirs("datasets/images", exist_ok=True)
os.makedirs("datasets/audio", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ========================
# 1. DATASET GENERATION
# ========================

def generate_text_data():
    data = []
    
    # Legitimate (label=0)
    for _ in range(5000):
        data.append({
            "text": f"Your order #{fake.random_number(digits=6)} will arrive on {fake.date_this_month()}",
            "label": 0
        })
    
    # Scams (label=1)
    scams = [
        f"URGENT! Your {fake.company()} account is locked. Verify now: {fake.url()}",
        f"You won {random.choice(['$1000','an iPhone'])}! Claim at {fake.url()}",
        f"ALERT: Suspicious login from {fake.country()}. Secure account: {fake.url()}"
    ]
    for _ in range(2000):
        data.append({
            "text": random.choice(scams),
            "label": 1
        })
    
    pd.DataFrame(data).to_csv("datasets/text_data.csv", index=False)

def generate_image_data():
    data = []
    
    # Legitimate (label=0)
    for i in range(700):
        img = Image.new("RGB", (300, 150), (230, 230, 230))
        d = ImageDraw.Draw(img)
        d.text((10, 50), f"Invoice #{fake.random_number(digits=6)}", fill="black")
        img.save(f"datasets/images/legit_{i}.png")
        data.append({"path": f"datasets/images/legit_{i}.png", "label": 0})
    
    # Scams (label=1)
    for i in range(300):
        img = Image.new("RGB", (300, 150), (255, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((10, 50), random.choice(["URGENT: Account Locked!", "WINNER! Claim Prize"]), fill="black")
        img.save(f"datasets/images/scam_{i}.png")
        data.append({"path": f"datasets/images/scam_{i}.png", "label": 1})
    
    pd.DataFrame(data).to_csv("datasets/image_data.csv", index=False)

def generate_audio_data():
    data = []
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    
    # Legitimate (label=0)
    for i in range(350):
        text = f"Your order #{fake.random_number(digits=6)} is confirmed"
        output_path = f"datasets/audio/legit_{i}.mp3"
        engine.save_to_file(text, output_path)
        data.append({"path": output_path, "label": 0})
    
    # Scams (label=1)
    for i in range(150):
        text = random.choice([
            "URGENT! Your account is compromised",
            "You won a prize! Call now"
        ])
        output_path = f"datasets/audio/scam_{i}.mp3"
        engine.save_to_file(text, output_path)
        data.append({"path": output_path, "label": 1})
    
    engine.runAndWait()
    pd.DataFrame(data).to_csv("datasets/audio_data.csv", index=False)

# ========================
# 2. MODEL TRAINING
# ========================

def train_text_model():
    df = pd.read_csv("datasets/text_data.csv")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    model = RandomForestClassifier()
    model.fit(X, df['label'])
    
    with open('models/text_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/text_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def train_image_model():
    df = pd.read_csv("datasets/image_data.csv")
    X = []
    for img_path in df['path']:
        img = Image.open(img_path).resize((64, 64))
        X.append(np.array(img))
    X = np.stack(X) / 255.0
    
    model = models.Sequential([
        Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    model.fit(X, df['label'], epochs=5)
    model.save("models/image_model.keras")

def train_audio_model():
    df = pd.read_csv("datasets/audio_data.csv")
    X = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            y, sr = librosa.load(row['path'])
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            X.append(mfcc.mean(axis=1))
            valid_indices.append(idx)
        except Exception as e:
            print(f"Skipping {row['path']}: {str(e)}")
            continue
    
    X = np.stack(X)
    y = df.iloc[valid_indices]['label'].values
    
    # Class weights for imbalanced data
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(weights))
    
    model = models.Sequential([
        Input(shape=(13,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    model.fit(X, y, epochs=10, class_weight=class_weights)
    model.save("models/audio_model.keras")

if __name__ == "__main__":
    print("🚀 Generating datasets...")
    generate_text_data()
    generate_image_data()
    generate_audio_data()
    
    print("🔧 Training models...")
    train_text_model()
    train_image_model()
    train_audio_model()
    
    print("✅ Training complete! Models saved to /models")