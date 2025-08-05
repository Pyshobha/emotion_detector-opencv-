import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib

# Path to your dataset
data_path = r"C:\Users\Owner\OneDrive\Documents\face_reco\All"

X, y = [], []

# Loop through each emotion folder
for emotion in os.listdir(data_path):
    emotion_path = os.path.join(data_path, emotion)
    if not os.path.isdir(emotion_path):
        continue

    for img_name in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # skip corrupted images
        img = cv2.resize(img, (48, 48))  # Resize all to 48x48
        X.append(img.flatten())         # Flatten image to 1D
        y.append(emotion)               # Save label

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train classifier (SVM)
model = SVC(kernel='linear', probability=True)
model.fit(X, y_encoded)

# Save model and label encoder
joblib.dump(model, "emotion_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model trained and saved successfully.")
