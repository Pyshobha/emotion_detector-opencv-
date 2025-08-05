import cv2
import numpy as np
import joblib
import time

# Load model and label encoder
model_path = "C:\\Users\\Owner\\OneDrive\\Documents\\face_reco\\emotion_model.pkl"
encoder_path = "C:\\Users\\Owner\\OneDrive\\Documents\\face_reco\\label_encoder.pkl"

model = joblib.load(model_path)
le = joblib.load(encoder_path)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# Set larger resolution (optional: 1280x720 or 960x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🔍 Starting real-time emotion detection. Press 'q' to quit.")

# Variables to slow down prediction
last_emotion = ""
last_confidence = 0.0
last_prediction_time = time.time()
prediction_interval = 0.7  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_flattened = face_resized.flatten().reshape(1, -1)

        # Only predict every 1.5 seconds
        if current_time - last_prediction_time > prediction_interval:
            prediction = model.predict(face_flattened)
            prob = model.predict_proba(face_flattened).max()
            last_emotion = le.inverse_transform(prediction)[0]
            last_confidence = prob
            last_prediction_time = current_time

        # Draw on screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{last_emotion} ({last_confidence*100:.1f}%)"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Real-time Emotion Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
