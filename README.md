# 😄 Real-Time Emotion Detection Using OpenCV and SVM

This project performs **real-time emotion recognition** using a webcam. It detects human faces and classifies their emotional expressions (happy, sad, shocked, etc.) using an **SVM model trained on custom face images**.

## 📸 Demo

![demo gif or screenshot here](demo.gif)  
*Add a screenshot or screen recording to show your app in action.*

---

## 🚀 Features

- Real-time face detection using OpenCV Haar Cascades
- Emotion classification using a trained SVM model
- Trained on custom dataset organized into folders (happy, sad, angry, etc.)
- Displays emotion label and confidence on live camera feed
- Adjustable prediction interval for faster or slower updates

---

## 🗂️ Project Structure

```
face_reco/
│
├── All/
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   └── shocked/
│       └── (images for training)
│
├── emotion_model.pkl
├── label_encoder.pkl
├── train_emotion_model.py
├── detect_emotion_realtime.py
├── README.md
```

---

## 🧠 How It Works

1. You organize face images by emotion into folders (`All/happy/`, `All/sad/`, etc.)
2. You train a Support Vector Machine (SVM) classifier using grayscale resized (48x48) face images.
3. In real-time, faces are detected using Haar Cascades.
4. Each face is preprocessed and passed into the trained model to classify emotion.

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/face-emotion-detector.git
cd face-emotion-detector
```

2. **Install dependencies**
```bash
pip install opencv-python numpy scikit-learn joblib
```

3. **(Optional)** Clean PNG files from ICC warnings:
```bash
pip install pillow
```

---

## 🧪 Train the Emotion Model

Ensure your dataset is stored in:
```
face_reco/All/emotion_name/image.png
```

Run the training script:

```bash
python train_emotion_model.py
```

It will save:
- `emotion_model.pkl`
- `label_encoder.pkl`

---

## 📷 Run Real-Time Emotion Detection

After training:
```bash
python detect_emotion_realtime.py
```

> Press **`q`** to quit the window.

---

## 🔧 Adjustable Settings

In `detect_emotion_realtime.py`, you can tune:

```python
prediction_interval = 0.3  # Time between emotion updates
emotion_display_duration = 1.0  # How long emotion is shown
```

---

## 📌 Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- joblib

---

## 📚 Credits

- OpenCV (for face detection)
- scikit-learn (for SVM model)
- Project by **Shobha Jangade**

---

## 📃 License

This project is open-source and free to use under the MIT License.