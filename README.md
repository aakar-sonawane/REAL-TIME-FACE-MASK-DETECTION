# Real-Time Face Mask Detection

A computer vision system that detects whether a person is wearing a face mask or not, using real-time webcam input.

## Overview
This project trains a deep learning model on a 2-class dataset (mask / no mask) and applies it for real-time face mask detection through webcam feed, combining face detection with mask classification.

## Tech Stack
- Python
- TensorFlow/Keras (model training)
- OpenCV (real-time face detection & inference)
- Haar Cascade Classifier (face detection)

## Dataset
- 2-class dataset: **Mask** and **No Mask**
- Preprocessed and used to train a CNN-based classification model

## Project Structure
```
├── DATASET/                                  # Mask / No-mask training images
├── FEATURES_TARGET/                           # Extracted features and target labels
├── facemask_model.h5                          # Trained model
├── haarcascade_frontalface_default.xml        # Face detection classifier
├── REAL_TIME_FACE_MASK_DETECTION.ipynb        # Training & real-time detection notebook
└── README.md
```

## How It Works
1. Face is detected in each webcam frame using a Haar Cascade classifier.
2. The detected face region is preprocessed (resized, normalized) to match model input requirements.
3. The trained CNN model classifies the face as **"Mask"** or **"No Mask"**.
4. The result is displayed in real-time on the video feed with a bounding box and label.

## Approach
- Preprocessed the 2-class dataset for training.
- Trained a CNN model to distinguish between masked and unmasked faces.
- Used Haar Cascade classifiers for real-time face detection.
- Tested the complete pipeline using live webcam input to validate real-time performance.

## Results
*(Add your model's training/validation accuracy here)*

## Future Improvements
- Extend to detect improper mask usage (e.g., mask below nose)
- Improve robustness under varying lighting conditions
- Deploy as a lightweight edge application
