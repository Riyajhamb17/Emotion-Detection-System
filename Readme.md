😄Emotion Detection System

An AI-powered Emotion Detection System built using Convolutional Neural Networks (CNNs) and Transfer Learning. This project classifies facial expressions into 7 distinct emotions using the FER-2013 dataset and compares the performance of four models: a Custom CNN, CNN with Augmentation, VGG16, and ResNet50.

🎯 Objective
 To develop a facial expression recognition system that can classify images into the following 7 emotional states:

😠 Angry

🤢 Disgust

😨 Fear

😄 Happy

😐 Neutral

😢 Sad

😲 Surprise


🗂 Dataset – FER-2013 Source: FER-2013

Format: 48x48 grayscale facial images in CSV format
Size: ~35,887 labeled images
Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

Preprocessing steps:

Conversion from pixel strings to images
Normalization (pixel values scaled to [0, 1])
Reshaping images to (48, 48, 1) for custom CNN and (224, 224, 3) for pretrained models

Models Implemented

🧱 Custom CNN (Built from Scratch) 
Architecture built using Conv2D, MaxPooling, Dropout, Dense layers
Used as baseline model
Trained for classification from scratch

🔄 CNN with Data Augmentation 
Custom CNN architecture with real-time augmentation using:
Horizontal flips
Zoom range
Rotation
Improves generalization and reduces overfitting

🧠 VGG16 (Transfer Learning) Pretrained VGG16 loaded without top classifier
Feature extractor with frozen base
Custom classification head with dense layers
Input resized to (224x224x3)

🔗 ResNet50 (Transfer Learning) 
Pretrained ResNet50 as feature extractor
Getting Started: Step 1: Clone the Repository image Step 2: Install Dependencies image Step 3: Download the Dataset You can download the FER-2013 dataset from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013 Step 4: Run the Notebook Open the emotion_detection.ipynb file and run the cells to train and evaluate models.

Requirements: 
tensorflow 
keras 
numpy 
pandas
matplotlib 
seaborn 
opencv-python 
scikit-learn

Key Learnings
Built and evaluated multiple CNN-based models for emotion recognition
Hands-on with preprocessing FER-2013 CSV dataset
Practical use of Transfer Learning and Data Augmentation
Performance comparison and tuning of various architectures

Future Work
Real-time webcam-based emotion detection using OpenCV
Streamlit/Flask Web App for interactive UI
Save and load models using .h5 or .tflite format
Deploy as REST API or mobile app


Author
Riya Jhamb 
Jhambriya23@gmail.com
