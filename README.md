
# Facial Emotion Detection using AI & ML

This project focuses on building a real-time **Facial Emotion Detection System** using advanced AI and ML techniques. The primary objective is to accurately identify emotions from live video feeds, leveraging the **DeepFace** library for deep learning-based emotion analysis and **OpenCV** for video processing.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Results and Discussion](#results-and-discussion)

## Introduction
The Facial Emotion Detection System is a dynamic AI application designed to recognize and display emotions in real time. By integrating **OpenCV** for video handling, **DeepFace** for emotion detection, and **Streamlit** for a user-friendly interface, this system identifies emotions such as happiness, sadness, anger, surprise, fear, and disgust, making it versatile for various applications like mental health monitoring, customer service, and virtual learning environments.

## Problem Statement
Building an emotion detection system that:
1. Enhances face recognition accuracy across various environments.
2. Adapts to diverse demographics and real-world conditions.
3. Ensures user privacy and data security.
4. Operates reliably in real-time.

## Objectives
1. **Real-Time Operation**: Immediate processing of live video feeds for timely emotion recognition.
2. **User-Friendly Interface**: Simple and intuitive UI to facilitate use by non-technical individuals.
3. **High Accuracy Emotion Detection**: Precise identification of a wide range of emotions.
4. **Robust Performance**: Consistent results under different lighting and demographic variations.

## Features
- **Real-Time Video Processing**: Capture and process live video to detect and display emotions in real-time.
- **Multi-Emotion Detection**: Recognize multiple emotions including happiness, sadness, anger, surprise, fear, and disgust.
- **High Accuracy and Reliability**: Uses DeepFace and OpenCV to ensure precise emotion detection.
- **Support for Diverse Conditions**: Handles varying lighting conditions and diverse demographic profiles.
- **Multi-Face Detection**: Detects and labels emotions for multiple faces within the same frame.

## Technologies Used
- **Python**: Programming language.
- **DeepFace**: Deep learning library for facial analysis.
- **OpenCV**: Computer vision library for video capture and processing.
- **Streamlit**: Framework for building interactive UIs.
- **Haar Cascade Classifier**: For initial face detection.

## Setup and Installation

### Prerequisites
- **Python 3.6+**
- Libraries: OpenCV, DeepFace, Streamlit

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/facial-emotion-detection.git
   cd facial-emotion-detection
   ```
2. **Install Required Libraries**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
4. **Access the App**:
   - The app will open in your browser at `http://localhost:8501`.

## Usage
1. **Start the Video Stream**: Once the app launches, start the video feed.
2. **Real-Time Emotion Detection**: The system will automatically detect faces and annotate each detected face with the corresponding emotion.
3. **View Detected Emotions**: Each face will display its emotion label and a confidence score in real-time.

## Screenshots
- **Live Emotion Detection**: Real-time display with emotion labels and confidence scores.
- **Multiple Faces**: Handles multiple faces within the same video frame.

## Results and Discussion
- **Real-Time Accuracy**: The system accurately processes and labels emotions, performing consistently across varied lighting and demographic conditions.
- **Multi-Face Capability**: The model detects and annotates emotions for multiple faces simultaneously, enhancing its utility for group settings.
- **Interactive Visualization**: Users can see detected emotions in real-time, making the system suitable for live analysis and feedback.
