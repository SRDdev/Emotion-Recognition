# Emotion Recognition App

## Description

The Emotion Recognition App is a real-time application that utilizes machine learning to detect and recognize human emotions from a webcam feed. The application uses a pre-trained deep learning model to classify facial expressions into seven different emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised. This project demonstrates the integration of computer vision and machine learning for emotion analysis, which can have various applications, including user experience enhancement and human-computer interaction.

## Why is it Important?

Understanding human emotions can be valuable in several domains, such as human-computer interaction, sentiment analysis, and mental health monitoring. This Emotion Recognition App provides a simple and interactive way to showcase the capabilities of emotion detection models. It can be used as a starting point for building more advanced applications or for educational purposes in the fields of computer vision and machine learning.

## How to Use

To use the Emotion Recognition App:

1. Ensure you have Python installed on your system.
2. Install the required libraries by running `pip install -r requirements.txt`.
3. Run the application using `python app.py`.

The application will open in your web browser, displaying a webcam feed with real-time emotion recognition. Emotions detected from the faces in the webcam feed will be displayed alongside the video.

## Technologies Used

- OpenCV: Computer vision library for image and video processing.
- TensorFlow/Keras: Deep learning framework for building and training the emotion detection model.
- Streamlit: Web application framework for creating interactive web applications with Python.

---

### Documentation

## Project Structure

The project structure is organized as follows:

- `src/app.py`: Main script to run the Emotion Recognition App.
- `src/emotion_model.h5`: Pre-trained deep learning model weights for emotion detection.
- `src/requirements.txt`: List of required Python libraries.

## Functionality

### 1. Emotion Detection Model

The emotion detection model is a Convolutional Neural Network (CNN) trained to recognize facial expressions. The model is loaded using the `load_emotion_model` function defined in `app.py`.

### 2. Streamlit Application

The Streamlit application is initialized using the `init_streamlit` function. It configures the layout, displays the webcam feed, and provides an interactive sidebar for project information and an exit button.

### 3. Real-time Emotion Recognition

The main loop (`run` function) captures frames from the webcam, processes them, and uses the emotion detection model to predict the emotion of each detected face. The detected emotion is then displayed in real-time.

### 4. Exit and Resource Release

The application can be exited by clicking the "Exit" button in the sidebar. This triggers the `release_resources` function, which releases the webcam resources and closes OpenCV windows.

## Usage

1. Run the application using `python app.py`.
2. Observe the webcam feed and the real-time emotion recognition display.
3. Click the "Exit" button to close the application.

---

This README provides a high-level overview of the Emotion Recognition App, its importance, usage, and technologies used. For more detailed information, refer to the documentation section below.
