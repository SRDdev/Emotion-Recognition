import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import time

def load_emotion_model():
    """
    Load the pre-trained emotion detection model.

    Returns:
    Sequential: Loaded emotion detection model.
    """
    emotion_model = Sequential()

    # Convolutional layers for feature extraction
    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    # Flatten and dense layers for classification
    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(7, activation='softmax'))

    # Load pre-trained weights for the model
    emotion_model.load_weights('emotion_model.h5')
    return emotion_model

def init_streamlit():
    """
    Initialize Streamlit components and configuration.

    Returns:
    tuple: Tuple containing Streamlit components - (emotion_text_placeholder, frame_window, exit_button).
    """
    # Set up Streamlit page configuration
    st.set_page_config(page_title="Emotion Recognition App", page_icon="😊", layout="centered")
    st.title("Emotion Recognition App 🎭")

    # Set up the sidebar with project information
    st.sidebar.title("ML Project 👩‍💻")
    st.sidebar.text("Real-time emotion recognition from webcam")
    st.sidebar.divider()
    st.sidebar.subheader('Members')
    st.sidebar.text('Shreyas Dixit')
    st.sidebar.text('Vedika Bhat')
    st.sidebar.text('Om Doshi')
    st.sidebar.text('Shonak Vaywhare')

    # Set up placeholders for Streamlit components
    emotion_text_placeholder = st.empty()
    frame_window = st.image([])
    exit_button = st.sidebar.button("Exit")

    return emotion_text_placeholder, frame_window, exit_button

def release_resources(cap):
    """
    Release resources - close camera and destroy OpenCV windows.

    Args:
    cap: cv2.VideoCapture object.
    """
    cap.release()
    cv2.destroyAllWindows()

def run():
    """
    Run the main application loop for real-time emotion recognition.
    """
    # Open the webcam (camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open the camera.")
        st.stop()

    # Load the pre-trained emotion detection model
    emotion_model = load_emotion_model()

    # Dictionary to map emotion indices to corresponding emojis
    emotion_dict = {0: "😠 Angry", 1: "😤 Disgusted", 2: "😨 Fearful", 3: "😄 Happy", 4: "😐 Neutral", 5: "😢 Sad", 6: "😲 Surprised"}

    # Initialize Streamlit components
    emotion_text_placeholder, frame_window, exit_button = init_streamlit()

    # Main loop for capturing and processing webcam frames
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Resize the frame for better display
        frame = cv2.resize(frame, (600, 500))

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)

        # Detect faces in the frame using Haarcascades classifier
        num_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            # Update the emotion text placeholder with the predicted emotion
            emotion_text_placeholder.text(emotion_dict[maxindex])

        # Display the processed frame in the Streamlit window
        frame_window.image(frame, channels="BGR", caption="Webcam Feed")

        # Check if the exit button is pressed
        if exit_button:
            # Release resources and stop the application
            release_resources(cap)
            st.balloons()
            st.success("Application successfully exited.")
            st.stop()

# Entry point of the script
if __name__ == "__main__":
    run()
