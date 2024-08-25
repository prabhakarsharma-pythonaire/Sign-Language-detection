import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import datetime

# Load the trained model
model = load_model('best_model.h5')

# Class labels
data_class = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R',
              'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Known words to recognize based on sequences
known_words = {
    "HELLO": ['H', 'E', 'L', 'L', 'O'],
    "HI": ['H', 'I'],
    "LOVE": ['L', 'O', 'V', 'E']
}

# Image properties
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Function to preprocess the image (can handle both file paths and numpy arrays)
def preprocess_image(img):
    if isinstance(img, str):
        img = cv2.imread(img)  # Read the image from the file path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Resize to match model's expected input size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function
def predict_class(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction)  # Get the index of the highest predicted probability
    predicted_class = data_class[predicted_class_index]
    return predicted_class

# Function to check for known words
def check_known_words(predicted_sequence):
    for word, sequence in known_words.items():
        if sequence == predicted_sequence:
            return word
    return None

# Function to handle image upload and prediction
def upload_image():
    if not is_operational_time():
        prediction_label.config(text="Model operational between 9 AM and 11 AM.")
        return

    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (250, 250))  # Resize for display
        img_display = ImageTk.PhotoImage(image=Image.fromarray(img_resized))

        panel = Label(root, image=img_display)
        panel.image = img_display  # Keep a reference to avoid garbage collection
        panel.grid(row=1, column=0, padx=10, pady=10)

        predicted_class = predict_class(file_path)
        prediction_label.config(text=f'Predicted Sign: {predicted_class}')

# Real-time video capture and prediction
def video_stream():
    if not is_operational_time():
        prediction_label.config(text="Model operational between 9 AM and 11 AM.")
        return

    _, frame = cap.read()
    if frame is not None:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (250, 250))  # Resize for display

        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        video_panel.imgtk = imgtk
        video_panel.config(image=imgtk)

        # Model prediction for real-time frame
        predicted_class = predict_class(frame)  # Directly pass the numpy array (frame)
        prediction_label.config(text=f'Predicted Sign: {predicted_class}')

        # If needed, track predicted sequence for known word recognition
        predicted_sequence.append(predicted_class)
        if len(predicted_sequence) > max(len(seq) for seq in known_words.values()):
            predicted_sequence.pop(0)

        known_word = check_known_words(predicted_sequence)
        if known_word:
            prediction_label.config(text=f'Recognized Word: {known_word}')

    root.after(10, video_stream)

# Check if the app is operational based on the current time (9 AM to 11 AM)
def is_operational_time():
    now = datetime.datetime.now()
    current_time = now.time()
    start_time = datetime.time(9, 0)  # 9 AM
    end_time = datetime.time(11, 0)   # 11 AM

    if start_time <= current_time <= end_time:
        return True
    return False

# GUI Setup
root = tk.Tk()
root.title("Sign Language Detection")

# Upload Button
upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=10, pady=10)

# Real-Time Video Button
video_button = Button(root, text="Start Video", command=video_stream)
video_button.grid(row=0, column=1, padx=10, pady=10)

# Prediction Label
prediction_label = Label(root, text="Predicted Sign: None")
prediction_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# Video Panel for real-time video display
video_panel = Label(root)
video_panel.grid(row=1, column=1, padx=10, pady=10)

# Initialize variables for tracking the predicted sequence of signs
predicted_sequence = []

# Check if the app is operational based on the time window
if is_operational_time():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
else:
    prediction_label.config(text="Model operational between 9 AM and 11 AM.")

root.mainloop()

# Release the webcam resource
cap.release()
cv2.destroyAllWindows()
