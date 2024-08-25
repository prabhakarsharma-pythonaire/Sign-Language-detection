# Sign Language Recognition with GUI

This project is a Sign Language Recognition system built using TensorFlow and Keras. It recognizes American Sign Language (ASL) letters from images or real-time video input. Additionally, it can recognize specific words like "HELLO," "HI," and "LOVE" based on the sequence of detected letters. The system is operational only between 9 AM and 11 AM daily.
![Working](https://github.com/prabhakarsharma-pythonaire/Sign-Language-detection/blob/main/Screenshot%202024-08-25%20094056.png)
![Working2](https://github.com/prabhakarsharma-pythonaire/Sign-Language-detection/blob/main/Screenshot%202024-08-25%20093944.png)

## Features

- **Image Upload**: Upload an image of a hand sign, and the model predicts the corresponding ASL letter.
- **Real-time Video Recognition**: Use your webcam to recognize ASL letters in real-time.
- **Known Word Detection**: Detect specific words like "HELLO," "HI," and "LOVE" based on sequences of detected letters.
- **Operational Time Window**: The model is active only between 9 AM and 11 AM.

## Requirements

- Python 3.8
- TensorFlow 2.10
- Keras
- OpenCV
- Tkinter (for the GUI)
- Pillow (for image handling in Tkinter)

You can install the required libraries using pip:

```bash
pip install -r requirements.txt


## clone reqo
git clone https://github.com/prabhakarsharma-pythonaire/Sign-Language-detection.git
cd <repository-directory>

