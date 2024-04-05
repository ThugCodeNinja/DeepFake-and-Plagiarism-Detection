import gradio as gr
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# Constants
MAX_TIME_STEPS = 109
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MODEL_PATH = "audio_classifier.h5"  # Replace with the actual path to your saved model

# Load the pre-trained model
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def visualize(mel_spectrogram):
    median_decibels = np.median(mel_spectrogram)
    median_human_voice_range = -65
    diff_decibels = abs(median_decibels - median_human_voice_range)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    librosa.display.specshow(mel_spectrogram, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Difference from Median Human Voice Range: {diff_decibels:.2f} dB')
    plt.savefig("mel_spectrogram.png")  # Save the image
    plt.close()

def classify_audio(audio):
    # Convert the audio data to NumPy array
    rate, ar = audio
    arone = ar.astype(np.float32)
    mel_spectrogram = librosa.feature.melspectrogram(y=arone, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    # Reshape for the model
    X_test = np.expand_dims(mel_spectrogram, axis=-1)
    X_test = np.expand_dims(X_test, axis=0)

    # Predict using the loaded model
    y_pred = model.predict(X_test)

    # Convert probabilities to predicted classes
    y_pred_classes = np.argmax(y_pred, axis=1)

    if y_pred_classes[0] == 1:
        prediction = "Not Spoof : High chances of original voice"
    else:
        prediction = "Spoof : Possible voice cloning"

    visualize(mel_spectrogram)
    return prediction,"mel_spectrogram.png"
    
title=" Group-2 Audio Spoof detection using CNN"
description="The model was trained on the ASVspoof 2019 dataset with an aim to detect spoof audios through deep learning.To use it please upload an audio file of suitable length. The Mel spectrogram used for inferencing is also available for the user to understand the classification and compare it with the median Human decibal range."
 
iface = gr.Interface(classify_audio, inputs=["audio"], outputs=["text","image"],title=title,description=description)
iface.launch()
