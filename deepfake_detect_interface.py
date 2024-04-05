import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from facenet_pytorch import MTCNN
from PIL import Image
import moviepy.editor as mp
import os
import zipfile

# Load face detector
mtcnn = MTCNN(margin=14, keep_all=True, factor=0.7, device='cpu')

# Face Detection function
class DetectionPipeline:
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize

    def __call__(self, filename):
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        faces = []
        frames = []
        dummy_data = np.zeros((224, 224, 3), dtype=np.uint8)
        face2 = dummy_data
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.resize is not None:
                    frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))

                frames.append(frame)

                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    boxes, _ = self.detector.detect(frames)
                    for i in range(len(frames)):
                        if boxes[i] is None:
                            faces.append(face2)
                            continue

                        box = boxes[i][0].astype(int)
                        frame = frames[i]
                        face = frame[box[1]:box[3], box[0]:box[2]]

                        if not face.any():
                            faces.append(face2)
                            continue

                        face2 = cv2.resize(face, (224, 224))
                        faces.append(face2)

                    frames = []

        v_cap.release()

        return faces

detection_pipeline = DetectionPipeline(detector=mtcnn, n_frames=20, batch_size=60)

model = tf.saved_model.load("p1")

def deepfakespredict(input_video):
    faces = detection_pipeline(input_video)

    total = 0
    real = 0
    fake = 0

    for face in faces:
        face2 = (face / 255).astype(np.float32)
        pred = model(np.expand_dims(face2, axis=0))[0]
        total += 1

        pred2 = pred[1]

        if pred2 > 0.5:
            fake += 1
        else:
            real += 1

    fake_ratio = fake / total

    text = ""
    text2 = "Deepfakes Confidence: " + str(fake_ratio * 100) + "%"

    if fake_ratio >= 0.5:
        text = "The video is FAKE."
    else:
        text = "The video is REAL."

    face_frames = []

    for face in faces:
        face_frame = Image.fromarray(face.astype('uint8'), 'RGB')
        face_frames.append(face_frame)

    face_frames[0].save('results.gif', save_all=True, append_images=face_frames[1:], duration=250, loop=100)
    clip = mp.VideoFileClip("results.gif")
    clip.write_videofile("video.mp4")

    return text, text2, "video.mp4"

title = "Group 2- EfficientNetV2 based Deepfake Video Detector"
description = '''Please upload videos responsibly and await the results in a gif. The approach in place includes breaking down the video into several frames followed by collecting
the frames that contain a face. Once these frames are collected the trained model attempts to predict if the face is fake or real and contribute to a deepfake confidence. This confidence level eventually 
determines if the video can be considered a fake or not.'''

gr.Interface(deepfakespredict,
             inputs=["video"],
             outputs=["text", "text", gr.Video(label="Detected face sequence")],
             title=title,
             description=description
             ).launch()