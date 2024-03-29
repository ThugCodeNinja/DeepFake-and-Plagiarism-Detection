import os
import cv2
import math
from tqdm import tqdm
import json
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
import tensorflow as tf
dataset_path = 'split_dataset'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from facenet_pytorch import MTCNN
base_path = '/kaggle/input/celeb-df-v2'
output_path = '/kaggle/working/temp/real'

os.makedirs(output_path, exist_ok=True) 

for filename in tqdm(os.listdir(os.path.join(base_path, 'Celeb-real'))):
    if filename.endswith(".mp4"):
        tmp_path = os.path.join(base_path, filename)
        count = 0
        video_file = os.path.join(os.path.join(base_path, 'Celeb-real'), filename)
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5)  # frame rate
        while cap.isOpened():
            frame_id = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif 1000 < frame.shape[1] <= 1900:
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                new_filename = os.path.join(output_path, '{}-{:03d}-{}.png'.format(os.path.splitext(filename)[0], count,"real"))
                count += 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
    else:
        continue

base_path = '/kaggle/input/celeb-df-v2'
output_path = '/kaggle/working/temp/fake'  # Specify the output directory path

os.makedirs(output_path, exist_ok=True)  # Create the output directory if it doesn't exist
ct=0
for filename in tqdm(os.listdir(os.path.join(base_path, 'Celeb-synthesis'))):
    if filename.endswith(".mp4"):
        ct+=1
        tmp_path = os.path.join(base_path, filename)
        count = 0
        video_file = os.path.join(os.path.join(base_path, 'Celeb-synthesis'), filename)
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5)  # frame rate
        while cap.isOpened():
            frame_id = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif 1000 < frame.shape[1] <= 1900:
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                new_filename = os.path.join(output_path, '{}-{:03d}-{}.png'.format(os.path.splitext(filename)[0], count,"fake"))
                count += 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
        if ct>590:
            break
    else:
        continue
fake_base_path = '/kaggle/working/temp/fake'
fake_faces_path= '/kaggle/working/train/fake'

for frame in tqdm(os.listdir(base_path)):
    detector = MTCNN()
    image = cv2.cvtColor(cv2.imread(os.path.join(base_path, frame)), cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    count = 0
        
    for result in results:
        bounding_box = result['box']
        confidence = result['confidence']
        if len(results) < 2 or confidence > 0.95:
            margin_x = bounding_box[2] * 0.3  
            margin_y = bounding_box[3] * 0.3 
            x1 = int(bounding_box[0] - margin_x)
            if x1 < 0:
                x1 = 0
            x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
            if x2 > image.shape[1]:
                x2 = image.shape[1]
            y1 = int(bounding_box[1] - margin_y)
            if y1 < 0:
                y1 = 0
            y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
            if y2 > image.shape[0]:
                y2 = image.shape[0]
            crop_image = image[y1:y2, x1:x2]
            new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, frame), count)
            count = count + 1
            cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
        else:
            pass

train_datagen = ImageDataGenerator(
    rescale = 1/255,    #rescale the tensor values to [0,1]
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary", 
    batch_size = batch_size_num,
    shuffle = True
    
)

val_datagen = ImageDataGenerator(
    rescale = 1/255    
)

val_generator = val_datagen.flow_from_directory(
    directory = val_path,
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = "binary",  
    batch_size = batch_size_num,
    shuffle = True
    
)

test_datagen = ImageDataGenerator(
    rescale = 1/255    
)

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = None,
    batch_size = 1,
    shuffle = False
)

# Train a CNN classifier
efficient_net = EfficientNetB2(
    weights = 'imagenet',
    input_shape = (input_size, input_size, 3),
    include_top = False,
    pooling = 'max'
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compile model
model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train network
num_epochs = 20
history = model.fit(
    train_generator,
    epochs = num_epochs,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator,
    validation_steps = len(val_generator),
)
model.save("p1")