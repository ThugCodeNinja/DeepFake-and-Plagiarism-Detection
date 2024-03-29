import keras
import tensorflow as tf
from tensorflow.keras import layers
from facenet_pytorch import MTCNN
test_dir='/kaggle/working/test'
txt_file = '/kaggle/input/celeb-df-v2/List_of_testing_videos.txt'
base_path='/kaggle/input/celeb-df-v2'
detector= MTCNN(margin=14,keep_all=True,factor=0.7)
test_model = tf.saved_model.load("/kaggle/input/efficientnet/keras/trial/1/p1")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
import warnings
warnings.filterwarnings("ignore")
ans=pd.DataFrame(columns=['Video', 'Label','Pred','real_avg','fake_avg','real_med','fake_med','Confidence'])
# for test dataset
with open(txt_file, 'r') as file:
    lines = file.readlines()
    for l in tqdm(lines):
        label=l[0]
        path=l.split()[1]
        tmp_path = os.path.join(base_path, path)
        # Create video reader and find length
        faces=detect_preprocess(tmp_path)
        total = 0
        real = 0
        fake = 0
        pred_1=0.0
        pred_2=0.0
        video_avg_real=[]
        video_avg_fake=[]
        video_median_fake=[]
        video_median_real=[]
        for face in faces:
            face2 = (face / 255).astype(np.float32)  # Normalize and convert to float32
            face2 = np.expand_dims(face2, axis=0)
            pred = test_model(face2) 
            # Use the model for prediction
            pred2 = pred[0,1].numpy()
            pred1 = pred[0,0].numpy()
            total+=1
            if pred2 > 0.5:
                fake+=1
            else:
                real+=1
            video_avg_fake.append(pred2)
            video_avg_real.append(pred1)
            video_median_fake.append(pred2)
            video_median_real.append(pred1)
        if total==0:
            total=-1
        fake_ratio = fake/total
        text =""
        text2 = "Deepfakes Confidence: " + str(fake_ratio*100) + "%"
        if fake_ratio >= 0.5:
            text = "fake"
        else:
            text = "real"
        new_row = pd.DataFrame({'Video': [path], 'Label': [label], 'Pred': [text],'real_avg':[np.mean(video_avg_real)],'fake_avg':[np.mean(video_avg_fake)],'real_med':[np.median(video_median_real)],'fake_med':[np.median(video_median_fake)],'Confidence': [fake_ratio]})
        ans = pd.concat([ans, new_row], ignore_index=True)
        faces.clear()


