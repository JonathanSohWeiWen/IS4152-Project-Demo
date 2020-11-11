import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from torchvision import datasets, models, transforms
import numpy as np
from numpy.ma import exp,sin,cos
import imutils
import torch
import torch.nn as nn
from best_model import *

class VideoCamera(object):

    def __init__(self):
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self):
        model = torch.load(os.getcwd()+'\\demo_githubpages\\best_model.pt' ,map_location='cpu')
        emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                        4: 'neutral', 5: 'sadness', 6: 'surprise'}
        val_transform = transforms.Compose([transforms.ToTensor()])
        frame = self.video.read()[1]
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_cascade = cv2.CascadeClassifier(os.getcwd()+'\\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            resize_frame = cv2.resize(gray[y:y + h, x:x + w], (224, 224))
            #X = resize_frame/256
            X = resize_frame  ##新增
            ###
            X = np.uint8(X)
            ###
            X = Image.fromarray((X))
            X = val_transform(X).unsqueeze(0)

            with torch.no_grad():
                model.eval()
                log_ps = model.cpu()(X)   #use cpu
                ##
                #print(type(log_ps))
                #print(log_ps)
                #log_ps=np.array(log_ps)
                #print(log_ps[1])
                #print(len(log_ps[1]))
                ##

                ps = torch.exp(log_ps[0])
                #ps =np.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                pred = emotion_dict[int(top_class.numpy())]
                #print(top_class)
                #print(top_p)
            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()