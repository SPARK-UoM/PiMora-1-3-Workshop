import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torch import nn
from torch.optim import Optimizer
from torch.optim import Adam
import numpy as np
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class MaskDetector(nn.Module):
    def __init__(self, loss_function):
        super(MaskDetector, self).__init__()

        self.loss_function = loss_function

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1), stride=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.linearLayers = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2)
        )

        for sequential in [self.conv2d_1, self.conv2d_2, self.conv2d_3, self.linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)
        out = self.conv2d_3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)

        return out
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

normed_weights = [0.015228536906614965, 0.984771463093385]
loss_function = nn.CrossEntropyLoss(weight=torch.tensor(normed_weights))

model = MaskDetector(loss_function)
m_state_dict = torch.load('Resources\large_model.pt', map_location=device)
model.load_state_dict(m_state_dict)
model.to(device)

""" Face detection using neural network
"""
from pathlib import Path

import numpy as np
from cv2 import resize
from cv2.dnn import blobFromImage, readNetFromCaffe


class FaceDetectorException(Exception):
    """ generic default exception
    """


class FaceDetector:
    """ Face Detector class
    """
    def __init__(self, prototype: Path=None, model: Path=None,
                 confidenceThreshold: float=0.6):
        self.prototype = prototype
        self.model = model
        self.confidenceThreshold = confidenceThreshold
        if self.prototype is None:
            raise FaceDetectorException("must specify prototype '.prototxt.txt' file "
                                        "path")
        if self.model is None:
            raise FaceDetectorException("must specify model '.caffemodel' file path")
        self.classifier = readNetFromCaffe(str(prototype), str(model))
    
    def detect(self, image):
        """ detect faces in image
        """
        net = self.classifier
        height, width = image.shape[:2]
        blob = blobFromImage(resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidenceThreshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            faces.append(np.array([startX, startY, endX-startX, endY-startY]))
        return faces

def get_inference(img):
    re_img = Resize((100, 100))(img).reshape((1, 3, 100, 100)).float()
    re_img = re_img.to(device)

    pred = model(re_img)
    res = pred.argmax(dim = 1)

    if res == 1:
        print("Masked", end="\r")
    else:
        print("No mask", end="\r")
    
    return res

faceDetector = FaceDetector(
        prototype='Resources/deploy.prototxt.txt',
        model='Resources/res10_300x300_ssd_iter_140000.caffemodel',
    )

labels = ['No mask', 'Mask']
labelColor = [(10, 0, 255), (10, 255, 0)]
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceDetector.detect(frame)

    # drawing rectangles
    for (x, y , w ,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0 , 0), 3)
        faceImg = frame[y:y+h, x:x+w]

        faceImg = torch.tensor(faceImg).permute(2, 0, 1)
        
        try:
            predicted = get_inference(faceImg)

            # center text according to the face frame
            textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            textX = x + w // 2 - textSize[0] // 2

            # draw prediction label
            cv2.putText(frame,
                        labels[predicted],
                        (textX, y-20),
                        font, 1, labelColor[predicted], 2)
        except:
            pass


    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()