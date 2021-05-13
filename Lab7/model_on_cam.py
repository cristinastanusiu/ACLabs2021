import numpy as np
# ML
from sklearn.metrics import accuracy_score

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary

# Plots
import matplotlib.pyplot as plt

# Utils
from collections import OrderedDict
from tqdm import tqdm, trange
import os

MODEL_PATH = os.path.join('models', 'my_models', 'my_resnet18_decent.pth')

if __name__ == '__main__':
    # Load model
    resnet18 = torchvision.models.resnet18(
        pretrained=False)  # Takes multiple of 32 as input
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 128)),
        ('dropout', nn.Dropout(p=.5)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(128, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    resnet18.fc = fc
    resnet18.cuda()

    resnet18.load_state_dict(torch.load(MODEL_PATH))
    resnet18.eval()

    summary(resnet18, input_size=(3, 128, 128), batch_size=32, device="cuda")

    # Camera
    cap = cv2.VideoCapture(1)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        sample = frame
        # Our operations on the frame come here
        # Display the resulting frame
        # cv2.imshow('frame', frame)
        # Preprocessing
        cv2.imshow('frame', sample)

        sample = cv2.resize(sample, dsize=(128, 128),
                            interpolation=cv2.INTER_AREA)
        # sample = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sample = sample / sample.max()
        # print(sample.shape)
        sample = sample.transpose(2, 0, 1)
        sample = torch.Tensor(sample)
        sample = sample.unsqueeze(0)
        # print(sample.shape)

        prediction = resnet18(sample.cuda())
        print(torch.argmax(prediction, axis=1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
