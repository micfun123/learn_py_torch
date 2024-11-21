import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CardRecognitionCNN(nn.Module):
    def __init__(self,num_classes=53):
        super(CardRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Global Average Pooling
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#use this to load the model
model = CardRecognitionCNN().to(device)
model.load_state_dict(torch.load('card_recognition/model.pth',weights_only=False))
model.eval()

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((300, 300))
    image = np.array(image)
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    
print(predict_image('card_recognition/test/ace of clubs/2.jpg'))

