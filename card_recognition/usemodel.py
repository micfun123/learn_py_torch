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
    def __init__(self):
        super(CardRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 53)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#use this to load the model
model = CardRecognitionCNN().to(device)
model.load_state_dict(torch.load('model.pth',weights_only=False))
model.eval()

#use opencv take a picture when space is pressed and predict the card
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite('temp.jpg', frame)
        break
cap.release()
cv2.destroyAllWindows()

#open the image and predict the card
image = Image.open('temp.jpg').convert('RGB')
image = image.resize((224, 224))
image = np.array(image)
image = image.astype(np.uint8)
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
output = model(image)
_, predicted = torch.max(output, 1)
print(f"Card ID {predicted.item()}")

