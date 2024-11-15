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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CardsDataSet(Dataset):
    
    def __init__(self, csv_file, transform=None, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        print(self.data.columns)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx]['filepaths']
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image)
        image = image.astype(np.uint8)
        
        
        
        # Load label if in training mode
        if self.is_train:
            label = torch.tensor(int(self.data.iloc[idx]['class index']), dtype=torch.long)
            sample = {'image': image, 'label': label}
        else:
            sample = {'image': image}


        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(),
])

train_dataset = CardsDataSet('cards.csv', transform=transform,is_train=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print(f"Train Size: {len(train_dataset)}")
print(f"{train_dataset.data['class index'].unique()}")
print("I AM HERE")




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
    


model = CardRecognitionCNN().to(device)
criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class. 
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




num_epochs = 100
print(f"Training Started on {device}")
for epoch in range(num_epochs):
    running_loss = 0.0 
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['image'], data['label']
        inputs, labels = inputs.to(device), labels.to(device)

        
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, "
                  f"Loss: {running_loss / 100:.4f}, "
                  f"Progress: {100 * (epoch + 1) / num_epochs:.2f}%")
            running_loss = 0.0

print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model Saved")
