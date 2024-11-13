import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class CustomMNISTDataset(Dataset):
    def __init__(self,csv_file,transform=None,is_test=False):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test = is_test

    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, index):
        item = self.data_frame.iloc[index]

        if self.is_test:
            image = item.values.reshape(28,28).astype(np.uint8)
            label = None
        else:
            image = item[1:].values.reshape(28,28).astype(np.uint8)
            label = item.iloc[0]

        image = transforms.ToPILImage()(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            return image,label
        

transform = transforms.Compose(
    [transforms.RandomRotation(15),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5),)]
)

train_dataset = CustomMNISTDataset(csv_file='train.csv',transform=transform,is_test=False)
test_dataset = CustomMNISTDataset(csv_file='test.csv',transform=transform,is_test=True)

print(f"Train Size: {len(train_dataset)} Test Size: {len(test_dataset)}")

batch_size = 300
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # num_workers=2 slows down the training

from models import SimpleCNN

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)   # Stochastic Gradient Descent


num_epochs = 150
running_loss = 0.0

for epoch in range(num_epochs):
    for i , data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.4f}, Progress: {100 * (epoch + 1) / num_epochs:.2f}%")
            running_loss = 0.0

print("Finished Training")


#save the model 
torch.save(model.state_dict(), 'model.pth')

print("Model Saved")