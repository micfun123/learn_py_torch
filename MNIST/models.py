import torch
import torch.nn as nn



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cov1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) #convolutional layer
        self.relu = nn.ReLU() #activation function
        self.cov2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)       
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1) #pooling layer 
        self.cov3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.cov4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.cov5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        

        # Update the input size for fc1
        self.fc1 = nn.Linear(128 * 26 * 26, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x): 
        x = self.cov1(x)
        x = self.relu(x)
        x = self.cov2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.cov3(x)
        x = self.relu(x)
        x = self.cov4(x)
        x = self.relu(x)
        x = self.cov5(x)
        x = self.relu(x)
        x = self.pool(x)
        

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x