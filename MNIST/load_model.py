import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader , Dataset
import torch.nn as nn
from models import SimpleCNN


# Initialize and load the model with the saved state dictionary
model = SimpleCNN()
model.load_state_dict(torch.load('MNIST\model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),  # Convert to grayscale if model is trained on grayscale images
    transforms.Resize((28, 28)),  # Resize to model's expected input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization, adjust as per model's training
])

# Start the webcam feed
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press SPACE to capture an image, or ESC to exit.")
while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the resulting frame
    cv2.imshow("Webcam Feed", frame)
    
    # Capture key press
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        # Preprocess the captured frame
        input_image = transform(frame).unsqueeze(0).to(device)  # Add batch dimension
        
        # Run the model prediction
        with torch.no_grad():
            output = model(input_image.float())
            _, predicted = torch.max(output.data, 1)
            prediction = predicted.item()
        
        # Display prediction result
        print(f"Predicted: {prediction}")
        
        # Display the image with prediction
        plt.imshow(frame[..., ::-1])  # Convert BGR (OpenCV) to RGB for matplotlib
        plt.title(f"Predicted: {prediction}")
        plt.axis('off')
        plt.show()
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()

