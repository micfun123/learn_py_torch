import csv
import numpy as np
from PIL import Image

def image_to_csv_row(image_path, label=0):
    """
    Converts an image to a single CSV row with a label and 784 pixel values.
    
    Parameters:
        image_path (str): The file path to the image.
        label (int): The label for the image, default is 0.
        
    Returns:
        list: A list where the first element is the label, followed by 784 grayscale pixel values.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels

    # Convert image to a 1D array of pixel values
    pixel_values = np.array(image).flatten()  # Flatten the 2D image into 1D array

    # Create a row with the label and pixel values
    csv_row = [label] + pixel_values.tolist()
    
    return csv_row

def save_images_to_csv(image_paths, csv_filename, labels=None):
    """
    Saves multiple images to a CSV file in the specified format.
    
    Parameters:
        image_paths (list of str): List of image file paths.
        csv_filename (str): Output CSV file path.
        labels (list of int): List of labels for the images; if None, labels will be 0.
    """
    # Open CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        header = ['label'] + [f'pixel{i}' for i in range(784)]
        writer.writerow(header)
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            label = labels[i] if labels else 0  # Use given label or default to 0
            csv_row = image_to_csv_row(image_path, label=label)
            writer.writerow(csv_row)
    
    print(f"CSV saved to {csv_filename}")

# Example usage:
# Convert and save images to CSV
image_paths = ['images.png', '3593441.png','4images.png']  # Replace with actual image paths
labels = [2, 2,4]  # Replace with actual labels if available
csv_filename = 'images.csv'
save_images_to_csv(image_paths, csv_filename, labels=labels)
