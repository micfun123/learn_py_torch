import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D tensor with shape (3, 2, 3)
tensor = np.zeros((3, 2, 3))

# Function to visualize the shape of a 3D tensor
def visualize_tensor_shape(tensor):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Get the dimensions of the tensor
    x_dim, y_dim, z_dim = tensor.shape

    # Create a grid to represent the tensor's shape
    x, y, z = np.meshgrid(
        np.arange(x_dim),
        np.arange(y_dim),
        np.arange(z_dim),
        indexing='ij'
    )

    # Flatten the grid for scatter plotting
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Plot the points
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_title(f'Tensor Shape: {tensor.shape}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()

visualize_tensor_shape(tensor)
