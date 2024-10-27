import numpy as np
import matplotlib.pyplot as plt

# Create a sample 2x2x3 tensor

def generate_tensor(shape):
    return np.random.rand(*shape)

tensor = generate_tensor((4, 2, 3)) 

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each element of the tensor as a point in 3D space
for i in range(tensor.shape[0]):
    for j in range(tensor.shape[1]):
        for k in range(tensor.shape[2]):
            ax.scatter(i, j, k, c=tensor[i, j, k], cmap='viridis')

plt.show()