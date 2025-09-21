import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

# Parameters
height, width = 256, 256
octaves = 8
scale = 180   # bigger = zoom out
seed = 42

# Create noise generator
noise = PerlinNoise(octaves=octaves, seed=seed)

# Fill an array
img = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        # coordinates normalized by scale
        val = noise([y/scale, x/scale])
        img[y][x] = (val + 1) / 2  # map from [-1,1] to [0,1]

# Show
for cm in ["terrain", "grey", "viridis", "inferno"]:
    plt.imshow(img, cmap=cm)
    plt.colorbar()
    plt.title(f"Perlin Noise (octaves={octaves}, seed={seed})")
    plt.axis("off")
    plt.show()
