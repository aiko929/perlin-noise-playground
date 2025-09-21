import numpy as np
import torch
from perlin_noise import PerlinNoise

class PerlinNoiseMaskNode:
    CATEGORY = "noise"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "perlinNoise"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 480}),
                "height": ("INT", {"default": 480}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 99, "step": 1}),
                "zoom": ("INT", {"default": 300, "min": 1}),
                "seed": ("INT", {"default": 42})
            },
            "optional": {  # optional sockets can be left unconnected
                "maybe_mask": ("MASK", {}),
            },
        }

    def perlinNoise(self, width, height, octaves, zoom, seed):
        noise = PerlinNoise(octaves=octaves, seed=seed)

        # Fill an array
        img = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                # coordinates normalized by zoom
                val = noise([y / zoom, x / zoom])
                img[y][x] = (val + 1) / 2  # map from [-1,1] to [0,1]
        return img

import matplotlib.pyplot as plt
import matplotlib.cm as cm

class PerlinNoiseImageNode:
    CATEGORY = "noise"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        # get available matplotlib colormaps
        cmap_names = sorted(plt.colormaps())

        return {
            "required": {
                "width":   ("INT",   {"default": 512, "min": 1}),
                "height":  ("INT",   {"default": 512, "min": 1}),
                "octaves": ("INT",   {"default": 4,   "min": 1, "max": 99, "step": 1}),
                "zoom":    ("FLOAT", {"default": 300.0, "min": 1.0, "step": 1.0}),
                "seed":    ("INT",   {"default": 42}),
                "colormap": (cmap_names, {"default": "viridis"}),  # dropdown of cmaps
            }
        }

    def run(self, width, height, octaves, zoom, seed, colormap):
        noise = PerlinNoise(octaves=octaves, seed=seed)

        # build Perlin values [H, W] in [0,1]
        img = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                v = noise([y / zoom, x / zoom])    # ~[-1,1]
                img[y, x] = (v + 1.0) * 0.5        # -> [0,1]

        # apply chosen colormap -> returns [H, W, 4] RGBA
        cmap = cm.get_cmap(colormap)
        rgba = cmap(img)[:, :, :3]  # take RGB only, drop alpha
        rgb = torch.from_numpy(rgba.astype(np.float32))

        # add batch dimension [1, H, W, 3]
        rgb = rgb.unsqueeze(0)

        return (rgb,)



