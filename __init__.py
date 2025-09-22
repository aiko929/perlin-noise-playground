from .nodes import PerlinNoiseMaskNode, PerlinNoiseImageNode

NODE_CLASS_MAPPINGS = {
    "PerlinNoiseMask": PerlinNoiseMaskNode,
    "PerlinNoiseImage": PerlinNoiseImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerlinNoiseMask": "Perlin Noise (Mask)",
    "PerlinNoiseImage": "Perlin Noise (Image)",
}

__all__ = ["NODE_CLASS_MAPPINGS"]
