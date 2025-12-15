import numpy as np
try:
    from pyxel.models.photon_collection.point_spread_function import apply_psf_2d
except ImportError:
    # Fallback for older versions (like 2.4.1)
    from pyxel.models.photon_collection.point_spread_function import apply_psf as apply_psf_2d

from .base import ImageDegradationModel

def gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    """
    Generates a 2D Gaussian kernel.
    """
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    
    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(x, y)
    
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    return kernel

class PyxelGaussianBlur(ImageDegradationModel):
    """
    Pyxel-based Gaussian blur degradation using apply_psf_2d.
    """

    def __init__(self, sigma: float = 1.5):
        self.sigma = sigma
        self.kernel = gaussian_kernel(sigma)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        image: numpy array (H,W) or (H,W,C)
        returns: numpy array degraded
        """
        # apply_psf_2d expects float array
        img_float = image.astype(float)
        
        if image.ndim == 3:
            # Apply to each channel
            channels = []
            for i in range(image.shape[2]):
                channels.append(apply_psf_2d(img_float[..., i], self.kernel))
            return np.stack(channels, axis=-1)
        else:
            return apply_psf_2d(img_float, self.kernel)