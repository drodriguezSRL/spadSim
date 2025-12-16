import numpy as np
from pyxel.models.photon_collection.point_spread_function import apply_psf_2d
from .base import ImageDegradationModel


def gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)

    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


class PyxelGaussianBlur(ImageDegradationModel):
    def __init__(self, sigma: float = 1.5):
        self.kernel = gaussian_kernel(sigma)

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32)

        if img.ndim == 3:
            return np.stack(
                [apply_psf_2d(img[..., c], self.kernel) for c in range(img.shape[2])],
                axis=-1
            )
        else:
            return apply_psf_2d(img, self.kernel)

class PyxelPhotonNoise(ImageDegradationModel):
    """
    Minimal physical noise model:
    - Shot noise (Poisson)
    - Dark current (Poisson)
    """

    def __init__(
        self,
        photons_per_pixel: float = 1000.0,
        exposure_time: float = 0.033,
        dark_rate: float = 5.0,
        rng: np.random.Generator | None = None,
    ):
        self.photons_per_pixel = photons_per_pixel
        self.exposure_time = exposure_time
        self.dark_rate = dark_rate
        self.rng = rng or np.random.default_rng()

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)

        # Expected photons from signal
        lambda_signal = img * self.photons_per_pixel

        # Expected dark electrons
        lambda_dark = self.dark_rate * self.exposure_time

        # Total Poisson process
        noisy = self.rng.poisson(lambda_signal + lambda_dark)

        return noisy.astype(np.float32)
