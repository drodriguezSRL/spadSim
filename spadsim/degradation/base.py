from abc import ABC, abstractmethod
import numpy as np

class ImageDegradationModel(ABC):
    """
    common interface for image degradation models
    """

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the degradation to an image numpy array.
        """
        pass
