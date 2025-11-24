from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class BaseFetcher(ABC):
    @abstractmethod
    def get_images(self, lat: float, lon: float, num_images: int = 20) -> List[Image.Image]:
        """
        Fetches reference images around given coordinates.
        Returns a list of PIL Images.
        """
        pass
