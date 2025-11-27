from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from PIL import Image


class BaseFetcher(ABC):
    @abstractmethod
    async def get_images(
        self, lat: float, lon: float, num_images: int = 20
    ) -> AsyncGenerator[Image.Image]:
        """
        Fetches reference images around given coordinates.
        Yields PIL Images as they are downloaded/fetched.
        """
        pass
