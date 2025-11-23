from fetchers.base_fetcher import BaseFetcher
from PIL import Image
# import requests


class MapillaryFetcher(BaseFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_images(
        self, lat: float, lon: float, num_images: int = 20
    ) -> list[Image.Image]:
        # Call Mapillary API to get URLs
        urls = self._query_api(lat, lon, num_images)
        # images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
        # return images

    def _query_api(self, lat: float, lon: float, num_images: int):
        return ["https://example.com/image1.jpg", "https://example.com/image2.jpg"][
            :num_images
        ]


if __name__ == "__main__":  # TODO: remove this block
    import os

    fetcher = MapillaryFetcher(os.getenv("MAPILLARY_API_KEY"))
    images = fetcher.get_images(37.7749, -122.4194, num_images=5)
    for img in images:
        img.show()
