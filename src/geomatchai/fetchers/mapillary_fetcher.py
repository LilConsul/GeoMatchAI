from PIL import Image
from io import BytesIO
import asyncio
import aiohttp
from typing import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from geomatchai.fetchers.base_fetcher import BaseFetcher

import mapillary as mly


class MapillaryFetcher(BaseFetcher):
    """
    Fetcher for Mapillary street-level imagery.

    Args:
        api_token: Your Mapillary client access token from
                     https://www.mapillary.com/dashboard/developers
    """

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.interface = mly.interface
        self.interface.set_access_token(api_token)

    async def get_images(
        self,
        lat: float,
        lon: float,
        *,
        distance: float = 50.0,
        num_images: int = 20,
    ) -> AsyncGenerator[Image.Image, None]:
        """
        Async generator that yields images as they're downloaded.
        Downloads images concurrently for better performance.

        Usage:
            async for img in fetcher.get_images(lat, lon):
                # Process image immediately as it arrives
                img.show()
        """
        loop = asyncio.get_event_loop()

        # Run synchronous SDK call in executor
        with ThreadPoolExecutor() as executor:
            images_json = await loop.run_in_executor(
                executor,
                lambda: self.interface.get_image_looking_at(
                    at={"lat": lat, "lng": lon},
                    radius=distance,
                    image_type="flat",
                ).to_dict()
            )

        image_ids = [
            images_json["features"][i]["properties"]["id"]
            for i in range(min(num_images, len(images_json["features"])))
        ]

        # Get thumbnail URLs and download images concurrently
        async with aiohttp.ClientSession() as session:
            # First, get all thumbnail URLs using SDK in parallel
            with ThreadPoolExecutor() as executor:
                thumbnail_url_tasks = [
                    loop.run_in_executor(
                        executor,
                        lambda img_id=img_id: self.interface.image_thumbnail(
                            img_id, resolution=1024
                        )
                    )
                    for img_id in image_ids
                ]
                thumbnail_urls = await asyncio.gather(*thumbnail_url_tasks)

            # Download images concurrently and yield as they complete
            tasks = [
                self._download_image(session, url)
                for url in thumbnail_urls if url
            ]

            # Use as_completed to yield images as soon as they're ready
            for coro in asyncio.as_completed(tasks):
                img = await coro
                if img is not None:
                    yield img

    async def _download_image(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Image.Image | None:
        """Download image from URL asynchronously."""
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                image_data = await response.read()
                img = Image.open(BytesIO(image_data))
                return img

        except Exception as e:
            print(f"Failed to download image from {url}: {e}")
            return None


if __name__ == "__main__":  # TODO: remove this block
    import os
    import time

    if not (mapillary_client_token := os.getenv("MAPILLARY_API_KEY")):
        raise ValueError(
            "MAPILLARY_API_KEY environment variable not set.\n"
            "Get your client token from: https://www.mapillary.com/dashboard/developers"
        )

    fetcher = MapillaryFetcher(mapillary_client_token)

    # Test Wawel Castle
    lat, lon = 50.054404, 19.935730

    async def test_async():
        start = time.time()
        count = 0
        async for img in fetcher.get_images(lat, lon, num_images=5):
            count += 1
            print(
                f"Image {count}: {img.size} (received after {time.time() - start:.2f}s)"
            )
            # Optionally show image as it arrives
            # img.show(title=f"Image {count}")
        print(f"Total images fetched: {count}")
        print(f"Total time: {time.time() - start:.2f}s")

    asyncio.run(test_async())

