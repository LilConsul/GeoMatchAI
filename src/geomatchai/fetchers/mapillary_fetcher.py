"""
Mapillary street-level imagery fetcher.

This module provides async fetching of street-level images from Mapillary API
around specified GPS coordinates.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import aiohttp
import mapillary as mly
from PIL import Image

from ..config import config
from ..exceptions import MapillaryFetcherError
from .base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)


class MapillaryFetcher(BaseFetcher):
    """
    Fetch street-level imagery around a coordinate using the Mapillary API.

    This fetcher uses the Mapillary SDK to find images near a location and
    downloads them concurrently for better performance.

    Args:
        api_token: Your Mapillary client access token from
                   https://www.mapillary.com/dashboard/developers
        request_timeout: Total timeout for HTTP requests in seconds
                        (defaults to config.fetcher.DEFAULT_REQUEST_TIMEOUT)
        max_retries: Maximum number of retries for failed downloads
                    (defaults to config.fetcher.DEFAULT_MAX_RETRIES)

    Example:
        fetcher = MapillaryFetcher(api_token="your_token")
        async for img in fetcher.get_images(50.054404, 19.935730, num_images=10):
            # Process image
            print(f"Downloaded image: {img.size}")
    """

    def __init__(
        self,
        api_token: str,
        request_timeout: float = config.fetcher.DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = config.fetcher.DEFAULT_MAX_RETRIES,
    ):
        if not api_token:
            raise ValueError("api_token cannot be empty")

        self.api_token = api_token
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._max_retries = max_retries

        # Initialize Mapillary interface
        self.interface = mly.interface
        self.interface.set_access_token(api_token)

        logger.info("MapillaryFetcher initialized successfully")

    async def get_images(
        self,
        lat: float,
        lon: float,
        *,
        distance: float = config.fetcher.DEFAULT_SEARCH_RADIUS,
        num_images: int = config.fetcher.DEFAULT_NUM_IMAGES,
    ) -> AsyncGenerator[Image.Image]:
        """
        Yield Mapillary images near the provided coordinate as they download.

        Downloads images concurrently for better performance and yields them
        as soon as each download completes.

        Args:
            lat: Latitude of the target location
            lon: Longitude of the target location
            distance: Search radius in meters
                     (defaults to config.fetcher.DEFAULT_SEARCH_RADIUS)
            num_images: Maximum number of images to fetch
                       (defaults to config.fetcher.DEFAULT_NUM_IMAGES)

        Yields:
            PIL.Image: Downloaded images as they become available

        Raises:
            MapillaryFetcherError: If no images found or API request fails
        """
        logger.info(f"Fetching {num_images} images near ({lat}, {lon}) within {distance}m radius")

        loop = asyncio.get_running_loop()

        # Run synchronous SDK call in executor to avoid blocking
        try:
            with ThreadPoolExecutor() as executor:
                images_json = await loop.run_in_executor(
                    executor,
                    lambda: self.interface.get_image_looking_at(
                        at={"lat": lat, "lng": lon},
                        radius=distance,
                        image_type="flat",
                    ).to_dict(),
                )
        except Exception as e:
            logger.error(f"Failed to fetch image metadata from Mapillary: {e}")
            raise MapillaryFetcherError(f"Mapillary API request failed: {e}") from e

        # Extract image IDs
        features = images_json.get("features", [])
        if not features:
            logger.warning(f"No images found near ({lat}, {lon})")
            raise MapillaryFetcherError(
                f"No images found near coordinates ({lat}, {lon}) within {distance}m radius"
            )

        image_ids = [features[i]["properties"]["id"] for i in range(min(num_images, len(features)))]
        logger.info(f"Found {len(image_ids)} image(s) to download")

        # Get thumbnail URLs in parallel
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            with ThreadPoolExecutor() as executor:
                thumbnail_url_tasks = [
                    loop.run_in_executor(
                        executor,
                        lambda img_id=img_id: self.interface.image_thumbnail(
                            img_id, resolution=config.fetcher.DEFAULT_THUMBNAIL_RESOLUTION
                        ),
                    )
                    for img_id in image_ids
                ]
                thumbnail_urls = await asyncio.gather(*thumbnail_url_tasks, return_exceptions=True)

            # Download images concurrently and yield as they complete
            download_tasks = [
                self._download_image(session, url, img_id)
                for img_id, url in zip(image_ids, thumbnail_urls, strict=False)
                if isinstance(url, str) and url
            ]

            if not download_tasks:
                logger.error("No valid thumbnail URLs obtained")
                raise MapillaryFetcherError("Failed to get any thumbnail URLs")

            # Use as_completed to yield images as soon as they're ready
            successful_downloads = 0
            for coro in asyncio.as_completed(download_tasks):
                img = await coro
                if img is not None:
                    successful_downloads += 1
                    yield img

            logger.info(
                f"Successfully downloaded {successful_downloads}/{len(download_tasks)} images"
            )

    async def _download_image(
        self, session: aiohttp.ClientSession, url: str, image_id: str
    ) -> Image.Image | None:
        """
        Download image from URL asynchronously with retry logic.

        Args:
            session: aiohttp client session
            url: Image URL to download
            image_id: Mapillary image ID for logging

        Returns:
            PIL.Image if successful, None otherwise
        """
        for attempt in range(self._max_retries):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    img = Image.open(BytesIO(image_data))
                    logger.debug(f"Downloaded image {image_id}: {img.size} (attempt {attempt + 1})")
                    return img

            except TimeoutError:
                logger.warning(
                    f"Timeout downloading image {image_id} "
                    f"(attempt {attempt + 1}/{self._max_retries})"
                )
            except aiohttp.ClientError as e:
                logger.warning(
                    f"HTTP error downloading image {image_id}: {e} "
                    f"(attempt {attempt + 1}/{self._max_retries})"
                )
            except Exception as e:
                logger.error(f"Unexpected error downloading image {image_id}: {e}")
                break  # Don't retry on unexpected errors

            if attempt < self._max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        logger.error(f"Failed to download image {image_id} after {self._max_retries} attempts")
        return None
