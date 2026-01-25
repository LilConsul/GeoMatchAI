"""Mapillary street-level imagery fetcher."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import aiohttp
import mapillary as mly
from PIL import Image

from .base_fetcher import BaseFetcher
from ..config import config
from ..exceptions import MapillaryFetcherError

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
        api_token: str = config.get_mapillary_api_key(),
        request_timeout: float = config.fetcher.DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = config.fetcher.DEFAULT_MAX_RETRIES,
    ):
        if not api_token:
            raise ValueError("API token required")

        self.api_token = api_token
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._max_retries = max_retries

        self.interface = mly.interface
        self.interface.set_access_token(api_token)

        # Suppress verbose mapillary logging
        logging.getLogger("mapillary.utils.client").setLevel(logging.WARNING)

        logger.info("MapillaryFetcher initialized")

    async def get_images(
        self,
        lat: float,
        lon: float,
        *,
        distance: float = config.fetcher.DEFAULT_SEARCH_RADIUS,
        num_images: int = config.fetcher.DEFAULT_NUM_IMAGES,
    ) -> AsyncGenerator[Image.Image, None]:
        """Yield images near coordinate as they download."""
        logger.info(f"Fetching {num_images} images near ({lat}, {lon}) within {distance}m")

        image_ids = await self._fetch_image_metadata(lat, lon, distance, num_images)

        # Get thumbnail URLs in parallel
        url_id_pairs = await self._get_thumbnail_urls(image_ids)

        # Download images concurrently and yield as they complete
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async for img in self._download_images(session, url_id_pairs):
                yield img

    async def _fetch_image_metadata(
        self, lat: float, lon: float, distance: float, num_images: int
    ) -> list[str]:
        """
        Fetch image metadata from Mapillary API and extract image IDs.

        Args:
            lat: Latitude of the target location
            lon: Longitude of the target location
            distance: Search radius in meters
            num_images: Maximum number of images to fetch

        Returns:
            List of Mapillary image IDs

        Raises:
            MapillaryFetcherError: If no images found or API request fails
        """
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

        features = images_json.get("features", [])
        if not features:
            raise MapillaryFetcherError(f"No images found near ({lat}, {lon}) within {distance}m")

        image_ids = [features[i]["properties"]["id"] for i in range(min(num_images, len(features)))]
        logger.info(f"Found {len(image_ids)} images")
        return image_ids

    async def _get_thumbnail_urls(self, image_ids: list[str]) -> list[tuple[str, str]]:
        """Get thumbnail URLs for image IDs in parallel."""
        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    lambda img_id=img_id: self.interface.image_thumbnail(
                        img_id, resolution=config.fetcher.DEFAULT_THUMBNAIL_RESOLUTION
                    ),
                )
                for img_id in image_ids
            ]
            thumbnail_urls = await asyncio.gather(*tasks, return_exceptions=True)

        url_id_pairs = [
            (url, img_id)
            for url, img_id in zip(thumbnail_urls, image_ids, strict=False)
            if isinstance(url, str) and url
        ]

        if not url_id_pairs:
            raise MapillaryFetcherError("No valid thumbnail URLs obtained")

        return url_id_pairs

    async def _download_images(
        self, session: aiohttp.ClientSession, url_id_pairs: list[tuple[str, str]]
    ) -> AsyncGenerator[Image.Image, None]:
        """Download images concurrently and yield as they complete."""
        tasks = [self._download_image(session, url, img_id) for url, img_id in url_id_pairs]

        successful = 0
        for coro in asyncio.as_completed(tasks):
            img = await coro
            if img is not None:
                successful += 1
                yield img

        logger.info(f"Downloaded {successful}/{len(tasks)} images")

    async def _download_image(
        self, session: aiohttp.ClientSession, url: str, image_id: str
    ) -> Image.Image | None:
        """Download image with retry logic."""
        for attempt in range(self._max_retries):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    return Image.open(BytesIO(image_data))

            except (TimeoutError, aiohttp.ClientError) as e:
                logger.warning(f"Download failed {image_id} (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error {image_id}: {e}")
                break

            if attempt < self._max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))

        return None
