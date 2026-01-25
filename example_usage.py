"""
GeoMatchAI minimal usage example.
"""

import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator
from PIL import Image

from geomatchai import GeoMatchAI, config
from geomatchai.fetchers import BaseFetcher, MapillaryFetcher


class LocalFolderFetcher(BaseFetcher):
    """Custom fetcher that reads images from local folder."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    async def get_images(
        self, lat: float, lon: float, num_images: int = 20
    ) -> AsyncGenerator[Image.Image, None]:
        for img_path in self.folder.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield Image.open(img_path).convert("RGB")


async def example_with_mapillary():
    """Example using MapillaryFetcher with config."""

    print("Example 1: MapillaryFetcher")

    # Configure library settings
    # It automatically reads from MAPILLARY_API_KEY env var
    # config.set_mapillary_api_key("YOUR_MAPILLARY_API_KEY")
    config.set_device("cuda")
    config.set_log_level("INFO")

    # Create fetcher
    fetcher = MapillaryFetcher(
        api_token=config.get_mapillary_api_key()
    )  # Default value will be retrieved from config

    # Create verifier (uses config defaults)
    verifier = await GeoMatchAI.create(fetcher=fetcher)

    # Verify image
    with open("examples/input/wawel/wawel1.jpg", "rb") as f:
        is_verified, score = await verifier.verify(50.054404, 19.935730, f.read())

    print(f"Verified: {is_verified}, Score: {score:.3f}")


async def example_with_custom_fetcher():
    """Example using custom LocalFolderFetcher with config."""
    print("Example 2: Custom LocalFolderFetcher")

    # Configure library settings
    config.set_log_level("INFO")
    config.set_device("auto")

    # Create custom fetcher
    fetcher = LocalFolderFetcher(folder=Path("examples/input/wawel"))

    # Create verifier (uses config defaults)
    verifier = await GeoMatchAI.create(fetcher=fetcher)

    # Verify image
    with open("examples/input/wawel/wawel1.jpg", "rb") as f:
        is_verified, score = await verifier.verify(50.054404, 19.935730, f.read())

    print(f"Verified: {is_verified}, Score: {score:.3f}")


async def main():
    # Run custom fetcher example (works without API key)
    if (config.get_mapillary_api_key() is None) or (config.get_mapillary_api_key() == ""):
        await example_with_custom_fetcher()

    # Run Mapillary example (requires API key)
    else:
        await example_with_mapillary()


if __name__ == "__main__":
    asyncio.run(main())
