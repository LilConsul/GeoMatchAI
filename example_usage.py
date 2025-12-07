"""
GeoMatchAI usage examples.

Example 1: Basic usage with MapillaryFetcher
Example 2: Custom fetcher implementation
"""

import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator

from PIL import Image

from geomatchai import GeoMatchAI
from geomatchai.fetchers import MapillaryFetcher, BaseFetcher


class LocalFolderFetcher(BaseFetcher):
    """
    Custom fetcher that reads images from a local folder.

    This demonstrates how to implement BaseFetcher for your own image sources.
    """

    def __init__(self, gallery_folder: Path):
        self.gallery_folder = Path(gallery_folder)

    async def get_images(
        self,
        lat: float,
        lon: float,
        num_images: int = 20
    ) -> AsyncGenerator[Image.Image, None]:
        """
        Yield images from local folder.

        In a real implementation, you might filter by lat/lon proximity.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        count = 0

        for image_path in self.gallery_folder.iterdir():
            if count >= num_images:
                break

            if image_path.suffix.lower() in image_extensions:
                try:
                    img = Image.open(image_path).convert("RGB")
                    yield img
                    count += 1
                except Exception as e:
                    print(f"Failed to load {image_path}: {e}")
                    continue


async def example_mapillary():
    """Example 1: Basic usage with Mapillary."""

    print("Example 1: Using MapillaryFetcher")
    print("-" * 50)

    # Create fetcher
    fetcher = MapillaryFetcher(api_token="YOUR_MAPILLARY_API_KEY")

    # Create verifier
    verifier = await GeoMatchAI.create(
        fetcher=fetcher,
        num_gallery_images=200,
        threshold=0.65,
        device="auto"
    )

    print(f"Verifier created: {verifier}")

    # Verify image
    test_image = Path("tests/input/wawel/wawel1.jpg")
    if test_image.exists():
        with open(test_image, "rb") as f:
            image_bytes = f.read()

        # Wawel Castle coordinates
        lat, lon = 50.054404, 19.935730

        is_verified, score = await verifier.verify(lat, lon, image_bytes)

        print(f"Location: ({lat}, {lon})")
        print(f"Verified: {is_verified}")
        print(f"Score: {score:.3f}")
    else:
        print(f"Test image not found: {test_image}")

    print()


async def example_custom_fetcher():
    """Example 2: Using custom fetcher."""

    print("Example 2: Using Custom LocalFolderFetcher")
    print("-" * 50)

    # Create custom fetcher
    gallery_folder = Path("tests/input/wawel")

    if not gallery_folder.exists():
        print(f"Gallery folder not found: {gallery_folder}")
        return

    fetcher = LocalFolderFetcher(gallery_folder=gallery_folder)

    # Create verifier
    verifier = await GeoMatchAI.create(
        fetcher=fetcher,
        num_gallery_images=50,
        threshold=0.65,
        skip_gallery_preprocessing=True
    )

    print(f"Verifier created: {verifier}")

    # Verify image
    test_image = Path("tests/input/wawel/wawel1.jpg")
    if test_image.exists():
        with open(test_image, "rb") as f:
            image_bytes = f.read()

        # Coordinates (not used by LocalFolderFetcher, but required by API)
        lat, lon = 50.054404, 19.935730

        is_verified, score = await verifier.verify(lat, lon, image_bytes)

        print(f"Location: ({lat}, {lon})")
        print(f"Verified: {is_verified}")
        print(f"Score: {score:.3f}")
        print(f"Cached locations: {verifier.cached_locations}")
    else:
        print(f"Test image not found: {test_image}")

    print()


async def main():
    """Run all examples."""

    # Run custom fetcher example (works without API key)
    await example_custom_fetcher()

    # Uncomment to run Mapillary example (requires API key)
    # await example_mapillary()



if __name__ == "__main__":
    asyncio.run(main())
