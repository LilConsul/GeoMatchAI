"""
Simple unified verifier for easy GeoMatchAI usage.

This module provides a simple, easy-to-use interface for the entire
GeoMatchAI verification pipeline. Just create an instance and call verify().

Example:
    >>> from geomatchai import GeoMatchAI
    >>> from geomatchai.fetchers import MapillaryFetcher
    >>>
    >>> # Initialize verifier
    >>> verifier = await GeoMatchAI.create(
    >>>     fetcher=MapillaryFetcher(api_token="YOUR_KEY"),
    >>>     num_gallery_images=200,
    >>>     threshold=0.65
    >>> )
    >>>
    >>> # Verify any image (provide lat, lon, and image bytes)
    >>> is_verified, score = await verifier.verify(lat, lon, image_bytes)
    >>> print(f"Verified: {is_verified}, Score: {score:.3f}")
"""

import io
import logging
from typing import Literal

import torch
from PIL import Image

from .config import config
from .fetchers.base_fetcher import BaseFetcher
from .gallery.gallery_builder import GalleryBuilder
from .verification.verifier import LandmarkVerifier

logger = logging.getLogger(__name__)


class GeoMatchAI:
    """
    Easy-to-use verifier for landmark verification.

    This class provides a simple interface that wraps all the complexity of:
    - Fetching reference images using a provided fetcher
    - Building a gallery of embeddings
    - Preprocessing user images
    - Extracting features
    - Verifying against the gallery

    Usage:
        from geomatchai import GeoMatchAI
        from geomatchai.fetchers import MapillaryFetcher

        # Create verifier with a fetcher
        fetcher = MapillaryFetcher(api_token="YOUR_KEY")
        verifier = await GeoMatchAI.create(
            fetcher=fetcher,
            num_gallery_images=200,
            threshold=0.65
        )

        # Verify any image (provide lat, lon, image bytes)
        is_verified, score = await verifier.verify(lat, lon, image_bytes)
    """

    def __init__(
        self,
        gallery_builder: GalleryBuilder,
        verifier: LandmarkVerifier,
        fetcher: BaseFetcher,
        num_gallery_images: int,
        search_radius: float,
        skip_gallery_preprocessing: bool,
        batch_size: int,
        threshold: float,
        device: str,
        model_type: str,
        model_variant: str | None
    ):
        """Private constructor. Use GeoMatchAI.create() instead."""
        self.gallery_builder = gallery_builder
        self.verifier = verifier
        self.fetcher = fetcher
        self.num_gallery_images = num_gallery_images
        self.search_radius = search_radius
        self.skip_gallery_preprocessing = skip_gallery_preprocessing
        self.batch_size = batch_size
        self._threshold = threshold
        self._device = device
        self._model_type = model_type
        self._model_variant = model_variant

        # Cache for gallery embeddings by location (lat, lon rounded to 4 decimals ~11m precision)
        self._gallery_cache: dict[tuple[float, float], torch.Tensor] = {}

    @classmethod
    async def create(
        cls,
        fetcher: BaseFetcher,
        num_gallery_images: int = 200,
        search_radius: float = 50.0,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        model_type: Literal["torchvision", "timm", "timm_ensemble"] = "timm",
        model_variant: str | None = None,
        threshold: float = 0.65,
        skip_gallery_preprocessing: bool = True,
        batch_size: int = 32,
    ) -> "GeoMatchAI":
        """
        Create and initialize a GeoMatchAI verifier.

        Args:
            fetcher: Instance of BaseFetcher (e.g., MapillaryFetcher) to fetch gallery images
            num_gallery_images: Number of images to fetch per landmark (default: 200)
            search_radius: Search radius in meters for fetching (default: 50.0)
            device: Device to run on - "auto", "cuda", or "cpu" (default: "auto")
            model_type: Model architecture (default: "timm"):
                - "timm": Best performance (RECOMMENDED)
                - "torchvision": Standard EfficientNet
                - "timm_ensemble": Ensemble model (slower)
            model_variant: Specific model variant (default: tf_efficientnet_b4.ns_jft_in1k for timm)
            threshold: Verification threshold 0.50-0.85 (default: 0.65)
            skip_gallery_preprocessing: Skip person removal for gallery images (default: True)
                                       Set False if gallery contains people
            batch_size: Batch size for gallery processing (default: 32)

        Returns:
            Initialized GeoMatchAI instance

        Example:
            from geomatchai import GeoMatchAI
            from geomatchai.fetchers import MapillaryFetcher

            # Create with Mapillary fetcher
            fetcher = MapillaryFetcher(api_token="YOUR_KEY")
            verifier = await GeoMatchAI.create(
                fetcher=fetcher,
                num_gallery_images=200,
                threshold=0.65
            )

            # Verify user image
            with open("selfie.jpg", "rb") as f:
                image_bytes = f.read()
            is_verified, score = await verifier.verify(50.054404, 19.935730, image_bytes)
        """
        logger.info("Initializing GeoMatchAI...")

        # Initialize gallery builder
        logger.info(f"Loading model: {model_type} ({model_variant or 'default variant'})")
        gallery_builder = GalleryBuilder(
            device=device,
            model_type=model_type,
            model_variant=model_variant
        )

        # Create a dummy verifier (will be updated on first verify call)
        # We need this to avoid building gallery at creation time
        dummy_embedding = torch.randn(1, config.model.EFFICIENTNET_B4_FEATURES)
        verifier = LandmarkVerifier(
            gallery_embeddings=dummy_embedding,
            t_verify=threshold
        )

        logger.info("✅ GeoMatchAI initialization complete!")

        return cls(
            gallery_builder=gallery_builder,
            verifier=verifier,
            fetcher=fetcher,
            num_gallery_images=num_gallery_images,
            search_radius=search_radius,
            skip_gallery_preprocessing=skip_gallery_preprocessing,
            batch_size=batch_size,
            threshold=threshold,
            device=device,
            model_type=model_type,
            model_variant=model_variant
        )


    async def _get_or_build_gallery(self, lat: float, lon: float) -> torch.Tensor:
        """
        Get gallery embeddings for a location, building and caching if needed.

        Args:
            lat: Latitude of the landmark
            lon: Longitude of the landmark

        Returns:
            Gallery embeddings tensor
        """
        # Round to 4 decimal places (~11m precision) for cache key
        cache_key = (round(lat, 4), round(lon, 4))

        # Check cache
        if cache_key in self._gallery_cache:
            logger.debug(f"Using cached gallery for location {cache_key}")
            return self._gallery_cache[cache_key]

        # Build new gallery
        logger.info(f"Building gallery for landmark ({lat}, {lon})...")
        logger.info(f"Fetching {self.num_gallery_images} images, radius: {self.search_radius}m")

        gallery_gen = self.fetcher.get_images(
            lat=lat,
            lon=lon,
            num_images=self.num_gallery_images
        )

        gallery_embeddings = await self.gallery_builder.build_gallery(
            gallery_gen,
            batch_size=self.batch_size,
            skip_preprocessing=self.skip_gallery_preprocessing
        )

        logger.info(f"Gallery built: {gallery_embeddings.shape[0]} images, {gallery_embeddings.shape[1]}D features")

        # Cache the gallery
        self._gallery_cache[cache_key] = gallery_embeddings

        return gallery_embeddings

    async def verify(
        self,
        lat: float,
        lon: float,
        image_bytes: bytes,
        skip_preprocessing: bool = False
    ) -> tuple[bool, float]:
        """
        Verify if an image shows the landmark at the given coordinates.

        This method:
        1. Fetches and builds gallery for the location (cached after first call)
        2. Loads image from bytes
        3. Extracts features from the image
        4. Compares against the gallery

        Args:
            lat: Latitude of the landmark to verify against
            lon: Longitude of the landmark to verify against
            image_bytes: Image data as bytes (e.g., from file.read())
            skip_preprocessing: Skip person removal (default: False)
                               Set True if image has no people in foreground

        Returns:
            Tuple of (is_verified: bool, similarity_score: float)
            - is_verified: True if verified at landmark
            - similarity_score: Confidence score [0, 1]

        Example:
            # Read image bytes
            with open("user_selfie.jpg", "rb") as f:
                image_bytes = f.read()

            # Verify at Wawel Castle
            is_verified, score = await verifier.verify(
                lat=50.054404,
                lon=19.935730,
                image_bytes=image_bytes
            )

            print(f"Verified: {is_verified}, Score: {score:.3f}")
        """
        # Get or build gallery for this location
        gallery_embeddings = await self._get_or_build_gallery(lat, lon)

        # Update verifier with current gallery
        self.verifier.gallery = gallery_embeddings

        # Load image from bytes
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image from bytes: {e}")
            raise ValueError(f"Invalid image bytes: {e}")

        # Extract embedding
        query_embedding = self.gallery_builder.extract_embedding(
            image,
            skip_preprocessing=skip_preprocessing
        )

        # Verify
        is_verified, score = self.verifier.verify(query_embedding)

        return is_verified, score


    def update_threshold(self, threshold: float):
        """
        Update the verification threshold.

        Args:
            threshold: New threshold value (0.50 - 0.85)
                - Lower (0.55): Stricter, fewer false positives
                - Higher (0.70): More lenient, fewer false negatives

        Example:
            verifier.update_threshold(0.70)  # More lenient
        """
        if not (0.50 <= threshold <= 0.85):
            logger.warning(f"Threshold {threshold} outside recommended range [0.50, 0.85]")

        self.verifier.set_threshold(threshold)
        self._threshold = threshold
        logger.info(f"Threshold updated to {threshold}")

    def clear_cache(self):
        """
        Clear the gallery cache to free memory.

        Use this if you've verified many different landmarks and want to
        free up memory.

        Example:
            verifier.clear_cache()
        """
        self._gallery_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Gallery cache cleared")

    @property
    def cached_locations(self) -> list[tuple[float, float]]:
        """Get list of cached landmark locations."""
        return list(self._gallery_cache.keys())

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    @property
    def model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_type": self._model_type,
            "model_variant": self._model_variant,
            "threshold": self._threshold,
            "device": self._device,
            "cached_locations": len(self._gallery_cache),
        }

    def __repr__(self) -> str:
        return (
            f"GeoMatchAI("
            f"model={self._model_type}/{self._model_variant or 'default'}, "
            f"threshold={self._threshold}, "
            f"device={self._device}, "
            f"cached_locations={len(self._gallery_cache)}"
            f")"
        )

