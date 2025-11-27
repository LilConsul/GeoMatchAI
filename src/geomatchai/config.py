"""
Configuration management for GeoMatchAI.

Centralizes all configurable parameters, constants, and environment-based settings.
"""

import os
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""

    # Image validation limits
    MAX_IMAGE_SIZE_MB: int = 50
    MAX_DIMENSION: int = 10000
    MIN_DIMENSION: int = 100
    TARGET_SIZE: tuple[int, int] = (520, 520)

    # Segmentation settings
    PERSON_CLASS_IDX: int = 15  # COCO class index for "person"

    # Normalization (ImageNet stats)
    MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Configuration for feature extraction models."""

    # Default model settings
    DEFAULT_MODEL_TYPE: str = "timm"
    DEFAULT_TIMM_VARIANT: str = "tf_efficientnet_b4.ns_jft_in1k"
    DEFAULT_TORCHVISION_VARIANT: str = "b4"

    # Feature dimensions
    EFFICIENTNET_B4_FEATURES: int = 1792
    EFFICIENTNET_B5_FEATURES: int = 2048


@dataclass
class VerificationConfig:
    """Configuration for landmark verification."""

    # Verification thresholds
    DEFAULT_THRESHOLD: float = 0.65
    MIN_THRESHOLD: float = 0.50
    MAX_THRESHOLD: float = 0.85

    # Recommended thresholds by use case
    STRICT_THRESHOLD: float = 0.55  # Fewer false positives
    LENIENT_THRESHOLD: float = 0.70  # Fewer false negatives

    # Acquisition threshold for continuous improvement
    ACQUISITION_THRESHOLD: float = 0.95


@dataclass
class GalleryConfig:
    """Configuration for gallery building."""

    DEFAULT_BATCH_SIZE: int = 32
    MAX_BATCH_SIZE: int = 64
    MIN_BATCH_SIZE: int = 1


@dataclass
class FetcherConfig:
    """Configuration for image fetchers."""

    # Mapillary settings
    DEFAULT_SEARCH_RADIUS: float = 50.0  # meters
    DEFAULT_NUM_IMAGES: int = 20
    DEFAULT_REQUEST_TIMEOUT: float = 30.0  # seconds
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_THUMBNAIL_RESOLUTION: int = 1024

    # Environment variable names
    MAPILLARY_API_KEY_ENV: str = "MAPILLARY_API_KEY"


@dataclass
class RuntimeConfig:
    """Configuration for runtime behavior (logging, device, etc.)."""

    # Logging
    DEFAULT_LOG_LEVEL: str = "INFO"
    VALID_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    LOG_LEVEL_ENV: str = "LOG_LEVEL"

    # Device
    DEFAULT_DEVICE: str = "auto"  # "auto", "cuda", or "cpu"
    DEVICE_ENV: str = "DEVICE"
    CUDA_VISIBLE_DEVICES_ENV: str = "CUDA_VISIBLE_DEVICES"


class Config:
    """
    Main configuration class aggregating all settings.

    Provides easy access to all configuration parameters and
    supports environment variable overrides.
    """

    def __init__(self):
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.verification = VerificationConfig()
        self.gallery = GalleryConfig()
        self.fetcher = FetcherConfig()
        self.runtime = RuntimeConfig()

    def get_mapillary_api_key(self) -> str | None:
        """
        Get Mapillary API key from environment.

        Returns:
            API key if set, None otherwise
        """
        return os.getenv(self.fetcher.MAPILLARY_API_KEY_ENV)

    def get_log_level(self) -> str:
        """
        Get log level from environment or use default.

        Returns:
            Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level = os.getenv(self.runtime.LOG_LEVEL_ENV, self.runtime.DEFAULT_LOG_LEVEL).upper()
        return level if level in self.runtime.VALID_LOG_LEVELS else self.runtime.DEFAULT_LOG_LEVEL

    def get_device(self) -> str:
        """
        Get device configuration from environment or use default.

        Returns:
            Device string ("auto", "cuda", or "cpu")
        """
        device = os.getenv(self.runtime.DEVICE_ENV, self.runtime.DEFAULT_DEVICE).lower()
        valid_devices = ("auto", "cuda", "cpu")
        return device if device in valid_devices else self.runtime.DEFAULT_DEVICE

    def get_cuda_visible_devices(self) -> str | None:
        """
        Get CUDA_VISIBLE_DEVICES from environment.

        Returns:
            CUDA device IDs if set, None otherwise
        """
        return os.getenv(self.runtime.CUDA_VISIBLE_DEVICES_ENV)

    def validate(self) -> list[str]:
        """
        Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate thresholds
        if not (
            self.verification.MIN_THRESHOLD
            <= self.verification.DEFAULT_THRESHOLD
            <= self.verification.MAX_THRESHOLD
        ):
            errors.append(
                f"Default threshold {self.verification.DEFAULT_THRESHOLD} "
                f"outside valid range [{self.verification.MIN_THRESHOLD}, "
                f"{self.verification.MAX_THRESHOLD}]"
            )

        # Validate batch sizes
        if not (
            self.gallery.MIN_BATCH_SIZE
            <= self.gallery.DEFAULT_BATCH_SIZE
            <= self.gallery.MAX_BATCH_SIZE
        ):
            errors.append(
                f"Default batch size {self.gallery.DEFAULT_BATCH_SIZE} "
                f"outside valid range [{self.gallery.MIN_BATCH_SIZE}, "
                f"{self.gallery.MAX_BATCH_SIZE}]"
            )

        # Validate image dimensions
        if self.preprocessing.MIN_DIMENSION >= self.preprocessing.MAX_DIMENSION:
            errors.append("MIN_DIMENSION must be less than MAX_DIMENSION")

        return errors


# Global config instance
config = Config()
