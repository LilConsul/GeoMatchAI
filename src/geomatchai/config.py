"""
Configuration management for GeoMatchAI.

Centralizes all configurable parameters, constants, and user settings.
Users can configure the library by setting values directly on the config object
or by passing parameters to individual classes.

Example:
    >>> from geomatchai import config
    >>> config.set_mapillary_api_key("your_key_here")
    >>> config.set_device("cuda")
    >>> config.set_log_level("DEBUG")
"""

import os
from dataclasses import dataclass
from typing import Literal


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


@dataclass
class RuntimeConfig:
    """Configuration for runtime behavior (logging, device, etc.)."""

    # Logging
    VALID_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

    # These are user-settable at runtime
    _log_level: str | None = None  # User can set via config.set_log_level()
    _device: str | None = None  # User can set via config.set_device()
    _mapillary_api_key: str | None = None  # User can set via config.set_mapillary_api_key()


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

    # =============================================================================
    # User Configuration Methods
    # =============================================================================

    def set_mapillary_api_key(self, api_key: str) -> None:
        """
        Set Mapillary API key for the library.

        Args:
            api_key: Your Mapillary client access token

        Example:
            >>> config.set_mapillary_api_key("your_key_here")
        """
        self.runtime._mapillary_api_key = api_key

    def get_mapillary_api_key(self) -> str | None:
        """
        Get configured Mapillary API key.

        Priority:
            1. User-set value (via set_mapillary_api_key)
            2. MAPILLARY_API_KEY environment variable (convenience for dev)
            3. None

        Returns:
            API key if configured, None otherwise
        """
        # Priority 1: User-configured value
        if self.runtime._mapillary_api_key:
            return self.runtime._mapillary_api_key

        # Priority 2: Environment variable (for development convenience)
        return os.getenv("MAPILLARY_API_KEY")

    def set_log_level(
        self, level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ) -> None:
        """
        Set logging level for the library.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Example:
            >>> config.set_log_level("DEBUG")

        Raises:
            ValueError: If invalid log level provided
        """
        level_upper = level.upper()
        if level_upper not in self.runtime.VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level: {level}. "
                f"Must be one of: {', '.join(self.runtime.VALID_LOG_LEVELS)}"
            )
        self.runtime._log_level = level_upper

    def get_log_level(self) -> str:
        """
        Get configured logging level.

        Priority:
            1. User-set value (via set_log_level)
            2. LOG_LEVEL environment variable (convenience for dev)
            3. Default: "INFO"

        Returns:
            Log level string
        """
        # Priority 1: User-configured value
        if self.runtime._log_level:
            return self.runtime._log_level

        # Priority 2: Environment variable (for development convenience)
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if env_level in self.runtime.VALID_LOG_LEVELS:
            return env_level

        # Priority 3: Default
        return "INFO"

    def set_device(self, device: Literal["auto", "cuda", "cpu"]) -> None:
        """
        Set default device for computation globally.

        This can be overridden per-instance when creating components.

        Args:
            device: Device to use ("auto", "cuda", or "cpu")

        Example:
            >>> config.set_device("cuda")  # Use GPU globally

        Raises:
            ValueError: If invalid device provided
        """
        device_lower = device.lower()
        valid_devices = ("auto", "cuda", "cpu")
        if device_lower not in valid_devices:
            raise ValueError(
                f"Invalid device: {device}. Must be one of: {', '.join(valid_devices)}"
            )
        self.runtime._device = device_lower

    def get_device(self) -> str | None:
        """
        Get globally configured device.

        Priority:
            1. User-set value (via set_device)
            2. DEVICE environment variable (convenience for dev)
            3. None (each class will auto-detect)

        Returns:
            Device string if configured, None if should auto-detect per-instance
        """
        # Priority 1: User-configured value
        if self.runtime._device:
            return self.runtime._device

        # Priority 2: Environment variable (for development convenience)
        env_device = os.getenv("DEVICE", "").lower()
        if env_device in ("auto", "cuda", "cpu"):
            return env_device

        # Priority 3: None (let each instance decide)
        return None

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
