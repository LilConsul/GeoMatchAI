"""
Configuration management for GeoMatchAI.

Example:
    >>> from geomatchai import config
    >>> config.set_mapillary_api_key("your_key_here")
    >>> config.set_device("cuda")
    >>> config.set_log_level("DEBUG")
"""

import os
from dataclasses import dataclass, field
from typing import Literal

# =============================================================================
# Constants - Static configuration values
# =============================================================================


@dataclass(frozen=True)
class PreprocessingConfig:
    """Image preprocessing constants."""

    MAX_IMAGE_SIZE_MB: int = 50
    MAX_DIMENSION: int = 10000
    MIN_DIMENSION: int = 100
    TARGET_SIZE: tuple[int, int] = (520, 520)
    PERSON_CLASS_IDX: int = 15  # COCO "person" class
    MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    STD: tuple[float, float, float] = (0.229, 0.224, 0.225)  # ImageNet


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration constants."""

    DEFAULT_MODEL_TYPE: str = "timm"
    DEFAULT_TIMM_VARIANT: str = "tf_efficientnet_b4.ns_jft_in1k"
    DEFAULT_TORCHVISION_VARIANT: str = "b4"
    EFFICIENTNET_B4_FEATURES: int = 1792
    EFFICIENTNET_B5_FEATURES: int = 2048


@dataclass(frozen=True)
class VerificationConfig:
    """Verification threshold constants."""

    DEFAULT_THRESHOLD: float = 0.65
    MIN_THRESHOLD: float = 0.50
    MAX_THRESHOLD: float = 0.85
    STRICT_THRESHOLD: float = 0.55
    LENIENT_THRESHOLD: float = 0.70
    ACQUISITION_THRESHOLD: float = 0.95


@dataclass(frozen=True)
class GalleryConfig:
    """Gallery building constants."""

    DEFAULT_BATCH_SIZE: int = 32
    MAX_BATCH_SIZE: int = 64
    MIN_BATCH_SIZE: int = 1


@dataclass(frozen=True)
class FetcherConfig:
    """Image fetcher constants."""

    DEFAULT_SEARCH_RADIUS: float = 50.0
    DEFAULT_NUM_IMAGES: int = 20
    DEFAULT_REQUEST_TIMEOUT: float = 30.0
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_THUMBNAIL_RESOLUTION: int = 1024


# =============================================================================
# Runtime State - User-configurable values
# =============================================================================


@dataclass
class RuntimeState:
    """Runtime configuration state (user-settable)."""

    mapillary_api_key: str | None = None
    log_level: str | None = None
    device: str | None = None

    VALID_LOG_LEVELS: tuple[str, ...] = field(
        default=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), init=False
    )
    VALID_DEVICES: tuple[str, ...] = field(default=("auto", "cuda", "cpu"), init=False)


# =============================================================================
# Main Config Class
# =============================================================================


class Config:
    """Configuration manager for GeoMatchAI."""

    def __init__(self):
        # Constants
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.verification = VerificationConfig()
        self.gallery = GalleryConfig()
        self.fetcher = FetcherConfig()

        # Runtime state
        self._state = RuntimeState()

    # -------------------------------------------------------------------------
    # Mapillary API Key
    # -------------------------------------------------------------------------

    def set_mapillary_api_key(self, api_key: str) -> None:
        """Set Mapillary API key."""
        self._state.mapillary_api_key = api_key

    def get_mapillary_api_key(self) -> str | None:
        """Get API key (priority: user-set > env var > None)."""
        return self._state.mapillary_api_key or os.getenv("MAPILLARY_API_KEY")

    # -------------------------------------------------------------------------
    # Log Level
    # -------------------------------------------------------------------------

    def set_log_level(
        self, level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ) -> None:
        """Set logging level."""
        level = level.upper()
        if level not in self._state.VALID_LOG_LEVELS:
            raise ValueError(f"Invalid log level: {level}")
        self._state.log_level = level

    def get_log_level(self) -> str:
        """Get log level (priority: user-set > env var > INFO)."""
        if self._state.log_level:
            return self._state.log_level

        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        return env_level if env_level in self._state.VALID_LOG_LEVELS else "INFO"

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    def set_device(self, device: Literal["auto", "cuda", "cpu"]) -> None:
        """Set global device."""
        device = device.lower()
        if device not in self._state.VALID_DEVICES:
            raise ValueError(f"Invalid device: {device}")
        self._state.device = device

    def get_device(self) -> str | None:
        """Get device (priority: user-set > env var > None/auto-detect)."""
        if self._state.device:
            return self._state.device

        env_device = os.getenv("DEVICE", "").lower()
        return env_device if env_device in self._state.VALID_DEVICES else None


# =============================================================================
# Validation Functions
# =============================================================================


def validate_config(cfg: Config) -> list[str]:
    """
    Validate the current configuration state.

    Args:
        cfg: Config instance to validate

    Returns:
        List of validation error messages (empty list if valid)

    Example:
        >>> from geomatchai import config, validate_config
        >>> errors = validate_config(config)
        >>> if errors:
        ...     print("Configuration errors:", errors)
    """
    errors = []

    # Check Mapillary API key (if needed for fetching)
    api_key = cfg.get_mapillary_api_key()
    if api_key is None:
        errors.append(
            "Mapillary API key not configured. Set it with config.set_mapillary_api_key() "
            "or via MAPILLARY_API_KEY environment variable."
        )
    elif not api_key.strip():
        errors.append("Mapillary API key is empty.")

    # Validate device setting if explicitly set
    if cfg._state.device:
        if cfg._state.device not in cfg._state.VALID_DEVICES:
            errors.append(f"Invalid device: {cfg._state.device}")

    # Validate log level if explicitly set
    if cfg._state.log_level:
        if cfg._state.log_level not in cfg._state.VALID_LOG_LEVELS:
            errors.append(f"Invalid log level: {cfg._state.log_level}")

    return errors


# Global config instance
config = Config()
