"""
Custom exception classes for GeoMatchAI library.

Provides specific exception types for different error scenarios
to enable better error handling and debugging.
"""


class GeoMatchAIError(Exception):
    """Base exception for all GeoMatchAI errors."""

    pass


class PreprocessingError(GeoMatchAIError):
    """Exception raised during image preprocessing."""

    pass


class FeatureExtractionError(GeoMatchAIError):
    """Exception raised during feature extraction."""

    pass


class GalleryBuildError(GeoMatchAIError):
    """Exception raised during gallery building."""

    pass


class VerificationError(GeoMatchAIError):
    """Exception raised during landmark verification."""

    pass


class FetcherError(GeoMatchAIError):
    """Exception raised by image fetchers."""

    pass


class ConfigurationError(GeoMatchAIError):
    """Exception raised for invalid configuration."""

    pass
