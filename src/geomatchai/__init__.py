"""
GeoMatchAI - High-Performance Visual Place Verification Library

A production-ready deep learning library that verifies whether a user is
physically present at a specific landmark by comparing their selfie against
reference imagery.
"""

import logging

__version__ = "0.1.0"
__author__ = "GeoMatchAI Contributors"
__all__ = [
    "Preprocessor",
    "GalleryBuilder",
    "LandmarkVerifier",
    "EfficientNetFeatureExtractor",
    "MapillaryFetcher",
    "MapillaryFetcherError",
    "BaseFetcher",
    "config",
    "Config",
    "validate_config",
]

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Core components
# Configuration
from geomatchai.config import Config, config, validate_config

# Exceptions
from geomatchai.exceptions import (
    ConfigurationError,
    FeatureExtractionError,
    FetcherError,
    GalleryBuildError,
    GeoMatchAIError,
    MapillaryFetcherError,
    PreprocessingError,
    VerificationError,
)
from geomatchai.fetchers.base_fetcher import BaseFetcher
from geomatchai.fetchers.mapillary_fetcher import MapillaryFetcher
from geomatchai.gallery.gallery_builder import GalleryBuilder
from geomatchai.models.efficientnet_timm import EfficientNetFeatureExtractor
from geomatchai.preprocessing.preprocessor import Preprocessor
from geomatchai.verification.verifier import LandmarkVerifier
