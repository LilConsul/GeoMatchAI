# GeoMatchAI 🌍🔍

**High-Performance Visual Place Verification Library**

**Authors:** Shevchenko Denys & Karabanov Yehor

A production-ready deep learning library that verifies whether a user is physically present at a specific landmark by comparing their selfie against reference imagery. Built with PyTorch, EfficientNet, and advanced computer vision techniques.

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Skill Issue](https://img.shields.io/badge/skill-issue-orange.svg)](https://github.com/LilConsul/GeoMatchAI)

---

## 🎯 Overview

GeoMatchAI is a Python library that solves the challenge of verifying a user's presence at a landmark (e.g., Wawel Castle) using deep learning-based visual recognition. The system handles occlusion from people in selfies, uses semantic segmentation to isolate landmark features, and achieves **87.5% accuracy** with **75.6ms inference time**.

### Key Features

- **Person Segmentation** - Automatically removes people from selfies using DeepLabV3 to focus on landmark features
- **High Accuracy** - 87.5% verification accuracy with 33.2% discrimination gap
- **Fast Inference** - ~75ms per verification (excluding gallery build)
- **Multiple Model Support** - TorchVision and TIMM EfficientNet variants (B4/B5)
- **Async Image Fetching** - Concurrent Mapillary API integration for reference gallery building
- **Production Ready** - Comprehensive error handling, validation, and security features
- **Extensible Architecture** - Clean abstractions for fetchers, models, and verification logic

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────┐
│  User Selfie    │
│   (with person) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Stage 1: Preprocessing         │
│  ─────────────────────────      │
│  • Semantic Segmentation        │
│  • Person Mask Extraction       │
│  • Neutral Mean Replacement     │
│  • Normalize & Resize (520x520) │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Stage 2: Feature Extraction    │
│  ─────────────────────────      │
│  • EfficientNet-B4 Backbone     │
│  • Extract 1792-D Embedding     │
│  • L2 Normalization             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Stage 3: Verification          │
│  ─────────────────────────      │
│  • Cosine Similarity vs Gallery │
│  • Max Score Comparison         │
│  • Threshold Decision (0.65)    │
└────────┬────────────────────────┘
         │
         ▼
┌────────────┐
│  VERIFIED  │
│  REJECTED  │
└────────────┘
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Preprocessor** | [`preprocessing/preprocessor.py`](src/geomatchai/preprocessing/preprocessor.py) | Semantic segmentation (DeepLabV3) to remove people from selfies |
| **GalleryBuilder** | [`gallery/gallery_builder.py`](src/geomatchai/gallery/gallery_builder.py) | Batch processing to build reference embedding matrix (N×D) |
| **LandmarkVerifier** | [`verification/verifier.py`](src/geomatchai/verification/verifier.py) | Cosine similarity-based verification against gallery |
| **Feature Extractors** | [`models/efficientnet_timm.py`](src/geomatchai/models/efficientnet_timm.py) | EfficientNet-B4 models (TorchVision/TIMM) |
| **Mapillary Fetcher** | [`fetchers/mapillary_fetcher.py`](src/geomatchai/fetchers/mapillary_fetcher.py) | Async street-level imagery fetcher |

---

## 📦 Installation

### Prerequisites

- **Python 3.13** (required)
- **CUDA 12.8** (optional, for GPU acceleration)
- **UV package manager** ([install here](https://docs.astral.sh/uv/getting-started/installation/))

### Install from GitHub

```powershell
# Add GeoMatchAI to your project
uv add git+https://github.com/LilConsul/GeoMatchAI.git

# For CPU-only PyTorch (smaller download, no GPU support)
uv add git+https://github.com/LilConsul/GeoMatchAI.git --index pytorch-cpu

# For CUDA 12.8 (GPU acceleration - default)
uv add git+https://github.com/LilConsul/GeoMatchAI.git
```

### Development Installation

If you want to contribute or modify the library:

```powershell
# Clone the repository
git clone https://github.com/LilConsul/GeoMatchAI.git
cd GeoMatchAI

# Install in editable mode with all dependencies
uv sync

# Install with development dependencies (includes plotting tools)
uv sync --all-groups
```

### Environment Variables

Set your Mapillary API key (get from [Mapillary Developer Dashboard](https://www.mapillary.com/dashboard/developers)):

```powershell
$env:MAPILLARY_API_KEY="YOUR_API_KEY_HERE"
```

---

## 🚀 Quick Start

### Basic Example

See [example_usage.py](example_usage.py) for complete working examples.

```python
import asyncio
from geomatchai import GeoMatchAI, config
from geomatchai.fetchers import MapillaryFetcher

async def main():
    # Configure library settings
    config.set_mapillary_api_key("YOUR_MAPILLARY_API_KEY")
    config.set_device("cuda")
    config.set_log_level("INFO")
    
    # Create fetcher
    fetcher = MapillaryFetcher(api_token=config.get_mapillary_api_key())
    
    # Create verifier (uses config defaults)
    verifier = await GeoMatchAI.create(fetcher=fetcher)
    
    # Verify image
    with open("user_selfie.jpg", "rb") as f:
        is_verified, score = await verifier.verify(50.054404, 19.935730, f.read())
    
    print(f"Verified: {is_verified}, Score: {score:.3f}")

asyncio.run(main())
```

The verifier automatically:
- Uses config defaults for all parameters (threshold, batch_size, num_images, etc.)
- Fetches reference images for each location (cached per location)
- Builds gallery embeddings (cached)
- Verifies images against the gallery

### Custom Fetcher

Implement `BaseFetcher` to use your own image source:

```python
from geomatchai.fetchers import BaseFetcher
from collections.abc import AsyncGenerator
from PIL import Image

class DatabaseFetcher(BaseFetcher):
    async def get_images(self, lat: float, lon: float, num_images: int = 20) -> AsyncGenerator[Image.Image, None]:
        for image_data in your_database.query(lat, lon):
            yield Image.open(image_data)

# Use it
fetcher = DatabaseFetcher()
verifier = await GeoMatchAI.create(fetcher=fetcher)
```

### Configuration

Configure library settings using the `config` object:

```python
from geomatchai import config

# Set Mapillary API key
config.set_mapillary_api_key("YOUR_KEY")

# Set device (auto, cuda, or cpu)
config.set_device("cuda")

# Set logging level
config.set_log_level("INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Access configuration constants
print(config.verification.DEFAULT_THRESHOLD)  # 0.65
print(config.model.DEFAULT_MODEL_TYPE)        # "timm"
print(config.fetcher.DEFAULT_SEARCH_RADIUS)   # 50.0
```

Configuration options:
- `config.set_mapillary_api_key(key)` - Set API key (or use `MAPILLARY_API_KEY` env var)
- `config.set_device(device)` - Set device: "auto", "cuda", or "cpu"
- `config.set_log_level(level)` - Set logging: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
- `config.get_mapillary_api_key()` - Get API key (checks env var if not set)
- `config.get_device()` - Get configured device
- `config.get_log_level()` - Get logging level

---

## 📊 Performance Metrics

### Best Configuration (Production Recommended)

Based on comprehensive testing with **6 Wawel Castle images** and **2 unrelated images**:
- **Test configurations:** 64 (8 images × 4 models × 2 preprocessing modes)
- **Test images:** 6 Wawel (positive examples) + 2 unrelated landmarks (negative examples)

```
Model:              TIMM-NoisyStudent (tf_efficientnet_b4.ns_jft_in1k)
Preprocessing:      ENABLED
Threshold:          0.65
Gallery Size:       198 images (Mapillary street-level imagery)
─────────────────────────────────────────────────────────────
Accuracy:           87.5% (7/8 correct)
Discrimination Gap: 0.332 (33.2%)
Avg Wawel Score:    0.783
Avg Unrelated:      0.451
Inference Time:     75.6ms (preprocessing + feature extraction + verification)
Gallery Build:      4.1-4.3s for 198 images
Overall Accuracy:   73.44% (47/64 across all configs)
```

### Model Comparison

All models tested with 8 images (6 Wawel + 2 unrelated), threshold = 0.65:

| Model | Preprocessing | Accuracy | Discrimination Gap | Avg Wawel | Avg Unrelated | Inference Time |
|-------|---------------|----------|-------------------|-----------|---------------|----------------|
| **TIMM-NoisyStudent** ⭐ | ✅ Enabled | **87.5% (7/8)** | **0.332** | **0.783** | **0.451** | **75.6ms** |
| TIMM-Standard | ✅ Enabled | 87.5% (7/8) | 0.332 | 0.783 | 0.451 | 77.7ms |
| TIMM-NoisyStudent | ❌ Disabled | 87.5% (7/8) | 0.285 | 0.771 | 0.486 | 22.7ms |
| TIMM-Standard | ❌ Disabled | 87.5% (7/8) | 0.285 | 0.771 | 0.486 | 23.4ms |
| TorchVision-B4 | ❌ Disabled | 87.5% (7/8) | 0.325 | 0.834 | 0.509 | 24.3ms |
| TorchVision-B4 | ✅ Enabled | 75.0% (6/8) | 0.083 | 0.909 | 0.826 | 90.2ms |
| TIMM-AdvProp | ✅ Enabled | 37.5% (3/8) | 0.137 | 0.611 | 0.474 | 78.5ms |
| TIMM-AdvProp | ❌ Disabled | 37.5% (3/8) | 0.042 | 0.578 | 0.536 | 22.2ms |

**Key Insights:**
- ⭐ **TIMM-NoisyStudent with preprocessing** achieves best balance of accuracy (87.5%) and discrimination gap (33.2%)
- **Without preprocessing**, TIMM models are faster (~23ms) with same accuracy but lower discrimination
- **TorchVision-B4 performs WORSE with preprocessing** (75% vs 87.5%) - high scores but poor discrimination
- **TIMM-AdvProp underperforms** significantly (37.5% accuracy) - not recommended
- Discrimination gap > 0.30 indicates excellent separation between matching/non-matching landmarks

---

## 🧪 Testing

> **Note:** Tests are included in the repository for development purposes. If you installed GeoMatchAI via `uv add`, you can use the library directly in your project without running these tests.

### Run Comprehensive Tests

```powershell
# Clone and set up development environment
git clone https://github.com/LilConsul/GeoMatchAI.git
cd GeoMatchAI
uv sync

# Set your Mapillary API key (required for tests that fetch images)
$env:MAPILLARY_API_KEY="YOUR_API_KEY"

# Run full test suite (tests all models, preprocessing modes, timing)
uv run python tests/test_comprehensive.py

# Run specific tests
uv run python tests/test_preprocessing.py
uv run python tests/test_gallery.py
uv run python tests/test_verifier.py
```

### Generate Visualizations

```powershell
# Install plotting dependencies
uv sync --group plot

# Generate all plots from test results
uv run python tests/plot_results.py
```

**Generated Outputs:**
- `tests/output/csv/` - Detailed CSV results
- `tests/output/plots/` - 7 visualization plots
- `tests/output/TEST_REPORT.txt` - Summary report

---

## 📖 API Reference

### GeoMatchAI

Main interface for verification. See [example_usage.py](example_usage.py) for complete examples.

#### `GeoMatchAI.create()`

```python
# All parameters are optional and use config defaults
verifier = await GeoMatchAI.create(
    fetcher=fetcher,                    # BaseFetcher instance (required)
    num_gallery_images=None,            # default: config.fetcher.DEFAULT_NUM_IMAGES (20)
    search_radius=None,                 # default: config.fetcher.DEFAULT_SEARCH_RADIUS (50.0)
    threshold=None,                     # default: config.verification.DEFAULT_THRESHOLD (0.65)
    device=None,                        # default: config.get_device() or "auto"
    model_type=None,                    # default: config.model.DEFAULT_MODEL_TYPE ("timm")
    model_variant=None,                 # default: config.model.DEFAULT_TIMM_VARIANT
    skip_gallery_preprocessing=True,    # Skip person removal for gallery
    batch_size=None                     # default: config.gallery.DEFAULT_BATCH_SIZE (32)
)
```

**Recommended usage:** Configure settings via `config`, then create verifier with just the fetcher:

```python
from geomatchai import GeoMatchAI, config

# Configure once
config.set_device("cuda")
config.set_log_level("INFO")

# Create verifier (uses all config defaults)
verifier = await GeoMatchAI.create(fetcher=fetcher)
```

#### `verifier.verify()`

```python
is_verified, score = await verifier.verify(
    lat=50.054404,                      # Landmark latitude (required)
    lon=19.935730,                      # Landmark longitude (required)
    image_bytes=image_bytes,            # Image data as bytes (required)
    skip_preprocessing=False            # Skip person removal for query
)
```

#### Other Methods

```python
# Update threshold
verifier.update_threshold(0.70)

# Clear cache
verifier.clear_cache()

# Properties
verifier.device                         # Device being used
verifier.model_info                     # Model information dict
verifier.cached_locations               # List of cached (lat, lon)
```

### BaseFetcher

Interface for custom image fetchers:

```python
from geomatchai.fetchers import BaseFetcher
from collections.abc import AsyncGenerator
from PIL import Image

class CustomFetcher(BaseFetcher):
    async def get_images(
        self, 
        lat: float, 
        lon: float, 
        num_images: int = 20
    ) -> AsyncGenerator[Image.Image, None]:
        # Yield PIL Images
        for img_data in your_source:
            yield Image.open(img_data)
```

### Configuration Constants

Access via `config` object:

```python
from geomatchai import config

# Verification thresholds
config.verification.DEFAULT_THRESHOLD      # 0.65
config.verification.STRICT_THRESHOLD       # 0.55
config.verification.LENIENT_THRESHOLD      # 0.70

# Model settings
config.model.DEFAULT_MODEL_TYPE            # "timm"
config.model.DEFAULT_TIMM_VARIANT          # "tf_efficientnet_b4.ns_jft_in1k"

# Fetcher settings
config.fetcher.DEFAULT_SEARCH_RADIUS       # 50.0
config.fetcher.DEFAULT_NUM_IMAGES          # 20

# Gallery settings
config.gallery.DEFAULT_BATCH_SIZE          # 32
```

---

## 🔧 Configuration

### Threshold Tuning

Adjust the verification threshold based on your use case:

| Threshold | Use Case | Trade-off |
|-----------|----------|-----------|
| **0.55-0.60** | High security applications | Fewer false positives, more false negatives |
| **0.65** | Balanced (RECOMMENDED) | Optimal accuracy (87.5%) |
| **0.70-0.75** | User-friendly applications | Fewer false negatives, more false positives |

### Device Selection

```python
from geomatchai import config

# Configure globally
config.set_device("cuda")   # Use GPU
config.set_device("cpu")    # Use CPU only
config.set_device("auto")   # Auto-detect (default)

# Or per-instance
verifier = await GeoMatchAI.create(fetcher=fetcher, device="cuda")
```

**Performance Impact (TIMM-NoisyStudent with preprocessing):**
- **GPU (CUDA):** 75.6ms inference
- **CPU:** ~200-500ms inference (slower)

### Logging

```python
from geomatchai import config

config.set_log_level("DEBUG")    # Verbose output
config.set_log_level("INFO")     # Normal output (default)
config.set_log_level("WARNING")  # Warnings only
```

---

## 🛠️ Development

### Project Structure

```
GeoMatchAI/
├── src/
│   └── geomatchai/
│       ├── __init__.py
│       ├── config.py                    # Configuration management
│       ├── exceptions.py                # Custom exceptions
│       ├── verifier_singleton.py        # Main GeoMatchAI class
│       ├── fetchers/                    # Image fetchers
│       │   ├── __init__.py
│       │   ├── base_fetcher.py         # Abstract fetcher interface
│       │   └── mapillary_fetcher.py    # Mapillary API integration
│       ├── gallery/                     # Reference gallery builder
│       │   ├── __init__.py
│       │   └── gallery_builder.py
│       ├── models/                      # Feature extraction models
│       │   ├── __init__.py
│       │   ├── efficientnet.py         # TorchVision EfficientNet
│       │   └── efficientnet_timm.py    # TIMM EfficientNet variants
│       ├── preprocessing/               # Image preprocessing
│       │   ├── __init__.py
│       │   └── preprocessor.py         # DeepLabV3 segmentation
│       └── verification/                # Verification logic
│           ├── __init__.py
│           └── verifier.py             # Cosine similarity verifier
├── tests/                               # Test suite
│   ├── __init__.py
│   ├── usertest_comprehensive.py        # Full test suite
│   ├── usertest_preprocessing.py        # Preprocessing tests
│   ├── usertest_gallery.py             # Gallery builder tests
│   ├── usertest_verifier.py            # Verifier tests
│   ├── input/                          # Test images
│   │   ├── wawel/                      # Wawel Castle test images
│   │   └── ...
│   └── output/                         # Test results
│       ├── refactored/                 # Processed images
│       └── wrapper/                    # Wrapper test outputs
├── example_usage.py                     # Usage examples
├── pyproject.toml                       # Project configuration & dependencies
├── uv.lock                             # Dependency lock file
├── README.md                           # This file
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # MIT License
└── goal.md                             # Project goals & architecture
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **EfficientNet** - Tan & Le (2019) - [Paper](https://arxiv.org/abs/1905.11946)
- **DeepLabV3** - Chen et al. (2017) - [Paper](https://arxiv.org/abs/1706.05587)
- **TIMM Library** - Ross Wightman - [GitHub](https://github.com/huggingface/pytorch-image-models)
- **Mapillary API** - Street-level imagery - [Website](https://www.mapillary.com/)

---

<div align="center">
  <strong>Built with PyTorch, EfficientNet, and DeepLabV3</strong>
</div>
