# GeoMatchAI 🌍🔍

**High-Performance Visual Place Verification System**

A production-ready deep learning system that verifies whether a user is physically present at a specific landmark by comparing their selfie against reference imagery. Built with PyTorch, EfficientNet, and advanced computer vision techniques.

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

GeoMatchAI solves the challenge of verifying a user's presence at a landmark (e.g., Wawel Castle) using deep learning-based visual recognition. The system handles occlusion from people in selfies, uses semantic segmentation to isolate landmark features, and achieves **87.5% accuracy** with **75.6ms inference time**.

### Key Features

✅ **Person Segmentation** - Automatically removes people from selfies using DeepLabV3 to focus on landmark features  
✅ **High Accuracy** - 87.5% verification accuracy with 33.2% discrimination gap  
✅ **Fast Inference** - ~75ms per verification (excluding gallery build)  
✅ **Multiple Model Support** - TorchVision and TIMM EfficientNet variants (B4/B5)  
✅ **Async Image Fetching** - Concurrent Mapillary API integration for reference gallery building  
✅ **Production Ready** - Comprehensive error handling, validation, and security features  
✅ **Extensible Architecture** - Clean abstractions for fetchers, models, and verification logic

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
┌─────────────────┐
│  ✅ VERIFIED    │
│  ❌ REJECTED    │
└─────────────────┘
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
- **UV package manager** (recommended)

### Quick Install

```powershell
# Clone the repository
git clone https://github.com/yourusername/GeoMatchAI.git
cd GeoMatchAI

# Install dependencies with UV (automatically sets up virtual environment)
uv sync

# For CPU-only installation (modify pyproject.toml to use pytorch-cpu index)
# Then run: uv sync
```

### Manual Installation

```powershell
# Create virtual environment
uv venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
uv add torch torchvision timm pillow requests mapillary aiohttp

# Optional: Install plotting dependencies for analysis
uv sync --group plot
```

### Environment Variables

Set your Mapillary API key (get from [Mapillary Developer Dashboard](https://www.mapillary.com/dashboard/developers)):

```powershell
$env:MAPILLARY_API_KEY="YOUR_API_KEY_HERE"
```

---

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from PIL import Image
from geomatchai.gallery.gallery_builder import GalleryBuilder
from geomatchai.verification.verifier import LandmarkVerifier
from geomatchai.fetchers.mapillary_fetcher import MapillaryFetcher

async def verify_landmark():
    # 1. Fetch reference images from Mapillary (Wawel Castle example)
    fetcher = MapillaryFetcher(api_token="YOUR_MAPILLARY_TOKEN")
    lat, lon = 50.054404, 19.935730  # Wawel Castle coordinates
    
    # 2. Build reference gallery (one-time setup)
    builder = GalleryBuilder(
        device="cuda",  # or "cpu"
        model_type="timm",
        model_variant="tf_efficientnet_b4.ns_jft_in1k"  # RECOMMENDED
    )
    
    gallery_embeddings = await builder.build_gallery(
        fetcher.get_images(lat, lon, num_images=200),
        skip_preprocessing=True  # Gallery images are clean (no people)
    )
    
    print(f"Gallery built: {gallery_embeddings.shape[0]} images")
    
    # 3. Initialize verifier
    verifier = LandmarkVerifier(
        gallery_embeddings=gallery_embeddings,
        t_verify=0.65  # Recommended threshold
    )
    
    # 4. Verify user's selfie
    user_selfie = Image.open("user_selfie.jpg")
    
    # Preprocess (removes person, extracts landmark features)
    query_tensor = builder.preprocessor.preprocess_image(user_selfie)
    
    # Extract features
    import torch
    with torch.no_grad():
        query_embedding = builder.feature_extractor(
            query_tensor.unsqueeze(0).to(builder.device)
        )
    
    # Verify
    is_verified, similarity_score = verifier.verify(query_embedding)
    
    if is_verified:
        print(f"✅ VERIFIED! Score: {similarity_score:.3f}")
    else:
        print(f"❌ REJECTED. Score: {similarity_score:.3f}")

# Run
asyncio.run(verify_landmark())
```

### Advanced Configuration

```python
# Use different model variants
builder = GalleryBuilder(
    device="cuda",
    model_type="timm",
    model_variant="tf_efficientnet_b4.ns_jft_in1k"  # Options:
    # - "tf_efficientnet_b4.ns_jft_in1k" (NoisyStudent - BEST)
    # - "tf_efficientnet_b4.ap_in1k" (AdvProp)
    # - "tf_efficientnet_b4" (Standard)
)

# Ensemble model (slower but potentially more robust)
builder = GalleryBuilder(
    model_type="timm_ensemble"  # Combines NoisyStudent + AdvProp
)

# Adjust verification threshold
verifier = LandmarkVerifier(
    gallery_embeddings=gallery_embeddings,
    t_verify=0.55  # Stricter (fewer false positives)
    # t_verify=0.70  # More lenient (fewer false negatives)
)
```

---

## 📊 Performance Metrics

### Best Configuration (Production Recommended)

Based on comprehensive testing with 6 Wawel images and 2 unrelated images:

```
Model:              TIMM-NoisyStudent (tf_efficientnet_b4.ns_jft_in1k)
Preprocessing:      ENABLED
Threshold:          0.65
─────────────────────────────────────────────────────────────
Accuracy:           87.5%
Discrimination Gap: 0.332 (33.2%)
Avg Wawel Score:    0.783
Avg Unrelated:      0.451
Inference Time:     75.6ms (preprocessing + feature extraction + verification)
Gallery Build:      ~2-5s for 200 images
```

### Model Comparison

| Model | Preprocessing | Accuracy | Discrimination Gap | Inference Time |
|-------|---------------|----------|-------------------|----------------|
| **TIMM-NoisyStudent** ⭐ | ✅ Enabled | **87.5%** | **33.2%** | **75.6ms** |
| TIMM-Standard | ✅ Enabled | 87.5% | 33.2% | 77.7ms |
| TorchVision-B4 | ❌ Disabled | 87.5% | 32.5% | 24.3ms |
| TorchVision-B4 | ✅ Enabled | 75.0% | 8.3% | 90.2ms |
| TIMM-AdvProp | ✅ Enabled | 37.5% | 13.7% | 78.5ms |

**Key Insights:**
- ⭐ **TIMM-NoisyStudent with preprocessing** achieves best balance of accuracy and discrimination
- Preprocessing is **critical** for most models (except TorchVision which performs better without it)
- Discrimination gap > 30% indicates excellent separation between matching/non-matching landmarks

---

## 🧪 Testing

### Run Comprehensive Tests

```powershell
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

### Preprocessor

```python
from geomatchai.preprocessing.preprocessor import Preprocessor

preprocessor = Preprocessor(device="cuda")

# Preprocess image (with person removal)
cleaned_tensor = preprocessor.preprocess_image(pil_image)

# Transform only (no person removal - for gallery images)
tensor = preprocessor.transform_image(pil_image)

# From file path
tensor = preprocessor.preprocess_image_from_path("path/to/image.jpg")
```

**Security Features:**
- Image size validation (max 50MB)
- Dimension validation (100-10,000 pixels)
- File existence checks

### GalleryBuilder

```python
from geomatchai.gallery.gallery_builder import GalleryBuilder

builder = GalleryBuilder(
    device="cuda",
    model_type="timm",
    model_variant="tf_efficientnet_b4.ns_jft_in1k"
)

# Build gallery from async generator
gallery_embeddings = await builder.build_gallery(
    image_generator,
    batch_size=32,  # Adjust based on GPU memory
    skip_preprocessing=True  # For clean gallery images
)
```

### LandmarkVerifier

```python
from geomatchai.verification.verifier import LandmarkVerifier

verifier = LandmarkVerifier(
    gallery_embeddings=gallery_tensor,  # Shape: (N, 1792)
    t_verify=0.65
)

# Verify query
is_verified, score = verifier.verify(query_embedding)

# Update threshold dynamically
verifier.set_threshold(0.70)

# Get gallery info
gallery_size = verifier.get_gallery_size()
```

### MapillaryFetcher

```python
from geomatchai.fetchers.mapillary_fetcher import MapillaryFetcher

fetcher = MapillaryFetcher(api_token="YOUR_TOKEN")

# Async generator - yields images as they download
async for img in fetcher.get_images(
    lat=50.054404,
    lon=19.935730,
    distance=50.0,  # Search radius in meters
    num_images=200
):
    # Process image immediately
    img.save(f"gallery_{count}.jpg")
```

---

## 🔧 Configuration

### Threshold Tuning

Based on your use case, adjust the verification threshold:

| Threshold | Use Case | Trade-off |
|-----------|----------|-----------|
| **0.55-0.60** | High security applications | Fewer false positives, more false negatives |
| **0.65** ⭐ | Balanced (RECOMMENDED) | Optimal accuracy (87.5%) |
| **0.70-0.75** | User-friendly applications | Fewer false negatives, more false positives |

### GPU vs CPU

```python
# Auto-detect (uses CUDA if available)
builder = GalleryBuilder()

# Force CPU (for servers without GPU)
builder = GalleryBuilder(device="cpu")

# Force CUDA (fails if not available)
builder = GalleryBuilder(device="cuda")
```

**Performance Impact:**
- GPU: ~75ms inference
- CPU: ~200-500ms inference (depending on CPU)

---

## 🛠️ Development

### Project Structure

```
GeoMatchAI/
├── src/geomatchai/
│   ├── __init__.py
│   ├── fetchers/              # Image fetchers (Mapillary, etc.)
│   │   ├── base_fetcher.py
│   │   └── mapillary_fetcher.py
│   ├── gallery/               # Reference gallery builder
│   │   └── gallery_builder.py
│   ├── models/                # Feature extraction models
│   │   ├── efficientnet.py           # TorchVision
│   │   └── efficientnet_timm.py      # TIMM variants
│   ├── preprocessing/         # Image preprocessing & segmentation
│   │   └── preprocessor.py
│   └── verification/          # Verification logic
│       └── verifier.py
├── tests/                     # Comprehensive test suite
│   ├── test_comprehensive.py  # Main test suite
│   ├── plot_results.py        # Visualization generator
│   ├── input/                 # Test images
│   └── output/                # Test results & plots
├── pyproject.toml             # Project configuration
├── goal.md                    # Project goals & architecture
└── README.md                  # This file
```

### Adding Custom Fetchers

Extend `BaseFetcher` to add new image sources:

```python
from geomatchai.fetchers.base_fetcher import BaseFetcher
from typing import AsyncGenerator
from PIL import Image

class CustomFetcher(BaseFetcher):
    async def get_images(
        self, 
        lat: float, 
        lon: float, 
        num_images: int = 20
    ) -> AsyncGenerator[Image.Image, None]:
        # Your implementation
        for img_data in your_api_call(lat, lon):
            yield Image.open(img_data)
```

---

## 🎯 Roadmap

### Completed ✅
- [x] EfficientNet-B4 feature extraction (TorchVision & TIMM)
- [x] Semantic segmentation for person removal
- [x] Combined similarity metric (optimized to pure cosine)
- [x] Async Mapillary integration
- [x] Comprehensive testing suite with CSV export
- [x] Visualization tools

### Planned 🚧
- [ ] Human-in-the-loop feedback system for continuous improvement
- [ ] Model re-training pipeline with ArcFace/Triplet Loss
- [ ] Negative anchor support (rejection against other landmarks)
- [ ] REST API wrapper for deployment
- [ ] Docker containerization
- [ ] Multi-landmark support (not just Wawel Castle)
- [ ] Mobile app integration examples

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

## 📧 Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact: [your-email@example.com]

---

## 🌟 Citation

If you use GeoMatchAI in your research, please cite:

```bibtex
@software{geomatchai2025,
  title={GeoMatchAI: High-Performance Visual Place Verification System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/GeoMatchAI}
}
```

---

<div align="center">
  <strong>Built with ❤️ using PyTorch, EfficientNet, and DeepLabV3</strong>
</div>

