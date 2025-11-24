"""
Comprehensive Model Comparison Test Suite

Tests all model variants and preprocessing combinations:
- torchvision: EfficientNet-B4 (ImageNet)
- timm: NoisyStudent, AdvProp, Standard
- Preprocessing: WITH (person removal) and WITHOUT (raw)

Displays results in user-friendly tables with clear recommendations.
"""
import asyncio
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Tuple

import torch
from PIL import Image

from geomatchai.fetchers.mapillary_fetcher import MapillaryFetcher
from geomatchai.gallery.gallery_builder import GalleryBuilder
from geomatchai.verification.verifier import LandmarkVerifier


# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIGS = [
    # (model_type, variant, display_name)
    ("torchvision", "b4", "TorchVision EfficientNet-B4"),
    ("timm", "tf_efficientnet_b4.ns_jft_in1k", "TIMM NoisyStudent"),
    ("timm", "tf_efficientnet_b4.ap_in1k", "TIMM AdvProp"),
    ("timm", "tf_efficientnet_b4", "TIMM Standard"),
]

THRESHOLD = 0.65
GALLERY_SIZE = 100


# ============================================================================
# Test Functions
# ============================================================================

async def test_single_model(
    model_type: str,
    model_variant: str,
    gallery_images: List[Image.Image],
    query_image: Image.Image,
    unrelated_image: Image.Image,
    device: str
) -> Dict:
    """Test one model with all preprocessing combinations."""

    # Initialize builder
    builder = GalleryBuilder(
        device=device,
        model_type=model_type,
        model_variant=model_variant
    )

    # Build gallery from cached images
    async def image_gen() -> AsyncGenerator[Image.Image, None]:
        for img in gallery_images:
            yield img

    gallery_embeddings = await builder.build_gallery(
        image_gen(),
        skip_preprocessing=True
    )

    # Initialize verifier
    verifier = LandmarkVerifier(gallery_embeddings, t_verify=THRESHOLD)

    results = {}

    # Test 4 combinations: (query_type, preprocessing_enabled)
    test_cases = [
        ("wawel", True, query_image),
        ("wawel", False, query_image),
        ("unrelated", True, unrelated_image),
        ("unrelated", False, unrelated_image),
    ]

    for img_type, use_prep, image in test_cases:
        # Preprocess or just transform
        if use_prep:
            tensor = builder.preprocessor.preprocess_image(image)
            prep_str = "with_prep"
        else:
            tensor = builder.preprocessor.transform_image(image)
            prep_str = "no_prep"

        # Extract features and verify
        with torch.no_grad():
            embedding = builder.feature_extractor(tensor.unsqueeze(0).to(device))

        is_verified, score = verifier.verify(embedding)

        # Store results
        key = f"{img_type}_{prep_str}"
        results[key] = {
            'score': score,
            'verified': is_verified
        }

    # Calculate derived metrics
    results['gap_with_prep'] = results['wawel_with_prep']['score'] - results['unrelated_with_prep']['score']
    results['gap_no_prep'] = results['wawel_no_prep']['score'] - results['unrelated_no_prep']['score']
    results['prep_impact_wawel'] = results['wawel_no_prep']['score'] - results['wawel_with_prep']['score']
    results['prep_impact_unrelated'] = results['unrelated_no_prep']['score'] - results['unrelated_with_prep']['score']
    results['gallery_size'] = gallery_embeddings.shape[0]

    return results


def print_header(title: str, width: int = 100):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_summary_table(all_results: Dict[str, Dict], display_names: Dict[str, str]):
    """Print comparison table with preprocessing ON."""
    print_header("📊 MODEL COMPARISON: WITH PREPROCESSING (Person Removal)")

    print(f"\n{'Model':<30} {'Wawel':<10} {'Unrelated':<10} {'Gap':<10} {'Pass':<6}")
    print("-" * 70)

    best_gap = 0
    best_model = None

    for model_key, results in all_results.items():
        if results is None:
            print(f"{display_names[model_key]:<30} {'FAILED':<10}")
            continue

        wawel = results['wawel_with_prep']['score']
        unrel = results['unrelated_with_prep']['score']
        gap = results['gap_with_prep']

        # Check if passes all tests
        wawel_pass = results['wawel_with_prep']['verified']
        unrel_reject = not results['unrelated_with_prep']['verified']
        pass_status = "✅" if (wawel_pass and unrel_reject) else "❌"

        print(f"{display_names[model_key]:<30} {wawel:<10.4f} {unrel:<10.4f} {gap:<10.4f} {pass_status:<6}")

        if gap > best_gap:
            best_gap = gap
            best_model = display_names[model_key]

    print("-" * 70)
    if best_model:
        print(f"🏆 Best: {best_model} (Gap: {best_gap:.4f} = {best_gap*100:.1f}%)")


def print_preprocessing_comparison(all_results: Dict[str, Dict], display_names: Dict[str, str]):
    """Print detailed preprocessing impact analysis."""
    print_header("🔬 PREPROCESSING IMPACT ANALYSIS")

    print(f"\n{'Model':<30} {'Scenario':<15} {'Score':<10} {'Gap':<10} {'Impact':<12}")
    print("-" * 80)

    for model_key, results in all_results.items():
        if results is None:
            continue

        model_name = display_names[model_key]

        # WITH preprocessing
        wawel_prep = results['wawel_with_prep']['score']
        unrel_prep = results['unrelated_with_prep']['score']
        gap_prep = results['gap_with_prep']

        # WITHOUT preprocessing
        wawel_no_prep = results['wawel_no_prep']['score']
        unrel_no_prep = results['unrelated_no_prep']['score']
        gap_no_prep = results['gap_no_prep']

        # Calculate impacts
        impact_wawel = results['prep_impact_wawel']
        impact_unrel = results['prep_impact_unrelated']

        # Print WITH preprocessing
        print(f"{model_name:<30} {'WITH prep':<15} {wawel_prep:<10.4f} {gap_prep:<10.4f} {impact_wawel:+.4f} ({impact_wawel/wawel_no_prep*100:+.1f}%)" if wawel_no_prep > 0 else f"{model_name:<30} {'WITH prep':<15} {wawel_prep:<10.4f} {gap_prep:<10.4f}")

        # Print WITHOUT preprocessing
        print(f"{'':30} {'WITHOUT prep':<15} {wawel_no_prep:<10.4f} {gap_no_prep:<10.4f}")
        print("-" * 80)


def print_detailed_results(all_results: Dict[str, Dict], display_names: Dict[str, str]):
    """Print complete detailed results for all models."""
    print_header("📋 DETAILED RESULTS: ALL TEST CASES")

    for model_key, results in all_results.items():
        if results is None:
            continue

        print(f"\n🔹 {display_names[model_key]}")
        print(f"   Gallery Size: {results['gallery_size']} images")
        print()

        # WITH Preprocessing
        print("   WITH Preprocessing (Person Removal):")
        print(f"      Wawel Query:    {results['wawel_with_prep']['score']:.4f} {'✅ PASS' if results['wawel_with_prep']['verified'] else '❌ FAIL'}")
        print(f"      Unrelated:      {results['unrelated_with_prep']['score']:.4f} {'✅ REJECT' if not results['unrelated_with_prep']['verified'] else '❌ ACCEPT'}")
        print(f"      Gap:            {results['gap_with_prep']:.4f} ({results['gap_with_prep']*100:.1f}%)")
        print()

        # WITHOUT Preprocessing
        print("   WITHOUT Preprocessing (Raw Images):")
        print(f"      Wawel Query:    {results['wawel_no_prep']['score']:.4f}")
        print(f"      Unrelated:      {results['unrelated_no_prep']['score']:.4f}")
        print(f"      Gap:            {results['gap_no_prep']:.4f} ({results['gap_no_prep']*100:.1f}%)")
        print()

        # Impact
        wawel_impact = results['prep_impact_wawel']
        wawel_no_prep_score = results['wawel_no_prep']['score']
        if wawel_no_prep_score > 0:
            wawel_impact_pct = (wawel_impact / wawel_no_prep_score) * 100
            impact_indicator = "📉" if wawel_impact > 0 else "📈"
            print(f"   Preprocessing Impact: {impact_indicator} {wawel_impact:+.4f} ({wawel_impact_pct:+.1f}%)")


def print_recommendations(all_results: Dict[str, Dict], display_names: Dict[str, str]):
    """Print actionable recommendations based on test results."""
    print_header("💡 RECOMMENDATIONS & DIAGNOSIS")

    # Find best model
    best_gap = 0
    best_model_key = None
    best_model_name = None

    for model_key, results in all_results.items():
        if results is None:
            continue
        gap = results['gap_with_prep']
        if gap > best_gap:
            best_gap = gap
            best_model_key = model_key
            best_model_name = display_names[model_key]

    # Overall assessment
    print()
    if best_gap < 0.05:
        print("❌ CRITICAL: Poor discrimination across all models (gap < 5%)")
        print("\n   Action Items:")
        print("   1. Use Google Landmarks pre-trained models")
        print("   2. Add GeM (Generalized Mean) pooling layer")
        print("   3. Fine-tune on landmark recognition dataset")
        print("   4. Increase gallery size to 200+ images")

    elif best_gap < 0.15:
        print("⚠️  WARNING: Weak discrimination (gap 5-15%)")
        print(f"\n   Best Model: {best_model_name} (Gap: {best_gap:.1%})")
        print("\n   Action Items:")
        print("   1. Lower threshold from 0.65 to 0.50-0.55")
        print("   2. Try ensemble approach (slower but more robust)")
        print("   3. Use multiple reference galleries per landmark")

    elif best_gap < 0.30:
        print("✅ GOOD: Acceptable discrimination (gap 15-30%)")
        print(f"\n   🏆 Recommended Model: {best_model_name}")
        print(f"   📊 Discrimination Gap: {best_gap:.1%}")
        print("\n   Production Settings:")
        print(f"      Model Type: {best_model_key.split('_')[0]}")
        print(f"      Variant: {best_model_key.split('_', 1)[1]}")
        print(f"      Threshold: {THRESHOLD}")
        print("      Preprocessing: ENABLE for user queries")

    else:
        print("🎯 EXCELLENT: Strong discrimination (gap > 30%)")
        print(f"\n   🏆 Recommended Model: {best_model_name}")
        print(f"   📊 Discrimination Gap: {best_gap:.1%}")
        print("\n   ✅ System is production-ready!")
        print("\n   Production Settings:")
        print(f"      Model Type: {best_model_key.split('_')[0]}")
        print(f"      Variant: {best_model_key.split('_', 1)[1]}")
        print(f"      Threshold: {THRESHOLD}")
        print("      Preprocessing: ENABLE for user queries")

    # Model-specific insights
    if best_model_key and all_results.get(best_model_key):
        print("\n   Model-Specific Insights:")
        results = all_results[best_model_key]

        # Preprocessing impact
        impact = results['prep_impact_wawel']
        impact_pct = (impact / results['wawel_no_prep']['score'] * 100) if results['wawel_no_prep']['score'] > 0 else 0

        if impact < -0.02:  # Preprocessing helps (score increases)
            print(f"      ✅ Preprocessing HELPS this model (+{-impact:.1%})")
        elif impact > 0.05:  # Preprocessing hurts significantly
            print(f"      ⚠️  Preprocessing HURTS this model (-{impact:.1%})")
            print("         Consider using raw images or detection-based approach")
        else:
            print("      ➡️  Preprocessing has minimal impact")


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Main test execution."""
    # Suppress HuggingFace symlink warning
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Starting Comprehensive Model Test Suite")
    print(f"   Device: {device}")
    print(f"   Threshold: {THRESHOLD}")
    print(f"   Gallery Size: {GALLERY_SIZE} images")

    # Get API token
    mapillary_token = os.getenv("MAPILLARY_API_KEY")
    if not mapillary_token:
        raise ValueError(
            "MAPILLARY_API_KEY environment variable not set.\n"
            "Get your token from: https://www.mapillary.com/dashboard/developers"
        )

    # Fetch gallery images
    print("\n📥 Fetching gallery images from Mapillary...")
    fetcher = MapillaryFetcher(mapillary_token)
    lat, lon = 50.054404, 19.935730  # Wawel Castle

    gallery_images = []
    async for img in fetcher.get_images(lat, lon, num_images=GALLERY_SIZE):
        gallery_images.append(img)
    print(f"✅ Cached {len(gallery_images)} gallery images")

    # Load test images
    test_image_path = Path(__file__).parent / "input" / "wawel" / "test.png"
    query_image = Image.open(test_image_path).convert("RGB")

    unrelated_image_path = Path(__file__).parent / "input" / "photo_2025-11-21_10-07-59.jpg"
    unrelated_image = Image.open(unrelated_image_path).convert("RGB")

    # Run tests for all models
    print(f"\n🧪 Testing {len(MODEL_CONFIGS)} model configurations...")
    all_results = {}
    display_names = {}

    for model_type, variant, display_name in MODEL_CONFIGS:
        model_key = f"{model_type}_{variant}"
        display_names[model_key] = display_name

        print(f"\n   Testing {display_name}...")
        try:
            results = await test_single_model(
                model_type, variant, gallery_images,
                query_image, unrelated_image, device
            )
            all_results[model_key] = results
            print(f"   ✅ Complete")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_results[model_key] = None

    # Display results
    print_summary_table(all_results, display_names)
    print_preprocessing_comparison(all_results, display_names)
    print_detailed_results(all_results, display_names)
    print_recommendations(all_results, display_names)

    print("\n" + "="*100)
    print("✅ Test suite completed!".center(100))
    print("="*100 + "\n")



if __name__ == "__main__":
    asyncio.run(main())
