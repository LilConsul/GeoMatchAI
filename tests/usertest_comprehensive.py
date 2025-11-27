"""
Comprehensive Test Suite with CSV Export and Timing

Tests all possible combinations:
- 7 test images (5 wawel + 2 unrelated)
- 4 models (torchvision + 3 timm variants)
- 2 preprocessing modes (with/without)
- Timing measurements for each operation

Outputs:
- results_summary.csv: Main results with timing
- results_by_image.csv: Per-image detailed results
- results_by_model.csv: Per-model aggregated results
"""

import asyncio
import csv
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import torch
from PIL import Image

from geomatchai.fetchers.mapillary_fetcher import MapillaryFetcher
from geomatchai.gallery.gallery_builder import GalleryBuilder
from geomatchai.verification.verifier import LandmarkVerifier

# ============================================================================
# Configuration
# ============================================================================

MODEL_CONFIGS = [
    ("torchvision", "b4", "TorchVision-B4"),
    ("timm", "tf_efficientnet_b4.ns_jft_in1k", "TIMM-NoisyStudent"),
    ("timm", "tf_efficientnet_b4.ap_in1k", "TIMM-AdvProp"),
    ("timm", "tf_efficientnet_b4", "TIMM-Standard"),
]

THRESHOLD = 0.65
GALLERY_SIZE = 200
OUTPUT_DIR = Path(__file__).parent / "output" / "csv"  # Relative to test file, not CWD


# ============================================================================
# Test Images Configuration
# ============================================================================


def get_test_images() -> list[tuple[str, str, bool]]:
    """
    Return list of test images with metadata.
    Returns: List of (image_path, image_name, is_wawel)
    """
    base_path = Path(__file__).parent / "input"

    test_images = []

    # Wawel images (related - should PASS)
    wawel_dir = base_path / "wawel"
    wawel_images = [
        "test.png",
        "wawel1.jpg",
        "wawel2.jpg",
        "wawel3.jpg",
        "wawel4.jpg",
        "wawel5.jpg",
    ]

    for img_name in wawel_images:
        img_path = wawel_dir / img_name
        if img_path.exists():
            test_images.append((str(img_path), f"wawel_{img_name}", True))

    # Unrelated images (should REJECT)
    unrelated_images = [
        "photo_2024-07-21_12-07-58.jpg",
        "photo_2025-11-21_10-07-59.jpg",
    ]

    for img_name in unrelated_images:
        img_path = base_path / img_name
        if img_path.exists():
            test_images.append((str(img_path), f"unrelated_{img_name}", False))

    return test_images


# ============================================================================
# Test Execution
# ============================================================================


async def test_single_configuration(
    model_type: str,
    model_variant: str,
    model_name: str,
    gallery_images: list[Image.Image],
    test_image_path: str,
    test_image_name: str,
    is_wawel: bool,
    use_preprocessing: bool,
    device: str,
) -> dict:
    """Test a single configuration and return detailed results with timing."""

    results = {
        "model_type": model_type,
        "model_variant": model_variant,
        "model_name": model_name,
        "image_name": test_image_name,
        "is_wawel": is_wawel,
        "preprocessing": use_preprocessing,
    }

    try:
        # Time: Gallery build
        t_start = time.time()

        builder = GalleryBuilder(device=device, model_type=model_type, model_variant=model_variant)

        async def image_gen() -> AsyncGenerator[Image.Image]:
            for img in gallery_images:
                yield img

        gallery_embeddings = await builder.build_gallery(image_gen(), skip_preprocessing=True)

        t_gallery_build = time.time() - t_start

        # Initialize verifier
        verifier = LandmarkVerifier(gallery_embeddings, t_verify=THRESHOLD)

        # Load test image
        test_image = Image.open(test_image_path).convert("RGB")

        # Time: Preprocessing/Transform
        t_start = time.time()
        if use_preprocessing:
            tensor = builder.preprocessor.preprocess_image(test_image)
        else:
            tensor = builder.preprocessor.transform_image(test_image)
        t_preprocess = time.time() - t_start

        # Time: Feature extraction
        t_start = time.time()
        with torch.no_grad():
            embedding = builder.feature_extractor(tensor.unsqueeze(0).to(device))
        t_feature_extract = time.time() - t_start

        # Time: Verification
        t_start = time.time()
        is_verified, similarity_score = verifier.verify(embedding)
        t_verify = time.time() - t_start

        # Calculate total time
        t_total = t_gallery_build + t_preprocess + t_feature_extract + t_verify

        # Determine if result is correct
        expected_result = is_wawel  # Wawel should verify, unrelated should not
        is_correct = is_verified == expected_result

        results.update(
            {
                "similarity_score": similarity_score,
                "is_verified": is_verified,
                "expected_verified": expected_result,
                "is_correct": is_correct,
                "gallery_size": gallery_embeddings.shape[0],
                "time_gallery_build_s": t_gallery_build,
                "time_preprocess_s": t_preprocess,
                "time_feature_extract_s": t_feature_extract,
                "time_verify_s": t_verify,
                "time_total_s": t_total,
                "time_inference_s": t_preprocess
                + t_feature_extract
                + t_verify,  # Without gallery build
                "error": None,
            }
        )

        # Clean up memory
        del builder
        del gallery_embeddings
        del embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        results.update(
            {
                "similarity_score": 0.0,
                "is_verified": False,
                "expected_verified": is_wawel,
                "is_correct": False,
                "gallery_size": 0,
                "time_gallery_build_s": 0.0,
                "time_preprocess_s": 0.0,
                "time_feature_extract_s": 0.0,
                "time_verify_s": 0.0,
                "time_total_s": 0.0,
                "time_inference_s": 0.0,
                "error": str(e),
            }
        )

    return results


# ============================================================================
# CSV Export Functions
# ============================================================================


def save_results_to_csv(all_results: list[dict], output_dir: Path):
    """Save all results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main results file (all data)
    csv_path = output_dir / "results_summary.csv"
    fieldnames = [
        "model_name",
        "model_type",
        "model_variant",
        "image_name",
        "is_wawel",
        "preprocessing",
        "similarity_score",
        "is_verified",
        "expected_verified",
        "is_correct",
        "gallery_size",
        "threshold",
        "time_gallery_build_s",
        "time_preprocess_s",
        "time_feature_extract_s",
        "time_verify_s",
        "time_inference_s",
        "time_total_s",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            row = result.copy()
            row["threshold"] = THRESHOLD
            writer.writerow(row)

    print(f"✅ Saved: {csv_path}")

    # 2. Per-image aggregated results
    csv_path = output_dir / "results_by_image.csv"
    image_stats = {}

    for result in all_results:
        key = (result["image_name"], result["preprocessing"])
        if key not in image_stats:
            image_stats[key] = {
                "image_name": result["image_name"],
                "is_wawel": result["is_wawel"],
                "preprocessing": result["preprocessing"],
                "scores": [],
                "correct_count": 0,
                "total_count": 0,
            }

        image_stats[key]["scores"].append(result["similarity_score"])
        if result["is_correct"]:
            image_stats[key]["correct_count"] += 1
        image_stats[key]["total_count"] += 1

    fieldnames = [
        "image_name",
        "is_wawel",
        "preprocessing",
        "avg_score",
        "min_score",
        "max_score",
        "std_score",
        "accuracy",
        "correct_count",
        "total_count",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in image_stats.values():
            import statistics

            scores = stats["scores"]
            row = {
                "image_name": stats["image_name"],
                "is_wawel": stats["is_wawel"],
                "preprocessing": stats["preprocessing"],
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "std_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                "accuracy": stats["correct_count"] / stats["total_count"]
                if stats["total_count"] > 0
                else 0,
                "correct_count": stats["correct_count"],
                "total_count": stats["total_count"],
            }
            writer.writerow(row)

    print(f"✅ Saved: {csv_path}")

    # 3. Per-model aggregated results
    csv_path = output_dir / "results_by_model.csv"
    model_stats = {}

    for result in all_results:
        key = (result["model_name"], result["preprocessing"])
        if key not in model_stats:
            model_stats[key] = {
                "model_name": result["model_name"],
                "preprocessing": result["preprocessing"],
                "wawel_scores": [],
                "unrelated_scores": [],
                "correct_count": 0,
                "total_count": 0,
                "time_inference": [],
            }

        if result["is_wawel"]:
            model_stats[key]["wawel_scores"].append(result["similarity_score"])
        else:
            model_stats[key]["unrelated_scores"].append(result["similarity_score"])

        if result["is_correct"]:
            model_stats[key]["correct_count"] += 1
        model_stats[key]["total_count"] += 1
        model_stats[key]["time_inference"].append(result["time_inference_s"])

    fieldnames = [
        "model_name",
        "preprocessing",
        "avg_wawel_score",
        "avg_unrelated_score",
        "discrimination_gap",
        "accuracy",
        "correct_count",
        "total_count",
        "avg_inference_time_s",
        "min_inference_time_s",
        "max_inference_time_s",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in model_stats.values():
            wawel_avg = (
                sum(stats["wawel_scores"]) / len(stats["wawel_scores"])
                if stats["wawel_scores"]
                else 0
            )
            unrelated_avg = (
                sum(stats["unrelated_scores"]) / len(stats["unrelated_scores"])
                if stats["unrelated_scores"]
                else 0
            )

            row = {
                "model_name": stats["model_name"],
                "preprocessing": stats["preprocessing"],
                "avg_wawel_score": wawel_avg,
                "avg_unrelated_score": unrelated_avg,
                "discrimination_gap": wawel_avg - unrelated_avg,
                "accuracy": stats["correct_count"] / stats["total_count"]
                if stats["total_count"] > 0
                else 0,
                "correct_count": stats["correct_count"],
                "total_count": stats["total_count"],
                "avg_inference_time_s": sum(stats["time_inference"]) / len(stats["time_inference"])
                if stats["time_inference"]
                else 0,
                "min_inference_time_s": min(stats["time_inference"])
                if stats["time_inference"]
                else 0,
                "max_inference_time_s": max(stats["time_inference"])
                if stats["time_inference"]
                else 0,
            }
            writer.writerow(row)

    print(f"✅ Saved: {csv_path}")


# ============================================================================
# Main Test Runner
# ============================================================================


async def main():
    """Main test execution."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 100)
    print("🧪 COMPREHENSIVE TEST SUITE WITH CSV EXPORT".center(100))
    print("=" * 100)
    print("\n📋 Configuration:")
    print(f"   Device: {device}")
    print(f"   Threshold: {THRESHOLD}")
    print(f"   Gallery Size: {GALLERY_SIZE} images")
    print(f"   Models: {len(MODEL_CONFIGS)}")

    # Get API token
    mapillary_token = os.getenv("MAPILLARY_API_KEY")
    if not mapillary_token:
        raise ValueError(
            "MAPILLARY_API_KEY environment variable not set.\n"
            "Get your token from: https://www.mapillary.com/dashboard/developers"
        )

    # Fetch gallery images (once)
    print("\n📥 Fetching gallery images from Mapillary...")
    t_start_gallery = time.time()
    fetcher = MapillaryFetcher(mapillary_token)
    lat, lon = 50.054404, 19.935730

    gallery_images = []
    async for img in fetcher.get_images(lat, lon, num_images=GALLERY_SIZE, distance=100):
        gallery_images.append(img)

    t_gallery_fetch = time.time() - t_start_gallery
    print(f"✅ Cached {len(gallery_images)} images (took {t_gallery_fetch:.2f}s)")

    # Get test images
    test_images = get_test_images()
    print(f"\n📸 Test Images: {len(test_images)}")
    for _img_path, img_name, is_wawel in test_images:
        status = "✓ Wawel" if is_wawel else "✗ Unrelated"
        print(f"   {status:<15} {img_name}")

    # Calculate total tests
    total_tests = len(MODEL_CONFIGS) * len(test_images) * 2  # 2 preprocessing modes
    print(f"\n🔢 Total test configurations: {total_tests}")
    print(f"   ({len(MODEL_CONFIGS)} models × {len(test_images)} images × 2 preprocessing modes)")

    # Run all tests
    print("\n🚀 Starting tests...")
    all_results = []
    test_count = 0
    t_start_all = time.time()

    for model_type, model_variant, model_name in MODEL_CONFIGS:
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_name}")
        print(f"{'=' * 80}")

        for img_path, img_name, is_wawel in test_images:
            for use_preprocessing in [True, False]:
                test_count += 1
                prep_str = "WITH" if use_preprocessing else "WITHOUT"

                print(
                    f"  [{test_count}/{total_tests}] {img_name} ({prep_str} preprocessing)...",
                    end=" ",
                )

                result = await test_single_configuration(
                    model_type,
                    model_variant,
                    model_name,
                    gallery_images,
                    img_path,
                    img_name,
                    is_wawel,
                    use_preprocessing,
                    device,
                )

                all_results.append(result)

                # Print result
                if result["error"]:
                    print(f"❌ ERROR: {result['error'][:50]}")
                else:
                    status = "✅" if result["is_correct"] else "❌"
                    print(
                        f"{status} Score: {result['similarity_score']:.4f}, Time: {result['time_inference_s']:.3f}s"
                    )

    t_total_all = time.time() - t_start_all

    # Save results to CSV
    print(f"\n{'=' * 100}")
    print("💾 SAVING RESULTS TO CSV FILES")
    print(f"{'=' * 100}")
    save_results_to_csv(all_results, OUTPUT_DIR)

    # Print summary statistics
    print(f"\n{'=' * 100}")
    print("📊 SUMMARY STATISTICS")
    print(f"{'=' * 100}")

    total_correct = sum(1 for r in all_results if r["is_correct"])
    accuracy = total_correct / len(all_results) * 100 if all_results else 0

    print(f"\n✅ Tests completed: {len(all_results)}")
    print(f"✅ Overall accuracy: {accuracy:.1f}% ({total_correct}/{len(all_results)})")
    print(f"⏱️  Total test time: {t_total_all:.2f}s")
    print(f"⏱️  Average per test: {t_total_all / len(all_results):.2f}s")

    # Best model by accuracy
    model_accuracy = {}
    for result in all_results:
        key = result["model_name"]
        if key not in model_accuracy:
            model_accuracy[key] = {"correct": 0, "total": 0}
        if result["is_correct"]:
            model_accuracy[key]["correct"] += 1
        model_accuracy[key]["total"] += 1

    print("\n🏆 Model Accuracy Rankings:")
    sorted_models = sorted(
        model_accuracy.items(), key=lambda x: x[1]["correct"] / x[1]["total"], reverse=True
    )
    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        acc = stats["correct"] / stats["total"] * 100
        print(f"   {rank}. {model_name:<30} {acc:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\n{'=' * 100}")
    print("✅ ALL TESTS COMPLETED!".center(100))
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    asyncio.run(main())
