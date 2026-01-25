"""
Comprehensive Test Suite with CSV Export and Timing

Automatically discovers and tests all image folders in ./input/:
- Folders with cord.txt: Uses coordinates to fetch gallery images from Mapillary
- Folders without cord.txt (e.g., 'unrelated'): Uses as negative test cases
- Tests all models with/without preprocessing
- Exports detailed CSV results with timing

Outputs (in ./output/csv/):
- results_summary.csv: Main results with timing
- results_by_image.csv: Per-image detailed results
- results_by_model.csv: Per-model aggregated results
- results_by_landmark.csv: Per-landmark aggregated results
"""

import asyncio
import csv
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from geomatchai.fetchers.mapillary_fetcher import MapillaryFetcher
from geomatchai.gallery.gallery_builder import GalleryBuilder
from geomatchai.verification.verifier import LandmarkVerifier


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LandmarkTestCase:
    """Represents a landmark folder with test images and gallery coordinates."""

    name: str
    folder_path: Path
    has_coordinates: bool
    lat: float | None = None
    lon: float | None = None
    test_images: list[Path] = None

    def __post_init__(self):
        if self.test_images is None:
            self.test_images = []


MODEL_CONFIGS = [
    # ----------------- TorchVision classic CNNs -----------------
    ("torchvision", "resnet50", "ResNet50"),
    ("torchvision", "resnet101", "ResNet101"),
    ("torchvision", "resnet152", "ResNet152"),
    ("torchvision", "densenet121", "DenseNet121"),
    ("torchvision", "densenet169", "DenseNet169"),
    ("torchvision", "mobilenet_v3_large", "MobileNetV3-Large"),
    ("torchvision", "inception_v3", "InceptionV3"),
    # ----------------- TIMM EfficientNets -----------------
    ("timm", "tf_efficientnet_b4", "TIMM-Standard"),
    ("timm", "tf_efficientnet_b4.ap_in1k", "TIMM-AdvProp"),
    ("timm", "tf_efficientnet_b4.ns_jft_in1k", "TIMM-NoisyStudent"),
    ("timm", "tf_efficientnet_b5", "TIMM-EfficientNetB5"),
    ("timm", "tf_efficientnet_b6", "TIMM-EfficientNetB6"),
    # ----------------- TIMM modern CNNs -----------------
    ("timm", "resnest50d", "ResNeSt50"),
    ("timm", "resnest101e", "ResNeSt101"),
    ("timm", "regnety_040", "RegNetY-040"),
    ("timm", "regnety_080", "RegNetY-080"),
    ("timm", "convnext_base", "ConvNeXt-Base"),
    ("timm", "convnext_large", "ConvNeXt-Large"),
    ("timm", "dm_nfnet_f0", "NFNet-F0"),
    # ----------------- TIMM Vision Transformers -----------------
    ("timm", "vit_base_patch16_224", "ViT-Base"),
    ("timm", "vit_large_patch16_224", "ViT-Large"),
    ("timm", "deit_base_distilled_patch16_224", "DeiT-Base"),
    ("timm", "swin_base_patch4_window7_224", "Swin-Base"),
    ("timm", "swin_large_patch4_window7_224", "Swin-Large"),
    # ----------------- CLIP embeddings -----------------
    ("timm", "clip_vit_b32", "CLIP-ViT-B32"),
    ("timm", "clip_vit_b16", "CLIP-ViT-B16"),
    ("timm", "clip_rn50", "CLIP-RN50"),
]

THRESHOLD = 0.65
GALLERY_SIZE = 200
GALLERY_DISTANCE = 100  # meters
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output" / "csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================================
# TUI Output Formatting
# ============================================================================


class TUIFormatter:
    """Clean TUI-style output formatting."""

    # Box drawing characters
    H_LINE = "-"
    V_LINE = "|"
    CORNER_TL = "+"
    CORNER_TR = "+"
    CORNER_BL = "+"
    CORNER_BR = "+"
    T_LEFT = "+"
    T_RIGHT = "+"
    T_TOP = "+"
    T_BOTTOM = "+"
    CROSS = "+"

    # Progress bar characters
    PROGRESS_FULL = "#"
    PROGRESS_EMPTY = "-"

    # Width settings
    WIDTH = 100

    @staticmethod
    def header(title: str, char: str = "=") -> str:
        """Create a centered header."""
        return f"\n{char * TUIFormatter.WIDTH}\n{title.center(TUIFormatter.WIDTH)}\n{char * TUIFormatter.WIDTH}"

    @staticmethod
    def subheader(title: str, char: str = "-") -> str:
        """Create a subheader."""
        return f"\n{char * TUIFormatter.WIDTH}\n{title}\n{char * TUIFormatter.WIDTH}"

    @staticmethod
    def progress_bar(current: int, total: int, width: int = 20) -> str:
        """Create a text progress bar."""
        if total == 0:
            return "[" + TUIFormatter.PROGRESS_EMPTY * width + "]"
        progress = current / total
        filled = int(width * progress)
        return (
            "["
            + TUIFormatter.PROGRESS_FULL * filled
            + TUIFormatter.PROGRESS_EMPTY * (width - filled)
            + "]"
        )

    @staticmethod
    def score_bar(score: float, width: int = 15) -> str:
        """Create a visual score bar."""
        filled = int(width * score)
        return TUIFormatter.PROGRESS_FULL * filled + TUIFormatter.PROGRESS_EMPTY * (width - filled)

    @staticmethod
    def table_header(columns: list[tuple[str, int]]) -> str:
        """Create a table header row."""
        parts = []
        for name, width in columns:
            parts.append(name.center(width))
        line = TUIFormatter.V_LINE + TUIFormatter.V_LINE.join(parts) + TUIFormatter.V_LINE
        separator = (
            TUIFormatter.CORNER_TL
            + TUIFormatter.T_TOP.join(TUIFormatter.H_LINE * w for _, w in columns)
            + TUIFormatter.CORNER_TR
        )
        return (
            separator
            + "\n"
            + line
            + "\n"
            + separator.replace(TUIFormatter.CORNER_TL, TUIFormatter.T_LEFT).replace(
                TUIFormatter.CORNER_TR, TUIFormatter.T_RIGHT
            )
        )

    @staticmethod
    def table_row(values: list[tuple[str, int]]) -> str:
        """Create a table row."""
        parts = []
        for value, width in values:
            parts.append(value.center(width))
        return TUIFormatter.V_LINE + TUIFormatter.V_LINE.join(parts) + TUIFormatter.V_LINE

    @staticmethod
    def table_footer(widths: list[int]) -> str:
        """Create a table footer."""
        return (
            TUIFormatter.CORNER_BL
            + TUIFormatter.T_BOTTOM.join(TUIFormatter.H_LINE * w for w in widths)
            + TUIFormatter.CORNER_BR
        )


def print_model_header(model_name: str, model_idx: int, total_models: int):
    """Print a prominent header for each model."""
    progress = TUIFormatter.progress_bar(model_idx, total_models, 30)
    print(TUIFormatter.header(f"MODEL [{model_idx}/{total_models}]: {model_name}"))
    print(f"  Progress: {progress} {model_idx}/{total_models} models")


def print_landmark_section_header(landmark_name: str, num_images: int, is_related: bool):
    """Print header for a landmark test section."""
    relation = "RELATED" if is_related else "UNRELATED"
    print(f"\n  {TUIFormatter.H_LINE * 100}")
    print(f"  TESTING: {landmark_name:<20} | Images: {num_images:<3} | Type: {relation}")
    print(f"  {TUIFormatter.H_LINE * 100}")

    header = (
        f"  {'#':>6}|"
        f"{'%':>6} |"
        f"{'STATUS':^8}|"
        f"{'PREP':^5}|"
        f"{'SCORE BAR':^17}|"
        f"{'SCORE':^8}|"
        f"{'TIME':^8}|"
        f" {'IMAGE':<28}"
    )
    print(header)
    print(f"  {TUIFormatter.H_LINE * 100}")


def print_test_result(
    test_count: int,
    total_tests: int,
    image_name: str,
    preprocessing: bool,
    result: dict,
):
    """Print a formatted test result line."""
    # Calculate progress
    progress = (test_count / total_tests * 100) if total_tests > 0 else 0

    # Status
    if result.get("error"):
        status = "ERROR"
    elif result.get("is_correct"):
        status = "PASS"
    else:
        status = "FAIL"

    # Preprocessing indicator
    prep = "ON" if preprocessing else "OFF"

    # Score and time
    score = result.get("similarity_score", 0)
    score_bar = TUIFormatter.score_bar(score, 15)
    time_ms = result.get("time_inference_s", 0) * 1000

    # Truncate image name if too long
    img_display = image_name[:28] if len(image_name) > 28 else image_name

    # Format row - must match header widths exactly
    row = (
        f"  {test_count:6d}|"
        f"{progress:5.1f}% |"
        f"{status:^8}|"
        f"{prep:^5}|"
        f" {score_bar} |"
        f" {score:.4f} |"
        f"{time_ms:6.1f}ms|"
        f" {img_display:<28}"
    )
    print(row)


def print_landmark_section_footer(pass_count: int, fail_count: int, error_count: int):
    """Print footer with section statistics."""
    total = pass_count + fail_count + error_count
    pass_rate = (pass_count / total * 100) if total > 0 else 0
    print(f"  {TUIFormatter.H_LINE * 100}")
    print(
        f"  SECTION SUMMARY: PASS={pass_count} FAIL={fail_count} ERROR={error_count} | Pass Rate: {pass_rate:.1f}%"
    )


def print_live_stats(all_results: list[dict], elapsed_time: float):
    """Print live running statistics."""
    if not all_results:
        return

    correct = sum(1 for r in all_results if r.get("is_correct"))
    total = len(all_results)
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_time = sum(r.get("time_inference_s", 0) for r in all_results) / total

    print(
        f"\n  [RUNNING STATS] Tests: {total} | Accuracy: {accuracy:.1f}% | Avg Time: {avg_time * 1000:.1f}ms | Elapsed: {elapsed_time:.1f}s"
    )


# ============================================================================
# Folder Discovery and Test Case Loading
# ============================================================================


def parse_coordinates(cord_file: Path) -> tuple[float, float] | None:
    """
    Parse coordinates from cord.txt file.
    Expected format: "lat, lon" on first line.
    Returns: (lat, lon) tuple or None if file doesn't exist/parse error
    """
    try:
        if not cord_file.exists():
            return None

        content = cord_file.read_text().strip()
        lat_str, lon_str = content.split(",")
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        return lat, lon
    except Exception as e:
        print(f"Warning: Could not parse {cord_file}: {e}")
        return None


def discover_test_cases(input_dir: Path) -> list[LandmarkTestCase]:
    """
    Automatically discover all landmark test cases from input directory.

    Returns:
        List of LandmarkTestCase objects, one per folder in input_dir
    """
    test_cases = []

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return test_cases

    # Iterate through all subdirectories
    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir():
            continue

        landmark_name = folder.name
        cord_file = folder / "cord.txt"

        # Parse coordinates if cord.txt exists
        coordinates = parse_coordinates(cord_file)
        has_coordinates = coordinates is not None
        lat, lon = coordinates if has_coordinates else (None, None)

        # Find all image files in folder
        test_images = []
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                test_images.append(file)

        # Create test case
        test_case = LandmarkTestCase(
            name=landmark_name,
            folder_path=folder,
            has_coordinates=has_coordinates,
            lat=lat,
            lon=lon,
            test_images=test_images,
        )

        test_cases.append(test_case)

        # Log discovery
        coord_status = f"({lat:.6f}, {lon:.6f})" if has_coordinates else "NO COORDINATES"
        print(f"   Found: {landmark_name:<15} {len(test_images)} images  {coord_status}")

    return test_cases


# ============================================================================
# Test Execution
# ============================================================================


async def test_single_configuration(
    builder: GalleryBuilder,
    gallery_embeddings: torch.Tensor,
    model_name: str,
    landmark_name: str,
    test_image_path: Path,
    is_related: bool,
    use_preprocessing: bool,
    device: str,
    threshold: float,
) -> dict:
    """Test a single configuration and return detailed results with timing."""

    results = {
        "model_type": builder.model_type,
        "model_variant": builder.model_variant,
        "model_name": model_name,
        "landmark_name": landmark_name,
        "image_name": test_image_path.name,
        "image_path": str(test_image_path),
        "is_related": is_related,
        "preprocessing": use_preprocessing,
    }

    try:
        # Initialize verifier
        verifier = LandmarkVerifier(gallery_embeddings, t_verify=threshold)

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

        # Calculate total time (inference only, since gallery built once per model)
        t_total = t_preprocess + t_feature_extract + t_verify

        # Determine if result is correct
        expected_result = is_related  # Related images should verify, unrelated should not
        is_correct = is_verified == expected_result

        results.update(
            {
                "similarity_score": similarity_score,
                "is_verified": is_verified,
                "expected_verified": expected_result,
                "is_correct": is_correct,
                "gallery_size": gallery_embeddings.shape[0],
                "time_preprocess_s": t_preprocess,
                "time_feature_extract_s": t_feature_extract,
                "time_verify_s": t_verify,
                "time_total_s": t_total,
                "time_inference_s": t_total,  # Same as total now
                "error": None,
            }
        )

        # Clean up memory
        del embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        results.update(
            {
                "similarity_score": 0.0,
                "is_verified": False,
                "expected_verified": is_related,
                "is_correct": False,
                "gallery_size": 0,
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
        "landmark_name",
        "image_name",
        "image_path",
        "is_related",
        "preprocessing",
        "similarity_score",
        "is_verified",
        "expected_verified",
        "is_correct",
        "gallery_size",
        "threshold",
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

    print(f"Results saved: {csv_path}")

    # 2. Per-image aggregated results
    csv_path = output_dir / "results_by_image.csv"
    image_stats = {}

    for result in all_results:
        key = (result["landmark_name"], result["image_name"], result["preprocessing"])
        if key not in image_stats:
            image_stats[key] = {
                "landmark_name": result["landmark_name"],
                "image_name": result["image_name"],
                "is_related": result["is_related"],
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
        "landmark_name",
        "image_name",
        "is_related",
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
                "landmark_name": stats["landmark_name"],
                "image_name": stats["image_name"],
                "is_related": stats["is_related"],
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

    print(f"Results saved: {csv_path}")

    # 3. Per-model aggregated results
    csv_path = output_dir / "results_by_model.csv"
    model_stats = {}

    for result in all_results:
        key = (result["model_name"], result["preprocessing"])
        if key not in model_stats:
            model_stats[key] = {
                "model_name": result["model_name"],
                "preprocessing": result["preprocessing"],
                "related_scores": [],
                "unrelated_scores": [],
                "correct_count": 0,
                "total_count": 0,
                "time_inference": [],
            }

        if result["is_related"]:
            model_stats[key]["related_scores"].append(result["similarity_score"])
        else:
            model_stats[key]["unrelated_scores"].append(result["similarity_score"])

        if result["is_correct"]:
            model_stats[key]["correct_count"] += 1
        model_stats[key]["total_count"] += 1
        model_stats[key]["time_inference"].append(result["time_inference_s"])

    fieldnames = [
        "model_name",
        "preprocessing",
        "avg_related_score",
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
            related_avg = (
                sum(stats["related_scores"]) / len(stats["related_scores"])
                if stats["related_scores"]
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
                "avg_related_score": related_avg,
                "avg_unrelated_score": unrelated_avg,
                "discrimination_gap": related_avg - unrelated_avg,
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

    print(f"Results saved: {csv_path}")

    # 4. Per-landmark aggregated results
    csv_path = output_dir / "results_by_landmark.csv"
    landmark_stats = {}

    for result in all_results:
        key = (result["landmark_name"], result["preprocessing"])
        if key not in landmark_stats:
            landmark_stats[key] = {
                "landmark_name": result["landmark_name"],
                "is_related": result["is_related"],
                "preprocessing": result["preprocessing"],
                "scores": [],
                "correct_count": 0,
                "total_count": 0,
            }

        landmark_stats[key]["scores"].append(result["similarity_score"])
        if result["is_correct"]:
            landmark_stats[key]["correct_count"] += 1
        landmark_stats[key]["total_count"] += 1

    fieldnames = [
        "landmark_name",
        "is_related",
        "preprocessing",
        "avg_score",
        "min_score",
        "max_score",
        "accuracy",
        "correct_count",
        "total_count",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in landmark_stats.values():
            scores = stats["scores"]
            row = {
                "landmark_name": stats["landmark_name"],
                "is_related": stats["is_related"],
                "preprocessing": stats["preprocessing"],
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "accuracy": stats["correct_count"] / stats["total_count"]
                if stats["total_count"] > 0
                else 0,
                "correct_count": stats["correct_count"],
                "total_count": stats["total_count"],
            }
            writer.writerow(row)

    print(f"Results saved: {csv_path}")


# ============================================================================
# Main Test Runner
# ============================================================================


async def main():
    """Main test execution."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 100)
    print("COMPREHENSIVE TEST SUITE WITH CSV EXPORT".center(100))
    print("=" * 100)
    print("\nConfiguration:")
    print(f"   Device: {device}")
    print(f"   Threshold: {THRESHOLD}")
    print(f"   Gallery Size: {GALLERY_SIZE} images")
    print(f"   Gallery Distance: {GALLERY_DISTANCE}m")
    print(f"   Models: {len(MODEL_CONFIGS)}")

    # Get API token
    mapillary_token = os.getenv("MAPILLARY_API_KEY")
    if not mapillary_token:
        raise ValueError(
            "MAPILLARY_API_KEY environment variable not set.\n"
            "Get your token from: https://www.mapillary.com/dashboard/developers"
        )

    # Discover test cases
    print("\n" + "=" * 100)
    print("DISCOVERING TEST CASES".center(100))
    print("=" * 100)
    test_cases = discover_test_cases(INPUT_DIR)

    if not test_cases:
        print(f"Error: No test cases found in {INPUT_DIR}")
        return

    # Fetch gallery images for each landmark with coordinates
    print("\n" + "=" * 100)
    print("FETCHING GALLERY IMAGES".center(100))
    print("=" * 100)

    fetcher = MapillaryFetcher(mapillary_token)
    galleries = {}  # landmark_name -> list[Image]

    for test_case in test_cases:
        if test_case.has_coordinates:
            print(f"\nFetching gallery for: {test_case.name}")
            print(f"   Coordinates: ({test_case.lat:.6f}, {test_case.lon:.6f})")
            print(f"   Distance: {GALLERY_DISTANCE}m, Target size: {GALLERY_SIZE} images")

            t_start = time.time()
            gallery_images = []
            try:
                async for img in fetcher.get_images(
                    test_case.lat, test_case.lon, num_images=GALLERY_SIZE, distance=GALLERY_DISTANCE
                ):
                    gallery_images.append(img)
            except Exception as e:
                print(f"   Error fetching gallery: {e}")
                continue

            t_fetch = time.time() - t_start
            galleries[test_case.name] = gallery_images
            print(f"   Fetched {len(gallery_images)} images in {t_fetch:.2f}s")
        else:
            print(f"\nSkipping gallery fetch for: {test_case.name} (no coordinates)")

    # Calculate total tests
    total_test_images = sum(len(tc.test_images) for tc in test_cases)
    total_tests = len(MODEL_CONFIGS) * total_test_images * 2  # 2 preprocessing modes

    print("\n" + "=" * 100)
    print("TEST EXECUTION PLAN".center(100))
    print("=" * 100)
    print(f"\nTotal test images: {total_test_images}")
    for tc in test_cases:
        print(
            f"   {tc.name:<15} {len(tc.test_images)} images  {'(with gallery)' if tc.has_coordinates else '(negative case)'}"
        )

    print(f"\nTotal test configurations: {total_tests}")
    print(f"   ({len(MODEL_CONFIGS)} models × {total_test_images} images × 2 preprocessing modes)")

    # Run all tests
    print("\n" + "=" * 100)
    print("RUNNING TESTS".center(100))
    print("=" * 100)

    all_results = []
    test_count = 0
    t_start_all = time.time()

    for model_idx, (model_type, model_variant, model_name) in enumerate(MODEL_CONFIGS, 1):
        print_model_header(model_name, model_idx, len(MODEL_CONFIGS))

        # Build galleries once per model (for each landmark with coordinates)
        model_galleries = {}  # landmark_name -> gallery_embeddings

        for test_case in test_cases:
            if not test_case.has_coordinates or test_case.name not in galleries:
                continue

            print(f"\nBuilding gallery for {test_case.name}...")
            t_start = time.time()

            try:
                builder = GalleryBuilder(
                    device=device, model_type=model_type, model_variant=model_variant
                )

                async def image_gen() -> AsyncGenerator[Image.Image, None]:
                    for img in galleries[test_case.name]:
                        yield img

                gallery_embeddings = await builder.build_gallery(
                    image_gen(), skip_preprocessing=True
                )
                model_galleries[test_case.name] = (builder, gallery_embeddings)

                t_gallery_build = time.time() - t_start
                print(
                    f"   Gallery built: {gallery_embeddings.shape[0]} embeddings in {t_gallery_build:.2f}s"
                )

            except Exception as e:
                print(f"   Error building gallery: {e}")
                continue

        # Test all images for this model
        for test_case in test_cases:
            # Determine gallery to use
            if test_case.has_coordinates and test_case.name in model_galleries:
                # Use own gallery
                builder, gallery_embeddings = model_galleries[test_case.name]
                is_related = True
            elif len(model_galleries) > 0:
                # Use first available gallery (for negative testing)
                first_landmark = next(iter(model_galleries.keys()))
                builder, gallery_embeddings = model_galleries[first_landmark]
                is_related = False
            else:
                print(f"\n  [SKIP] {test_case.name} - no gallery available")
                continue

            # Print section header
            print_landmark_section_header(test_case.name, len(test_case.test_images), is_related)

            # Track section stats
            section_pass = 0
            section_fail = 0
            section_error = 0

            # Test each image
            for test_image_path in test_case.test_images:
                for use_preprocessing in [True, False]:
                    test_count += 1

                    result = await test_single_configuration(
                        builder,
                        gallery_embeddings,
                        model_name,
                        test_case.name,
                        test_image_path,
                        is_related,
                        use_preprocessing,
                        device,
                        THRESHOLD,
                    )

                    all_results.append(result)

                    # Print formatted result
                    print_test_result(
                        test_count,
                        total_tests,
                        test_image_path.name,
                        use_preprocessing,
                        result,
                    )

                    # Track section stats
                    if result.get("error"):
                        section_error += 1
                    elif result.get("is_correct"):
                        section_pass += 1
                    else:
                        section_fail += 1

            # Print section footer
            print_landmark_section_footer(section_pass, section_fail, section_error)

        # Clean up after model
        for builder, gallery_embeddings in model_galleries.values():
            del builder
            del gallery_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_total_all = time.time() - t_start_all

    # Save results to CSV
    print("\n" + "=" * 100)
    print("SAVING RESULTS".center(100))
    print("=" * 100)
    save_results_to_csv(all_results, OUTPUT_DIR)

    # Print summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS".center(100))
    print("=" * 100)

    total_correct = sum(1 for r in all_results if r["is_correct"])
    accuracy = total_correct / len(all_results) * 100 if all_results else 0

    print(f"\nTests completed: {len(all_results)}")
    print(f"Overall accuracy: {accuracy:.1f}% ({total_correct}/{len(all_results)})")
    print(f"Total test time: {t_total_all:.2f}s")
    print(f"Average per test: {t_total_all / len(all_results):.2f}s")

    # Best model by accuracy
    model_accuracy = {}
    for result in all_results:
        key = result["model_name"]
        if key not in model_accuracy:
            model_accuracy[key] = {"correct": 0, "total": 0}
        if result["is_correct"]:
            model_accuracy[key]["correct"] += 1
        model_accuracy[key]["total"] += 1

    print("\nModel Accuracy Rankings:")
    sorted_models = sorted(
        model_accuracy.items(), key=lambda x: x[1]["correct"] / x[1]["total"], reverse=True
    )
    for rank, (model_name, stats) in enumerate(sorted_models[:10], 1):  # Top 10
        acc = stats["correct"] / stats["total"] * 100
        print(f"   {rank:2d}. {model_name:<30} {acc:.1f}% ({stats['correct']}/{stats['total']})")

    # Per-landmark breakdown
    print("\nAccuracy by Landmark:")
    landmark_accuracy = {}
    for result in all_results:
        key = result["landmark_name"]
        if key not in landmark_accuracy:
            landmark_accuracy[key] = {"correct": 0, "total": 0}
        if result["is_correct"]:
            landmark_accuracy[key]["correct"] += 1
        landmark_accuracy[key]["total"] += 1

    for landmark_name, stats in sorted(landmark_accuracy.items()):
        acc = stats["correct"] / stats["total"] * 100
        print(f"   {landmark_name:<15} {acc:.1f}% ({stats['correct']}/{stats['total']})")

    print("\n" + "=" * 100)
    print("ALL TESTS COMPLETED".center(100))
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
