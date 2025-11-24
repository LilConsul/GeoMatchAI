from pathlib import Path

import torch

from src.geomatchai.gallery.gallery_builder import GalleryBuilder
from src.geomatchai.verification.verifier import LandmarkVerifier

if __name__ == "__main__":
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    builder = GalleryBuilder(device=device)

    # Build small gallery (using test image as reference)
    image_paths = [
        Path("input/wawel/wawel1.jpg"),
        Path("input/wawel/wawel2.jpg"),
        Path("input/wawel/wawel3.jpg"),
        Path("input/wawel/wawel4.jpg"),
        Path("input/wawel/wawel5.jpg"),
    ]
    # CRITICAL FIX: Skip preprocessing for gallery (clean photos without people)
    # This preserves landmark features instead of destroying them with mean replacement
    gallery_embeddings = builder.build_gallery(image_paths, skip_preprocessing=True)
    print(f"Gallery built with shape: {gallery_embeddings.shape}")
    print("✓ Gallery built WITHOUT preprocessing (preserves landmark features)")

    # Initialize verifier
    verifier = LandmarkVerifier(gallery_embeddings, t_verify=0.65)

    # SANITY CHECK: Test with identical image from gallery (should be ~1.0)
    print("\n=== SANITY CHECK: Identical Image Test (No Preprocessing) ===")
    # Load WITHOUT preprocessing to match gallery
    from PIL import Image
    import torchvision.transforms as T

    image = Image.open("input/wawel/wawel1.jpg").convert("RGB")
    image = T.Resize((520, 520))(image)
    tensor = T.ToTensor()(image)
    tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        tensor
    ).to(device)

    with torch.no_grad():
        identical_embedding = builder.feature_extractor(tensor.unsqueeze(0))
    is_identical, identical_score = verifier.verify(identical_embedding)
    print(f"Identical image score: {identical_score:.4f} (should be 1.0)")
    if identical_score < 0.99:
        print("⚠️  WARNING: Identical image scores < 0.99. Gallery build failed!")

    print("\n=== ACTUAL TEST: Similar Location (WITH Preprocessing) ===")
    # For query selfies, we MUST use preprocessing to remove person
    query_tensor = builder.preprocessor.preprocess_image("input/wawel/test.png")
    with torch.no_grad():
        query_embedding = builder.feature_extractor(
            query_tensor.unsqueeze(0).to(device)
        )

    # Verify
    is_verified, max_score = verifier.verify(query_embedding)
    print(f"With person removal: {is_verified}, Score: {max_score:.4f}")
    print(f"Note: Score lower due to feature loss from preprocessing")

    # Compare: Same image WITHOUT preprocessing
    print("\n=== Comparison: Same Image WITHOUT Preprocessing ===")
    image_no_prep = Image.open("input/wawel/test.png").convert("RGB")
    image_no_prep = T.Resize((520, 520))(image_no_prep)
    tensor_no_prep = T.ToTensor()(image_no_prep)
    tensor_no_prep = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        tensor_no_prep
    ).to(device)
    with torch.no_grad():
        query_embedding_no_prep = builder.feature_extractor(tensor_no_prep.unsqueeze(0))
    is_verified_no_prep, score_no_prep = verifier.verify(query_embedding_no_prep)
    print(f"Without person removal: {is_verified_no_prep}, Score: {score_no_prep:.4f}")
    print(f"Expected: >0.85 for same location without preprocessing")

    # Test with different threshold
    verifier.set_threshold(0.75)
    is_verified_high, max_score_high = verifier.verify(query_embedding)
    print(f"\nWith threshold 0.75: {is_verified_high}, Score: {max_score_high:.4f}")

    print("\n=== NEGATIVE TEST: Completely Different Image ===")
    # Test with a different image (should score low)
    try:
        different_tensor = builder.preprocessor.preprocess_image(
            "input/photo_2024-07-21_12-07-58.jpg"
        )
        with torch.no_grad():
            different_embedding = builder.feature_extractor(
                different_tensor.unsqueeze(0).to(device)
            )
        is_different, different_score = verifier.verify(different_embedding)
        print(f"Different image test: {is_different}, Score: {different_score:.4f}")
        print(f"Expected: <0.60 for completely different location")
    except Exception as e:
        print(f"⚠️  Could not test different image: {e}")
        different_score = None

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(
        f"  - Identical image (no preprocessing): {identical_score:.4f} (target: 1.0)"
    )
    print(f"  - Similar location WITH preprocessing: {max_score:.4f}")
    print(f"  - Similar location WITHOUT preprocessing: {score_no_prep:.4f}")
    if different_score is not None:
        print(
            f"  - Different location (with preprocessing): {different_score:.4f} (target: <0.60)"
        )
    print("=" * 60)

    # Evaluation
    print("\nEVALUATION:")
    if identical_score >= 0.99:
        print("  ✅ Gallery build: PASS (identical score = 1.0)")
    else:
        print("  ❌ Gallery build: FAIL (identical should be 1.0)")

    print(f"\n  📊 Preprocessing Impact: {score_no_prep - max_score:.4f} score loss")
    if score_no_prep - max_score > 0.05:
        print("     ⚠️  Mean replacement destroys 5%+ of features!")

    if score_no_prep >= 0.85:
        print("  ✅ Location matching (without preprocessing): EXCELLENT")
    elif score_no_prep >= 0.75:
        print("  ✅ Location matching (without preprocessing): GOOD")
    else:
        print("  ⚠️  Location matching: NEEDS IMPROVEMENT")

    if different_score is not None:
        gap = score_no_prep - different_score
        print(f"\n  📊 Discrimination Gap: {gap:.4f}")
        if gap > 0.15:
            print("     ✅ Can distinguish Wawel from other locations")
        elif gap > 0.05:
            print("     ⚠️  Weak discrimination (gap < 0.15)")
        else:
            print("     ❌ Cannot distinguish locations (gap < 0.05)")

    print("=" * 60)
    print("\nVerification test completed!")
