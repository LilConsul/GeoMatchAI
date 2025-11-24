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
    gallery_embeddings = builder.build_gallery(image_paths)
    print(f"Gallery built with shape: {gallery_embeddings.shape}")

    # Initialize verifier
    verifier = LandmarkVerifier(gallery_embeddings, t_verify=0.65)

    # SANITY CHECK: Test with identical image from gallery (should be ~1.0)
    print("\n=== SANITY CHECK: Identical Image Test ===")
    identical_tensor = builder.preprocessor.preprocess_image("input/wawel/wawel1.jpg")
    with torch.no_grad():
        identical_embedding = builder.feature_extractor(
            identical_tensor.unsqueeze(0).to(device)
        )
    is_identical, identical_score = verifier.verify(identical_embedding)
    print(f"Identical image score: {identical_score:.4f} (should be ~0.95-1.0)")
    if identical_score < 0.90:
        print("⚠️  WARNING: Identical image scores < 0.90. Pipeline has issues!")

    print("\n=== ACTUAL TEST: Similar Location, Different Angle ===")
    # Process query image using the SAME preprocessor as gallery
    query_tensor = builder.preprocessor.preprocess_image("input/wawel/test.png")
    with torch.no_grad():
        query_embedding = builder.feature_extractor(
            query_tensor.unsqueeze(0).to(device)
        )

    # Verify
    is_verified, max_score = verifier.verify(query_embedding)
    print(f"Similar location test: {is_verified}, Score: {max_score:.4f}")
    print(f"Expected: 0.75-0.90 for same location, different angle")

    # Test with different threshold
    verifier.set_threshold(0.75)
    is_verified_high, max_score_high = verifier.verify(query_embedding)
    print(
        f"\nWith threshold 0.75: {is_verified_high}, Score: {max_score_high:.4f}"
    )

    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  - Identical image: {identical_score:.4f} (target: >0.90)")
    print(f"  - Similar location: {max_score:.4f} (target: >0.75)")
    print("="*60)
    print("\nVerification test completed!")
