import asyncio
import os
from pathlib import Path

import torch
from PIL import Image

from geomatchai.fetchers.mapillary_fetcher import MapillaryFetcher
from geomatchai.gallery.gallery_builder import GalleryBuilder
from geomatchai.verification.verifier import LandmarkVerifier


async def main():
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    builder = GalleryBuilder(device=device)

    # Get Mapillary API token
    mapillary_token = os.getenv("MAPILLARY_API_KEY")
    if not mapillary_token:
        raise ValueError(
            "MAPILLARY_API_KEY environment variable not set.\n"
            "Get your token from: https://www.mapillary.com/dashboard/developers"
        )

    # Initialize fetcher
    fetcher = MapillaryFetcher(mapillary_token)

    # Wawel Castle coordinates
    lat, lon = 50.054404, 19.935730

    print("Fetching gallery images from Mapillary...")
    # Build gallery from Mapillary images (skip preprocessing for clean street photos)
    gallery_embeddings = await builder.build_gallery(
        fetcher.get_images(lat, lon, num_images=100),
        skip_preprocessing=True
    )
    print(f"✓ Gallery built with {gallery_embeddings.shape[0]} images")

    # Initialize verifier
    verifier = LandmarkVerifier(gallery_embeddings, t_verify=0.65)

    # Test with query image (with preprocessing to remove person)
    print("\n=== Testing query image with person removal ===")
    test_image_path = Path(__file__).parent / "input" / "wawel" / "test.png"
    query_image = Image.open(test_image_path).convert("RGB")
    query_tensor = builder.preprocessor.preprocess_image(query_image)
    with torch.no_grad():
        query_embedding = builder.feature_extractor(
            query_tensor.unsqueeze(0).to(device)
        )

    is_verified, max_score = verifier.verify(query_embedding)
    print(f"Verification result: {is_verified}")
    print(f"Similarity score: {max_score:.4f}")
    print(f"Threshold: {verifier.t_verify}")

    # Test with completely unrelated image (should be rejected)
    print("\n=== Testing unrelated image (negative test) ===")
    unrelated_image_path = Path(__file__).parent / "input" / "photo_2025-11-21_10-07-59.jpg"
    unrelated_image = Image.open(unrelated_image_path).convert("RGB")
    unrelated_tensor = builder.preprocessor.preprocess_image(unrelated_image)
    with torch.no_grad():
        unrelated_embedding = builder.feature_extractor(
            unrelated_tensor.unsqueeze(0).to(device)
        )

    is_unrelated_verified, unrelated_score = verifier.verify(unrelated_embedding)
    print(f"Verification result: {is_unrelated_verified}")
    print(f"Similarity score: {unrelated_score:.4f}")
    print(f"Expected: False (score < {verifier.t_verify})")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Wawel query: {'✓ PASS' if is_verified else '✗ FAIL'} (score: {max_score:.4f})")
    print(f"  Unrelated:   {'✓ PASS' if not is_unrelated_verified else '✗ FAIL'} (score: {unrelated_score:.4f})")
    print(f"  Discrimination gap: {max_score - unrelated_score:.4f}")
    print("=" * 60)
    print("\nVerification test completed!")


if __name__ == "__main__":
    asyncio.run(main())
