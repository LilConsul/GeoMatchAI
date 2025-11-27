"""
Complete usage example for GeoMatchAI library.

This script demonstrates the full pipeline:
1. Configure logging
2. Fetch reference images from Mapillary
3. Build a gallery of embeddings
4. Verify a user's selfie against the gallery
"""

import asyncio
import logging
import os
from pathlib import Path

from geomatchai import (
    GalleryBuilder,
    LandmarkVerifier,
    MapillaryFetcher,
    Preprocessor,
    config,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main function demonstrating GeoMatchAI usage."""

    # Configuration
    LANDMARK_LAT = 50.054404  # Wawel Castle latitude
    LANDMARK_LON = 19.935730  # Wawel Castle longitude
    MAPILLARY_API_KEY = os.getenv("MAPILLARY_API_KEY")

    if not MAPILLARY_API_KEY:
        logger.error("MAPILLARY_API_KEY environment variable not set!")
        logger.info("Get your API key from: https://www.mapillary.com/dashboard/developers")
        return

    logger.info("Starting GeoMatchAI pipeline...")
    logger.info(f"Target location: ({LANDMARK_LAT}, {LANDMARK_LON})")

    # Step 1: Initialize components
    logger.info("\n=== Step 1: Initializing Components ===")
    fetcher = MapillaryFetcher(api_token=MAPILLARY_API_KEY, request_timeout=30.0, max_retries=3)

    builder = GalleryBuilder(
        model_type="timm",
        model_variant="tf_efficientnet_b4.ns_jft_in1k",  # NoisyStudent
    )

    preprocessor = Preprocessor()

    # Step 2: Fetch reference images and build gallery
    logger.info("\n=== Step 2: Building Reference Gallery ===")
    logger.info(f"Fetching images within {config.fetcher.DEFAULT_SEARCH_RADIUS}m radius...")

    try:
        image_generator = fetcher.get_images(
            LANDMARK_LAT, LANDMARK_LON, distance=50.0, num_images=20
        )

        gallery_embeddings = await builder.build_gallery(
            image_generator,
            batch_size=config.gallery.DEFAULT_BATCH_SIZE,
            skip_preprocessing=True,  # Gallery images are clean (no people)
        )

        logger.info(f"Gallery built: {gallery_embeddings.shape}")

    except Exception as e:
        logger.error(f"Failed to build gallery: {e}")
        return

    # Step 3: Initialize verifier
    logger.info("\n=== Step 3: Initializing Verifier ===")
    verifier = LandmarkVerifier(
        gallery_embeddings=gallery_embeddings, t_verify=config.verification.DEFAULT_THRESHOLD
    )
    logger.info(f"Verifier ready with threshold: {config.verification.DEFAULT_THRESHOLD}")

    # Step 4: Verify user selfie
    logger.info("\n=== Step 4: Verifying User Selfie ===")

    # Example: Load a test image (replace with actual user selfie)
    test_image_path = Path("tests/input/wawel/wawel1.jpg")

    if not test_image_path.exists():
        logger.warning(f"Test image not found: {test_image_path}")
        logger.info("Skipping verification step - no test image available")
        return

    try:
        # Preprocess the user's selfie (removes people)
        from PIL import Image

        user_image = Image.open(test_image_path)
        preprocessed = preprocessor.preprocess_image(user_image)

        # Extract features
        with builder.feature_extractor.eval():
            import torch

            with torch.no_grad():
                query_embedding = builder.feature_extractor(
                    preprocessed.unsqueeze(0).to(builder.device)
                )

        # Verify
        is_verified, similarity_score = verifier.verify(query_embedding)

        logger.info("\n=== Verification Result ===")
        logger.info(f"Image: {test_image_path.name}")
        logger.info(f"Similarity Score: {similarity_score:.4f}")
        logger.info(f"Threshold: {config.verification.DEFAULT_THRESHOLD}")
        logger.info(f"Result: {'✅ VERIFIED' if is_verified else '❌ REJECTED'}")

        if is_verified:
            logger.info("User is at the landmark location!")
        else:
            logger.info("User is NOT at the landmark location.")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
