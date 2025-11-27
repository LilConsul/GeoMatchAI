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

    # Validate configuration
    validation_errors = config.validate()
    if validation_errors:
        logger.error("Configuration validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        return

    # Configuration
    LANDMARK_LAT = 50.054404  # Wawel Castle latitude
    LANDMARK_LON = 19.935730  # Wawel Castle longitude
    MAPILLARY_API_KEY = config.get_mapillary_api_key()

    if not MAPILLARY_API_KEY:
        logger.error("MAPILLARY_API_KEY environment variable not set!")
        logger.info("Get your API key from: https://www.mapillary.com/dashboard/developers")
        return

    logger.info("Starting GeoMatchAI pipeline...")
    logger.info(f"Target location: ({LANDMARK_LAT}, {LANDMARK_LON})")
    logger.info(f"Using config defaults:")
    logger.info(f"  - Model: {config.model.DEFAULT_MODEL_TYPE}/{config.model.DEFAULT_TIMM_VARIANT}")
    logger.info(f"  - Verification threshold: {config.verification.DEFAULT_THRESHOLD}")
    logger.info(f"  - Search radius: {config.fetcher.DEFAULT_SEARCH_RADIUS}m")
    logger.info(f"  - Max images: {config.fetcher.DEFAULT_NUM_IMAGES}")

    # Step 1: Initialize components
    logger.info("\n=== Step 1: Initializing Components ===")
    # All parameters use config defaults
    fetcher = MapillaryFetcher(api_token=MAPILLARY_API_KEY)

    # Model type and variant use config defaults (timm + NoisyStudent)
    builder = GalleryBuilder()

    preprocessor = Preprocessor()

    # Step 2: Fetch reference images and build gallery
    logger.info("\n=== Step 2: Building Reference Gallery ===")
    logger.info(f"Fetching images within {config.fetcher.DEFAULT_SEARCH_RADIUS}m radius...")

    try:
        # distance and num_images use config defaults
        image_generator = fetcher.get_images(LANDMARK_LAT, LANDMARK_LON)

        # batch_size uses config default
        gallery_embeddings = await builder.build_gallery(
            image_generator,
            skip_preprocessing=True,  # Gallery images are clean (no people)
        )

        logger.info(f"Gallery built: {gallery_embeddings.shape}")

    except Exception as e:
        logger.error(f"Failed to build gallery: {e}")
        return

    # Step 3: Initialize verifier
    logger.info("\n=== Step 3: Initializing Verifier ===")
    # t_verify uses config default
    verifier = LandmarkVerifier(gallery_embeddings=gallery_embeddings)
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
