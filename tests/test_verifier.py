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
    verifier = LandmarkVerifier(gallery_embeddings, t_verify=0.8)

    # Process query image using the SAME preprocessor as gallery
    query_tensor = builder.preprocessor.preprocess_image("input/wawel/test.png")
    with torch.no_grad():
        query_embedding = builder.feature_extractor(
            query_tensor.unsqueeze(0).to(device)
        )

    # Verify
    is_verified, max_score = verifier.verify(query_embedding)
    print(f"Verification result: {is_verified}, Score: {max_score:.4f}")

    # Test with different threshold
    verifier.set_threshold(0.9)
    is_verified_high, max_score_high = verifier.verify(query_embedding)
    print(
        f"With higher threshold (0.9): {is_verified_high}, Score: {max_score_high:.4f}"
    )

    print("Verification test completed!")
