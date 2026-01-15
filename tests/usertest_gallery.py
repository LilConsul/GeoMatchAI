from geomatchai.gallery.gallery_builder import GalleryBuilder
import asyncio


async def main():
    # Initialize gallery builder
    builder = GalleryBuilder()

    # Example: Build gallery from sample images
    # Using the test photo (in a real scenario, you'd have multiple reference images)
    image_paths = [
        "input/photo_2024-07-21_12-07-58.jpg",
        "input/photo_2025-11-21_10-07-59.jpg",
    ]

    try:
        gallery_embeddings = await builder.build_gallery(image_paths)
        print(f"Gallery shape: {gallery_embeddings.shape}")
        print("Gallery built successfully.")

        # Optional: Save embeddings for inspection
        import torch

        torch.save(gallery_embeddings, "output/gallery_embeddings.pt")
        print("Embeddings saved to output/gallery_embeddings.pt")

    except Exception as e:
        print(f"Failed to build gallery: {e}")


if __name__ == "__main__":
    asyncio.run(main())
