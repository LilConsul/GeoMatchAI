from PIL import Image
from geojson import GeoJSON

from geomatchai.fetchers.base_fetcher import BaseFetcher

import mapillary as mly


class MapillaryFetcher(BaseFetcher):
    """
    Fetcher for Mapillary street-level imagery.

    Args:
        api_token: Your Mapillary client access token from
                     https://www.mapillary.com/dashboard/developers
    """

    def __init__(self, api_token: str):
        self.interface = mly.interface
        self.interface.set_access_token(api_token)

    def get_images(
        self,
        lat: float,
        lon: float,
        *,
        distance: float = 50.0,
        num_images: int = 20,
    ) -> list[Image.Image]:
        images_json = self.interface.get_image_looking_at(
            at={
                "lat": lat,
                "lng": lon,
            },
            radius=distance,
            image_type="flat",
        ).to_dict()

        print(f"{images_json=}")
        image_ids = [
            images_json["features"][i]["properties"]["id"]
            for i in range(min(num_images, len(images_json["features"])))
        ]

        images = [
            self.interface.image_thumbnail(image_id, resolution=256)
            for image_id in image_ids
        ]
        print(images)


if __name__ == "__main__":  # TODO: remove this block
    import os

    if not (mapillary_client_token := os.getenv("MAPILLARY_CLIENT_TOKEN")):
        raise ValueError(
            "MAPILLARY_CLIENT_TOKEN environment variable not set.\n"
            "Get your client token from: https://www.mapillary.com/dashboard/developers"
        )

    # fetcher = MapillaryFetcher(mapillary_client_token)
    #
    # # Test Wawel Castle
    # lat, lon = 50.055191, 19.937365
    #
    # images = fetcher.get_images(lat, lon, num_images=1)
    # print(f"Fetched {len(images)} images.")
    #
    # for i, img in enumerate(images):
    #     img.show(title=f"Image {i + 1}")

    interface = mly.interface
    interface.set_access_token(mapillary_client_token)

    img = interface.image_thumbnail(image_id="846885049236095", resolution=256)
    print(img)