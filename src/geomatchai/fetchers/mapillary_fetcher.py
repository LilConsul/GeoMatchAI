from io import BytesIO

import requests
from PIL import Image

from geomatchai.fetchers.base_fetcher import BaseFetcher


class MapillaryFetcher(BaseFetcher):
    """
    Fetcher for Mapillary street-level imagery.

    Args:
        client_token: Your Mapillary client access token from
                     https://www.mapillary.com/dashboard/developers
        base_url: Base URL for Mapillary API (default: https://graph.mapillary.com/)
    """

    def __init__(
        self, client_token: str, base_url: str = "https://graph.mapillary.com/"
    ):
        self.client_token = client_token
        self.base_url = base_url.rstrip("/")

    def get_images(
        self,
        lat: float,
        lon: float,
        num_images: int = 20,
        distance: float = 20,
        is_pano: bool = False,
        thumb_size: str = "1024",
    ) -> list[Image.Image]:
        """
        Fetch images from Mapillary near the given coordinates.

        Args:
            lat: Latitude
            lon: Longitude
            num_images: Maximum number of images to fetch
            distance: Search radius in METERS (default 20m)
            is_pano: If True, only fetch 360° panoramic images
            thumb_size: Thumbnail size - "256", "1024", or "2048"

        Returns:
            List of PIL Image objects
        """
        # Convert meters to km for bbox calculation
        distance_km = distance / 1000.0

        image_data = self._query_api(
            lat, lon, num_images, distance_km, is_pano, thumb_size
        )
        images = []

        thumb_field = f"thumb_{thumb_size}_url"
        for img_info in image_data:
            thumb_url = img_info.get(thumb_field)
            if not thumb_url:
                continue

            try:
                resp = requests.get(thumb_url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to download image: {e}")
                continue

        return images

    def _query_api(
        self,
        lat: float,
        lon: float,
        num_images: int,
        distance_km: float = 0.02,
        is_pano: bool = False,
        thumb_size: str = "1024",
    ) -> list[dict]:
        """
        Query Mapillary API for images in a bounding box.

        Args:
            lat: Latitude
            lon: Longitude
            num_images: Maximum number of images to return
            distance_km: Search radius in kilometers (default 0.02 = 20m)
            is_pano: If True, only return 360° panoramic images
            thumb_size: Thumbnail size to fetch

        Returns:
            List of image metadata dictionaries
        """
        # Calculate bounding box corners (minLon, minLat, maxLon, maxLat)
        bbox_corners = self._get_bounding_box_corners(lat, lon, distance_km)
        bbox = ",".join(map(str, bbox_corners))

        # Construct API URL
        url = f"{self.base_url}/images"

        # Build parameters
        thumb_field = f"thumb_{thumb_size}_url"
        params = {
            "access_token": self.client_token,
            "fields": f"id,geometry,{thumb_field},captured_at,compass_angle",
            "bbox": bbox,
            "limit": min(num_images, 2000),  # API max is 2000
        }

        # Add is_pano filter if requested
        if is_pano:
            params["is_pano"] = "true"

        print(f"Querying Mapillary API...")
        print(f"  Location: ({lat}, {lon})")
        print(f"  Distance: {distance_km * 1000:.0f}m ({distance_km}km)")
        print(f"  Bounding box: {bbox}")
        print(f"  Panoramic only: {is_pano}")

        try:
            response = requests.get(url, params=params, timeout=30)
            print(f"  Response Status: {response.status_code}")

            response.raise_for_status()

            data = response.json()
            images = data.get("data", [])
            print(f"  Found {len(images)} images")

            if images and len(images) > 0:
                print(
                    f"  First image: {images[0].get('id')} at {images[0].get('geometry')}"
                )

            return images

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response Status: {e.response.status_code}")
                print(f"Response Body: {e.response.text}")
            raise
        except Exception as e:
            print(f"Error querying Mapillary API: {e}")
            raise

    def _get_bounding_box_corners(
        self, lat: float, lon: float, distance: float = 0.02
    ) -> tuple[float, float, float, float]:
        """
        Calculate bounding box corners from center point and distance.
        Based on: https://stackoverflow.com/a/25025590/84369

        Args:
            lat: Center latitude
            lon: Center longitude
            distance: Distance in degrees (default 0.02)

        Returns:
            Tuple of (minLon, minLat, maxLon, maxLat)
        """
        import math

        if distance < 0:
            raise ValueError("Distance must be positive")

        # Helper functions
        def deg_to_rad(v):
            return v * (math.pi / 180)

        def rad_to_deg(v):
            return (180 * v) / math.pi

        # Coordinate limits
        MIN_LAT = deg_to_rad(-90)
        MAX_LAT = deg_to_rad(90)
        MIN_LON = deg_to_rad(-180)
        MAX_LON = deg_to_rad(180)

        # Earth's radius (km)
        R = 6378.1

        # Angular distance in radians on a great circle
        rad_dist = distance / R

        # Center point coordinates (rad)
        rad_lat = deg_to_rad(lat)
        rad_lon = deg_to_rad(lon)

        # Minimum and maximum latitudes for given distance
        min_lat = rad_lat - rad_dist
        max_lat = rad_lat + rad_dist

        # Define deltaLon to help determine min and max longitudes
        delta_lon = math.asin(math.sin(rad_dist) / math.cos(rad_lat))

        if min_lat > MIN_LAT and max_lat < MAX_LAT:
            min_lon = rad_lon - delta_lon
            max_lon = rad_lon + delta_lon
            if min_lon < MIN_LON:
                min_lon = min_lon + 2 * math.pi
            if max_lon > MAX_LON:
                max_lon = max_lon - 2 * math.pi
        else:
            # A pole is within the given distance
            min_lat = max(min_lat, MIN_LAT)
            max_lat = min(max_lat, MAX_LAT)
            min_lon = MIN_LON
            max_lon = MAX_LON

        return (
            rad_to_deg(min_lon),
            rad_to_deg(min_lat),
            rad_to_deg(max_lon),
            rad_to_deg(max_lat),
        )


if __name__ == "__main__":  # TODO: remove this block
    import os

    if not (mapillary_client_token := os.getenv("MAPILLARY_CLIENT_TOKEN")):
        raise ValueError(
            "MAPILLARY_CLIENT_TOKEN environment variable not set.\n"
            "Get your client token from: https://www.mapillary.com/dashboard/developers"
        )

    fetcher = MapillaryFetcher(mapillary_client_token)

    # Test Wawel Castle
    lat, lon = 50.055191, 19.937365

    print(f"\n{'=' * 70}")
    print(f"Testing: Wawel Castle, Krakow ({lat}, {lon})")
    print(f"{'=' * 70}\n")

    try:
        # Fetch images with 100m distance, NOT panoramic only
        images = fetcher.get_images(
            lat,
            lon,
            num_images=5,
            distance=100,
            is_pano=False,  # not panoramic only
            thumb_size="256",
        )

        if images:
            print(f"\n✓ Successfully fetched {len(images)} images!")
            print(f"  First image size: {images[0].size}")
            print(f"  First image mode: {images[0].mode}")

            # Show images
            for i, img in enumerate(images):
                print(f"  Displaying image {i + 1}...")
                img.show()
        else:
            print(f"\n✗ No images found")
            print(f"  This could mean:")
            print(f"  1. No Mapillary coverage at this location")
            print(f"  2. Invalid API token")
            print(f"  3. Try increasing distance parameter")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
