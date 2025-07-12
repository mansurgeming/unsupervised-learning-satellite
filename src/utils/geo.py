import numpy as np
from geopy.distance import distance
from geopy.point import Point

def generate_random_utm_positions(center_latlon, size_km, num_points):
    """
    Generate random lat/lon positions within a square of size_km × size_km centered at center_latlon.
    """
    center = Point(center_latlon)
    radius_km = size_km / 2

    positions = []
    for _ in range(num_points):
        bearing = np.random.uniform(0, 360)
        dist = np.random.uniform(0, radius_km)
        pt = distance(kilometers=dist).destination(center, bearing)
        positions.append((pt.latitude, pt.longitude))

    return np.array(positions)  # shape: (num_points, 2)