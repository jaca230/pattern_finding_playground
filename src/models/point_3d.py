import math

class Point3D:
    def __init__(self, x: float, y: float, z: float):
        """
        Initialize a 3D point with x, y, and z coordinates.

        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.
            z (float): z-coordinate of the point.
        """
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Return a string representation of the Point3D."""
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"

    def distance_to(self, other: "Point3D") -> float:
        """
        Calculate the Euclidean distance between this point and another Point3D.

        Args:
            other (Point3D): Another Point3D to calculate distance to.

        Returns:
            float: Euclidean distance between the two points.
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx**2 + dy**2 + dz**2)
