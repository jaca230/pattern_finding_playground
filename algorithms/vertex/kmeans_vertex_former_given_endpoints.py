import math
from models.point_3d import Point3D
from algorithms.vertex.kmeans_vertex_former import KMeansVertexFormer
from models.tracklet import Tracklet

class KMeansVertexFormerGivenEndpoints(KMeansVertexFormer):
    def determine_endpoints(self, tracklet: Tracklet) -> tuple[Point3D, Point3D]:
        """Returns the endpoints that are already set on the tracklet, fixing individual invalid coordinates."""
        p0, p1 = tracklet.get_endpoints()
        return self.sanitize_point(p0), self.sanitize_point(p1)

    def sanitize_point(self, point: Point3D) -> Point3D:
        """Replaces invalid coordinates in a Point3D with 0."""
        if point is None:
            return Point3D(0, 0, 0)
        x = point.x if math.isfinite(point.x) else 0
        y = point.y if math.isfinite(point.y) else 0
        z = point.z if math.isfinite(point.z) else 0
        return Point3D(x, y, z)
