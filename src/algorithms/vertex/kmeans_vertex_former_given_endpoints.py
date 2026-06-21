import numpy as np
import random
from sklearn.cluster import DBSCAN
from models.point_3d import Point3D
from algorithms.vertex.kmeans_vertex_former import KMeansVertexFormer
from models.tracklet import Tracklet

class KMeansVertexFormerGivenEndpoints(KMeansVertexFormer):
    def determine_endpoints(self, tracklet: Tracklet) -> tuple[Point3D, Point3D]:
        return tracklet.get_endpoints()

    def BIC(self, sigma, k, cluster_centers, end_points):
        """Override BIC to ignore non-finite coordinates in the residual calculation."""
        N = len(end_points)
        d = 3
        term1 = -(N * d / 2) * np.log(2 * np.pi * sigma ** 2)

        residuals = []
        for p, c in zip(end_points, cluster_centers):
            # if not (np.all(np.isfinite(p)) and np.all(np.isfinite(c))):
            #     print("⚠️ Non-finite values detected in the input endpoints and cluster centers:")
            #     print("k:", k)
            #     print("End Points:", end_points)
            #     print("Cluster Centers:", cluster_centers)
            residuals.append(self.compute_partial_norm(np.array(p), np.array(c)))

        term2 = -1 / (2 * sigma ** 2) * np.sum(residuals)
        logL = term1 + term2
        return k * np.log(N) - 2 * logL

    def compute_partial_norm(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute norm between two vectors, ignoring non-finite coordinates."""
        valid_mask = np.isfinite(vec1) & np.isfinite(vec2)

        # if not np.all(valid_mask):
        #     print("⚠️ Non-finite values detected in a vector pair!")
        #     print(f"Cluster Centers: {vec2}")
        #     print(f"End Points: {vec1}")
        #     print(f"Valid mask: {valid_mask}")
        #     print(f"Filtered vec1: {vec1[valid_mask]}")
        #     print(f"Filtered vec2: {vec2[valid_mask]}")

        valid_vec1 = vec1[valid_mask]
        valid_vec2 = vec2[valid_mask]
        diffs = valid_vec1 - valid_vec2

        return np.linalg.norm(diffs)

    def create_vertex_guess_random(self, k, tracklet_end_points_vec):
        """Randomly generate k vertex guesses within the bounding box of all endpoints, ignoring invalid points."""
        x_values = []
        y_values = []
        z_values = []
        invalid = False

        for endpoints in tracklet_end_points_vec:
            for end_point in endpoints:
                if np.isfinite(end_point[0]) and np.isfinite(end_point[1]) and np.isfinite(end_point[2]):
                    x_values.append(end_point[0])
                    y_values.append(end_point[1])
                    z_values.append(end_point[2])
                else:
                    # print(f"⚠️ Invalid endpoint detected: {end_point}")
                    invalid = True

        if not x_values or not y_values or not z_values:
            # print("⚠️ No valid points available for seeding!")
            return []

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        min_z, max_z = min(z_values), max(z_values)

        vertices = []
        for _ in range(k):
            rand_x = random.uniform(min_x, max_x)
            rand_y = random.uniform(min_y, max_y)
            rand_z = random.uniform(min_z, max_z)
            vertices.append(np.array([rand_x, rand_y, rand_z]))

        # if invalid:
        #     print(vertices)

        while len(vertices) < k:
            vertices.append(np.array([0.0, 0.0, 0.0]))
        return vertices

    def create_vertex_guess_by_distance(self, k, tracklet_end_points_vec):
        """Cluster endpoints (excluding intra-tracklet groups), then fill with random guesses if needed, ignoring invalid points."""
        all_endpoints = []
        tracklet_ids = []
        for tracklet_idx, endpoints in enumerate(tracklet_end_points_vec):
            for endpoint in endpoints:
                if np.isfinite(endpoint[0]) and np.isfinite(endpoint[1]) and np.isfinite(endpoint[2]):
                    all_endpoints.append(np.array(endpoint))
                    tracklet_ids.append(tracklet_idx)

        if not all_endpoints:
            # print("⚠️ No valid endpoints available for distance-based guessing!")
            return self.create_vertex_guess_random(k, tracklet_end_points_vec)

        all_endpoints = np.array(all_endpoints)
        tracklet_ids = np.array(tracklet_ids)

        db = DBSCAN(eps=0.5, min_samples=1).fit(all_endpoints)
        labels = db.labels_
        unique_labels = set(labels)

        cluster_centroids = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_tracklet_ids = tracklet_ids[indices]

            if len(set(cluster_tracklet_ids)) == len(cluster_tracklet_ids):
                cluster_points = all_endpoints[indices]
                centroid = np.mean(cluster_points, axis=0)
                cluster_centroids.append(centroid)

        if len(cluster_centroids) >= k:
            return cluster_centroids[:k]

        remaining = k - len(cluster_centroids)
        random_guesses = self.create_vertex_guess_random(remaining, tracklet_end_points_vec)
        return cluster_centroids + random_guesses

    def compute_new_vertices(self, vertices_vec, tracklet_end_points_vec, vertices_end_points_map):
        """Compute new vertex positions as the mean of assigned tracklet endpoints, ignoring invalid points."""
        new_vertices = np.zeros(np.shape(vertices_vec))
        n_vertices_vec = np.zeros((np.shape(vertices_vec)[0], 1))

        for i, tracklet_end_points in enumerate(tracklet_end_points_vec):
            for j, end_point in enumerate(tracklet_end_points):
                k = vertices_end_points_map[i][j]

                if np.isfinite(end_point[0]) and np.isfinite(end_point[1]) and np.isfinite(end_point[2]):
                    new_vertices[k] += tracklet_end_points_vec[i][j]
                    n_vertices_vec[k] += 1
                else:
                    # print(f"⚠️ Invalid endpoint at ({end_point}) for vertex {k}.")
                    pass

        for k, vertex in enumerate(new_vertices):
            if n_vertices_vec[k] != 0:
                new_vertices[k] /= n_vertices_vec[k]
            else:
                new_vertices[k] = self.create_vertex_guess(1, tracklet_end_points_vec)[0]

        return new_vertices
