import random
import numpy as np
from typing import Set, Optional
from models.tracklet import Tracklet
from models.vertex import Vertex
from algorithms.vertex.vertex_former import VertexFormer
from models.point_3d import Point3D
from utils.utils import fit_tracklet_hits
from collections import defaultdict

class KMeansVertexFormer(VertexFormer):
    def __init__(self, n_iters=5, sigma=0.5):
        """KMeans-based vertex formation."""
        self.n_iters = n_iters
        self.sigma = sigma

    def BIC(self, sigma, k, cluster_centers, end_points):
        """Compute the Bayesian Information Criterion for a clustering solution."""
        N = len(end_points)
        d = 3
        term1 = -(N * d / 2) * np.log(2 * np.pi * sigma ** 2)
        term2 = -1 / (2 * (sigma ** 2)) * np.sum([np.linalg.norm(vec) for vec in end_points - cluster_centers])
        logL = term1 + term2
        return k * np.log(N) - 2 * logL

    def assign_vertices(self, vertices_vec, tracklet_end_points_vec):
        """Assign tracklet endpoints to the nearest vertices."""
        n_tracklets = len(tracklet_end_points_vec)
        n_vertices = len(vertices_vec)
        vertices_end_points_map = []
        min_dist_end_points_map = []

        for i, tracklet_end_points in enumerate(tracklet_end_points_vec):
            vertices_end_points_map.append([])
            min_dist_end_points_map.append([])
            dist_map = []
            dist_index_map = []

            for j, end_point in enumerate(tracklet_end_points):
                vertices_end_points_map[i].append(0)
                min_dist_end_points_map[i].append(0)
                min_delta_r = 1e9
                min_k = -1
                for k, vertex in enumerate(vertices_vec):
                    delta_r_ijk = np.linalg.norm(np.array(end_point) - np.array(vertex))
                    dist_map.append(delta_r_ijk)
                    dist_index_map.append((j, k))

            sorted_indices = np.argsort(dist_map)
            dist_index_map_sorted = np.array(dist_index_map)[sorted_indices]
            dist_map_sorted = np.array(dist_map)[sorted_indices]

            j_used = []
            k_used = []
            for index, tup in enumerate(dist_index_map_sorted):
                j = tup[0]
                k = tup[1]
                if j not in j_used:
                    if k not in k_used:
                        vertices_end_points_map[i][j] = k
                        min_dist_end_points_map[i][j] = dist_map_sorted[index]
                        j_used.append(j)
                        k_used.append(k)
        return vertices_end_points_map

    def create_vertex_guess(self, k, tracklet_end_points_vec):
        """Generate random guesses for k vertices."""
        x_values = []
        y_values = []
        z_values = []

        for i, tracklet_end_points in enumerate(tracklet_end_points_vec):
            for j, end_point in enumerate(tracklet_end_points):
                x_values.append(end_point[0])
                y_values.append(end_point[1])
                z_values.append(end_point[2])

        min_x = min(x_values)
        min_y = min(y_values)
        min_z = min(z_values)

        max_x = max(x_values)
        max_y = max(y_values)
        max_z = max(z_values)

        vertices = []
        for i in range(k):
            rand_x = random.uniform(min_x, max_x)
            rand_y = random.uniform(min_y, max_y)
            rand_z = random.uniform(min_z, max_z)
            vertices.append(np.array([rand_x, rand_y, rand_z]))

        return vertices

    def compute_new_vertices(self, vertices_vec, tracklet_end_points_vec, vertices_end_points_map):
        """Compute new vertex positions as the mean of assigned tracklet endpoints."""
        new_vertices = np.zeros(np.shape(vertices_vec))
        n_vertices_vec = np.zeros((np.shape(vertices_vec)[0], 1))

        for i, tracklet_end_points in enumerate(tracklet_end_points_vec):
            for j, end_point in enumerate(tracklet_end_points):
                k = vertices_end_points_map[i][j]
                new_vertices[k] += tracklet_end_points_vec[i][j]
                n_vertices_vec[k] += 1

        for k, vertex in enumerate(new_vertices):
            if n_vertices_vec[k] != 0:
                new_vertices[k] /= n_vertices_vec[k]
            else:
                new_vertices[k] = self.create_vertex_guess(1, tracklet_end_points_vec)[0]

        return new_vertices

    def constrained_k_means(self, tracklet_end_points_vec, n_iters=5, sigma=0.3):
        """Run constrained k-means on the tracklet endpoints."""
        n_end_points = 0
        end_points = []
        for i, tracklet_end_points in enumerate(tracklet_end_points_vec):
            for j, end_point in enumerate(tracklet_end_points):
                n_end_points += 1
                end_points.append(tracklet_end_points_vec[i][j])

        min_bic = 1e9
        min_k = 1e9
        min_vertices_vec = []
        min_vertices_endpoints_map = []

        for k in range(1, n_end_points + 1):
            vertices_vec = self.create_vertex_guess(k, tracklet_end_points_vec)
            vertices_end_points_map = self.assign_vertices(vertices_vec, tracklet_end_points_vec)

            for iter in range(n_iters):
                vertices_vec = self.compute_new_vertices(vertices_vec, tracklet_end_points_vec, vertices_end_points_map)
                vertices_end_points_map = self.assign_vertices(vertices_vec, tracklet_end_points_vec)

            vertices_indices = []
            for i, vertices_end_points in enumerate(vertices_end_points_map):
                for j, index in enumerate(vertices_end_points):
                    vertices_indices.append(vertices_end_points_map[i][j])

            vertices_map = []
            for index in vertices_indices:
                vertices_map.append(vertices_vec[index])

            vertices_map = np.array(vertices_map)
            end_points = np.array(end_points)

            bic = self.BIC(sigma, k, vertices_map, end_points)
            if bic < min_bic:
                min_bic = bic
                min_k = k
                min_vertices_vec = vertices_vec
                min_vertices_endpoints_map=vertices_end_points_map

        return min_bic, min_k, min_vertices_vec, min_vertices_endpoints_map
        
    def determine_endpoints(self, tracklet: Tracklet) -> tuple[Optional[Point3D], Optional[Point3D]]:
        """Determines the endpoints of a tracklet based on the fit results and stores the endpoints."""
        
        # Apply the fitting function
        tracklet.fitter = fit_tracklet_hits  # Assign the fitting function
        tracklet.fit()  # Fit the tracklet using the fitter
        
        # Get the fitted y_min, y_max for back hits and x_min, x_max for front hits from fit_results
        y_min = tracklet.get_fit_results().get("y_z_fit", {}).get("y_min", None)
        y_max = tracklet.get_fit_results().get("y_z_fit", {}).get("y_max", None)
        
        x_min = tracklet.get_fit_results().get("x_z_fit", {}).get("x_min", None)
        x_max = tracklet.get_fit_results().get("x_z_fit", {}).get("x_max", None)
        
        # Get the min and max z values from fit_results
        min_z = tracklet.get_fit_results().get("min_z", None)
        max_z = tracklet.get_fit_results().get("max_z", None)

        # Create the endpoints using the fitted values and the z values from the fit results
        endpoint_0 = Point3D(x_min, y_min, min_z)
        endpoint_1 = Point3D(x_max, y_max, max_z)

        # Set the endpoints in the tracklet
        tracklet.set_endpoints(endpoint_0, endpoint_1)

        return endpoint_0, endpoint_1


    def form_vertices(self, tracklets: Set[Tracklet]) -> tuple[Set[Vertex], dict]:
        """Find vertices for the given set of tracklets using the constrained k-means algorithm."""

        tracklets = list(tracklets)
        front_endpoints = []
        back_endpoints = []

        for tracklet in tracklets:
            endpoint_0, endpoint_1 = self.determine_endpoints(tracklet)

            tracklet_front_endpoints = []
            tracklet_back_endpoints = []

            if endpoint_0:
                if endpoint_0.x is not None and endpoint_0.z is not None:
                    tracklet_front_endpoints.append(np.array([endpoint_0.x, 0, endpoint_0.z]))
                if endpoint_0.y is not None and endpoint_0.z is not None:
                    tracklet_back_endpoints.append(np.array([0, endpoint_0.y, endpoint_0.z]))

            if endpoint_1:
                if endpoint_1.x is not None and endpoint_1.z is not None:
                    tracklet_front_endpoints.append(np.array([endpoint_1.x, 0, endpoint_1.z]))
                if endpoint_1.y is not None and endpoint_1.z is not None:
                    tracklet_back_endpoints.append(np.array([0, endpoint_1.y, endpoint_1.z]))

            if tracklet_front_endpoints:
                front_endpoints.append(tracklet_front_endpoints)

            if tracklet_back_endpoints:
                back_endpoints.append(tracklet_back_endpoints)

        # Run constrained k-means clustering
        min_BIC_f, min_k_f, centroids_f, vertex_endpoint_map_f = self.constrained_k_means(front_endpoints, n_iters=self.n_iters, sigma=self.sigma)
        min_BIC_b, min_k_b, centroids_b, vertex_endpoint_map_b = self.constrained_k_means(back_endpoints, n_iters=self.n_iters, sigma=self.sigma)

        front_vertex_to_tracklets = defaultdict(set)
        back_vertex_to_tracklets = defaultdict(set)

        for i, vertex_indices in enumerate(vertex_endpoint_map_f):
            for v_idx in vertex_indices:
                front_vertex_to_tracklets[v_idx].add(tracklets[i])

        for i, vertex_indices in enumerate(vertex_endpoint_map_b):
            for v_idx in vertex_indices:
                back_vertex_to_tracklets[v_idx].add(tracklets[i])

        vertices_f = set()
        vertices_b = set()

        for v_idx, tracklets_set in front_vertex_to_tracklets.items():
            vertex = Vertex(vertex_id=v_idx)
            for tracklet in tracklets_set:
                vertex.add_tracklet(tracklet)
            vertices_f.add(vertex)

        for v_idx, tracklets_set in back_vertex_to_tracklets.items():
            vertex = Vertex(vertex_id=v_idx)
            for tracklet in tracklets_set:
                vertex.add_tracklet(tracklet)
            vertices_b.add(vertex)

        # Vertex set comparison logic
        result_info = {}
        result_info["vertex_comparison"] = {
            "only_in_front": vertices_f - vertices_b,
            "only_in_back": vertices_b - vertices_f,
            "match": vertices_f == vertices_b,
        }

        # Store BIC and k values in the result map
        result_info["stats"] = {
            "front": {
                "BIC": min_BIC_f,
                "k": min_k_f,
                "centroids": centroids_f,
            },
            "back": {
                "BIC": min_BIC_b,
                "k": min_k_b,
                "centroids": centroids_b,
            }
        }

        return vertices_f, result_info





