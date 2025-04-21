import random
import numpy as np
from typing import Set, Optional
from models.tracklet import Tracklet
from models.vertex import Vertex
from algorithms.vertex.vertex_former import VertexFormer
from models.point_3d import Point3D
from utils.utils import fit_tracklet_hits
from collections import defaultdict
from sklearn.cluster import DBSCAN

class KMeansVertexFormer(VertexFormer):
    def __init__(self, n_iters=5, sigma=0.5, plane="front", planes_to_run={"front", "back", "both"}, seed_method = "random"):
        """KMeans-based vertex formation for specified plane: 'front', 'back', or 'both'."""
        self.n_iters = n_iters
        self.sigma = sigma
        self.plane = plane  # 'front', 'back', or 'both'
        self.seed_method = seed_method # 'random' or 'at_endpoints'
        
        # Ensure the requested output plane is actually run
        if plane not in planes_to_run:
            planes_to_run = set(planes_to_run)  # Convert to set if not already
            planes_to_run.add(plane)

        self.planes_to_run = planes_to_run  # Subset of planes to actually run k-means on



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
        """Wrapper that selects vertex guessing strategy."""
        if self.seed_method == "distance":
            return self.create_vertex_guess_by_distance(k, tracklet_end_points_vec)
        else:
            return self.create_vertex_guess_random(k, tracklet_end_points_vec)

    def create_vertex_guess_random(self, k, tracklet_end_points_vec):
        """Randomly generate k vertex guesses within the bounding box of all endpoints."""
        x_values = []
        y_values = []
        z_values = []

        for endpoints in tracklet_end_points_vec:
            for end_point in endpoints:
                x_values.append(end_point[0])
                y_values.append(end_point[1])
                z_values.append(end_point[2])

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        min_z, max_z = min(z_values), max(z_values)

        vertices = []
        for _ in range(k):
            rand_x = random.uniform(min_x, max_x)
            rand_y = random.uniform(min_y, max_y)
            rand_z = random.uniform(min_z, max_z)
            vertices.append(np.array([rand_x, rand_y, rand_z]))

        return vertices

    def create_vertex_guess_by_distance(self, k, tracklet_end_points_vec):
        """Cluster endpoints (excluding intra-tracklet groups), then fill with random guesses if needed."""
        # Flatten all endpoints and keep track of their parent tracklet
        all_endpoints = []
        tracklet_ids = []
        for tracklet_idx, endpoints in enumerate(tracklet_end_points_vec):
            for endpoint in endpoints:
                all_endpoints.append(np.array(endpoint))
                tracklet_ids.append(tracklet_idx)

        if not all_endpoints:
            return self.create_vertex_guess_random(k, tracklet_end_points_vec)

        all_endpoints = np.array(all_endpoints)
        tracklet_ids = np.array(tracklet_ids)

        # Perform DBSCAN clustering
        db = DBSCAN(eps=0.5, min_samples=1).fit(all_endpoints)
        labels = db.labels_
        unique_labels = set(labels)

        cluster_centroids = []
        for label in unique_labels:
            # Get indices of points in this cluster
            indices = np.where(labels == label)[0]

            # Extract the tracklet IDs in this cluster
            cluster_tracklet_ids = tracklet_ids[indices]

            # If no repeated tracklet IDs, accept the cluster
            if len(set(cluster_tracklet_ids)) == len(cluster_tracklet_ids):
                cluster_points = all_endpoints[indices]
                centroid = np.mean(cluster_points, axis=0)
                cluster_centroids.append(centroid)

        # Return as many centroids as we can, fill the rest with random guesses
        if len(cluster_centroids) >= k:
            return cluster_centroids[:k]

        remaining = k - len(cluster_centroids)
        random_guesses = self.create_vertex_guess_random(remaining, tracklet_end_points_vec)
        return cluster_centroids + random_guesses



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
    
    def check_if_empty_vertex(self, k, indices):
        for i in range(0,k):
            if i not in indices:
                return True
        return False

    def constrained_k_means(self, tracklet_end_points_vec, n_iters=5, sigma=0.3):
        """Run the constrained k-means algorithm over the end points."""

        n_end_points = 0
        end_points = []
        for i, tracklet_end_points in enumerate(tracklet_end_points_vec):
            for j, end_point in enumerate(tracklet_end_points):
                n_end_points += 1
                end_points.append(tracklet_end_points_vec[i][j])

        min_bic = 1e9
        min_k = 1e9
        min_iter = 0
        min_vertices_vec = []
        all_results = []  # To store BIC, k, and vertices for each attempt

        for k in range(1, n_end_points + 1):
            vertices_vec = self.create_vertex_guess(k, tracklet_end_points_vec)
            vertices_end_points_map = self.assign_vertices(vertices_vec, tracklet_end_points_vec)

            for iter in range(n_iters):
                iteration_index = iter + 1
                vertices_vec = self.compute_new_vertices(vertices_vec, tracklet_end_points_vec, vertices_end_points_map)
                vertices_end_points_map = self.assign_vertices(vertices_vec, tracklet_end_points_vec)

                vertices_indices = []
                for i, vertices_end_points in enumerate(vertices_end_points_map):
                    for j, index in enumerate(vertices_end_points):
                        vertices_indices.append(vertices_end_points_map[i][j])

                unique_vertex_indices = np.unique(vertices_indices)
                is_empty_vertex = self.check_if_empty_vertex(k, unique_vertex_indices)
                if is_empty_vertex:
                    continue

                vertices_map = []
                for index in vertices_indices:
                    vertices_map.append(vertices_vec[index])
                vertices_map = np.array(vertices_map)
                end_points_arr = np.array(end_points)

                bic = self.BIC(sigma, k, vertices_map, end_points_arr)

                all_results.append({
                    "bic": bic,
                    "k": k,
                    "vertices": vertices_vec.copy(),  # or deepcopy if needed
                    "iteration": iteration_index
                })

                if bic < min_bic:
                    min_bic = bic
                    min_k = k
                    min_vertices_vec = vertices_vec
                    min_iter = iteration_index

        vertices_end_points_map = self.assign_vertices(min_vertices_vec, tracklet_end_points_vec)
        return min_bic, min_k, min_vertices_vec, min_iter, vertices_end_points_map, all_results
    
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

    def create_vertices_from_map(self, vertex_endpoint_map, tracklets):
        """Helper function to create Vertex objects from the vertex-endpoint map."""
        vertex_to_tracklets = defaultdict(set)
        for i, vertex_indices in enumerate(vertex_endpoint_map):
            for v_idx in vertex_indices:
                vertex_to_tracklets[v_idx].add(tracklets[i])

        vertices = set()
        for v_idx, tracklets_set in vertex_to_tracklets.items():
            vertex = Vertex(vertex_id=v_idx)
            for tracklet in tracklets_set:
                vertex.add_tracklet(tracklet)
            vertices.add(vertex)

        return vertices
    
    def form_vertices(self, tracklets: Set[Tracklet]) -> tuple[Set[Vertex], dict]:
        """Find vertices for the given set of tracklets using the constrained k-means algorithm."""

        tracklets = list(tracklets)
        front_endpoints = []
        back_endpoints = []
        both_endpoints = []

        for tracklet in tracklets:
            endpoint_0, endpoint_1 = self.determine_endpoints(tracklet)

            tracklet_front_endpoints = []
            tracklet_back_endpoints = []
            tracklet_both_endpoints = []

            if endpoint_0:
                if endpoint_0.x is not None and endpoint_0.z is not None:
                    tracklet_front_endpoints.append(np.array([endpoint_0.x, 0, endpoint_0.z]))
                if endpoint_0.y is not None and endpoint_0.z is not None:
                    tracklet_back_endpoints.append(np.array([0, endpoint_0.y, endpoint_0.z]))
                if endpoint_0.x is not None and endpoint_0.y is not None and endpoint_0.z is not None:
                    tracklet_both_endpoints.append(np.array([endpoint_0.x, endpoint_0.y, endpoint_0.z]))

            if endpoint_1:
                if endpoint_1.x is not None and endpoint_1.z is not None:
                    tracklet_front_endpoints.append(np.array([endpoint_1.x, 0, endpoint_1.z]))
                if endpoint_1.y is not None and endpoint_1.z is not None:
                    tracklet_back_endpoints.append(np.array([0, endpoint_1.y, endpoint_1.z]))
                if endpoint_1.x is not None and endpoint_1.y is not None and endpoint_1.z is not None:
                    tracklet_both_endpoints.append(np.array([endpoint_1.x, endpoint_1.y, endpoint_1.z]))

            if tracklet_front_endpoints:
                front_endpoints.append(tracklet_front_endpoints)

            if tracklet_back_endpoints:
                back_endpoints.append(tracklet_back_endpoints)

            if tracklet_both_endpoints:
                both_endpoints.append(tracklet_both_endpoints)

        result_info = {"vertex_comparison": {}, "stats": {}}
        vertices_front = vertices_back = vertices_both = set()

        if "front" in self.planes_to_run:
            min_BIC_f, min_k_f, centroids_f, min_iter_f, vertex_endpoint_map_f, all_results_f = self.constrained_k_means(
                front_endpoints, n_iters=self.n_iters, sigma=self.sigma
            )
            vertices_front = self.create_vertices_from_map(vertex_endpoint_map_f, tracklets)
            result_info["vertex_comparison"]["front_vertices"] = vertices_front
            result_info["stats"]["front"] = {
                "BIC": min_BIC_f,
                "k": min_k_f,
                "centroids": centroids_f,
                "iteration": min_iter_f,
                "all_iterations": all_results_f,
            }

        if "back" in self.planes_to_run:
            min_BIC_b, min_k_b, centroids_b, min_iter_b, vertex_endpoint_map_b, all_results_b = self.constrained_k_means(
                back_endpoints, n_iters=self.n_iters, sigma=self.sigma
            )
            vertices_back = self.create_vertices_from_map(vertex_endpoint_map_b, tracklets)
            result_info["vertex_comparison"]["back_vertices"] = vertices_back
            result_info["stats"]["back"] = {
                "BIC": min_BIC_b,
                "k": min_k_b,
                "centroids": centroids_b,
                "iteration": min_iter_b,
                "all_iterations": all_results_b,
            }

        if "both" in self.planes_to_run:
            min_BIC_both, min_k_both, centroids_both, min_iter_both, vertex_endpoint_map_both, all_results_both = self.constrained_k_means(
                both_endpoints, n_iters=self.n_iters, sigma=self.sigma
            )
            vertices_both = self.create_vertices_from_map(vertex_endpoint_map_both, tracklets)
            result_info["vertex_comparison"]["both_vertices"] = vertices_both
            result_info["stats"]["both"] = {
                "BIC": min_BIC_both,
                "k": min_k_both,
                "centroids": centroids_both,
                "iteration": min_iter_both,
                "all_iterations": all_results_both,
            }


        # Return results for the requested plane
        if self.plane == "front":
            vertices = vertices_front
        elif self.plane == "back":
            vertices = vertices_back
        else:  # "both"
            vertices = vertices_both

        return vertices, result_info

