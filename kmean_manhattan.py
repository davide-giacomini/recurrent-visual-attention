import argparse
import os
import numpy as np
import pandas as pd

class KMeansManhattan:
    def __init__(self, X, K, max_iters=100, num_inits=10, print_deb=False):
        self.X = X
        self.K = K
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.print_deb = print_deb

    def process(self):
        """ Processes the K-Means quantization.

        Args:
            X: matrix with rows of 1-dimensional vectors (N,M)
            K: number of clusters
            max_iters: maximum number of iterations
            num_inits: how many initializations

        Returns:
            labels: a matrix (N,1) which matches the rows to the correspondent cluster
            best_centers: a matrix (K,M) which matches the clusters to the correspondent center vector
            best_sse: the SSE correspondent to the resulting centers
        """
        best_centers = None
        best_sse = np.inf

        # Perform multiple initializations and keep the best set of cluster centers
        for i in range(self.num_inits):
            labels, centers = self.__kmeans_clustering(self.X, self.K, self.max_iters)

            # Compute the sum of squared errors (SSE) for the current set of cluster centers
            sse = np.sum([np.sum((self.X[labels == k, :] - centers[k])**2) for k in range(self.K)])

            # Update the best set of cluster centers if the SSE is smaller than the previous best
            if sse < best_sse:
                best_centers = centers
                best_sse = sse

            if self.print_deb:
                print(i)

        return labels, best_centers, best_sse

    def __kmeans_plusplus(self, X, K):
        # Initialize the first center randomly
        centers = np.array([ X[np.random.choice(X.shape[0]), :] ])
        # centers = np.expand_dims(centers, axis=0)

        # Initialize the remaining centers using the K-means++ method
        for i in range(K - 1):
            # Calculate the Manhattan distance from each row to its nearest center
            distances = np.array([min(self.__compute_distances(centers, x)) for x in X])

            # Calculate the probability of each point being the next center (FIXME: if there are too classes (K too high), distances.sum()==0)
            probabilities = distances / distances.sum()

            # Choose the next center randomly with the calculated probabilities
            centers = np.vstack([centers, X[np.random.choice(X.shape[0], p=probabilities)]])

        return centers

    def __kmeans_clustering(self, X, K, max_iters):
        # Initialize K cluster centers using K-means++
        centers = X[np.random.choice(X.shape[0], size=K, replace=False)]

        # Initialize variables
        N = X.shape[0]
        labels = np.zeros(N)
        distances = np.zeros((N, K))

        # Iterate until convergence or maximum number of iterations
        for i in range(max_iters):
            # Assign each data point to its nearest cluster center, Manhattan distance
            for k in range(K):
                # Compute distances for each row of X from each center
                distances[:, k] = self.__compute_distances(X, centers[k])   # (N,K) --> N data points, K centroids

            labels = np.argmin(distances, axis=1)   # (N,K).argmin(dim=-1) = (N,) --> N data points, 1 closest centroid

            # Recalculate the centroid of each cluster
            new_centers = np.empty_like(centers)
            for k in range(K):
                # Check if the cluster is empty, by checking if at least 1 data point has the closest centroid == k
                if not np.any(labels == k):
                    filled_clusters = []

                    closest_cluster, closest_point = self.__find_closest_cluster(distances, k, filled_clusters, labels)
                    self.__move_point(closest_cluster, k, closest_point, labels, distances, filled_clusters)

                new_centers[k] = np.mean(X[labels == k, :], axis=0)

            # Check if the cluster centers have converged
            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        return labels, centers
    
    def __move_point(self, old_cluster, new_cluster, closest_point, labels, distances, filled_clusters):
        labels[closest_point] = new_cluster
        filled_clusters.append(new_cluster)

        # Check if old cluster is now empty, by checking if at least 1 data point has the closest centroid == old_cluster
        if not np.any(labels == old_cluster):
            closest_cluster_to_old_cluster, closest_point_to_old_cluster = self.__find_closest_cluster(distances, old_cluster, filled_clusters, labels)
            self.__move_point(closest_cluster_to_old_cluster, old_cluster, closest_point_to_old_cluster, labels, distances, filled_clusters)

    def __find_closest_cluster(self, distances, cluster, filled_clusters, labels):
        # Find the cluster closest to cluster k
        distances_to_cluster = np.zeros(distances.shape[0]) # (N,) --> I can't modify "distances"
        distances_to_cluster[:] = distances[:, cluster]    # (N,) --> distances from each point to centroid of cluster

        for filled in filled_clusters:
            distances_to_cluster[labels == filled] = np.inf # Exclude clusters already visited

        closest_point = np.argmin(distances_to_cluster) # (N,).argmin() --> closest point from centroid of cluster
        closest_cluster = labels[closest_point]   # Cluster of the closest point, which is the cluster closest to our cluster

        return closest_cluster, closest_point

    def __compute_distances(self, X: np.ndarray, t: np.ndarray, a: float = 29.98, b: float = 16.08, c: float = 9.93) -> np.ndarray:
        """
        Compute weighted Manhattan distances between the vectors in X and the target vector t.
        
        Args:
        - X (np.ndarray): An array of shape (N, M) representing the N vectors in the dataset with M features.
        - t (np.ndarray): An array of shape (M,) representing a single target vector with M features.
        - a (float): A weighting factor.
        - b (float): A weighting factor.
        - c (float): A weighting factor.
        
        Returns:
        - distances (np.ndarray): An array of shape (N,) representing the weighted Manhattan distances between each vector in X and the target vector t.
        """
        X_h = X[:, :64]
        X_l = X[:, 64:66]
        X_p = X[:, 66:114]

        t_h = t[:64]
        t_l = t[64:66]
        t_p = t[66:114]

        p_distances = np.sum(np.abs(X_p - t_p), axis=-1)
        h_distances = np.sum(np.abs(X_h - t_h), axis=-1)
        l_distances = np.sum(np.abs(X_l - t_l), axis=-1)

        distances = (a*p_distances + b*h_distances + c*l_distances) / (a+b+c)

        return distances
    
# Define the command line arguments
parser = argparse.ArgumentParser(description='Read from a CSV file')
parser.add_argument('filename', type=str, nargs='?', default='training_table_50.csv', help='Path to the CSV file')

# Parse the command line arguments
args, unparsed = parser.parse_known_args()

# Create the output directory if it does not exist
root, extension = os.path.splitext(args.filename)
base_dir = root + '_clusters'
os.makedirs(base_dir, exist_ok=True)

K = 32

# Create first level of clusters
cluster_df = pd.read_csv(args.filename, header=None)

X = cluster_df.values
kmeans = KMeansManhattan(X, K)
labels, centers, best_sse = kmeans.process()


for c in range(centers.shape[0]):
    # Select the rows of X that belong to cluster k
    x_cluster = X[labels == c]

    # Select the k-th row of the cluster centers
    center_row = np.reshape(centers[c], (1,-1))

    # Create a Pandas DataFrame for the cluster and center data
    cluster_df = pd.DataFrame(x_cluster)
    center_df = pd.DataFrame(center_row)

    # Save the cluster and center data to separate CSV files
    cluster_filename = os.path.join(base_dir, f'{c}_cluster.csv')
    center_filename = os.path.join(base_dir, f'{c}_center.csv')
    cluster_df.to_csv(cluster_filename, index=False, header=False)
    center_df.to_csv(center_filename, index=False, header=False)


def build_clusters(base_dir, k):
    # Check if k_cluster.csv exists and has at least k rows. If it doesn't exist it means that k>K
    cluster_path = os.path.join(base_dir, f'{k}_cluster.csv')
    if not os.path.exists(cluster_path):
        return

    cluster_df = pd.read_csv(cluster_path, header=None)

    if len(cluster_df) <= K:
        return

    # Perform clustering
    X = cluster_df.values
    kmeans = KMeansManhattan(X, K)
    labels, centers, best_sse = kmeans.process()

    # Create the output directory if it does not exist
    output_dir = os.path.join(base_dir, str(k))
    os.makedirs(output_dir, exist_ok=True)

    for c in range(centers.shape[0]):
        # Select the rows of X that belong to cluster k
        x_cluster = X[labels == c]

        # Select the k-th row of the cluster centers
        center_row = np.reshape(centers[c], (1,-1))

        # Create a Pandas DataFrame for the cluster and center data
        cluster_df = pd.DataFrame(x_cluster)
        center_df = pd.DataFrame(center_row)

        # Save the cluster and center data to separate CSV files
        cluster_filename = os.path.join(output_dir, f'{c}_cluster.csv')
        center_filename = os.path.join(output_dir, f'{c}_center.csv')
        cluster_df.to_csv(cluster_filename, index=False, header=False)
        center_df.to_csv(center_filename, index=False, header=False)

    for clust_number in range(K):
        build_clusters(output_dir, clust_number)
        clust_number += 1


for c in range(K):
    build_clusters(base_dir, c)
