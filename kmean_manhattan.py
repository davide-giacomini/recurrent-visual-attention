import numpy as np

class KMeansManhattan:
    def __init__(self, X, K, max_iters=100, num_inits=10, print=False):
        self.X = X
        self.K = K
        self.max_iters = max_iters
        self.num_inits = num_inits
        self.print = print

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
            sse = np.sum([np.sum(np.abs(self.X[labels == k, :] - centers[k])**2) for k in range(self.K)])

            # Update the best set of cluster centers if the SSE is smaller than the previous best
            if sse < best_sse:
                best_centers = centers
                best_sse = sse

            if print:
                print(i)

        return labels, best_centers, best_sse

    def __kmeans_plusplus(self, X, K):
        # Initialize the first center randomly
        centers = [ X[np.random.choice(X.shape[0]), :] ]

        # Initialize the remaining centers using the K-means++ method
        for i in range(K - 1):
            # Calculate the Manhattan distance from each row to its nearest center
            distances = np.array([min([np.sum(np.abs(x - c)) for c in centers]) for x in X])

            # Calculate the probability of each point being the next center (FIXME: if there are too classes (K too high), distances.sum()==0)
            probabilities = distances / distances.sum()

            # Choose the next center randomly with the calculated probabilities
            centers.append(X[np.random.choice(X.shape[0], p=probabilities)])

        return np.array(centers)

    def __kmeans_clustering(self, X, K, max_iters):
        # Initialize K cluster centers using K-means++
        centers = self.__kmeans_plusplus(X, K)

        # Initialize variables
        N = X.shape[0]
        labels = np.zeros(N)
        distances = np.zeros((N, K))

        # Iterate until convergence or maximum number of iterations
        for i in range(max_iters):
            # Assign each data point to its nearest cluster center, Manhattan distance
            for k in range(K):
                distances[:, k] = np.sum(np.abs(X - centers[k]), axis=1)
            labels = np.argmin(distances, axis=1)

            # Recalculate the centroid of each cluster
            for k in range(K):
                centers[k] = np.mean(X[labels == k, :], axis=0)

        return labels, centers