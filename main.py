import numpy as np


def initialize_centroids(data, k, random_state=42):
    """
    Initializes k centroids by randomly sampling from the data.
    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_features).
        k (int): Number of clusters.
        random_state (int): Seed for reproducibility.

    Returns:
        centroids (np.ndarray): Initial centroids of shape (k, n_features).
    """
    np.random.seed(random_state)
    # Randomly choose k indices
    random_indices = np.random.choice(len(data), size=k, replace=False)
    centroids = data[random_indices]



    return centroids

def l2_distance(a, b):
    """
    Euclidean (L2) distance between vectors a and b.
    """
    return np.linalg.norm(a - b)

def l1_distance(a, b):
    """
    Manhattan (L1) distance between vectors a and b.
    """
    return np.sum(np.abs(a - b))

# 2. Dictionary to map distance "names" to actual functions
DISTANCE_FUNCTIONS = {
    "l2": l2_distance,
    "l1": l1_distance,
    # Add more if needed, e.g., "cosine", "chebyshev", etc.
}

def get_distance_func(name="l2"):
    """
    Retrieves a distance function from the DISTANCE_FUNCTIONS dict.
    Raises a ValueError if the requested name is not found.
    """
    if name not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown distance function: {name}")
    return DISTANCE_FUNCTIONS[name]


def assign_clusters(data, centroids, distance_type="l2"):
    """
    Assigns each data point to the nearest centroid using the specified distance metric.

    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_features).
        centroids (np.ndarray): 2D array of shape (k, n_features).
        distance_type (str): The name of the distance function to use.
                             Possible values: 'l2', 'l1', etc.

    Returns:
        np.ndarray: A 1D array of cluster labels (integers).
    """
    dist_func = get_distance_func(distance_type)
    labels = []

    for point in data:
        # Compute distance to each centroid
        distances = [dist_func(point, c) for c in centroids]
        # Find the centroid with the minimum distance
        label = np.argmin(distances)
        labels.append(label)

    return np.array(labels)


def update_centroids(data, labels, k):
    """
    Given the current cluster assignments, compute the new centroid of each cluster.
    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_features).
        labels (np.ndarray): 1D array of cluster labels.
        k (int): Number of clusters.

    Returns:
        new_centroids (np.ndarray): 2D array of updated centroid positions.
    """
    # TODO: Implement logic to recalculate each centroid as the mean of its cluster members
    pass


def kmeans(data, k, max_iter=100, tolerance=1e-4, random_state=42):
    """
    Runs K-Means clustering on the input data.

    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_features).
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tolerance (float): Convergence threshold.
        random_state (int): Seed for centroid initialization.

    Returns:
        (centroids, labels)
    """
    # 1. Initialize centroids
    centroids = initialize_centroids(data, k, random_state)

    for i in range(max_iter):
        # 2. Assign clusters
        labels = assign_clusters(data, centroids)

        # 3. Compute new centroids
        new_centroids = update_centroids(data, labels, k)

        # 4. Check for convergence
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tolerance:
            print(f"Converged after {i} iterations.")
            break

        centroids = new_centroids

    return centroids, labels


if __name__ == "__main__":
    # Load or generate data
    data = np.loadtxt('kmeans_data.csv', delimiter=',')
    # Attempt to run K-Means
    final_centroids, cluster_labels = kmeans(data, k=3)
    print("Centroids:", final_centroids)
