import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=1500, random_state=42):
    """
    Generates three 2D clusters, each with n_samples//3 points.
    Returns three arrays (each with shape (n_samples//3, 2)).
    """
    np.random.seed(random_state)
    # Let’s create three clusters centered around (2, 2), (6, 6), and (10, 2).
    cluster1 = np.random.randn(n_samples // 3, 2) + np.array([2, 2])
    cluster2 = np.random.randn(n_samples // 3, 2) + np.array([6, 6])
    cluster3 = np.random.randn(n_samples // 3, 2) + np.array([10, 2])
    return cluster1, cluster2, cluster3


def create_labeled_dataset(cluster1, cluster2, cluster3):
    """
    Stacks the three clusters into one array and adds a label column:
    cluster1 -> label 1
    cluster2 -> label 2
    cluster3 -> label 3

    Returns a (total_points, 3) array: [x, y, label].
    """
    # Add label column to each cluster
    cluster1_with_label = np.hstack((cluster1, np.ones((cluster1.shape[0], 1)) * 1))
    cluster2_with_label = np.hstack((cluster2, np.ones((cluster2.shape[0], 1)) * 2))
    cluster3_with_label = np.hstack((cluster3, np.ones((cluster3.shape[0], 1)) * 3))

    # Combine all clusters
    data = np.vstack((cluster1_with_label, cluster2_with_label, cluster3_with_label))
    return data


def visualize_dataset_with_labels(data):
    """
    Visualizes the dataset in different colors for each ground-truth label.
    Expects data to have shape (n_samples, 3) => columns: x, y, label.
    """
    # Extract unique labels
    labels = np.unique(data[:, 2]).astype(int)

    # Optional: define colors for each cluster label
    cluster_colors = ['red', 'blue', 'green']

    plt.title("Three Generated Clusters with Ground Truth Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Plot each label in a different color
    for i, lbl in enumerate(labels):
        points = data[data[:, 2] == lbl]
        plt.scatter(points[:, 0], points[:, 1],
                    color=cluster_colors[i % len(cluster_colors)],
                    alpha=0.7,
                    label=f"Cluster {lbl}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 1. Generate three separate 2D clusters
    c1, c2, c3 = generate_data(n_samples=1500, random_state=42)

    # 2. Add ground-truth labels (1, 2, 3) as a third column
    data_with_labels = create_labeled_dataset(c1, c2, c3)

    # 3. Save the data (x, y, label) to a CSV file
    #    Each row looks like: x_value, y_value, label
    # np.savetxt('kmeans_data_with_labels.csv', data_with_labels, delimiter=',')

    # 4. Visualize the dataset in different colors by label
    visualize_dataset_with_labels(data_with_labels)
