from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np


def init_kmeans_sklearn(n_clusters, batch_size, seed, init_centroids='random'):
    # init kmeans
    if batch_size is not None:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=100,
            tol=0,  # 0.0001 (if not zero, adds compute overhead)
            n_init=1,
            # verbose=True,
            batch_size=batch_size,
            compute_labels=True,
            max_no_improvement=100,  # None
            init_size=None,
            reassignment_ratio=0.1 / n_clusters,
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=100,
            tol=0.001,
            n_init=1,
            # verbose=True,
            precompute_distances=True,
            algorithm='full',  # 'full',  # 'elkan',
        )
    return kmeans


def test_sk():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
    # print(X)
    print(X.shape)
    print(kmeans.labels_)
    print(kmeans.predict([[0, 0], [12, 3]]))
    print(kmeans.cluster_centers_)


if __name__ == "__main__":
    test_sk()
