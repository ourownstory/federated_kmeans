import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

project_dir = os.path.dirname(os.getcwd())


def davies_bouldin(x, labels, centroids, verbose=False):
    # DIY
    NUM_CLUSTERS = centroids.shape[0]
    distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))

    # centroid distances
    centroid_dist_matrix = np.expand_dims(centroids, axis=0) - np.expand_dims(centroids, axis=1)
    centroid_dist_matrix = np.sqrt(np.sum(np.square(centroid_dist_matrix), axis=2))
    # print(centroid_dist_matrix - metrics.pairwise.euclidean_distances(X=centroids, Y=centroids))

    centroid_dist_matrix[range(NUM_CLUSTERS), range(NUM_CLUSTERS)] = float("inf")
    # print(centroid_dist_matrix)

    # intra cluster dist
    intra_dist = np.zeros(NUM_CLUSTERS)

    for i in range(NUM_CLUSTERS):
        intra_dist[i] = np.mean(distances[i == labels])

    s_ij = np.expand_dims(intra_dist, axis=0) + np.expand_dims(intra_dist, axis=1)
    d_i = np.nanmax(s_ij / centroid_dist_matrix, axis=1)
    db_score = np.nanmean(d_i)
    if verbose:
        print("centroid_min_dist", np.amin(centroid_dist_matrix, axis=1))
        print("intra_dist", intra_dist)
    return db_score


def euclidean_dist(x, labels, centroids):
    distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))
    dist = np.mean(distances)
    return dist


def evaluate(kmeans, x, splits, use_metric='euclidean', federated=False, verbose=False):
    scores = {}
    centroids = kmeans.cluster_centers_
    for split in splits:
        if federated:
            x[split] = np.concatenate(x[split], axis=0)
        labels = kmeans.predict(x[split])
        if verbose:
            print(split, use_metric)
        if "davies_bouldin" == use_metric:
            score = davies_bouldin(x[split], labels, centroids, verbose)
        elif "silhouette" == use_metric:
            # silhouette (takes forever) -> need metric with linear execution time wrt data size
            score = metrics.silhouette_score(x[split], labels)
        else:
            assert use_metric == 'euclidean'
            score = euclidean_dist(x[split], labels, centroids)
        scores[split] = score
        if verbose:
            print(score)
    return scores


def plot_stats(stats, x_variable, x_variable_name, metric_name):
    for spl, spl_dict in stats.items():
        for stat, stat_values in spl_dict.items():
            stats[spl][stat] = np.array(stat_values)

    if x_variable[-1] is None:
        x_variable[-1] = 1
    x_variable = ["single" if i == 0.0 else i for i in x_variable]
    x_axis = np.array(range(len(x_variable)))

    plt.plot(stats['train']['avg'], 'r-', label='Train')
    plt.plot(stats['test']['avg'], 'b-', label='Test')
    plt.fill_between(
        x_axis,
        stats['train']['avg'] - stats['train']['std'],
        stats['train']['avg'] + stats['train']['std'],
        facecolor='r',
        alpha=0.3,
    )
    plt.fill_between(
        x_axis,
        stats['test']['avg'] - stats['test']['std'],
        stats['test']['avg'] + stats['test']['std'],
        facecolor='b',
        alpha=0.2,
    )
    plt.xticks(x_axis, x_variable)
    plt.xlabel(x_variable_name)
    plt.ylabel(metric_name)
    plt.legend()
    fig_path = os.path.join(project_dir, "results")
    plt.savefig(os.path.join(fig_path, "stats_{}.png".format(x_variable_name)), dpi=600, bbox_inches='tight')
    plt.show()


def plot_progress(progress_means, progress_stds, record_at):
    #  NOTE: only for dummy data
    # print(len(progress_means), progress_means[0].shape)
    # print(len(progress_stds), progress_stds[0].shape)
    num_clusters = progress_means[0].shape[0]
    num_records = len(progress_means)
    true_means = np.arange(1, num_clusters+1)
    fig = plt.figure()
    for i in range(num_clusters):
        ax = fig.add_subplot(1, 1, 1)
        x_axis = np.array(range(num_records))
        true_means_i = np.repeat(true_means[i], num_records)
        means = np.array([x[i] for x in progress_means])
        stds = np.array([x[i] for x in progress_stds])
        ax.plot(means, 'r-', label='centroid mean')
        ax.plot(true_means_i, 'b-', label='true mean')
        ax.fill_between(
            x_axis,
            means - stds,
            means + stds,
            facecolor='r',
            alpha=0.4,
            label='centroid std',
        )
        # ax.fill_between(
        #     x_axis,
        #     true_means_i - 0.1,
        #     true_means_i + 0.1,
        #     facecolor='b',
        #     alpha=0.1,
        #     label='true std',
        # )
        plt.xticks(x_axis, record_at)
    plt.xlabel("Round")
    plt.ylabel("Cluster distribution")
    # plt.legend()
    fig_path = os.path.join(project_dir, "results")
    plt.savefig(os.path.join(fig_path, "stats_{}.png".format("progress")), dpi=600, bbox_inches='tight')
    plt.show()



