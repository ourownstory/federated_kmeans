import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

project_dir = os.path.dirname(os.getcwd())


def load(subsample_train_frac=None, num_train=None, num_test=None, verbose=False, seed=None):
    np.random.seed(seed)
    # params
    # print(project_dir)
    train_path = os.path.join(project_dir, "data", "pecan_train")
    test_path = os.path.join(project_dir, "data", "pecan_test")
    x = {}
    x['train'] = pd.read_csv(
        os.path.join(train_path, "x.csv"),
        index_col=False,
        header=None,
        nrows=num_train
    ).values
    x['test'] = pd.read_csv(
        os.path.join(test_path, "x.csv"),
        index_col=False,
        header=None,
        nrows=num_test
    ).values
    if subsample_train_frac:
        house_ids = pd.read_csv(
            os.path.join(train_path, "house_ids.csv"),
            index_col=False,
            header=None,
            nrows=num_train
        ).values[:, 0]
        unique_ids = np.unique(house_ids)
        total_ids_num = len(unique_ids)
        subsample_num = max(1, int(subsample_train_frac * total_ids_num))
        train_ids = np.random.choice(unique_ids, size=subsample_num, replace=False)

        # val_ids = list(set(unique_ids) - set(train_ids))
        # print(len(unique_ids))
        # print(len(train_ids))
        # print(len(val_ids))

        train_mask = np.isin(house_ids, train_ids)
        val_mask = np.invert(train_mask)
        x['val'] = x['train'][val_mask, :]
        x['train'] = x['train'][train_mask, :]

    if verbose:
        rand_plot_idx = np.random.choice(range(x['train'].shape[0]), size=5, replace=False)
        plt.plot(x['train'][rand_plot_idx, :].T)
        plt.show()
    return x


def create_dummy_data(dims=1, clients_per_cluster=10, samples_each=10, clusters=10, scale=0.5, verbose=False):
    num_clients = clients_per_cluster * clusters
    # create gaussian data set, per client one mean
    means = np.arange(1, clusters+1)
    means = np.tile(A=means, reps=clients_per_cluster)
    noise = np.random.normal(loc=0.0, scale=scale, size=(num_clients, samples_each, dims))
    data = np.expand_dims(np.expand_dims(means, axis=1), axis=2) + noise
    if verbose:
        # print(means)
        # print(noise)
        print("dummy data shape: ", data.shape)
    data = [data[i] for i in range(num_clients)]
    return data, means


def load_federated_dummy(seed=None, verbose=False, clients_per_cluster=10, clusters=10):
    # assert dims == 1, "only one dimension implemented"
    np.random.seed(seed)
    x = {}
    ids = {}
    data, means = create_dummy_data(clients_per_cluster=2*clients_per_cluster, clusters=clusters, verbose=verbose)
    mid = clients_per_cluster * clusters
    x["train"], ids["train"] = data[:mid], means[:mid]
    x["test"], ids["test"] = data[mid:], means[mid:]
    # print(len(x['train']), x['train'][0].shape)
    return x, ids


def load_federated(limit_csv=None, verbose=False, seed=None, dummy=False, clusters=None):
    if dummy:
        return load_federated_dummy(seed=seed, verbose=verbose, clusters=clusters)
    else:
        return load_federated_real(limit_csv, verbose, seed)


def load_federated_real(limit_csv=None, verbose=False, seed=None):
    np.random.seed(seed)
    x = {}
    client_ids = {}
    for spl in ['train', 'test']:
        path = os.path.join(project_dir, "data", "pecan_{}".format(spl))
        data = pd.read_csv(
            os.path.join(path, "x.csv"),
            index_col=False,
            header=None,
            nrows=limit_csv
        )
        house_ids = pd.read_csv(
            os.path.join(path, "house_ids.csv"),
            index_col=False,
            header=None,
            nrows=limit_csv
        ).values[:, 0]

        data.set_index(house_ids, inplace=True)

        # group by, return as list
        df = data.groupby(level=0, sort=True)
        df = df.apply(lambda y: y.values)
        x[spl] = list(df.values)
        client_ids[spl] = list(df.index.values)

        # print([d.shape for d in x[spl]])
        # print(client_ids[spl])

    if verbose:
        rand_plot_idx = np.random.choice(range(x['train'][0].shape[0]), size=5, replace=False)
        plt.plot(x['train'][0][rand_plot_idx, :].T)
        plt.show()
    return x, client_ids


def gaussian_density(mean, var, x):
    return np.exp(-np.square(x - mean) / (2*var)) / np.sqrt(2*np.pi*var)


def gaussian_24(mean, var):
    x = np.arange(24)
    return gaussian_density(mean, var, x)


def get_random_gaussian_mixtures(mix_num=2, repeats=1, dims=24, seed=None):
    np.random.seed(seed=seed)
    centroids = np.zeros((repeats, dims))
    for r in range(repeats):
        mean = np.random.randint(-2, dims+2, mix_num)
        var = np.random.randint(1, dims, mix_num)
        for i in range(mix_num):
            centroids[r, :] += gaussian_24(mean[i], var[i]) * (0.5 + np.random.random())
    centroids += 0.5
    centroids /= np.expand_dims(np.sum(centroids, axis=1), axis=1)
    centroids *= dims * (0.5 + np.random.random())
    return centroids


def init_centroids_gmm(init_centroids, num_clusters, seed, dims, verbose=False):
    if init_centroids == "GMM":
        init_centroids = get_random_gaussian_mixtures(
            mix_num=3,
            repeats=num_clusters,
            dims=dims,
            seed=seed,
        )
        if verbose:
            plt.plot(init_centroids.T)
            plt.show()
    return init_centroids


def test():
    # centroids = get_random_gaussian_mixtures(mix_num=2, repeats=3, dims=24)
    # plt.plot(centroids.T)
    # plt.show()
    # load_federated_real(verbose=True)
    create_dummy_data(dims=1, clients_per_cluster=2, samples_each=5, clusters=3, verbose=True)


if __name__ == "__main__":
    test()
