import numpy as np
import torch
from torch.utils.data import Dataset

norm = np.linalg.norm


def onehot(x):
    # if the array x is filled with integers then it makes each of this integer a class returning the onehot
    # representation in the last dimension of x
    x_unique = np.unique(x)
    y = np.zeros((x.shape[0], x_unique.shape[0]))
    for x_el, idx in enumerate(x_unique): y[np.where(x == x_el), int(idx)] = 1
    return y


def onehot2(x, N):
    # if the array x is filled with integers then it makes each of this integer a class returning the onehot
    # representation in the last dimension of x
    x_unique = np.arange(N)
    y = np.zeros((x.shape[0], x_unique.shape[0]))
    for x_el, idx in enumerate(x_unique): y[np.where(x == x_el), int(idx)] = 1
    return y


class InpData(Dataset):
    """Simple class for drawing samples from a pair of arrays X and Y."""

    def __init__(self, X, Y):
        """

        Parameters
        ----------
        X
            The data array
        Y
            The array of labels
        """
        # self.X = torch.from_numpy(X).float()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Gaussian_Spheres(Dataset):
    """ Class for drawing samples from isotropic gaussian distribution with multiple centers.
    """

    def __init__(self, centers, center_labels, final_time, max_samples=None, noise_sig=0.1, nonzero_time_points=None,
                 squeeze=False):
        self.centers = torch.from_numpy(centers).float()
        self.center_labels = torch.from_numpy(center_labels).long()
        self.max_samples = max_samples
        self.num_class_labels = len(np.unique(center_labels))
        self.final_time = final_time
        self.nonzero_time_points = nonzero_time_points
        self.noise_sig = noise_sig
        self.squeeze = squeeze
        self.cluster_identity = None
        self.seed_cnt = 0

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        X = self.centers.clone()
        Y = self.center_labels.clone()

        idx_singleton = False
        if isinstance(idx, slice):
            num_draws = len(range(*idx.indices(self.__len__())))
        elif hasattr(idx, '__len__'):
            num_draws = len(idx)
        else:
            num_draws = 1
            idx_singleton = True

        # This is a uniform draw from the clusters. There is no guarantee of draws being spread out evenly.
        # sequence = np.random.choice(self.centers.shape[0], num_draws, True)

        # This is a draw that tries to hit clusters as evenly as possible.
        m = np.mod(num_draws, self.centers.shape[0])
        sequence = np.mod(np.arange(num_draws), self.centers.shape[0])
        if m > 0:
            leftover = np.random.choice(self.centers.shape[0], m, False)
            sequence[-m:] = leftover
        np.random.shuffle(sequence)

        X = X[sequence]
        Y = Y[sequence]

        noise = self.noise_sig * torch.randn(*X.shape)
        X = X + noise
        if not self.squeeze:
            X = X[:, None]
            X = X.repeat(1, self.final_time, 1)
            Y = Y[:, None]
            Y = Y.repeat(1, self.final_time)
            mask = np.ones(X.shape[1], np.bool)
            mask[self.nonzero_time_points] = 0
            mask = np.nonzero(mask)[0]
            X[:, mask, :] = 0

        self.cluster_identity = sequence

        if idx_singleton:
            return X[0], Y[0]
        else:
            return X, Y


def draw_centers_hypercube(num_clusters, dim, min_sep):
    """Draw points uniformly distributed on hypercube, while ensuring a minimum separation distance."""
    X = []
    p = 4 * (np.random.rand(dim) - 0.5)
    X.append(p)
    counter = 0
    for i1 in range(num_clusters - 1):
        min_sep_p = min_sep - 1
        while min_sep_p < min_sep:
            p = 4 * (np.random.rand(dim) - 0.5)
            min_sep_p = 100000  # Just a very large number...
            for x in X:
                sep = norm(np.array(x) - p)
                min_sep_p = min(min_sep_p, sep)
            counter = counter + 1
        X.append(p)
    X = np.array(X)
    # print("minimum cluster separation allowed: " + str(min_sep))
    from scipy.spatial.distance import pdist
    # print("minimum cluster separation generated: " + str(np.min(pdist(X))))
    return np.array(X)


def delayed_mixed_gaussian(num_train, num_test, X_dim, Y_classes, X_clusters, n_hold, final_time_point, noise_sigma=0,
                           cluster_seed=None, assignment_and_noise_seed=None, avg_magn=0.3, min_sep=None,
                           freeze_input=True):
    """This function returns the data for a classification task where a stimulus is presented to the network (
    either sustained or only on the first timestep) for n_lag timesteps. Each stimulus is sampled from X_clusters
    labelled into Y_classes. The number of classes can be smaller but not bigger than X_clusters. The center of each
    cluster is generated randomly while its variance is controlled by noise_sigma.
    Each center is drawn uniformly from the hypersphere centered at the origin with side length 4.

    Args:
        num_train (int): Number of training input points to draw.
        num_test (int): Number of test points to draw.
        X_dim (int): Dimension of the ambient space in which clusters are generated
        Y_classes (int): Number of class labels
        X_clusters (int): Number of clusters
        n_hold (int): Number of timesteps for which the input is presented
        final_time_point (int): Final timestep, and the number of timesteps from stimulus onset to end of loss
            evaluation.
        noise_sigma (float): Standard deviation of each cluster
        cluster_seed (int): rng seed for cluster center locations
        avg_magn (float): The average magnutide for the datapoints.
        assignment_and_noise_seed (int): rng seed for assignment of class labels to clusters
        min_sep (float): Minimal separation of centers if using cluster_method 3.
        freeze_input (bool): If False, data are redrawn every time it is called (so data is in the "online learning"
        setting) and the data will be different every epoch. If True, the data are the same across epochs. This
        means, for instance, that class_datasets['train'][0] will be the same sample if called twice, while you'll
        get two different numbers if freeze_input=False.

    Returns:
        dict[str, torch.dataset]: Dictionary with keys 'train' and 'val' for training and validation datasets,
            respectively. Referred to as "class_datasets" in the code below.
        torch.Tensor: Locations for the centers of the clusters
        torch.Tensor: Class labels for the clusters

    """
    if min_sep is None:
        min_sep_defined = False
        min_sep = noise_sigma * 10
    else:
        min_sep_defined = True

    # Get clusters that don't overlap
    if not min_sep_defined:
        min_sep = noise_sigma * (0.59943537 * (X_dim - 1) ** 0.57832581 + 4.891638480163717)
    np.random.seed(cluster_seed)
    centers = draw_centers_hypercube(X_clusters, X_dim, min_sep)

    cluster_class_label = np.mod(np.arange(X_clusters), Y_classes).astype(int)
    nonzero_time_points = torch.arange(n_hold)
    # if n_hold == 0:
    #     squeeze = True
    torch.manual_seed(assignment_and_noise_seed)
    np.random.seed(assignment_and_noise_seed)
    if freeze_input:
        dataset = Gaussian_Spheres(centers, cluster_class_label, final_time_point,
                                   max_samples=num_train+num_test,
                                   noise_sig=noise_sigma, nonzero_time_points=nonzero_time_points,
                                   squeeze=final_time_point == 0)
        X, Y = dataset[:]
        class_datasets = {'train': InpData(X[:num_train], Y[:num_train]),
                          'val': InpData(X[num_train:], Y[num_train:])}
    else:
        class_datasets = {
            'train': Gaussian_Spheres(centers, cluster_class_label, final_time_point, max_samples=num_train,
                                      noise_sig=noise_sigma, nonzero_time_points=nonzero_time_points,
                                      squeeze=final_time_point == 0),
            'val': Gaussian_Spheres(centers, cluster_class_label, final_time_point, max_samples=num_test,
                                    noise_sig=noise_sigma, nonzero_time_points=nonzero_time_points,
                                    squeeze=final_time_point == 0)}
    return class_datasets, centers, cluster_class_label

