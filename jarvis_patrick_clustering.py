"""
Work with Jarvis-Patrick clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle
from scipy.spatial import distance

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################

def adjusted_rand_index(labels_true, computed_labels) -> float:
    """
    Compute the adjusted Rand index.

    Parameters:
    - labels_true: The true labels of the data points.
    - labels_pred: The predicted labels of the data points.

    Returns:
    - ari: The adjusted Rand index value.

    The adjusted Rand index is a measure of the similarity between two data clusterings.
    It takes into account both the similarity of the clusters themselves and the similarity
    of the data points within each cluster. The adjusted Rand index ranges from -1 to 1,
    where a value of 1 indicates perfect agreement between the two clusterings, 0 indicates
    random agreement, and -1 indicates complete disagreement.
    """
    
    # Create contingency table
    contingency_table = np.histogram2d(labels_true, computed_labels, bins=(np.unique(labels_true).size, np.unique(computed_labels).size))[0]
    def comb(n):
        return n * (n - 1) / 2
    # Sum over rows and columns
    sum_combinations_rows = np.sum([comb(n) for n in np.sum(contingency_table, axis=1)])
    sum_combinations_cols = np.sum([comb(n) for n in np.sum(contingency_table, axis=0)])

    # Sum of combinations for all elements
    N = np.sum(contingency_table)
    sum_combinations_total = comb(N)

    # Calculate ARI
    sum_combinations_within = np.sum([comb(n_ij) for n_ij in contingency_table.flatten()])
    ari = (
        sum_combinations_within
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    ) / (
        (sum_combinations_rows + sum_combinations_cols) / 2
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    )

    return ari

def comp_sse(data, labels):
    uniq_labels = set(labels) - {-1}
    sse = 0
    for label in uniq_labels:
        cluster_data = data[labels == label]
        if cluster_data.size == 0:
            continue
        centroid = np.mean(cluster_data, axis = 0)
        sse += np.sum((cluster_data - centroid) ** 2)
    return sse


def jarvis_patrick(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[NDArray[np.int32], float, float]:
    """
    Implementation of the Jarvis-Patrick algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'k', 'smin', There could be others.
    - params_dict['k']: the number of nearest neighbors to consider. This determines the size of the neighborhood used to assess the similarity between datapoints. Choose values in the range 3 to 8
    - params_dict['smin']:  the minimum number of shared neighbors to consider two points in the same cluster.
       Choose values in the range 4 to 10.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index

    Notes:
    - the nearest neighbors can be bidirectional or unidirectional
    - Bidirectional: if point A is a nearest neighbor of point B, then point B is a nearest neighbor of point A).
    - Unidirectional: if point A is a nearest neighbor of point B, then point B is not necessarily a nearest neighbor of point A).
    - In this project, only consider unidirectional nearest neighboars for simplicity.
    - The metric  used to compute the the k-nearest neighberhood of all points is the Euclidean metric
    """
    """k =int(params_dict['k'])
    smin = int(params_dict['smin'])

    num_points = data.shape[0]
    dist_matrix = distance.squareform(distance.pdist(data, 'euclidean'))
    neighbors = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
    computed_labels = -np.ones(num_points, dtype=int)
    cluster_id = 0
    
    for i in range(num_points):
        if computed_labels[i] == -1:
            computed_labels[i] = cluster_id
            for j in range(num_points):
                if i != j:
                    shared_neighbors = np.intersect1d(neighbors[i], neighbors[j]).shape[0]
                    if shared_neighbors >= smin:
                        computed_labels[j] = cluster_id
            cluster_id += 1

    SSE = np.sum((data - np.mean(data[computed_labels == computed_labels[:, None]], axis=1))**2)
    ARI = adjusted_rand_index(labels, computed_labels)"""


    k = int(params_dict['k'])
    smin = int(params_dict['smin'])

    num_points = data.shape[0]
    dist_matrix = distance.squareform(distance.pdist(data, 'euclidean'))
    neighbors = np.argsort(dist_matrix, axis=1)[:, 1:k+1]

    computed_labels = -np.ones(num_points, dtype=int)
    cluster_id = 0

    for i in range(num_points):
        if computed_labels[i] == -1:
            computed_labels[i] = cluster_id
            for j in range(num_points):
                if i != j:
                    shared_neighbors = np.intersect1d(neighbors[i], neighbors[j]).shape[0]
                    if shared_neighbors >= smin:
                        computed_labels[j] = cluster_id
            cluster_id += 1

    ARI = adjusted_rand_index(labels, computed_labels)
    SSE = comp_sse(data, labels)

    computed_labels: NDArray[np.int32]
    SSE: float
    ARI: float

    return computed_labels, SSE, ARI


def jarvis_patrick_clustering():
    """
    Performs Jarvis-Patrick clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    data = np.load("question1_cluster_data.npy")
    labels = np.load("question1_cluster_labels.npy")
    slice_size = 30
    num_pairs = 12
    answers = {}

    # Return your `jarvis_patrick` function
    answers["jarvis_patrick_function"] = jarvis_patrick

    # Work with the first 1,000 data points: data[0:1000]
    # Do a parameter study of this data using Jarvis-Patrick.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    
    # data for data group 0: data[0:1000]. For example,
    # groups[0] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}
    smins = np.linspace(1, 5, 5)
    ks = np.linspace(1, 10, 10)
    index = 0
    for k in ks:
        for smin in smins:
            params_dict = {'k': k, 'smin': smin}
            # Correct slicing using `index` instead of `i`
            computed_labels, SSE, ARI = jarvis_patrick(data[slice_size * index: slice_size * (index + 1)], labels[slice_size * index: slice_size * (index + 1)], params_dict)
            groups[index] = {'k': k, 'smin': smin, 'ARI': ARI, 'SSE': SSE}
            index = index + 1

    highest_ARI = max(groups.items(), key=lambda x: x[1]["ARI"])
    lowest_SSE = min(groups.items(), key=lambda x: x[1]["SSE"])
    print(highest_ARI)
    print(lowest_SSE)
    # data for data group i: data[1000*i: 1000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "xi": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with # \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    plot_ARI = plt.scatter([1,2,3], [4,5,6])
    plot_SSE = plt.scatter([1,2,3], [4,5,6])
    answers["cluster scatterplot with largest ARI"] = plot_ARI
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.

    # A single float
    answers["mean_ARIs"] = 0.

    # A single float
    answers["std_ARIs"] = 0.

    # A single float
    answers["mean_SSEs"] = 0.

    # A single float
    answers["std_SSEs"] = 0.

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = jarvis_patrick_clustering()
    with open("jarvis_patrick_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
