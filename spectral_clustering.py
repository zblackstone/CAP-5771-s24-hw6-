"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################


def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    sigma = params_dict['sigma']
    k = params_dict['k']

    dist_sq = np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=-1)
    affinity_matrix = np.exp(-dist_sq / (2 * sigma ** 2))

    D = np.diag(affinity_matrix.sum(axis=1))
    L = D - affinity_matrix

    eigenvalues, eigenvectors = eigh(L)
    idx = np.argsort(eigenvalues)[:k]
    V = eigenvectors[:, idx]

    from scipy.cluster.vq import kmeans2
    centroids, labels_pred = kmeans2(V, k, iter=20, minit='points')

    SSE = np.sum((V - centroids[labels_pred]) ** 2)

    def adjusted_rand_index(true_labels, predicted_labels):
        """
        Compute the Adjusted Rand Index (ARI) between two clusterings.
    
        Arguments:
        - true_labels: true cluster labels
        - predicted_labels: predicted cluster labels
    
        Returns:
        - ARI: Adjusted Rand Index
        """
        contingency_matrix = np.zeros((len(np.unique(true_labels)), len(np.unique(predicted_labels))))
        
        for i, true_label in enumerate(np.unique(true_labels)):
            true_idx = (true_labels == true_label)
            for j, pred_label in enumerate(np.unique(predicted_labels)):
                pred_idx = (predicted_labels == pred_label)
                contingency_matrix[i, j] = np.sum(true_idx & pred_idx)
    
        a = np.sum(contingency_matrix, axis=1)
        b = np.sum(contingency_matrix, axis=0)
        n = np.sum(contingency_matrix)
        a_choose_2 = np.sum(a * (a - 1)) / 2
        b_choose_2 = np.sum(b * (b - 1)) / 2
        n_choose_2 = n * (n - 1) / 2
    
        # Compute Rand Index
        rand_index = (np.sum(contingency_matrix ** 2) - (a_choose_2 + b_choose_2)) / 2
    
        # Expected Rand Index under the null hypothesis
        expected_rand_index = (a_choose_2 * b_choose_2) / n_choose_2
    
        # Adjusted Rand Index
        ARI = (rand_index - expected_rand_index) / (0.5 * (a_choose_2 + b_choose_2) - expected_rand_index)

    return ARI

    ARI = adjusted_rand_index(labels, computed_labels)

    computed_labels: NDArray[np.int32] | None = None
    SSE: float | None = None
    ARI: float | None = None
    eigenvalues: NDArray[np.floating] | None = None

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs Spectral clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups
    answers["1st group, SSE"] = {}

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
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

    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    plot_eig = plt.plot([1,2,3], [4,5,6])
    answers["eigenvalue plot"] = plot_eig

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
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
