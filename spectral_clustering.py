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

    # Helper function to calculate combinations of 2
    

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


def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    'sigma': float, 'xi': float, 'k': int
]:

    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 1,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.
       params_dict['xi']: xi

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    computed_labels: NDArray[np.int32]
    SSE: float 
    ARI: float
    eigenvalues: NDArray[np.floating]

    sigma = params_dict['sigma']
    k = params_dict.get('k', 5)

    dist_sq = np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=-1)
    affinity_matrix = np.exp(-dist_sq / (2 * sigma ** 2))

    D = np.diag(affinity_matrix.sum(axis=1))
    L = D - affinity_matrix

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)[:k]
    V = eigenvectors[:, idx]

    from scipy.cluster.vq import kmeans2
    centroids, computed_labels = kmeans2(V, k, iter=20, minit='points')
    
    SSE = np.sum((V - centroids[computed_labels]) ** 2)

    ARI = adjusted_rand_index(labels, computed_labels)

    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs Spectral clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """
    data = np.load("question1_cluster_data.npy")
    labels = np.load("question1_cluster_labels.npy")
    slice_size = 1000
    answers = {}
    num_pairs = 12
    # Return your `spectral` function
    answers["spectral_function"] = spectral

    # Work with the first 1,000 data points: data[0:1000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}

    # Generate 'num_pairs' pairs of ('sigma', 'xi') within the specified range
    sigmas = np.linspace(0.1, 10, num_pairs)
    xis = np.linspace(0.1, 10, num_pairs)  # Assuming 'xi' is another parameter you want to vary
    ks = np.linspace(1, 5, 5)
    # Perform clustering for each pair of parameters
    for i, (sigma, k) in enumerate(zip(sigmas, ks)):
        params_dict = {"sigma": sigma, "k": 5}
        computed_labels, SSE, ARI, eigenvalues = spectral(data[slice_size*i:slice_size*(i+1)], labels[slice_size*i:slice_size*(i+1)], params_dict)
        groups[i] = {"sigma": sigma, "ARI": ARI, "SSE": SSE}
    highest_ARI = max(groups.items(), key=lambda x: x[1]["ARI"])

    sigmas_plot = [group["sigma"] for group in groups.values()]
    SSEs_plot = [group["SSE"] for group in groups.values()]
    ARIs_plot = [group["ARI"] for group in groups.values()]
    
    # For the spectral method, perform your calculations with 5 clusters.
    # In this case, there is only a single parameter, σ.

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
    ARI_min_cluster = min(groups.keys(), key = lambda x: abs(groups[x]['ARI']))
    # print('Cluster with lowest ARI: ', ARI_min_cluster)

    ARI_max_graph = max(groups.keys(), key=lambda x: abs(groups[x]['ARI']))
    # print("Cluster with highest ARI:", ARI_max_graph)

    SSE_max_graph = max(groups.keys(), key = lambda x: abs(groups[x]['SSE']))
    # print("Cluster with highest SSE:", SSE_min_graph)
    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.
    plt.figure()
    plot_SSE = plt.scatter(sigmas_plot, SSEs_plot)
    plt.xlabel('sigma')
    plt.ylabel('SSE')
    plt.title('Scatter Plot of SSE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    answers["cluster scatterplot with smallest SSE"] = plot_SSE

    plt.figure()
    plot_ARI = plt.scatter(sigmas_plot, ARIs_plot)
    plt.xlabel('sigma')
    plt.ylabel('ARI')
    plt.title('Scatter Plot of ARI')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    answers["cluster scatterplot with largest ARI"] = plot_ARI

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.
    data_slice_ARI_max = data[slice_size * ARI_max_graph: slice_size * (ARI_max_graph + 1)]
    plt.figure()
    plt.scatter(data_slice_ARI_max[:, 0], data_slice_ARI_max[:, 1])
    plt.xlabel("X-Label")
    plt.ylabel("Y-Label")
    plt.title(f'Cluster with highest ARI: {ARI_max_graph}')
    plt.grid(True)
    plt.tight_layout()
    plt.show


    data_slice_SSE_max = data[slice_size * SSE_max_graph: slice_size * (SSE_max_graph + 1)]
    print(data_slice_SSE_max)
    plt.figure()
    plt.scatter(data_slice_SSE_max[:, 0], data_slice_SSE_max[:, 1])
    plt.xlabel('X-Label')
    plt.ylabel('Y-Label')
    plt.title(f'Cluster with highest SSE: {SSE_max_graph}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot is the return value of a call to plt.scatter()
    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    plt.figure()
    plot_eig = plt.plot(eigenvalues)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of Laplacian Matrix')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    best_params = groups[highest_ARI[0]]
    ARIs_datasets = []
    for i in range(1, 5):
        _, _, ARI = spectral(data[slice_size*i:slice_size*(i+1)], labels[slice_size*i:slice_size*(i+1)], best_params)
        ARIs_datasets.append(ARI)

    ARIs = np.array([group["ARI"] for group in groups.values()])
    SSEs = np.array([group["SSE"] for group in groups.values()])
    answers["mean_ARIs"] = np.mean(ARIs)
    answers["std_ARIs"] = np.std(ARIs)
    answers["mean_SSEs"] = np.mean(SSEs)
    answers["std_SSEs"] = np.std(SSEs)
    

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
