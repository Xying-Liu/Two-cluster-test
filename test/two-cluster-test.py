# two_cluster_test.py

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binom_test, chi2

def find_mutual_boundary_points(Z, A, B, k=7):
    """
    Statistical test to determine whether two subsets A and B belong to the same cluster,
    based on mutual nearest neighbor boundary points.
    
    The method finds pairs of points that are mutual nearest neighbors across subsets,
    calculates binomial test p-values for the number of same-cluster neighbors of each boundary point,
    and combines these p-values using Fisher's method to obtain an overall significance level.

    Parameters:
        Z (ndarray): Embedding matrix of all samples, shape (n_samples, n_features)
        A (list[int]): Indices of the first subset in Z
        B (list[int]): Indices of the second subset in Z
        k (int): Number of nearest neighbors used in k-NN graph (default 7)

    Returns:
        combined_p_val (float): Combined global p-value. Smaller values indicate stronger
                                evidence to reject the null hypothesis that A and B are from the same cluster.
    """
    if len(A) == 0 or len(B) == 0:
        return 1.0
    Z_merge = np.vstack([Z[A], Z[B]])
    labels_merge = np.array([0] * len(A) + [1] * len(B))
    n1 = len(A)
    if len(A) < k or len(B) < k:
        print(f"Skip test: too small subsets (|A|={len(A)}, |B|={len(B)}, k={k})")
        return 1.0
    dists_AB = pairwise_distances(Z_merge[:n1], Z_merge[n1:])
    nn_B_for_A = np.argmin(dists_AB, axis=1)
    dists_BA = pairwise_distances(Z_merge[n1:], Z_merge[:n1])
    nn_A_for_B = np.argmin(dists_BA, axis=1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z_merge)
    distances, knn_indices = nbrs.kneighbors(Z_merge)
    boundary_pairs = []
    boundary_indices = set()
    for i, j in enumerate(nn_B_for_A):
        if nn_A_for_B[j] == i:
            idx_A = i
            idx_B = j + n1
            boundary_pairs.append((idx_A, idx_B))
            boundary_indices.add(idx_A)
            boundary_indices.add(idx_B)
    p_list = []
    for idx in boundary_indices:
        own_label = labels_merge[idx]
        neighbors = knn_indices[idx][1:]  # exclude self
        same = sum(labels_merge[nbr] == own_label for nbr in neighbors)
        if same == 0:
            continue
        p_val = binom_test(same, k, p=0.5, alternative='greater')
        p_list.append(p_val)  
    # Combine p-values using Fisher's method
    T = -2 * np.sum(np.log(p_list))
    df = 2 * len(p_list)
    combined_p_val = 1 - chi2.cdf(T, df)

    return combined_p_val
