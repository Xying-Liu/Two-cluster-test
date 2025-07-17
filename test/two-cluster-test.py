# two_cluster_test.py

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binom_test, chi2

def find_mutual_boundary_points(Z, A, B, k=7):
    """
    Statistical test that computes the p-value to determine whether two subsets A and B belong to the same cluster.

    Parameters:
        Z: Embedding matrix of all samples, shape (n_samples, n_features)
        A: Indices of the first subset in Z
        B: Indices of the second subset in Z
        k: Number of nearest neighbors used in k-NN graph (default 7)

    Returns:
        p_val: Combined global p-value. Smaller values indicate stronger
        evidence to reject the null hypothesis that A and B are from the same cluster.
    """
   if len(A) == 0 or len(B) == 0:
        return 1.0
       
    Z_merge = np.vstack([Z[A], Z[B]])
    labels_merge = np.array([0] * len(A) + [1] * len(B))
    n1 = len(A)
    Z_A = Z_merge[labels_merge == 0]
    Z_B = Z_merge[labels_merge == 1]

    if len(Z_A) < k or len(Z_B) < k:
        print(f"Skip test: too small (|A|={len(Z_A)}, |B|={len(Z_B)}, k={k})")
        return 1.0
        
    dists_AB = pairwise_distances(Z_A, Z_B)
    nn_B_for_A = np.argmin(dists_AB, axis=1)
    dists_BA = pairwise_distances(Z_B, Z_A)
    nn_A_for_B = np.argmin(dists_BA, axis=1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(Z_merge)
    _, knn_indices = nbrs.kneighbors(Z_merge)
    boundary_indices = set()

    for i, j in enumerate(nn_B_for_A):
        if nn_A_for_B[j] == i:
            boundary_indices.add(i)
            boundary_indices.add(j + n1)
    if not boundary_indices:
        return 1.0
    p_list = []

    for idx in boundary_indices:
        own_label = labels_merge[idx]
        neighbors = knn_indices[idx][1:]
        same = sum(labels_merge[nbr] == own_label for nbr in neighbors)
        if same == 0:
            continue
        p_list.append(binom_test(same, k, p=0.5, alternative='greater'))
    if not p_list:
        return 1.0
        
    T = -2 * np.sum(np.log(p_list))
    df = 2 * len(p_list)
    p_val = 1 - chi2.cdf(T, df)

    return p_val
