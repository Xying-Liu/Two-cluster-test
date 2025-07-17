import numpy as np
from two_cluster_test import find_mutual_boundary_points

def different_cluster_pairs(Z, true_labels, p_thresh=0.05, k=7):
    """
    Compute p-values for all pairs of subsets from different clusters,
    testing whether they belong to the same cluster using two-cluster-test.py.
    
    Parameters:
    - Z: embeddings of all samples (n_samples, n_features)
    - true_labels: ground truth cluster labels (n_samples,)
    - p_thresh: significance threshold (default 0.05)
    - k: number of neighbors for the test (default 7)
    
    Returns:
    - eval_results: list of int (1 if prediction correct, 0 otherwise)
    """
    unique_labels = np.unique(true_labels)
    eval_results = []

   for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            indices_i = np.where(true_labels == label_i)[0]
            indices_j = np.where(true_labels == label_j)[0]

            if len(indices_i) < k or len(indices_j) < k:
                p_val = 1.0
            else:
                p_val = find_mutual_boundary_points(Z, indices_i.tolist(), indices_j.tolist(), k=k)

            is_same_cluster = 0
            predict_different = int(p_val < p_thresh)
            eval_results.append(1 if predict_different == 1 else 0)

    return eval_results
