import numpy as np
from two_cluster_test import find_mutual_boundary_points

def same_cluster_split(Z, true_labels, p_thresh=0.05, k=7):
    """
    Calculate splitting each true cluster into two parts 
    and testing if they belong to the same cluster using two-cluster-test.py.
    
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

    for label in unique_labels:
        indices = np.where(true_labels == label)[0]
        Z_sub = Z[indices]
        median_val = np.median(Z_sub[:, 0])
        pseudo_labels = (Z_same[:, 0] >= mean_value).astype(int)
        left_ids = [label_indices[i] for i, val in enumerate(pseudo_labels) if val == 0]
        right_ids = [label_indices[i] for i, val in enumerate(pseudo_labels) if val == 1]

        if len(left_ids) < k or len(right_ids) < k:
            p_val = 1.0
        else:
            p_val = find_mutual_boundary_points(Z, left_ids, right_ids, k=k)

        true_labels_sub = true_labels[np.concatenate([left_ids, right_ids])]
        is_same_cluster = int(np.all(true_labels_sub == true_labels_sub[0]))
        predict_same = int(p_val >= p_thresh)

        eval_results.append(1 if predict_same == is_same_cluster else 0)

    return eval_results
