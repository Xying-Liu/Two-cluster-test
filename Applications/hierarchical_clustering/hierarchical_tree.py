import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from test.two_cluster_test import find_mutual_boundary_points

class TreeNode:
    def __init__(self, indices):
        self.indices = indices
        self.left = None
        self.right = None
        self.pval = None

def top_down_split(Z, indices, cluster_labels, cluster_id, p_thresh=0.05, k=7):
    """
    Recursively split data using hierarchical clustering and significance testing.

    Parameters
    ----------
    Z : Embedded data matrix.
    indices : Indices of current subset.
    cluster_labels : Array to store final cluster assignments.
    cluster_id : Current cluster label.
    p_thresh : P-value threshold to determine significant split.
    k : Number of nearest neighbors for boundary point test.

    Returns
    -------
    cluster_id : Updated cluster label.
    """
    if len(indices) <= 2:
        cluster_labels[indices] = cluster_id
        return cluster_id + 1

    Z_sub = Z[indices]
    link = linkage(Z_sub, method='ward')
    tree = to_tree(link)

    left_ids = [indices[i] for i in tree.get_left().pre_order()]
    right_ids = [indices[i] for i in tree.get_right().pre_order()]
    
    pval = find_mutual_boundary_points(Z, left_ids, right_ids, k=k)

    if pval >= p_thresh:
        cluster_labels[indices] = cluster_id
        return cluster_id + 1
    else:
        cluster_id = top_down_split(Z, left_ids, cluster_labels, cluster_id, p_thresh, k)
        cluster_id = top_down_split(Z, right_ids, cluster_labels, cluster_id, p_thresh, k)
        return cluster_id
