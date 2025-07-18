import numpy as np
from find_best import find_best_split

class TreeNode:
    def __init__(self, indices, left=None, right=None, pval=None, split_value=None, feature=None):
        """
        Tree node representing a cluster or a split decision.
        """
        self.indices = indices
        self.left = left
        self.right = right
        self.pval = pval
        self.split_value = split_value
        self.feature = feature
      
def sig_divide(X, num_node, pi1, indices=None, p_thresh=0.05):
    """
    Parameters
    ----------
    X : Current data matrix to split.
    num_node : Current number of nodes in the tree.
    pi1 : Array storing final cluster assignments.
    indices : indices of the samples in the original dataset. Defaults to all samples.

    Returns
    -------
    node : Root node of the current subtree.
    num_node : Updated node count.
    pi1 : Updated cluster assignment array.
    """
    if indices is None:
        indices = np.arange(len(X))

    node = TreeNode(indices=indices)

    min_pval, best_pi, best_m, best_cat = find_best_split(X, k=7, outliers=7)

    if num_node != 1 and min_pval > p_thresh:
        pi1[-1] += 1
        pi1[indices] = pi1[-1]
        node.pval = min_pval
        node.feature = None
        node.split_value = None
        return node, num_node, pi1

    # Otherwise, proceed to split
    X_left = X[best_pi == 1, :]
    X_right = X[best_pi == 2, :]
    indices_left = indices[best_pi == 1]
    indices_right = indices[best_pi == 2]

    node.pval = min_pval
    node.split_value = (best_m, best_cat)
    node.feature = best_m

    node.left, num_node, pi1 = sig_divide(X_left, num_node, pi1, indices_left, p_thresh)
    node.right, num_node, pi1 = sig_divide(X_right, num_node, pi1, indices_right, p_thresh)

    return node, num_node, pi1
