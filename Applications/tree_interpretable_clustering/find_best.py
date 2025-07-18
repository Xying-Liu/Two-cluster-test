import numpy as np
from test.two_cluster_test import find_mutual_boundary_points

def find_best_split(X, k=7, outliers=7):
    """
    Find the best split of the data based on the minimum p-value.
    
    Parameters
    ----------
    X :  Input data matrix (n_samples, n_features)
    k : Number of nearest neighbors for the test (default=7)
    outliers : Minimum number of samples required on each side of a potential split. (default=7)

    Returns
    -------
    minpval : The minimum p-value found across all tested splits.
    bpi : An array indicating cluster assignments (1 or 2) for the best split.
    bm : The feature index (column) that resulted in the best split.
    bcat : The split value used to divide feature bm.
    """
    N, M = X.shape

    pv = [[] for _ in range(M)]  
    pi_list = [[] for _ in range(M)]
    discat = [[] for _ in range(M)] 

    for m in range(M):
        uList = np.unique(X[:, m])
        numQ = len(uList)

        if numQ != 1:
            for cat in uList:
                pi_current, pv_current = find_mutual_boundary_points(X, m, cat, k)
                num_tests += 1
                if isinstance(pv_current, (list, np.ndarray)):
                    pv[m].extend(pv_current)
                    pi_list[m].extend(pi_current)
                    discat[m].extend([cat] * len(pv_current))
                else:
                    pv[m].append(pv_current)
                    pi_list[m].append(pi_current)
                    discat[m].append(cat)
        else:
            cat = uList[0]
            pi_list[m].append(np.ones(N, dtype=int))
            pv[m].append(1.0)
            discat[m].append(cat)

    # Select best split
    t_pv = 1.0
    best_indices = (0, 0)

    for x in range(len(pv)):
        for y in range(len(pv[x])):
            pi_col = pi_list[x][y]
            p_val = pv[x][y]

            count_1 = np.sum(pi_col == 1)
            count_2 = np.sum(pi_col == 2)

            if p_val < t_pv and min(count_1, count_2) > outliers:
                t_pv = p_val
                best_indices = (x, y)

    bm, bq = best_indices
    minpval = t_pv
    bcat = discat[bm][bq]
    bpi = pi_list[bm][bq]

    return minpval, bpi, bm, bcat
