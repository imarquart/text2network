import numpy as np


def calculate_cutoffs(x, method="mean", percent=100, max_degree=100,min_cut=0.001):
    """
    Different methods to calculate cutoff probability and number.

    :param x: Contextual vector
    :param method: mean: Only accept entries above the mean; percent: Take the k biggest elements that explain X% of mass.
    :return: cutoff_number and probability
    """
    if method == "mean":
        cutoff_probability = max(np.mean(x), min_cut)
        cutoff_number = max(np.int(len(x) / 100), 100)
    elif method == "percent":
        sortx = np.sort(x)[::-1]
        cum_sum = np.cumsum(sortx)
        cutoff = cum_sum[-1] * percent/100
        cutoff_number = np.where(cum_sum >= cutoff)[0][0]
        if cutoff_number == 0: cutoff_number=max_degree
        cutoff_probability = min_cut
    else:
        cutoff_probability = min_cut
        cutoff_number = 0

    return min(cutoff_number,max_degree), cutoff_probability


def get_weighted_edgelist(token, x, time, cutoff_number=100, cutoff_probability=0, seq_id=0, pos=0,max_degree=100):
    """
    Sort probability distribution to get the most likely neighbor nodes.
    Return a networksx weighted edge list for a given focal token as node.

    :param token: Numerical, token which to add
    :param x: Probability distribution
    :param cutoff_number: Number of neighbor token to consider. Not used if 0.
    :param cutoff_probability: Lowest probability to consider. Not used if 0.
    :return: List of tuples compatible with networkx
    """
    # Get the most pertinent words
    if cutoff_number > 0:
        neighbors = np.argsort(-x)[:cutoff_number]
    else:
        neighbors = np.argsort(-x)[:max_degree]

    # Cutoff probability (zeros)
    if len(neighbors > 0):
        if cutoff_probability>0:
            neighbors = neighbors[x[neighbors] > cutoff_probability]
        weights = x[neighbors]
        return [(int(token), int(x[0]), int(time), {'weight': float(x[1]), 'p1': int(seq_id), 'p2': int(pos)}) for x in list(zip(neighbors, weights))]
    else:
        return None
