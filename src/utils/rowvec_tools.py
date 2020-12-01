import numpy as np
from src.utils.delwords import create_stopword_strings


def inverse_edge_weight(u, v, d):
    edge_wt = d.get('weight', 1)
    if edge_wt > 0.01:
        return 1 / edge_wt
    else:
        return 1000000

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x - np.max(x))
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def simple_norm(x, min_zero=True):
    """
    Norms a vector (or list)
    :param x: vector
    :param min_zero: If True, the smallest non-zero value is substracted before normalization
    :return: normed np.array
    """
    x_org=x
    x=np.array(x)
    if min_zero==True:
        x = x - np.min(x[x>0], axis=-1)
        x[x<0]=0
    if np.sum(x, axis=-1) > 0:
        return x / np.sum(x, axis=-1)
    else:
        return x_org


