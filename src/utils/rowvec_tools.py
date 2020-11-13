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


def simple_norm(x):
    """Just want to start at zero and sum to 1, without norming anything else"""
    x_org=x
    x = x - np.min(x, axis=-1)
    if np.sum(x, axis=-1) > 0:
        return x / np.sum(x, axis=-1)
    else:
        return x_org


