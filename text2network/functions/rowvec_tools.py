import numpy as np
import pandas as pd
from typing import Union


# %% Utilities
def cutoff_percentage(x: Union[pd.DataFrame, np.ndarray], percent: int = 100)->Union[pd.DataFrame,np.ndarray]:
    """
    Cuts a proximity table or adjancency matrix such that only
    z-percent of the total mass of ties are retained.
    This helps to sparsify the network

    Parameters
    ----------
    percent : int
        Percent (in xx%) of mass of row-vectors to retain
    x : Union[pd.DataFrame, np.ndarray]
        Adjacency matrix or dataframe of proximities


    """

    sortx = - np.sort(-x, axis=-1)
    # Get cumulative sum
    cum_sum = np.cumsum(sortx, axis=-1)
    # Get cutoff value as fraction of largest cumulative element (in case vector does not sum to 1)
    # This is the threshold to cross to explain 'percent' of mass in the vector
    cutoff = cum_sum[..., -1] * percent / 100
    cutoffs = np.tile(cutoff, (x.shape[-1], 1)).T
    # Determine first position where cumulative sum crosses cutoff threshold
    # Python indexing - add 1
    cutoff_degrees = np.sum(np.where(cum_sum <= cutoffs, cum_sum, 0) > 0, axis=-1)
    # Calculate corresponding probability
    if x.ndim>1:
        cutoff_values = sortx[(np.arange(0, len(cutoff_degrees)), cutoff_degrees)]
        cutoff_values = np.tile(cutoff_values, (x.shape[-1], 1)).T
    else:
        if cutoff_degrees<len(sortx):
            cutoff_values= sortx[cutoff_degrees]
        else:
            cutoff_values=sortx[-1]
    x[x < cutoff_values] = 0

    return x

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
    x_org=x.copy()
    x=np.array(x)
    if min_zero:
        x = x - np.min(x[x>0], axis=-1)
        x[x<0]=0
    if np.sum(x, axis=-1) > 0:
        return x / np.sum(x, axis=-1)
    else:
        return x_org


