import logging
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm

from text2network.classes.neo4jnw import neo4j_network


def extract_temporal_cosine_similarity(snw: neo4j_network, tokens: Union[list, str, int], depth: int = 0,
                                       symmetric: Optional[bool] = False,
                                       reverse: Optional[bool] = False, compositional: Optional[bool] = False,
                                       times: Optional[Union[list, int]] = None,
                                       symmetric_method: Optional[str] = None,
                                       prune_min_frequency: Optional[int] = None, filename: Optional[str]=None):
    """

    Extracts a matrix where relationships between time points is determined by cosine similarity
    of their adjacency matrices for nodes given in tokens parameter.

    Parameters
    ----------
    snw : neo4j network

    tokens : str, int or list thereof
        Tokens

    symmetric: bool, optional
        If True, sets network to symmetric.
        Use symmetric_method to specify how. See snw.to_symmetric()

    reverse: bool, optional
        If True, reverses ties in the direction occurrence -> substitute

    compositional: bool, optional
        If True, divides tie weight by the frequency of the occurring tie

    max_degree: int, optional
        If not None, cap the ties of a given node to a maximum the max_degree strongest (before symmetrization)

    times: Union[list,int], optional
        The times to check

    prune_min_frequency : int, optional
        Will remove nodes entirely which occur less than  prune_min_frequency+1 times

    filename : str, optional
        If given, will export as xlsx

    Returns
    -------

    """

    yoy_adjacency = extract_temporal_adjacency_matrices(snw=snw, tokens=tokens, depth=depth, symmetric=symmetric,
                                                        reverse=reverse, compositional=compositional, times=times,
                                                        symmetric_method=symmetric_method,
                                                        prune_min_frequency=prune_min_frequency)

    row_list = []
    n = len(tokens)
    prior_tokens = None
    for time in yoy_adjacency.keys():
        indices = np.triu_indices(n)
        current_tokens = list(yoy_adjacency[time].columns)
        row = yoy_adjacency[time].to_numpy()[indices]
        row_list.append(row)
        if prior_tokens is not None:
            if not current_tokens == prior_tokens:
                msg = "Mismatch of column/row order of adjacency matrix returned!"
                logging.error()
                raise np.AxisError()
            prior_tokens = current_tokens

    matrix = np.stack(row_list)
    cosine_similarities = 1 - squareform(pdist(matrix, metric='cosine'))
    cosine_similarities = pd.DataFrame(cosine_similarities, columns=list(yoy_adjacency.keys()),
                                       index=list(yoy_adjacency.keys()))
    if filename is not None:
        cosine_similarities.to_excel(filename, merge_cells=False)

    return cosine_similarities


def extract_temporal_adjacency_matrices(snw: neo4j_network, tokens: Union[list, str, int], depth: int = 0,
                                        symmetric: Optional[bool] = False,
                                        reverse: Optional[bool] = False, compositional: Optional[bool] = False,
                                        times: Optional[Union[list, int]] = None,
                                        symmetric_method: Optional[str] = None,
                                        prune_min_frequency: Optional[int] = None) -> dict:
    """

    Extracts a network formed by the words in the tokens parameter for each time point (optionally given by times).
    Returns a dict with numpy matrices.

    Parameters
    ----------
    snw : neo4j network

    tokens : str, int or list thereof
        Tokens

    symmetric: bool, optional
        If True, sets network to symmetric.
        Use symmetric_method to specify how. See snw.to_symmetric()

    reverse: bool, optional
        If True, reverses ties in the direction occurrence -> substitute

    compositional: bool, optional
        If True, divides tie weight by the frequency of the occurring tie

    max_degree: int, optional
        If not None, cap the ties of a given node to a maximum the max_degree strongest (before symmetrization)

    times: Union[list,int], optional
        The times to check

    prune_min_frequency : int, optional
        Will remove nodes entirely which occur less than  prune_min_frequency+1 times

    Returns
    -------

    """

    if times is not None:
        year_list = times
    else:
        logging.info("Getting years")
        year_list = snw.get_times_list()
        year_list.append(None)

    mat_dict = {}

    pbar = tqdm(year_list, desc="Exctracting years")
    for year in tqdm(year_list, desc="Extracting years"):
        pbar.desc = "Extracting year: {}".format(year)
        snw.decondition()
        snw.condition(tokens=tokens, keep_only_tokens=True, times=year, batchsize=5000,
                      prune_min_frequency=prune_min_frequency)

        if reverse:
            snw.to_reverse()
        if compositional:
            snw.to_compositional(times=year)
        if symmetric:
            snw.to_symmetric(technique=symmetric_method)
        # extract

        mat = nx.to_pandas_adjacency(snw.graph, nodelist=snw.ensure_ids(tokens))
        rows = snw.ensure_tokens(mat.columns)
        mat.columns = rows
        mat = mat.set_axis(rows, axis='index')
        mat_dict[year] = mat.copy()

    return mat_dict


def extract_yearly_networks(snw: neo4j_network, folder: str, symmetric: Optional[bool] = False,
                            reverse_ties: Optional[bool] = False, compositional: Optional[bool] = False,
                            max_degree: Optional[int] = None, times: Optional[Union[list, int]] = None,
                            symmetric_method: Optional[str] = None,
                            prune_min_frequency: Optional[int] = None):
    """

    Conditions, for each year, a network. By default with ties giving the aggregate probability of substitute -> occurrence relations.
    Saves as gexf and edge-list.

    Parameters
    ----------
    snw : neo4j network

    folder : str
        The export folder

    symmetric: bool, optional
        If True, sets network to symmetric.
        Use symmetric_method to specify how. See snw.to_symmetric()

    reverse_ties: bool, optional
        If True, reverses ties in the direction occurrence -> substitute

    compositional: bool, optional
        If True, divides tie weight by the frequency of the occurring tie

    max_degree: int, optional
        If not None, cap the ties of a given node to a maximum the max_degree strongest (before symmetrization)

    times: Union[list,int], optional
        The times to check

    prune_min_frequency : int, optional
        Will remove nodes entirely which occur less than  prune_min_frequency+1 times


    Returns
    -------
    edgelist: pd.DataFrame
    """

    if times is not None:
        year_list = times
    else:
        logging.info("Getting years")
        year_list = snw.get_times_list()
        year_list.append(None)

    pbar = tqdm(year_list, desc="Exctracting years")
    for year in tqdm(year_list, desc="Extracting years"):
        pbar.desc = "Extracting year: {}".format(year)
        snw.decondition()
        snw.condition(times=year, max_degree=max_degree, batchsize=5000, prune_min_frequency=prune_min_frequency)

        if reverse_ties:
            snw.to_reverse()
        if compositional:
            snw.to_compositional(times=year)
        if symmetric:
            snw.to_symmetric(technique=symmetric_method)

        snw.export_gefx(path=folder)
        snw.export_edgelist(path=folder)


def extract_yearly_ego_networks(snw: neo4j_network, folder: str, ego_token: Union[list, int, str],
                                symmetric: Optional[bool] = False,
                                reverse_ties: Optional[bool] = False, compositional: Optional[bool] = False,
                                max_degree: Optional[int] = None, times: Optional[Union[list, int]] = None,
                                symmetric_method: Optional[str] = None,
                                prune_min_frequency: Optional[int] = None):
    """

    Conditions, for each year, an ego network. By default with ties giving the aggregate probability of substitute -> occurrence relations.
    Saves as gexf and edge-list csv.

    Parameters
    ----------
    snw : neo4j network

    folder : str
        The export folder

    ego_token: list,int,str
        Tokens on which to condition

    symmetric: bool, optional
        If True, sets network to symmetric.
        Use symmetric_method to specify how. See snw.to_symmetric()

    reverse_ties: bool, optional
        If True, reverses ties in the direction occurrence -> substitute

    compositional: bool, optional
        If True, divides tie weight by the frequency of the occurring tie

    max_degree: int, optional
        If not None, cap the ties of a given node to a maximum the max_degree strongest (before symmetrization)

    times: Union[list,int], optional
        The times to check

    prune_min_frequency : int, optional
        Will remove nodes entirely which occur less than  prune_min_frequency+1 times


    Returns
    -------
    edgelist: pd.DataFrame
    """

    if times is not None:
        year_list = times
    else:
        logging.info("Getting years")
        year_list = snw.get_times_list()
        year_list.append(None)

    pbar = tqdm(year_list, desc="Extracting years")
    for year in tqdm(year_list, desc="Extracting years"):
        pbar.desc = "Extracting ego network year: {}".format(year)
        snw.decondition()
        snw.condition(times=year, tokens=ego_token, depth=1, max_degree=max_degree, batchsize=5000,
                      prune_min_frequency=prune_min_frequency)

        if reverse_ties:
            snw.to_reverse()
        if compositional:
            snw.to_compositional(times=year)
        if symmetric:
            snw.to_symmetric(technique=symmetric_method)

        snw.export_gefx(path=folder)
        snw.export_edgelist(path=folder)
