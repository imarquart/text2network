import logging
from typing import Optional, Union

from tqdm import tqdm

from text2network.classes.neo4jnw import neo4j_network
import pandas as pd


def extract_yearly_networks(snw:neo4j_network, folder:str, symmetric: Optional[bool] = False, reverse_ties: Optional[bool] = False, compositional: Optional[bool] = False, max_degree: Optional[int]= None, times: Optional[Union[list, int]] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """

    Conditions, for each year, a network. By default with ties giving the aggregate probability of substitute -> occurrence relations.
    Saves as gexf and return edge-list DataFrame.

    Parameters
    ----------
    snw : neo4j network
    folder : str
        The export folder
    symmetric: bool
        If True, sets network to symmetric
    reverse_ties: bool
        If True, reverses ties in the direction occurrence -> substitute
    compositional: bool
        If True, divides tie weight by the frequency of the occurring tie
    max_degree: int
        If not None, cap the ties of a given node to a maximum the max_degree strongest (before symmetrization)
    times: Union[list,int]
        The times to check
    seed: int
        Random seed

    Returns
    -------
    edgelist: pd.DataFrame
    """

    if times is not None:
        year_list= times
    else:
        logging.info("Getting years")
        year_list = snw.get_times_list()
        year_list.append(None)

    pbar = tqdm(year_list, desc="Exctracting years")
    for year in tqdm(year_list, desc="Extracting years"):
        pbar.desc = "Extracting year: {}".format(year)
        snw.decondition()
        snw.condition(times=year, max_degree=max_degree, batchsize=5000)
        snw.add_frequencies(times=year)
        if reverse_ties:
            snw.to_reverse()
        if compositional:
            snw.to_compositional(times=year)
        if symmetric:
            snw.to_symmetric(technique="sum")
        # Prune frequencies
        prune_list = [x for x in snw.graph.nodes if snw.graph.nodes[x]['freq'] < 2]
        logging.info("Found {} nodes with frequency less than 2, pruning".format(len(prune_list)))
        snw.graph.remove_nodes_from(prune_list)
        snw.export_gefx(path=folder)

    #edge_list=pd.DataFrame()
    #return edge_list