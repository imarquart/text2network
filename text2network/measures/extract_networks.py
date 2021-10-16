from typing import Optional, Union

import pandas as pd

# from src.classes import neo4jnw
# from src.classes.neo4jnw import neo4j_network
# from src.classes import neo4jnw

def extract_yearly_networks(snw, symmetric: Optional[bool] = False, reverse_ties: Optional[bool] = False, compositional: Optional[bool] = False, max_degree: Optional[int]= None, times: Optional[Union[list, int]] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """

    Conditions, for each year, a network. By default with ties giving the aggregate probability of substitute -> occurrence relations.
    Saves as gexf and return edge-list DataFrame.

    Parameters
    ----------
    snw : neo4j network
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



    edge_list=pd.DataFrame()
    return edge_list