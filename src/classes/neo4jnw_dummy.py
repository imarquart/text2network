# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:26:32 2020

@author: marquart
"""
import copy
import logging
from collections.abc import MutableSequence
from src.utils.twowaydict import TwoWayDict
from src.functions.backout_measure import backout_measure
from src.utils.network_tools import make_symmetric
import numpy as np
from src.functions.node_measures import proximity, centrality
from src.utils.input_check import input_check
try:
    import networkx as nx
except:
    nx = None

try:
    import igraph as ig
except:
    ig = None
from _collections import defaultdict
import logging
from typing import Optional, Callable, Tuple, List, Dict, Union

from src.utils.network_tools import make_symmetric
import networkx as nx
from community import best_partition
from src.functions.graph_clustering import *

# Type definition
try:
    from typing import TypedDict
    class GraphDict(TypedDict):
        graph: nx.DiGraph
        name: str
        parent: int
        level: int
        measures: List
        metadata: [Union[Dict,defaultdict]]
except:
    GraphDict = Dict[str, Union[str,int,Dict,List,defaultdict]]


class neo4j_network_dummy(MutableSequence):

    # %% Initialization functions
    def __init__(self, neo4j_creds=None, graph_type="networkx", agg_operator="SUM",
                 write_before_query=True,
                 neo_batch_size=10000, queue_size=100000, tie_query_limit=100000, tie_creation="UNSAFE",
                 logging_level=logging.NOTSET):
        # Set logging level
        logging.disable(logging_level)

        self.db = None

        # Conditioned graph information
        self.graph_type = graph_type
        self.graph = None
        self.conditioned = True
        self.years = []

        # Dictionaries and token/id saved in memory
        self.token_id_dict = TwoWayDict()
        # Since both are numerical, we need to use a single way dict here
        self.id_index_dict = dict()
        self.tokens = []
        self.ids = []
        # Copies to be used during conditioning
        self.neo_token_id_dict = TwoWayDict()
        self.neo_ids = []
        self.neo_tokens = []
        # Init tokens

        # Create graph
        starttime = 2000
        endtime = 2001

        self.starttime = starttime
        self.endtime = endtime

        self.graph = self.create_empty_graph()
        edgelist = [("chancellor", "president",  {'weight': 0.25, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                    ("king", "president",  {
                     'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                    ("tyrant", "president",  {
                     'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                    ("ceo", "president",  {'weight': 0.3, 'time': starttime,
                                           'start': starttime, 'end': endtime, 'occurrences': 1}),
                    ("father", "president",  {'weight': 0.25, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})]

        x = "chancellor"
        edgelist.extend([("president", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("king", x,  {'weight': 0.2, 'time': starttime,
                                       'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.1, 'time': starttime,
                                         'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("father", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "king"
        edgelist.extend([("president", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.6, 'time': starttime,
                                         'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("chancellor", x,  {'weight': 0.1, 'time': starttime,
                                             'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("father", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "tyrant"
        edgelist.extend([("president", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("king", x,  {'weight': 0.6, 'time': starttime,
                                       'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("chancellor", x,  {'weight': 0.05, 'time': starttime,
                                             'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("ceo", x,  {'weight': 0.15, 'time': starttime,
                                      'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("father", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "ceo"
        edgelist.extend([("president", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("king", x,  {'weight': 0.05, 'time': starttime,
                                       'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.05, 'time': starttime,
                                         'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("chancellor", x,  {'weight': 0.1, 'time': starttime,
                                             'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("father", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "father"
        edgelist.extend([("president", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("king", x,  {'weight': 0.4, 'time': starttime,
                                       'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.3, 'time': starttime,
                                         'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("chancellor", x,  {'weight': 0.2, 'time': starttime,
                                             'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("ceo", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        # verb-noun

        x = "judge"
        edgelist.extend([("king", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.25, 'time': starttime,
                                         'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("father", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "delegate"
        edgelist.extend([("ceo", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("president", x,  {'weight': 0.5, 'time': starttime,
                                            'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("chancellor", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "manage"
        edgelist.extend([("ceo", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("president", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "teach"
        edgelist.extend([("father", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("president", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "rule"
        edgelist.extend([("king", x,  {'weight': 0.6, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        # verb-verb

        x = "rule"
        edgelist.extend([("judge", x,  {'weight': 0.8, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("manage", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "teach"
        edgelist.extend([("manage", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("delegate", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "delegate"
        edgelist.extend([("manage", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("teach", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "manage"
        edgelist.extend([("delegate", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("rule", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "judge"
        edgelist.extend([("rule", x,  {'weight': 0.45, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("manage", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        # man women
        x = "man"
        edgelist.extend([("president", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("king", x,  {'weight': 0.4, 'time': starttime,
                                       'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.4, 'time': starttime,
                                         'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("chancellor", x,  {'weight': 0.1, 'time': starttime,
                                             'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("ceo", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "woman"
        edgelist.extend([("president", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("tyrant", x,  {'weight': 0.1, 'time': starttime,
                                         'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("chancellor", x,  {'weight': 0.4, 'time': starttime,
                                             'start': starttime, 'end': endtime, 'occurrences': 1}),
                         ("ceo", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x = "woman"
        edgelist.extend([("man", x,  {'weight': 0.6, 'time': starttime,
                                      'start': starttime, 'end': endtime, 'occurrences': 1})])
        x = "man"
        edgelist.extend([("woman", x,  {
                        'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        self.graph.add_edges_from(edgelist)

        self.update()

        # Init parent class
        super().__init__()

    def update(self):
        self.ids = list(self.graph.nodes)
        self.tokens = list(self.graph.nodes)

    def condition(self, years=None, tokens=None, weight_cutoff=None, depth=None, context=None, norm=False,
                  batchsize=10000):
        self.conditioned = True

    def decondition(self):
        self.conditioned = False
        return True

    def ensure_ids(self, tokens):
        return tokens

    def yearly_centralities(self, year_list, focal_tokens=None,  types=["PageRank", "normedPageRank"], ego_nw_tokens=None, depth=1, context=None, weight_cutoff=None, norm_ties=True):
        """
        Compute directly year-by-year centralities for provided list.

        Parameters
        ----------
        semantic_network : semantic network class
            semantic network to use.
        year_list : list
            List of years for which to calculate centrality.
        focal_tokens : list, str
            List of tokens of interest. If not provided, centralities for all tokens will be returned.
        types : list, optional
            types of centrality to calculate. The default is ["PageRank"].
        ego_nw_tokens : list, optional - used when conditioning
             List of tokens for an ego-network if desired. Only used if no graph is supplied. The default is None.
        depth : TYPE, optional - used when conditioning
            Maximal path length for ego network. Only used if no graph is supplied. The default is 1.
        context : list, optional - used when conditioning
            List of tokens that need to appear in the context distribution of a tie. The default is None.
        weight_cutoff : float, optional - used when conditioning
            Only links of higher weight are considered in conditioning.. The default is None.
        norm_ties : bool, optional - used when conditioning
            Please see semantic network class. The default is True.

        Returns
        -------
        dict
            Dict of years with dict of centralities for focal tokens.

        """
        cent_year = {}
        assert isinstance(year_list, list), "Please provide list of years."
        for year in year_list:
            cent_measures = self.centralities(focal_tokens=focal_tokens, types=types, years=[
                                              year], ego_nw_tokens=ego_nw_tokens, depth=depth, context=context, weight_cutoff=weight_cutoff, norm_ties=norm_ties)
            cent_year.update({year: cent_measures})

        return {'yearly_centralities': cent_year}

    def centralities(self, focal_tokens=None,  types=["PageRank", "normedPageRank"], years=None, ego_nw_tokens=None, depth=1, context=None, weight_cutoff=None, norm_ties=True):
        """
        Calculate centralities for given tokens over an aggregate of given years.
        If not conditioned, the semantic network will be conditioned according to the parameters given.

        Parameters
        ----------
        focal_tokens : list, str, optional
            List of tokens of interest. If not provided, centralities for all tokens will be returned.
        types : list, optional
            Types of centrality to calculate. The default is ["PageRank", "normedPageRank"].
        years : dict, int, optional - used when conditioning
            Given year, list of year, or an interval dict {"start":int,"end":int}. The default is None.
        ego_nw_tokens : list, optional - used when conditioning
             List of tokens for an ego-network if desired. Only used if no graph is supplied. The default is None.
        depth : TYPE, optional - used when conditioning
            Maximal path length for ego network. Only used if no graph is supplied. The default is 1.
        context : list, optional - used when conditioning
            List of tokens that need to appear in the context distribution of a tie. The default is None.
        weight_cutoff : float, optional - used when conditioning
            Only links of higher weight are considered in conditioning.. The default is None.
        norm_ties : bool, optional - used when conditioning
            Please see semantic network class. The default is True.

        Returns
        -------
        dict
            Dict of centralities for focal tokens.

        """
        input_check(tokens=focal_tokens)
        input_check(tokens=ego_nw_tokens)
        input_check(tokens=context)
        input_check(years=years)

        if not self.conditioned:
            was_conditioned = False
            if ego_nw_tokens == None:
                logging.debug("Conditioning year(s) {} with all tokens, measures on {}".format(
                    years, focal_tokens))
                self.condition(years=years, tokens=None, weight_cutoff=weight_cutoff,
                               depth=depth, context=context, norm=norm_ties)
                logging.debug("Finished conditioning, {} nodes and {} edges in graph".format(
                    len(self.graph.nodes), len(self.graph.edges)))
            else:
                logging.debug("Conditioning ego-network for {} tokens with depth {}, for year(s) {} , measures on {}".format(
                    len(ego_nw_tokens), depth, years, focal_tokens))
                self.condition(years=years, tokens=ego_nw_tokens, weight_cutoff=weight_cutoff,
                               depth=depth, context=context, norm=norm_ties)
                logging.debug("Finished ego conditioning, {} nodes and {} edges in graph".format(
                    len(self.graph.nodes), len(self.graph.edges)))
        else:
            logging.debug(
                "Network already conditioned! No reconditioning attempted, parameters unused.")
            was_conditioned = True

        cent_dict = centrality(
            self.graph, focal_tokens=focal_tokens,  types=types)

        if not was_conditioned:
            # Decondition
            self.decondition()

        return cent_dict


    def cluster(self, levels:int = 1, name:Optional[str]="base", metadata: Optional[Union[dict,defaultdict]] = None, algorithm: Optional[Callable[[nx.DiGraph], List]] = None,to_measure: Optional[List[Callable[[nx.DiGraph], Dict]]] = None, ego_nw_tokens:Optional[List]=None, depth:Optional[int]=1,years:Optional[Union[int,Dict,List]]=None, context:Optional[List]=None, weight_cutoff:float=None, norm_ties:bool=True):
        """
        Cluster the network, run measures on the clusters and return them as networkx subgraph in packaged dicts with metadata.
        Use the levels variable to determine how often to cluster hierarchically.

        Function takes the usual conditioning argument when the network is not yet conditioned.
        If it is conditioned, then the current graph will be used to cluster

        Parameters
        ----------
        levels: int
            Number of hierarchy levels to cluster
        name: str. Optional.
            Base name of the cluster. Further levels will add -i. Default is "base".
        metadata: dict. Optional.
            A dict of metadata that is kept for all clusters.
        algorithm: callable.  Optional.
            Any algorithm taking a networkx graph and return a list of lists of tokens.
        to_measure: list of callables. Optional.
            Functions that take a networkx graph as argument and return a formatted dict with measures.
        ego_nw_tokens: list. Optional.
            Ego network tokens. Used only when conditioning.
        depth: int. Optional.
            Depth of ego network. Used only when conditioning.
        years: int, list, str. Optional
            Given year, list of year, or an interval dict {"start":int,"end":int}. The default is None.
        context : list, optional - used when conditioning
            List of tokens that need to appear in the context distribution of a tie. The default is None.
        weight_cutoff : float, optional - used when conditioning
            Only ties of higher weight are considered. The default is None.
        norm_ties : bool, optional - used when conditioning
            Norm ties to get correct probabilities. The default is True.

        Returns
        -------
        list of cluster-dictionaries.
        """

        input_check(tokens=ego_nw_tokens)
        input_check(tokens=context)
        input_check(years=years)

        # Check and fix up token lists
        if ego_nw_tokens is not None:
            ego_nw_tokens = self.ensure_ids(ego_nw_tokens)
        if context is not None:
            context = self.ensure_ids(context)



        # Prepare metadata with standard additions
        # TODO Add standard metadata for conditioning
        metadata_new=defaultdict(list)
        if metadata is not None:
            for (k, v) in metadata.items():
                metadata_new[k].append(v)


        if not self.conditioned:
            was_conditioned = False
            if ego_nw_tokens == None:
                logging.debug("Conditioning year(s) {} with all tokens".format(
                    years))
                self.condition(years=years, tokens=None, weight_cutoff=weight_cutoff,
                               depth=depth, context=context, norm=norm_ties)
                logging.debug("Finished conditioning, {} nodes and {} edges in graph".format(
                    len(self.graph.nodes), len(self.graph.edges)))
            else:
                logging.debug("Conditioning ego-network for {} tokens with depth {}, for year(s) {} with focus on tokens {}".format(
                    len(ego_nw_tokens), depth, years, ego_nw_tokens))
                self.condition(years=years, tokens=ego_nw_tokens, weight_cutoff=weight_cutoff,
                               depth=depth, context=context, norm=norm_ties)
                logging.debug("Finished ego conditioning, {} nodes and {} edges in graph".format(
                    len(self.graph.nodes), len(self.graph.edges)))
        else:
            logging.debug(
                "Network already conditioned! No reconditioning attempted, parameters unused.")
            was_conditioned = True

        # Prepare base cluster
        base_cluster=return_cluster(self.graph, name, "", 0, to_measure, metadata_new)
        cluster_list=[]
        step_list = []
        base_step_list = []
        prior_list = [base_cluster]
        for t in range(0,levels):
            step_list=[]
            base_step_list=[]
            for base in prior_list:
                base,new_list=cluster_graph(base, to_measure, algorithm)
                base_step_list.append(base)
                step_list.extend(new_list)
            prior_list=step_list
            cluster_list.extend(base_step_list)
        # Add last hierarchy
        cluster_list.extend(step_list)

        if not was_conditioned:
            # Decondition
            self.decondition()

        return cluster_list


    def proximities(self, focal_tokens=None,  alter_subset=None, years=None, context=None, weight_cutoff=None, norm_ties=True):
        """
        Calculate proximities for given tokens.
        Conditions if network is not already conditioned.

        Parameters
        ----------
        focal_tokens : list, str, optional
            List of tokens of interest. If not provided, centralities for all tokens will be returned.
        alter_subset : list, str optional
            List of alters to show. Others are hidden. The default is None.
        years : dict, int, optional - used when conditioning
            Given year, list of year, or an interval dict {"start":int,"end":int}. The default is None.
        context : list, optional - used when conditioning
            List of tokens that need to appear in the context distribution of a tie. The default is None.
        weight_cutoff : float, optional - used when conditioning
            Only ties of higher weight are considered. The default is None.
        norm_ties : bool, optional - used when conditioning
            Norm ties to get correct probabilities. The default is True.

        Returns
        -------
        proximity_dict : dict
            Dictionary of form {token_id:{alter_id: proximity}}.

        """

        input_check(tokens=focal_tokens)
        input_check(tokens=alter_subset)
        input_check(tokens=context)
        input_check(years=years)

        if alter_subset is not None:
            alter_subset = self.ensure_ids(alter_subset)
        if focal_tokens is not None:
            focal_tokens = self.ensure_ids(focal_tokens)
        else:
            focal_tokens = self.ids

        if self.conditioned:
            logging.debug(
                "Network already conditioned! No reconditioning attempted, parameters unused.")
        proximity_dict = {}
        for token in focal_tokens:
            if self.conditioned:
                was_conditioned = True
            else:
                logging.debug(
                    "Conditioning year(s) {} with focus on token {}".format(years, token))
                self.condition(years, tokens=[
                    token], weight_cutoff=weight_cutoff, depth=1, context=context, norm=norm_ties)
                was_conditioned = False

            tie_dict = proximity(self.graph, focal_tokens=focal_tokens, alter_subset=alter_subset)[
                'proximity'][token]

            proximity_dict.update({token: tie_dict})

        if not was_conditioned:
            # Decondition
            self.decondition()

        return {"proximity": proximity_dict}

    def to_backout(self, decay=None, method="invert", stopping=25):
        """
        If each node is defined by the ties to its neighbors, and neighbors
        are equally defined in this manner, what is the final composition
        of each node?

        Function redefines neighborhood of a node by following all paths
        to other nodes, weighting each path according to its length by the 
        decay parameter:
            a_ij is the sum of weighted, discounted paths from i to j

        Row sum then corresponds to Eigenvector or Bonacich centrality.


        Parameters
        ----------
        graph : networkx graph
            Supplied networkx graph.
        nodelist : list, array, optional
            List of nodes to subset graph.
        decay : float, optional
            Decay parameter determining the weight of higher order ties. The default is None.
        method : "invert" or "series", optional
            "invert" tries to invert the adjacency matrix.
            "series" uses a series computation. The default is "invert".
        stopping : int, optional
            Used if method is "series". Determines the maximum order of series computation. The default is 25.
        """
        if not self.conditioned:
            logging.warning(
                "Network is not conditioned. Conditioning on all data...")
            self.condition()

        self.graph = backout_measure(
            self.graph, decay=decay, method=method, stopping=stopping)

    def to_symmetric(self, technique="avg-sym"):
        """
        Make graph symmetric

        Parameters
        ----------
        technique : string, optional
            transpose: Transpose and average adjacency matrix. Note: Loses other edge parameters!
            min-sym: Retain minimum direction, no tie if zero / unidirectional.
            max-sym: Retain maximum direction; tie exists even if unidirectional.
            avg-sym: Average ties. 
            min-sym-avg: Average ties if link is bidirectional, otherwise no tie.
            The default is "avg-sym".

        Returns
        -------
        None.

        """

        if not self.conditioned:
            logging.warning(
                "Network is not conditioned. Conditioning on all data...")
            self.condition()

        self.graph = make_symmetric(self.graph, technique)

    def ensure_tokens(self, tokens):
        return tokens
    # %% Graph abstractions - for now only network

    def create_empty_graph(self):
        return nx.DiGraph()

    def delete_graph(self):
        self.graph = None

    def __delitem__(self, key):
        """
        Deletes a node and all its ties
        :param key: Token or Token id
        :return:
        """
        self.remove(key)
        self.update()

    def remove(self, key):
        """
        Deletes a node and all its ties
        :param key: Token or Token id
        :return:
        """
        self.graph.remove_node(key)

    def __setitem__(self, key, value):
        """
        Set links of node
        :param key: id or token of node.
        :param value: To add links to node:  [(neighbor,time, {'weight':weight,'p1':p1,'p2':p2}))]. To add node itself, token_id (int)
        :return:
        """
        try:
            neighbors = [x[0] for x in value]
            weights = [{'weight': x[2], 'time': self.starttime, 'start': self.starttime,
                        'end': self.endtime, 'occurrences': 1} for x in value]
            token = map(str, np.repeat(key, len(neighbors)))
        except:
            raise ValueError(
                "Adding requires an iterable over tuples e.g. [(neighbor,time, weight))]")

        # Check if all neighbor tokens present
        assert set(neighbors) < set(
            self.ids), "ID of node to connect not found. Not in network?"
        ties = list(zip(token, neighbors, weights))

        # TODO Dispatch if graph conditioned

        # Add ties to query
        self.graph.add_edges_from(ties)
        self.update()

    def __getitem__(self, i):
        """
        Retrieve node information with input checking
        :param i: int or list or nodes, or tuple of nodes with timestamp. Format as int YYYYMMDD, or dict with {'start:'<YYYYMMDD>, 'end':<YYYYMMDD>.
        :return: NetworkX compatible node format
        """
        return(dict(self.graph[i]))

    def __len__(self):
        return len(self.graph.nodes)

    def insert(self, token):
        """
        Insert a new token
        :param token: Token string
        :param token_id: Token ID
        :return: None
        """
        assert isinstance(
            token, str), "Please add <token>,<token_id> as str,int"
        self.graph.add_node(token)
        self.update()
