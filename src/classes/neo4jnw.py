import copy
import inspect
from collections.abc import Sequence
import random

import numpy as np
from networkx import compose_all
from tqdm import tqdm

# import neo4j utilities and classes
from src.classes.neo4db import neo4j_database
from src.functions.backout_measure import backout_measure
from src.functions.file_helpers import check_create_folder
from src.functions.format import pd_format
# Clustering
from src.functions.graph_clustering import *
from src.functions.node_measures import proximity, centrality
from src.measures.measures import proximities, centralities
from src.utils.input_check import input_check
from src.utils.network_tools import make_reverse, sparsify_graph
from src.utils.twowaydict import TwoWayDict
import pandas as pd

# Type definition
try:  # Python 3.8+
    from typing import TypedDict


    class GraphDict(TypedDict):
        graph: nx.DiGraph
        name: str
        parent: int
        level: int
        measures: List
        metadata: [Union[Dict, defaultdict]]
except:
    GraphDict = Dict[str, Union[str, int, Dict, List, defaultdict]]

try:
    import networkx as nx
except:
    nx = None

try:
    import igraph as ig
except:
    ig = None


class neo4j_network(Sequence):

    # %% Initialization functions
    def __init__(self, config=None, neo4j_creds=None, graph_type="networkx", agg_operator="SUM",
                 write_before_query=True,
                 neo_batch_size=None, queue_size=100000, tie_query_limit=100000, tie_creation="UNSAFE",
                 logging_level=None, norm_ties=False, connection_type=None, consume_type=None, seed=100):
        # Fill parameters from configuration file
        if logging_level is not None:
            self.logging_level = logging_level
        else:
            if config is not None:
                self.logging_level = config['General'].getint('logging_level')
            else:
                msg = "Please provide valid logging level."
                logging.error(msg)
                raise AttributeError(msg)
        # Set logging level
        logging.disable(self.logging_level)

        if neo_batch_size is not None:
            self.neo_batch_size = neo_batch_size
        else:
            if config is not None:
                self.neo_batch_size = int(config['General']['neo_batch_size'])
            else:
                msg = "Please provide valid neo_batch_size."
                logging.error(msg)
                raise AttributeError(msg)

        if connection_type is not None:
            self.connection_type = connection_type
        else:
            if config is not None:
                self.connection_type = config['NeoConfig']['protocol']
            else:
                msg = "Please provide valid protocol."
                logging.error(msg)
                raise AttributeError(msg)

        if neo4j_creds is not None:
            self.neo4j_creds = neo4j_creds
        else:
            if config is not None:
                if self.connection_type == "http":
                    self.neo4j_creds = (
                        config['NeoConfig']["http_uri"], (config['NeoConfig']["db_db"], config['NeoConfig']["db_pwd"]))
                else:
                    self.neo4j_creds = (
                        config['NeoConfig']["db_uri"], (config['NeoConfig']["db_db"], config['NeoConfig']["db_pwd"]))
            else:
                msg = "Please provide valid neo4j_creds."
                logging.error(msg)
                raise AttributeError(msg)

        self.db = neo4j_database(neo4j_creds=self.neo4j_creds, agg_operator=agg_operator,
                                 write_before_query=write_before_query, neo_batch_size=self.neo_batch_size,
                                 queue_size=queue_size,
                                 tie_query_limit=tie_query_limit, tie_creation=tie_creation,
                                 logging_level=logging_level, connection_type=self.connection_type, consume_type=consume_type)

        # Conditioned graph information
        self.graph_type = graph_type
        self.graph = None
        self.conditioned = False
        self.norm_ties = norm_ties
        self.years = []
        self.seed=seed
        self.set_random_seed(seed)

        # Dictionaries and token/id saved in memory
        self.cond_dict=defaultdict(lambda: None)
        self.filename=""
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
        self.init_tokens()
        # Init parent class
        super().__init__()

    # %% Interface

    def get_times_list(self):
        query = "MATCH (n) WHERE EXISTS(n.time) RETURN DISTINCT  n.time AS time"
        res = self.db.receive_query(query)
        times = [x['time'] for x in res]
        return times

    def set_norm_ties(self, norm=None):
        """
        Sets or switches the normalization of ties.
        This switches the default behavior if no argument is passed explicitly
        Parameters
        ----------
        norm

        Returns
        -------
        None

        """
        # If no norm is supplied, we switch
        if norm is None:
            if self.norm_ties:
                norm = False
            else:
                norm = True
        # Set norm ties accordingly.
        if norm:
            self.norm_ties = True
            logging.info("Switched default to normalization to compositional mode.")
        elif not norm:
            self.norm_ties = False
            logging.info("Switched default to normalization to aggregate mode.")

    def pd_format(self, output: Union[List, Dict], ids_to_tokens: bool = True) -> List:
        """
        Formats a list of measures, or a single measure, into pandas for output.

        Accepted measures are proximity, centrality, and yearly centrality

        centrality returns a nxk DataFrame, where n is the number of focal tokens, and k is the number of centrality measures

        proximities are returned as a nxk DataFrame, where n is the number of focal tokens, and k is the alter subset
        such that [nk] represents the tie from n to k in the graph.
        Note that when calculating proximities for the whole graph, [nk] is not necessarily symmetric nor square.
        This is so, because tokens that have no incoming ties will not show up in the column dimension, but will show
        up in the row dimensions.

        Parameters
        ----------
        output: list or dict
            The output measures to format
        ids_to_tokens: bool
            Whether to output tokens as words rather than ids. Default is True.

        Returns
        -------
        List of DataFrames.
        """
        # Format output into pandas
        format_output = pd_format(output)
        # Transform ids to token names
        if not isinstance(format_output,list):
            format_output=[format_output]
        if ids_to_tokens:
            for idx, pd_tbl in enumerate(format_output):

                if isinstance(pd_tbl.index, pd.core.indexes.multi.MultiIndex):
                    cvals = pd_tbl.index.values
                    cvals2 = pd_tbl.index.values
                    for i, col in enumerate(cvals):
                        col = list(col)
                        for j,c in enumerate(col):
                            try:
                                col[j]=self.ensure_tokens(c)
                            except:
                                col[j]=c
                                for k, cold in enumerate(cvals):
                                    cold = list(cold)
                                    cold_mix = list(cvals2[k])
                                    cold_mix[j]=cold[j]
                                    cvals2[k] = tuple(cold_mix)
                        cvals2[i]=tuple(col)
                    pd_tbl.index = pd.MultiIndex.from_tuples(cvals.tolist())
                else:
                    pd_tbl.index = self.ensure_tokens(list(pd_tbl.index))

                if isinstance(pd_tbl.columns, pd.core.indexes.multi.MultiIndex):
                    cvals = pd_tbl.columns.values
                    for i, col in enumerate(cvals):
                        col = list(col)
                        for j,c in enumerate(col):
                            col[j]=self.ensure_tokens(c)
                        cvals[i]=tuple(col)
                    pd_tbl.columns = pd.MultiIndex.from_tuples(cvals.tolist())
                else:
                    pd_tbl.columns = self.ensure_tokens(list(pd_tbl.columns))

                format_output[idx] = pd_tbl

        return format_output

    # %% Conditoning functions



    def context_condition(self, times:Optional[Union[int,list]]=None, tokens:Optional[Union[int,str,list]]=None, weight_cutoff:Optional[float]=None, depth:Optional[int]=None,batchsize:Optional[int]=None,cond_type:Optional[str]=None,occurrence=False,asyncio=False):
        """

        Derive context network by conditioning on tokens that are likely to

        Parameters
        ----------
        times: int or list, optional
            Times (Years) to condition on
        tokens: int, str or list, optional
            tokens or token_ids.
            If given, ego networks of these tokens will be derived.
        weight_cutoff: float, optinal
            Ties below this cutoff value are discarded
        depth: int, optional
            The depth of the ego networks to consider.

        batchsize: int, optional
            If given, queries of this size are sent to the database.
        cond_type: str, optional
            Manually set how ego network is queried. If not set, heuristically determined.
                "subset": queries the entire network into memory and uses network x to find the ego networks
                "search": queries the ego networks in a "beam search" fashion. Slower for high depths.
        Returns
        -------

        """

        # Set up dict of configuration of the conditioning
        cond_dict = defaultdict(lambda: None)

        if batchsize is None:
            batchsize = self.neo_batch_size

        input_check(tokens=tokens)
        input_check(years=times)

        # Check and fix up token lists
        if tokens is not None:
            cond_dict.update({'tokens': tokens})
            tokens = self.ensure_ids(tokens)
            if not isinstance(tokens, list):
                tokens = [tokens]
        else:
            cond_dict.update({'tokens': None})


        # Without times, we query all
        if times is None:
            years = self.get_times_list()
            cond_dict.update({'years': 'ALL'})
        else:
            cond_dict.update({'years': times})

        cond_dict.update({'type':"context", 'cutoff': weight_cutoff, 'depth': depth})

        self.cond_dict=cond_dict


        if tokens is None:
            logging.debug("Context Conditioning dispatch: Yearly")
            if asyncio:
                self.__ayearly_context_condition(times=times, weight_cutoff=weight_cutoff, batchsize=batchsize, occurrence=occurrence)
            else:
                self.__yearly_context_condition(times=times, weight_cutoff=weight_cutoff,batchsize=batchsize, occurrence=occurrence)
        else:
            logging.debug("Context Conditioning dispatch: Ego")
            self.__context_ego_conditioning(times=times, tokens=tokens, weight_cutoff=weight_cutoff, depth=depth, batchsize=batchsize, cond_type=cond_type, occurrence=occurrence)



    def condition(self, times:Optional[Union[int,list]]=None, tokens:Optional[Union[int,str,list]]=None, weight_cutoff:Optional[float]=None, depth:Optional[int]=None, context:Optional[Union[int,str,list]]=None, compositional:Optional[bool]=None,
                  batchsize:Optional[int]=None, cond_type:Optional[str]=None, reverse_ties:Optional[bool]=False,
                  post_cutoff:Optional[float]=None, post_norm:Optional[bool]=False, query_mode="new"):
        """

        Condition the network: Pull ties from database and aggregate according to given parameters.

        Parameters
        ----------
        times: int or list, optional
            Times (Years) to condition on
        tokens: int, str or list, optional
            tokens or token_ids.
            If given, ego networks of these tokens will be derived.
        weight_cutoff: float, optinal
            Ties below this cutoff value are discarded
        depth: int, optional
            The depth of the ego networks to consider.
        context: int, str or list, optional
            tokens or token_ids.
            If given, only ties that occur within a sentence including these tokens are considered
        compositional: bool, optional
            Whether to condition as compositional ties, or aggregate ties
        batchsize: int, optional
            If given, queries of this size are sent to the database.
        cond_type: str, optional
            Manually set how the queries are sent. If not set, heuristically determined.
                "subset": queries the entire network into memory and uses network x to find the ego networks
                "search": queries the ego networks in a "beam search" fashion. Slower for high depths.
        reverse_ties: bool, optional
            If true, ties will be reversed after conditioning.
        post_cutoff: float, optional
            If set, network edges will be pruned by edges smaller than the given value.
            Note that this does not affect the conditioning itself
        post_norm: bool, optional
            If true, network will be normed after post_cutoff is applied.
            Network is normed such that the in-degree of each node is 1.
        Returns
        -------
        Nothing
        """

        # Set up dict of configuration of the conditioning
        cond_dict = defaultdict(lambda: None)

        # Get default normation behavior
        if compositional is None:
            compositional = self.norm_ties

        if batchsize is None:
            batchsize = self.neo_batch_size

        input_check(tokens=tokens)
        input_check(tokens=context)
        input_check(years=times)

        # Check and fix up token lists
        if tokens is not None:
            cond_dict.update({'tokens': tokens})
            tokens = self.ensure_ids(tokens)
            if not isinstance(tokens, list):
                tokens = [tokens]
        else:
            cond_dict.update({'tokens': None})


        # Without times, we query all
        if times is None:
            cond_dict.update({'years': 'ALL'})
        else:
            cond_dict.update({'years': times})

        cond_dict.update({'type':"replacement", 'compositional':compositional, 'cutoff': weight_cutoff, 'context': context, 'depth': depth})

        self.cond_dict=cond_dict

        if tokens is None:
            logging.debug("Conditioning dispatch: Yearly")
            self.__year_condition(years=times, weight_cutoff=weight_cutoff, context=context, norm=compositional, batchsize=batchsize)
        else:
            logging.debug("Conditioning dispatch: Ego")
            if depth is None:
                checkdepth=1000
            else:
                checkdepth=depth
            if cond_type is None:
                if checkdepth <= 2 and len(tokens) <= 5:
                    cond_type="search"
                elif checkdepth == 0: # just need proximities
                    cond_type="search"
                else:
                    cond_type="subset"
            if cond_type=="subset":
                logging.debug("Conditioning dispatch: Ego, subset, depth {}".format(depth))
                self.__ego_condition(years=times, token_ids=tokens, weight_cutoff=weight_cutoff, depth=depth,
                                     context=context, norm=compositional, batchsize=batchsize, query_mode=query_mode)
            elif cond_type=="search":
                logging.debug("Conditioning dispatch: Ego, search, depth {}".format(depth))
                self.__ego_condition_old(years=times, token_ids=tokens, weight_cutoff=weight_cutoff, depth=depth,
                                         context=context, norm=compositional, batchsize=batchsize, query_mode=query_mode)
            else:
                msg="Conditioning type {} requested. Please use either search or subset.".format(cond_type)
                logging.debug(msg)
                raise NotImplementedError(msg)
        if reverse_ties:
            self.to_reverse()
        self.__cut_and_norm(post_cutoff,post_norm)

    def decondition(self):
        # Reset token lists to original state.
        if self.conditioned:
            self.ids = self.neo_ids
            self.tokens = self.neo_tokens
            self.token_id_dict = self.neo_token_id_dict

            # Decondition
            logging.debug("Deconditioning graph.")
            self.delete_graph()
            self.conditioned = False
            self.cond_dict=defaultdict(lambda: None)
            self.filename=""

    # %% Clustering
    def cluster(self, levels: int = 1, name: Optional[str] = "base", interest_list: Optional[list] = None,
                metadata: Optional[Union[dict, defaultdict]] = None,
                algorithm: Optional[Callable[[nx.DiGraph], List]] = None,
                to_measure: Optional[List[Callable[[nx.DiGraph], Dict]]] = None,
                reverse_ties:Optional[bool]=False, add_ego_tokens:Optional[Union[str,int,list]]=None):
        """
        Cluster the network, run measures on the clusters and return them as networkx subgraph in packaged dicts with metadata.
        Use the levels variable to determine how often to cluster hierarchically.

        Function requires network to be conditioned!

        Parameters
        ----------
        levels: int
            Number of hierarchy levels to cluster
        name: str. Optional.
            Base name of the cluster. Further levels will add -i. Default is "base".
        interest_list: list. Optional.
            List of tokens or token_ids of interest. Clusters not including any of these terms are discarded
        metadata: dict. Optional.
            A dict of metadata that is kept for all clusters.
        algorithm: callable.  Optional.
            Any algorithm taking a networkx graph and return a list of lists of tokens.
        to_measure: list of callables. Optional.
            Functions that take a networkx graph as argument and return a formatted dict with measures.
        add_ego_tokens: str, int or list. Optional
            Optional list of ego tokens (or ids). If given, they will be added to
            each graph before further clustering occurs.

        Returns
        -------
        list of cluster-dictionaries.
        """


        input_check(tokens=interest_list)
        input_check(tokens=add_ego_tokens)

        # # Check and fix up token lists
        # input_check(tokens=context)
        # input_check(years=years)
        # input_check(tokens=ego_nw_tokens)
        # if ego_nw_tokens is not None:
        #     ego_nw_tokens = self.ensure_ids(ego_nw_tokens)
        #     if not isinstance(ego_nw_tokens, list):
        #         ego_nw_tokens = [ego_nw_tokens]
        # if context is not None:
        #     context = self.ensure_ids(context)
        # # Get default normation behavior
        # if norm_ties is None:
        #     norm_ties = self.norm_ties

        if interest_list is not None:
            interest_list = self.ensure_ids(interest_list)
            if not isinstance(interest_list, list):
                interest_list = [interest_list]

        if add_ego_tokens is not None:
            add_ego_tokens = self.ensure_ids(add_ego_tokens)
            if not isinstance(add_ego_tokens, list):
                add_ego_tokens = [add_ego_tokens]


        # Prepare metadata with standard additions
        # TODO Add standard metadata for conditioning
        metadata_new = defaultdict(list)
        if metadata is not None:
            for (k, v) in metadata.items():
                metadata_new[k].append(v)

        if not self.conditioned:
            # Previously, the conditioning function would be invoked here.
            # We have since decided to require the user to condition before calling clustering
            self.__condition_error(call=inspect.stack()[1][3])

            #was_conditioned = False
            #if ego_nw_tokens is None:
            #    logging.debug("Conditioning year(s) {} with all tokens".format(
            #        years))
            #    self.condition(years=years, tokens=None, weight_cutoff=weight_cutoff,
            #                    depth=depth, context=context, norm=norm_ties)
            #     logging.debug("Finished conditioning, {} nodes and {} edges in graph".format(
            #         len(self.graph.nodes), len(self.graph.edges)))
            # else:
            #     logging.debug(
            #         "Conditioning ego-network for {} tokens with depth {}, for year(s) {} with focus on tokens {}".format(
            #             len(ego_nw_tokens), depth, years, ego_nw_tokens))
            #     self.condition(years=years, tokens=ego_nw_tokens, weight_cutoff=weight_cutoff,
            #                    depth=depth, context=context, norm=norm_ties)
            #     logging.debug("Finished ego conditioning, {} nodes and {} edges in graph".format(
            #         len(self.graph.nodes), len(self.graph.edges)))

        # We allow to reverse ties in network
        if reverse_ties:
            self.to_reverse()


        # Prepare base cluster
        base_cluster = return_cluster(self.graph, name, "", 0, to_measure, metadata_new)
        cluster_list = []
        step_list = []
        prior_list = [base_cluster]
        # This goes as follows
        # prior_list is a list of clusters from the prior level - starting with level 0
        # base_step_list is a list of the same prior clusters, but modified such that each node has t+1 cluster information
        # step_list is a list of clusters populated from current level
        # cluster_list is a list of clusters for all levels

        # Since we want the clusters of level t to have node-information of level t+1, we have to run in a specific way
        # Start with level t, run the cluster function for each cluster
        # Get the cluster at level t, plus all its subclusters
        # save these in base_step_list (level t) and step_list (level t+1) respectively
        # Now add base_step_list (level t) to the overall cluster list
        # Set step list as new base list and run level t+1
        for t in range(0, levels):
            step_list = []
            base_step_list = []
            for base in prior_list:
                base, new_list, cluster_dict = cluster_graph(base, to_measure, algorithm, add_ego_tokens=add_ego_tokens)
                base_step_list.append(base)
                # Add assignment (level) to graph of snw
                nx.set_node_attributes(self.graph, cluster_dict, 'clusterl'+str(t))
                if interest_list is not None:  # We want to proceed only on clusters of interest.
                    for cl in new_list:
                        cl_nodes = self.ensure_ids(list(cl['graph'].nodes))
                        if len(np.intersect1d(cl_nodes, interest_list)) > 0:  # Cluster of interest, add
                            step_list.append(cl)
                else:  # add all
                    step_list.extend(new_list)
            prior_list = step_list
            cluster_list.extend(base_step_list)
        # Add last hierarchy
        cluster_list.extend(step_list)

        # Redo reverse ties
        if reverse_ties:
            self.to_reverse()

        return cluster_list

    # %% Measures

    def centralities(self, focal_tokens=None, types=["PageRank", "normedPageRank"], reverse_ties: Optional[bool] = False):
        """
        See measures.centralities
        """
        return centralities(self, focal_tokens=focal_tokens, types=types,reverse_ties=reverse_ties)

    def proximities(self,focal_tokens: Optional[List] = None, alter_subset: Optional[List] = None,
              reverse_ties: Optional[bool] = False) -> Dict:
        """
        See measures.proximities
        """
        return proximities(self, focal_tokens=focal_tokens, alter_subset=alter_subset, reverse_ties=reverse_ties)


    def get_node_context(self, tokens:Union[list,int,str], years:Union[int,list,dict]=None, weight_cutoff:Optional[float]=None, occurrence:Optional[bool]=False):

        if years is None:
            years=self.get_times_list()

        input_check(tokens=tokens)
        input_check(years=years)

        if isinstance(tokens,(str,int)):
            tokens=[tokens]

        tokens=self.ensure_ids(tokens)

        res=self.db.query_context_of_node(tokens,times=years,weight_cutoff=weight_cutoff, occurrence=occurrence)

        tokens=np.array([x[0] for x in res])
        alters=np.array([x[1] for x in res])
        weights=np.array([x[2]['weight'] for x in res])

        pd_dict={}
        for token in np.unique(tokens):
            mask=tokens==token
            cidx=list(alters[mask])
            cweights=list(weights[mask])
            contexts=dict(zip(cidx,cweights))
            pd_dict.update({int(token): contexts})

        return {'proximity': pd_dict}

    def get_dyad_context(self, dyads:Optional[Union[list,tuple]]=None, occurrence:Optional[Union[list,str,int]]=None, replacement:Optional[Union[list,str,int]]=None, years:Union[int,list,dict]=None, weight_cutoff:Optional[float]=None):
        """
        Specify a dyad and get the distribution of contextual words.
        dyads are tuples of (occurrence,replacement), that is, in the graph, ties that go replacement->occurrence.
        You can pass a list of tuples.

        You can also pass occurrences and replacement tokens as list. However, these have to
        Parameters
        ----------
        dyads
        occurrence
        replacement
        years
        weight_cutoff

        Returns
        -------

        """

        # Untangle dyad lists
        if dyads is not None:
            if isinstance(dyads, list):
                occurrence=[x[0] for x in dyads]
                replacement=[x[1] for x in dyads]
            else:
                occurrence=dyads[0]
                replacement=dyads[1]
        elif replacement is None or occurrence is None:
            msg="Please provide either dyads as list of tuples (occurrence, replacement) or individual lists of occurrences and replacements!"
            logging.error(msg)
            raise AttributeError(msg)

        if years is None:
            years=self.get_times_list()

        input_check(tokens=occurrence)
        input_check(tokens=replacement)
        input_check(years=years)

        if isinstance(occurrence,(str,int)):
            occurrence=[occurrence]
        if isinstance(replacement, (str, int)):
            replacement = [replacement]


        occurrence=np.array(self.ensure_ids(occurrence))
        replacement = np.array(self.ensure_ids(replacement))

        try:
            dyads=np.concatenate([occurrence.reshape(-1, 1), replacement.reshape(-1, 1)], axis=1)
        except:
            msg="Could not broadcast occurrences to replacements"
            logging.error(msg)
            raise AttributeError(msg)


        pd_dict={}
        for dyad in dyads:
            cidx,cweights=self.db.query_tie_context(int(dyad[0]),int(dyad[1]),times=years,weight_cutoff=weight_cutoff)
            contexts=dict(zip(cidx,cweights))
            pd_dict.update({tuple(dyad): contexts})

        return {'dyad_context': pd_dict}







    # %% Graph manipulation

    def sparsify(self,percentage:int=100):
        """
        Sparsify the graph as follows:
        For each node, delete low strength ties until percentage of the prior aggregate outgoing ties mass is retained

        Parameters
        ----------
        percentage: int
            Percentage of out-degree mass to retain
        Returns
        -------

        """
        if not self.conditioned:
            # Previously, the conditioning function would be invoked here.
            # We have since decided to require the user to condition before calling clustering
            self.__condition_error(call=inspect.stack()[1][3])

        self.graph=sparsify_graph(self.graph, percentage)

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
        decay : float, optional
            Decay parameter determining the weight of higher order ties. The default is None.
        method : "invert" or "series", optional
            "invert" tries to invert the adjacency matrix.
            "series" uses a series computation. The default is "invert".
        stopping : int, optional
            Used if method is "series". Determines the maximum order of series computation. The default is 25.


        Returns
        -------
        None.

        """
        if not self.conditioned:
            # Previously, the conditioning function would be invoked here.
            # We have since decided to require the user to condition before calling clustering
            self.__condition_error(call=inspect.stack()[1][3])


        self.graph = backout_measure(
            self.graph, decay=decay, method=method, stopping=stopping)

    def to_reverse(self):
        """
        Reverses a previously conditioned graph

        Returns
        -------
        None.
        """
        if not self.conditioned:
            # Previously, the conditioning function would be invoked here.
            # We have since decided to require the user to condition before calling clustering
            self.__condition_error(call=inspect.stack()[1][3])


        self.graph = make_reverse(self.graph)

    def to_symmetric(self, technique="avg-sym"):
        """
        Make graph symmetric

        Parameters
        ----------
        technique : string, optional
            transpose: Transpose and average adjacency matrix. Note: Loses other edge parameters!
            min-sym: Retain minimum direction, no tie if zero OR directed.
            max-sym: Retain maximum direction; tie exists even if directed.
            avg-sym: Average ties. 
            min-sym-avg: Average ties if link is bidirectional, otherwise no tie.
            The default is "avg-sym".

        Returns
        -------
        None.

        """

        if not self.conditioned:
            # Previously, the conditioning function would be invoked here.
            # We have since decided to require the user to condition before calling clustering
            self.__condition_error(call=inspect.stack()[1][3])


        self.graph = make_symmetric(self.graph, technique)

        # %% Sequence Interface implementations

    def __getitem__(self, i):
        """
        Retrieve node information with input checking
        :param i: int or list of nodes, or tuple of nodes with timestamp. Format as int YYYYMMDD, or dict with {'start:'<YYYYMMDD>, 'end':<YYYYMMDD>.
        :return: NetworkX compatible node format
        """
        # If so desired, induce a queue write before any query
        if self.db.write_before_query:
            self.db.write_queue()
        # Are time formats submitted? Handle those and check inputs
        if isinstance(i, tuple):
            assert len(
                i) == 2, "Please format a call as (<tokens>,<time>) or (<tokens>,{'start:'<time>, 'end':<time>})"
            # if not isinstance(i[1], dict):
            #    assert isinstance(
            #        i[1], int), "Please timestamp as <time>, or {'start:'<time>, 'end':<time>}"
            input_check(years=i[1], tokens=i[0])
            year = i[1]
            i = i[0]
        else:
            year = None
            input_check(tokens=i)

        i = self.ensure_ids(i)
        if not self.conditioned:
            return self.query_nodes(i, times=year, norm_ties=self.norm_ties)
        else:
            if isinstance(i, (list, tuple, np.ndarray)):
                returndict = []
                for token in i:
                    neighbors = dict(self.graph[token])
                    returndict.extend({token: neighbors})
            else:
                neighbors = dict(self.graph[i])
                returndict = {i: neighbors}
            return returndict

    def __len__(self):
        return len(self.tokens)

    # %% Conditioning sub-functions

    def __year_condition(self, years, weight_cutoff=None, context=None, norm=None, batchsize=None, query_mode="old"):
        """ Condition the entire network over all years """

        # Get default normation behavior
        if norm is None:
            norm = self.norm_ties

        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        if not self.conditioned:  # This is the first conditioning
            # Build graph
            self.graph = self.create_empty_graph()


            # All tokens
            worklist = self.ids
            # Add all tokens to graph
            self.graph.add_nodes_from(worklist)

            # Loop batched over all tokens to condition
            for i in tqdm(range(0, len(worklist), batchsize), leave=False,position=0):

                token_ids = worklist[i:i + batchsize]
                logging.debug(
                    "Conditioning by query batch {} of {} tokens.".format(i, len(token_ids)))
                # Query Neo4j
                self.graph.add_edges_from(
                    self.query_nodes(token_ids, context=context, times=years, weight_cutoff=weight_cutoff,
                                     norm_ties=norm, query_mode=query_mode))
            try:
                all_ids = list(self.graph.nodes)
            except:
                logging.error("Could not condition graph by query method.")

            # Update IDs and Tokens to reflect conditioning

            all_tokens = [self.get_token_from_id(x) for x in all_ids]
            # Add final properties
            att_list = [{"token": x} for x in all_ids]
            att_dict = dict(list(zip(all_ids, att_list)))
            nx.set_node_attributes(self.graph, att_dict)

            # Set conditioning true
            self.__complete_conditioning()

        else:  # Remove conditioning and recondition
            self.decondition()
            self.__year_condition(years, weight_cutoff, context, norm)

    def __ego_condition(self, years, token_ids, weight_cutoff=None, depth=None, context=None, norm=None,
                        batchsize=None, query_mode="old"):

        # Get default normation behavior
        if norm is None:
            norm = self.norm_ties
        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        # First, do a year conditioning
        logging.debug("Full year conditioning before ego subsetting.")
        self.__year_condition(years=years, weight_cutoff=weight_cutoff, context=context, norm=norm, batchsize=batchsize, query_mode=query_mode)

        if depth is not None:
            # Create ego graph for each node
            graph_list=[]

            for focal_token in token_ids:
                temp_graph=nx.generators.ego.ego_graph(self.graph, focal_token, radius=depth, center=True, undirected=False)
                graph_list.append(temp_graph)
            # Compose ego graphs
            self.graph = compose_all(graph_list)
        # Set conditioning true
        self.__complete_conditioning(copy_ids=False)

    def __ego_condition_old(self, years, token_ids, weight_cutoff=None, depth=None, context=None, norm=None,
                            batchsize=None,query_mode="old"):



        if not self.conditioned:  # This is the first conditioning
            # Build graph
            self.graph = self.create_empty_graph()
            self.db.open_session()
            if depth is None:
                logging.debug("Depth is None, but search conditioning is requested. Setting depth to 1.")
                depth = 1
            # Check one level deeper
            or_depth=depth
            depth += 1
            # Create a dict to hold previously queried ids
            prev_queried_ids = list()
            # ids to check
            ids_to_check=token_ids
            logging.debug(
                "Start of Depth {} conditioning: {} tokens".format(or_depth, len(ids_to_check)))
            while depth > 0:
                if not isinstance(ids_to_check, (list, np.ndarray)):
                    ids_to_check = [ids_to_check]
                # Work from ID list, give error if tokens are not in database
                ids_to_check = self.ensure_ids(ids_to_check)
                # Do not consider already added tokens
                ids_to_check = np.setdiff1d(ids_to_check, prev_queried_ids)
                logging.debug(
                    "Depth {} conditioning: {} new found tokens, where {} already added.".format(depth, len(ids_to_check),
                                                                                                 len(prev_queried_ids)))
                # Add ids_to_check to list since they will be queried this iteration
                prev_queried_ids.extend(ids_to_check)
                if isinstance(ids_to_check, (np.ndarray)):
                    ids_to_check=list(ids_to_check)
                ids_to_check.reverse()
                # Add starting nodes
                self.graph.add_nodes_from(ids_to_check)
                for i in tqdm(range(0, len(ids_to_check), batchsize), leave=False,position=0):

                    id_batch = ids_to_check[i:i + batchsize]
                    id_batch
                    logging.debug(
                        "Conditioning by query batch {} of {} tokens.".format(i, len(ids_to_check)))
                    # Query Neo4j
                    try:
                        self.graph.add_edges_from(
                            self.query_nodes(id_batch, context=context, times=years, weight_cutoff=weight_cutoff,
                                             norm_ties=norm,query_mode=query_mode))
                    except:
                        logging.error("Could not condition graph by query method.")

                # Delete disconnected nodes
                remove = [node for node, degree in dict(self.graph.degree).items() if degree <= 0]
                self.graph.remove_nodes_from(remove)

                # Update IDs and Tokens to reflect conditioning
                all_ids = list(self.graph.nodes)
                all_tokens = [self.get_token_from_id(x) for x in all_ids]

                # Set the next set of tokens as those that have not been previously queried
                ids_to_check = np.setdiff1d(all_ids, prev_queried_ids)

                # print("{} tokens post setdiff: {}".format(len(token_ids),token_ids))
                # Set additional attributes
                att_list = [{"token": x} for x in all_tokens]
                att_dict = dict(list(zip(all_ids, att_list)))
                nx.set_node_attributes(self.graph, att_dict)

                # decrease depth
                depth = depth - 1
                # No more ids to check
                if not ids_to_check.size > 0:
                    depth = 0

            # Close session
            self.db.close_session()

            # Create ego graph for each node and compose
            if or_depth > 0:
                self.graph = compose_all([nx.generators.ego.ego_graph(self.graph, x, radius=or_depth, center=True,
                                                         undirected=False) for x in token_ids])

            # Set conditioning true
            self.__complete_conditioning()

        else:  # Remove conditioning and recondition
            # TODO: "Allow for conditioning on conditioning"
            self.decondition()
            self.condition(years, token_ids, weight_cutoff,
                           depth, context, norm)

        # Continue conditioning

    def __yearly_context_condition(self, times, weight_cutoff=None, batchsize=None, occurrence=False):
        """ Condition the entire network over all years """


        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        if not self.conditioned:  # This is the first conditioning


            logging.info("Called into yearly conditioning, batch size: {}, cutoff: {}, occurrence: {}".format(batchsize,weight_cutoff,occurrence))
            # Build graph
            self.graph = self.create_empty_graph()


            # All tokens
            worklist = self.ids
            # Add all tokens to graph
            self.graph.add_nodes_from(worklist)

            # Loop batched over all tokens to condition
            for i in tqdm(range(0, len(worklist), batchsize), leave=False,position=0):

                token_ids = worklist[i:i + batchsize]
                logging.debug(
                    "Conditioning by query batch {} of {} tokens.".format(i, len(token_ids)))
                # Query Neo4j
                self.graph.add_edges_from(
                    self.query_context(token_ids, times=times, weight_cutoff=weight_cutoff, occurrence=occurrence))
            try:
                all_ids = list(self.graph.nodes)
            except:
                logging.error("Could not context-condition graph by query method.")

            # Update IDs and Tokens to reflect conditioning

            all_tokens = [self.get_token_from_id(x) for x in all_ids]
            # Add final properties
            att_list = [{"token": x} for x in all_ids]
            att_dict = dict(list(zip(all_ids, att_list)))
            nx.set_node_attributes(self.graph, att_dict)

            # Set conditioning true
            self.__complete_conditioning()

        else:  # Remove conditioning and recondition
            self.decondition()
            self.__yearly_context_condition(times, weight_cutoff, batchsize,occurrence)


    def __ayearly_context_condition(self, times, weight_cutoff=None, batchsize=None,occurrence=False):
        """ Condition the entire network over all years """


        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        if not self.conditioned:  # This is the first conditioning
            # Build graph
            self.graph = self.create_empty_graph()


            # All tokens
            worklist = self.ids
            # Add all tokens to graph
            self.graph.add_nodes_from(worklist)

            # Loop batched over all tokens to condition
            for i in tqdm(range(0, len(worklist), batchsize), leave=False,position=0):

                token_ids = worklist[i:i + batchsize]
                logging.debug(
                    "Conditioning by query batch {} of {} tokens.".format(i, len(token_ids)))
                # Query Neo4j
                self.graph.add_edges_from(
                    self.query_context(token_ids, times=times, weight_cutoff=weight_cutoff))


            try:
                all_ids = list(self.graph.nodes)
            except:
                logging.error("Could not context-condition graph by query method.")

            # Update IDs and Tokens to reflect conditioning

            all_tokens = [self.get_token_from_id(x) for x in all_ids]
            # Add final properties
            att_list = [{"token": x} for x in all_ids]
            att_dict = dict(list(zip(all_ids, att_list)))
            nx.set_node_attributes(self.graph, att_dict)

            # Set conditioning true
            self.__complete_conditioning()

        else:  # Remove conditioning and recondition
            self.decondition()
            self.__yearly_context_condition(times, weight_cutoff, batchsize)



    def __context_ego_conditioning(self, times:Optional[Union[int,list]]=None, tokens:Optional[Union[int,str,list]]=None, weight_cutoff:Optional[float]=None, depth:Optional[int]=None,batchsize:Optional[int]=None,cond_type:Optional[str]=None, occurrence:Optional[bool]=False):


        if not self.conditioned:  # This is the first conditioning
            # Save original depth variable
            or_depth = depth
            if not isinstance(times, (list, np.ndarray)):
                times=[times]
            if cond_type is None:
                if depth is not None:
                    if (depth <= 2 and len(tokens) <= 5):
                        cond_type="search"
                else:
                    cond_type="subset"
            logging.info("Context clustering mode: {} and batch size: {}".format(cond_type, batchsize))
            if cond_type=="search":
                # Build graph
                self.graph = self.create_empty_graph()

                # Depth 0 and Depth 1 really mean the same thing here
                if depth is None:
                    depth = 1
                # Check one level deeper

                depth += 1
                # Create a dict to hold previously queried ids
                prev_queried_ids = list()
                # ids to check
                ids_to_check = tokens
                logging.info(
                    "Start of Depth {} conditioning: {} tokens".format(or_depth, len(ids_to_check)))
                while depth > 0:
                    if not isinstance(ids_to_check, (list, np.ndarray)):
                        ids_to_check = [ids_to_check]
                    # Work from ID list, give error if tokens are not in database
                    ids_to_check = self.ensure_ids(ids_to_check)
                    # Do not consider already added tokens
                    ids_to_check = np.setdiff1d(ids_to_check, prev_queried_ids)
                    logging.info(
                        "Depth {} Context conditioning: {} new found tokens, where {} already added.".format(depth,
                                                                                                     len(ids_to_check),
                                                                                                     len(
                                                                                                         prev_queried_ids)))
                    # Add ids_to_check to list since they will be queried this iteration
                    prev_queried_ids.extend(ids_to_check)
                    # Add starting nodes
                    self.graph.add_nodes_from(ids_to_check)
                    for i in tqdm(range(0, len(ids_to_check), batchsize), leave=False, position=0):

                        id_batch = ids_to_check[i:i + batchsize]
                        logging.debug(
                            "Conditioning by query batch {} of {} tokens.".format(i, len(id_batch)))
                        # Query Neo4j
                        try:
                            self.graph.add_edges_from(self.query_context(ids=id_batch, times=times, weight_cutoff=weight_cutoff, occurrence=occurrence))
                        except:
                            logging.error("Could not context condition graph by query search method.")

                    # Delete disconnected nodes
                    remove = [node for node, degree in dict(self.graph.degree).items() if degree <= 0]
                    self.graph.remove_nodes_from(remove)

                    # Update IDs and Tokens to reflect conditioning
                    all_ids = list(self.graph.nodes)
                    all_tokens = [self.get_token_from_id(x) for x in all_ids]

                    # Set the next set of tokens as those that have not been previously queried
                    ids_to_check = np.setdiff1d(all_ids, prev_queried_ids)

                    # print("{} tokens post setdiff: {}".format(len(token_ids),token_ids))
                    # Set additional attributes
                    att_list = [{"token": x} for x in all_tokens]
                    att_dict = dict(list(zip(all_ids, att_list)))
                    nx.set_node_attributes(self.graph, att_dict)

                    # decrease depth
                    depth = depth - 1
                    # No more ids to check
                    if not ids_to_check.size > 0:
                        depth = 0
            elif cond_type=="subset":
                self.__yearly_context_condition(times=times,weight_cutoff=weight_cutoff,batchsize=batchsize, occurrence=occurrence)
            else:
                msg="Conditioning type {} requested. Please use either search or subset.".format(cond_type)
                logging.debug(msg)
                raise NotImplementedError(msg)

            # Create ego graph for each node and compose
            self.graph = compose_all([nx.generators.ego.ego_graph(self.graph, x, radius=or_depth, center=True,
                                                                  undirected=False) for x in tokens])
            # Set conditioning true
            self.__complete_conditioning()

        else:  # Remove conditioning and recondition
            self.decondition()
            self.__context_ego_conditioning(times=times,tokens=tokens,weight_cutoff=weight_cutoff,depth=depth,batchsize=batchsize,cond_type=cond_type)

    def __complete_conditioning(self, copy_ids=True):

        if copy_ids:
            # Copy pre conditioning IDs
            self.neo_ids = copy.deepcopy(self.ids)
            self.neo_tokens = copy.deepcopy(self.tokens)
            self.neo_token_id_dict = copy.deepcopy(self.token_id_dict)
        # Get IDs from graph
        all_ids = list(self.graph.nodes)
        self.tokens = [self.get_token_from_id(x) for x in all_ids]
        self.ids = all_ids
        self.update_dicts()
        self.conditioned=True
        self.filename= self.__create_filename(self.cond_dict)


    # %% Graph abstractions - for now only networkx

    @staticmethod
    def create_empty_graph() -> nx.DiGraph:
        return nx.DiGraph()

    def delete_graph(self):
        self.graph = None

    # %% Internal functions


    def __create_filename(self, cond_dict):

        if cond_dict['type'] is not None:
            fn = str(cond_dict['type'])+"-"
        else:
            fn = ""

        if cond_dict['tokens'] is not None:
            if not isinstance(cond_dict['tokens'],list):
                cond_dict['tokens']=[cond_dict['tokens']]
            fn=fn+"EGO-"
            fn=fn+'-'.join([str(x) for x in cond_dict['tokens']])+'-'

        if cond_dict['years'] is not "ALL":
            if not isinstance(cond_dict['years'],list):
                cond_dict['years']=[cond_dict['years']]
            fn = fn + 'Y-' + '-'.join([str(x) for x in cond_dict['years']])
        else:
            fn = fn + 'Y-' + "ALL"

        if cond_dict['depth']:
            fn = fn + '-depth' + str(cond_dict['depth'])

        if cond_dict['cutoff']:
            fn = fn + '-cut' + str(cond_dict['cutoff'])

        if cond_dict['compositional']:
            fn = fn + '-comp' + str(cond_dict['compositional'])

        if cond_dict['context'] is not None:
            if not isinstance(cond_dict['context'],list):
                cond_dict['context']=[cond_dict['context']]
            fn="context-"
            fn=fn+'-'.join([str(x) for x in cond_dict['context']])

        return fn

    def __condition_error(self, call:Optional[str] = None):
        """
        Raises an error that the network is to be conditioned.
        Tries to find the call that was used.

        Parameters
        ----------
        call : str
            Optionally pass the call or more information.

        Returns
        -------
        Nothing

        """
        if not call:
            try:
                call=inspect.stack()[2][3]
            except:
                call=None
        if call is not None:
            msg = "You tried to invoke an operation without conditioning the network. Please use condition() before doing so! The call was: {}".format(call)
        else:
            msg= "You tried to invoke an operation without conditioning the network. Please use condition() before doing so!"
        logging.error(msg)
        raise RuntimeError(msg)


    def __cut_and_norm(self, cutoff:Optional[float]=None, norm:Optional[bool] = False):

        if cutoff is not None:
            cut_edges= [(u,v) for u, v,wt in self.graph.edges.data('weight') if wt<=cutoff]
            self.graph.remove_edges_from(cut_edges)
        if norm:
            # Get in-degrees
            in_deg=dict(self.graph.in_degree(weight='weight'))
            in_deg=pd.Series(list(in_deg.values()), index=list(in_deg.keys()))
            # Normalize per in degree
            for (u, v, wt) in self.graph.edges.data('weight'):
                if in_deg[v] > 0:
                    self.graph[u][v]['weight']=wt/in_deg[v]




    # %% Utility functioncs


    def update_dicts(self):
        """Simply update dictionaries"""
        # Update dictionaries
        self.token_id_dict.update(dict(zip(self.tokens, self.ids)))

    def get_token_from_id(self, id):
        """Token of id in data structures used"""
        # id should be int
        assert np.issubdtype(type(id), np.integer)
        try:
            token = self.token_id_dict[id]
        except:
            # Graph is possibly conditioned or in the process of being conditioned
            # Check Backup dict from neo database.
            try:
                token = self.neo_token_id_dict[id]
            except:
                raise LookupError("".join(["Token with ID ", str(
                    id), " missing. Token not in network or database?"]))
        return token

    def get_id_from_token(self, token):
        """Id of token in data structures used"""
        # Token has to be string
        assert isinstance(token, str)
        try:
            id = int(self.token_id_dict[token])
        except:
            # Graph is possibly conditioned or in the process of being conditioned
            # Check Backup dict from neo database.
            try:
                id = int(self.neo_token_id_dict[token])
            except:
                raise LookupError(
                    "".join(["ID of token ", token, " missing. Token not in network or database?"]))
        return id

    def ensure_ids(self, tokens):
        """This is just to confirm mixed lists of tokens and ids get converted to ids"""
        if isinstance(tokens, (list, tuple, np.ndarray)):
            # Transform strings to corresponding IDs
            tokens = [self.get_id_from_token(x) if not np.issubdtype(
                type(x), np.integer) else x for x in tokens]
            # Make sure np arrays get transformed to int lists
            return [int(x) if not isinstance(x, int) else x for x in tokens]
        else:
            if not np.issubdtype(type(tokens), np.integer):
                return self.get_id_from_token(tokens)
            else:
                return int(tokens)

    def ensure_tokens(self, ids):
        """This is just to confirm mixed lists of tokens and ids get converted to ids"""
        if isinstance(ids, list):
            return [self.get_token_from_id(x) if not isinstance(x, str) else x for x in ids]
        else:
            if not isinstance(ids, str):
                return self.get_token_from_id(ids)
            else:
                return ids

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.seed=seed


    def export_gefx(self, filename=None, path=None,  delete_isolates=True):
        if self.conditioned:
            if filename is None:
                filename=self.filename
            if path is None:
                path=filename
            else:
                path=path+"/"+filename
                path=check_create_folder(path)
            try:

                # Relabel nodes
                labeldict = dict(
                    zip(self.ids, [self.get_token_from_id(x) for x in self.ids]))
                reverse_dict = dict(
                    zip([self.get_token_from_id(x) for x in self.ids], self.ids))
                if delete_isolates:
                    cleaned_graph = self.graph.copy()
                    cleaned_graph = nx.relabel_nodes(cleaned_graph, labeldict)
                    # need to put strings on edges for gefx to write (not sure why)
                    # confirm that edge attributes are int
                    for n1, n2, d in cleaned_graph.edges(data=True):
                        for att in ['time', 'start', 'end']:
                            d[att] = int(d[att])
                    isolates = list(nx.isolates(self.graph))
                    logging.debug(
                        "Found {} isolated nodes in graph, deleting.".format(len(isolates)))
                    cleaned_graph.remove_nodes_from(isolates)
                    nx.write_gexf(cleaned_graph, path)
                else:
                    cleaned_graph = self.graph.copy()
                    cleaned_graph = nx.relabel_nodes(cleaned_graph, labeldict)
                    # confirm that edge attributes are int
                    for n1, n2, d in cleaned_graph.edges(data=True):
                        for att in ['time', 'start', 'end']:
                            d[att] = int(d[att])
                    nx.write_gexf(cleaned_graph, path)
            except:
                raise SystemError("Could not save to %s " % path)

    # %% Graph Database Aliases
    def setup_neo_db(self, tokens, token_ids):
        """
        Creates tokens and token_ids in Neo database. Does not delete existing network!
        :param tokens: list of tokens
        :param token_ids: list of corresponding token IDs
        :return: None
        """
        self.db.setup_neo_db(tokens, token_ids)
        self.init_tokens()

    def init_tokens(self):
        """
        Gets all tokens and token_ids in the database
        and sets up two-way dicts
        :return:
        """
        # Run neo query to get all nodes
        ids, tokens = self.db.init_tokens()
        # Update results
        self.ids = ids
        self.tokens = tokens
        self.update_dicts()

    def query_context(self, ids, times=None, weight_cutoff=None, occurrence=False):
        """
        Query context of ids

        Parameters
        ----------
        ids: list of ids or tokens
        times: list of times
        weight_cutoff: float in 0,1
        occurrence: bool

        Returns
        -------
        dict of context ties

        """
        if times is None:
            times=self.get_times_list()

        # Make sure we have a list of ids and times
        # Get rid of numpy here as well
        ids = self.ensure_ids(ids)
        if not isinstance(ids, (list, np.ndarray)):
            ids = [int(ids)]
        else:
            ids = [int(x) for x in ids]
        if not isinstance(times, (list, np.ndarray)):
            times = [int(times)]
        else:
            times = [int(x) for x in times]

        return self.db.query_context_of_node(ids=ids, times=times, weight_cutoff=weight_cutoff, occurrence=occurrence)


    def query_nodes(self, ids, context=None, times=None, weight_cutoff=None, norm_ties=None, query_mode="old"):
        """
        Query multiple nodes by ID and over a set of time intervals, return distinct occurrences
        If provided with context, return under the condition that elements of context are present in the context element distribution of
        this occurrence


        Parameters
        ----------
        :param ids:
            list of ids
        :param context:
            list of ids
        :param times:
            either a number format YYYY, or an interval dict {"start":YYYY,"end":YYYY}
        :param weight_cutoff:
            float in 0,1
        :param: norm_ties:
            bool
        :return: list of tuples (u,occurrences)

        """

        if norm_ties is None:
            norm_ties = self.norm_ties

        if times is not None:
            if not isinstance(times, (list, np.ndarray)):
                times = [int(times)]
            else:
                times = [int(x) for x in times]

        # Make sure we have a list of ids and times
        # Get rid of numpy here as well
        ids = self.ensure_ids(ids)
        if not isinstance(ids, (list, np.ndarray)):
            ids = [int(ids)]
        else:
            ids = [int(x) for x in ids]


        # Dispatch with or without context
        if context is not None:
            context = self.ensure_ids(context)
            if not isinstance(context, (list, np.ndarray)):
                context = [int(context)]
            else:
                context = [int(x) for x in context]
            return self.db.query_multiple_nodes(ids=ids, context=context, times=times, weight_cutoff=weight_cutoff, norm_ties=norm_ties, mode=query_mode)
        else:
            return self.db.query_multiple_nodes(ids=ids, times=times, weight_cutoff=weight_cutoff, norm_ties=norm_ties, mode=query_mode)

    def query_multiple_nodes(self, ids, times=None, weight_cutoff=None, norm_ties=None):
        """
        Old interface
        """
        return self.query_nodes(ids, times, weight_cutoff, norm_ties)

    def query_multiple_nodes_context(self, ids, context, times=None, weight_cutoff=None, norm_ties=None):
        """
        Old interface
        """
        return self.query_nodes(ids, context, times, weight_cutoff, norm_ties)

    def insert_edges_context(self, ego, ties, contexts):
        return self.db.insert_edges_context(ego, ties, contexts)

