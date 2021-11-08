import copy
import inspect
import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
from networkx import compose_all
from tqdm import tqdm

# import neo4j utilities and classes
from text2network.classes.neo4db import neo4j_database
from text2network.functions.backout_measure import backout_measure
from text2network.functions.format import pd_format
# Clustering
from text2network.functions.graph_clustering import *
from text2network.functions.network_tools import make_reverse, sparsify_graph, renorm_graph
from text2network.measures.centrality import centralities
from text2network.measures.proximity import proximities
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.input_check import input_check
from text2network.utils.twowaydict import TwoWayDict

# Type definition
# try:  # Python 3.8+
#     from typing import TypedDict
#
#
#     class GraphDict(TypedDict):
#         graph: nx.DiGraph
#         name: str
#         parent: int
#         level: int
#         measures: List
#         metadata: [Union[Dict, defaultdict]]
# except:
#     GraphDict = Dict[str, Union[str, int, Dict, List, defaultdict]]

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
                 logging_level=None, connection_type=None, consume_type=None, seed=100):
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
                                 logging_level=logging_level, connection_type=self.connection_type,
                                 consume_type=consume_type)

        # Conditioned graph information
        self.graph_type = graph_type
        self.graph = None
        self.conditioned = False
        self.is_compositional = False
        self.years = []
        self.seed = seed
        self.set_random_seed(seed)

        # Dictionaries and token/id saved in memory
        self.cond_dict = defaultdict(lambda: None)
        self.filename = ""
        self.token_id_dict = TwoWayDict()
        # Since both are numerical, we need to use a single way dict here
        self.id_index_dict = {}
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
        query = "MATCH (n) WHERE EXISTS(n.time) RETURN DISTINCT  n.time AS time ORDER BY time"
        res = self.db.receive_query(query)
        times = [x['time'] for x in res]
        return times

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
        if not isinstance(format_output, list):
            format_output = [format_output]
        if ids_to_tokens:
            for idx, pd_tbl in enumerate(format_output):

                if isinstance(pd_tbl.index, pd.core.indexes.multi.MultiIndex):
                    cvals = pd_tbl.index.values
                    cvals2 = pd_tbl.index.values
                    for i, col in enumerate(cvals):
                        col = list(col)
                        for j, c in enumerate(col):
                            try:
                                col[j] = self.ensure_tokens(c)
                            except:
                                col[j] = c
                                for k, cold in enumerate(cvals):
                                    cold = list(cold)
                                    cold_mix = list(cvals2[k])
                                    cold_mix[j] = cold[j]
                                    cvals2[k] = tuple(cold_mix)
                        cvals2[i] = tuple(col)
                    pd_tbl.index = pd.MultiIndex.from_tuples(cvals.tolist())
                else:
                    pd_tbl.index = self.ensure_tokens(list(pd_tbl.index))

                if isinstance(pd_tbl.columns, pd.core.indexes.multi.MultiIndex):
                    cvals = pd_tbl.columns.values
                    for i, col in enumerate(cvals):
                        col = list(col)
                        for j, c in enumerate(col):
                            col[j] = self.ensure_tokens(c)
                        cvals[i] = tuple(col)
                    pd_tbl.columns = pd.MultiIndex.from_tuples(cvals.tolist())
                else:
                    pd_tbl.columns = self.ensure_tokens(list(pd_tbl.columns))

                format_output[idx] = pd_tbl

        return format_output

    # %% Conditoning functions

    def context_condition(self, times: Optional[Union[int, list]] = None,
                          tokens: Optional[Union[int, str, list]] = None, weight_cutoff: Optional[float] = None,
                          depth: Optional[int] = None, max_degree: Optional[int] = None,
                          prune_min_frequency: Optional[int] = None, keep_only_tokens: Optional[bool] = False,
                          batchsize: Optional[int] = None, cond_type: Optional[str] = None, occurrence=False):
        """

        Derive context network by conditioning on tokens that are likely to occur together with the focal token(s)

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

        max_degree: int, optional
            For each node that is conditioned upon, retain only 'max_degree' largest ties. Can significantly speed up conditioning!
            This sparsifies the network for tokens that have many neighbors and is appropriate if these
            ties are highly skewed. On the other hand, for non-skewed weight distributions, it would
            likely cut significant ties without account for the amount of probability mass preserved. Use with care!

        prune_min_frequency : int, optional
            Will remove nodes entirely which occur less than  prune_min_frequency+1 times

        keep_only_tokens : bool, optional
            If True, the network will only include tokens provided in "tokens" and their ties (according to depth).

        Technical Parameters
        ----------

        batchsize: int, optional
            If given, queries of this size are sent to the database.

        cond_type: str, optional
            Manually set how ego network is queried. If not set, heuristically determined.
                "subset": queries the entire network into memory and uses network x to find the ego networks
                "search": queries the ego networks in a "beam search" fashion. Slower for high depths.

        Returns
        -------

        """

        if batchsize is None:
            batchsize = self.neo_batch_size

        input_check(tokens=tokens)
        input_check(years=times)

        if isinstance(tokens, (list, np.ndarray)):
            nr_tokens = len(tokens)
        elif tokens is not None:
            nr_tokens = 1
        else:
            nr_tokens = 0

        if keep_only_tokens and (depth is None or depth > 0):
            depth = 0
            if depth > 0:
                logging.warning(
                    "Only focal tokens are to be kept, but depth is more than zero. This is not necessary. For performance reasons, depth is reduced to 0")

        # Create Conditioning Dict to keep track of state and create filename
        cond_dict_list = [('type', "context"),
                          ('cutoff', weight_cutoff),
                          ('depth', depth), ('max_degree', max_degree)]

        self.cond_dict = self.__make_condition_dict(tokens, times, cond_dict_list)

        if tokens is None:
            logging.debug("Context Conditioning dispatch: Yearly")
            self.__yearly_context_condition(times=times, weight_cutoff=weight_cutoff, batchsize=batchsize,
                                            occurrence=occurrence, max_degree=max_degree)
        else:
            logging.debug("Context Conditioning dispatch: Ego")
            self.__context_ego_conditioning(times=times, tokens=tokens, weight_cutoff=weight_cutoff, depth=depth,
                                            batchsize=batchsize, cond_type=cond_type, occurrence=occurrence,
                                            max_degree=max_degree)
        if keep_only_tokens:
            self.__prune_by_tokens(tokens)
        if prune_min_frequency is not None:
            self.__prune_by_frequency(prune_min_frequency, times=times)

        # Set conditioning true
        self.__complete_conditioning()

    def condition(self, times: Optional[Union[int, list]] = None, tokens: Optional[Union[int, str, list]] = None,
                  weight_cutoff: Optional[float] = None, depth: Optional[int] = None,
                  context: Optional[Union[int, str, list]] = None, compositional: Optional[bool] = False,
                  reverse: Optional[bool] = False,
                  max_degree: Optional[int] = None, prune_min_frequency: Optional[int] = None,
                  keep_only_tokens: Optional[bool] = False,
                  batchsize: Optional[int] = None, cond_type: Optional[str] = None,
                  post_cutoff: Optional[float] = None, post_norm: Optional[bool] = False):
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
            Occurrence-level ties below this cutoff value are discarded (in the query)

        depth: int, optional
            The depth of the ego networks to consider.

        context: int, str or list, optional
            tokens or token_ids.
            If given, only ties that occur within a sentence including these tokens are considered

        compositional: bool, optional
            Whether to condition as compositional ties, or aggregate ties

        max_degree: int, optional
            For each node that is conditioned upon, retain only 'max_degree' largest ties. Can significantly speed up conditioning!
            This sparsifies the network for tokens that have many neighbors and is appropriate if these
            ties are highly skewed. On the other hand, for non-skewed weight distributions, it would
            likely cut significant ties without account for the amount of probability mass preserved. Use with care!

        prune_min_frequency : int, optional
            Will remove nodes entirely which occur less than  prune_min_frequency+1 times

        keep_only_tokens : bool, optional
            If True, the network will only include tokens provided in "tokens" and their ties (according to depth).

        Technical Parameters
        ----------

        batchsize: int, optional
            If given, queries of this size are sent to the database.

        cond_type: str, optional
            Manually set how the queries are sent. If not set, heuristically determined.
                "subset": queries the entire network into memory and uses network x to find the ego networks
                "search": queries the ego networks in a "beam search" fashion. Slower for high depths.

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

        if batchsize is None:
            batchsize = self.neo_batch_size

        input_check(tokens=tokens)
        input_check(tokens=context)
        input_check(years=times)
        tokens = self.ensure_ids(tokens)
        context = self.ensure_ids(context)

        if isinstance(tokens, (list, np.ndarray)):
            nr_tokens = len(tokens)
        elif tokens is not None:
            nr_tokens = 1
        else:
            nr_tokens = 0

        if keep_only_tokens and (depth is None or depth > 0):
            depth = 0
            if depth > 0:
                logging.warning(
                    "Only focal tokens are to be kept, but depth is more than zero. This is not necessary. For performance reasons, depth is reduced to 0")

        # Create Conditioning Dict to keep track of state and create filename
        cond_dict_list = [('type', "replacement"),
                          ('cutoff', weight_cutoff), ('context', context),
                          ('depth', depth), ('max_degree', max_degree)]

        self.cond_dict = self.__make_condition_dict(tokens, times, cond_dict_list)

        if tokens is None:
            logging.debug("Conditioning dispatch: Yearly")
            self.__year_condition(years=times, weight_cutoff=weight_cutoff, context=context, batchsize=batchsize,
                                  max_degree=max_degree)
        else:
            logging.debug("Conditioning dispatch: Ego")
            if depth is None:
                checkdepth = 1000
            else:
                checkdepth = depth
            if cond_type is None:
                if checkdepth <= 2 and nr_tokens <= 5:
                    cond_type = "search"
                elif checkdepth == 0:  # just need proximities
                    cond_type = "search"
                else:
                    cond_type = "subset"
            if cond_type == "subset":
                logging.debug("Conditioning dispatch: Ego, subset, depth {}".format(depth))
                self.__ego_condition_subset(years=times, token_ids=tokens, weight_cutoff=weight_cutoff, depth=depth,
                                            context=context, batchsize=batchsize,
                                            max_degree=max_degree)
            elif cond_type == "search":
                logging.debug("Conditioning dispatch: Ego, search, depth {}".format(depth))
                self.__ego_condition_search(years=times, token_ids=tokens, weight_cutoff=weight_cutoff, depth=depth,
                                            context=context, batchsize=batchsize,
                                            max_degree=max_degree)
            else:
                msg = "Conditioning type {} requested. Please use either search or subset.".format(cond_type)
                logging.debug(msg)
                raise NotImplementedError(msg)
        if keep_only_tokens:
            self.__prune_by_tokens(tokens)
        self.__cut_and_norm(post_cutoff, post_norm)
        self.__prune_by_frequency(prune_min_frequency, times=times, context=context)

        # Set conditioning true
        self.__complete_conditioning()

        if compositional:
            self.to_compositional(times=times, context=context)

        if reverse:
            self.to_reverse()

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
            self.cond_dict = defaultdict(lambda: None)
            self.filename = ""

    # %% Clustering
    def cluster(self, levels: int = 1, name: Optional[str] = "base", interest_list: Optional[list] = None,
                metadata: Optional[Union[dict, defaultdict]] = None,
                algorithm: Optional[Callable[[nx.DiGraph], List]] = None,
                to_measure: Optional[List[Callable[[nx.DiGraph], Dict]]] = None,
                add_ego_tokens: Optional[Union[str, int, list]] = None):
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
                nx.set_node_attributes(self.graph, cluster_dict, 'clusterl' + str(t))
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

        return cluster_list

    # %% Measures

    def centralities(self, focal_tokens=None, types=None) -> Dict:
        """
        See measures.centralities
        """
        if types is None:
            types = ["PageRank", "normedPageRank"]
        return centralities(self, focal_tokens=focal_tokens, types=types)

    def proximities(self, focal_tokens: Optional[List] = None, alter_subset: Optional[List] = None) -> Dict:
        """
        See measures.proximities
        """
        return proximities(self, focal_tokens=focal_tokens, alter_subset=alter_subset)

    def get_node_context(self, tokens: Union[list, int, str], years: Union[int, list, dict] = None,
                         weight_cutoff: Optional[float] = None, occurrence: Optional[bool] = False):

        if years is None:
            years = self.get_times_list()

        input_check(tokens=tokens)
        input_check(years=years)

        if isinstance(tokens, (str, int)):
            tokens = [tokens]

        tokens = self.ensure_ids(tokens)

        res = self.db.query_context_of_node(tokens, times=years, weight_cutoff=weight_cutoff, occurrence=occurrence)

        tokens = np.array([x[0] for x in res])
        alters = np.array([x[1] for x in res])
        weights = np.array([x[2]['weight'] for x in res])

        pd_dict = {}
        for token in np.unique(tokens):
            mask = tokens == token
            cidx = list(alters[mask])
            cweights = list(weights[mask])
            contexts = dict(zip(cidx, cweights))
            pd_dict.update({int(token): contexts})

        return {'proximity': pd_dict}

    def get_dyad_context(self, dyads: Optional[Union[list, tuple]] = None,
                         occurrence: Optional[Union[list, str, int]] = None,
                         replacement: Optional[Union[list, str, int]] = None, years: Union[int, list, dict] = None,
                         weight_cutoff: Optional[float] = None, part_of_speech: Optional[str] = None,
                         context_mode: Optional[str] = "bidirectional", return_sentiment: Optional[bool] = True):
        """
        Specify a dyad and get the distribution of contextual words.
        dyads are tuples of (occurrence,replacement), that is, in the graph, ties that go replacement->occurrence.
        You can pass a list of tuples.
        You can also pass occurrences and replacement tokens as lists or individual tokens


        Parameters
        ----------
        dyads
        occurrence
        replacement
        years
        weight_cutoff
        part_of_speech
        context_mode
        return_sentiment


        Returns
        -------

        """

        # Untangle dyad lists
        if dyads is not None:
            if isinstance(dyads, list):
                occurrence = [x[0] for x in dyads]
                replacement = [x[1] for x in dyads]
            else:
                occurrence = dyads[0]
                replacement = dyads[1]
        elif replacement is None or occurrence is None:
            msg = "Please provide either dyads as list of tuples (occurrence, replacement) or individual lists of occurrences and replacements!"
            logging.error(msg)
            raise AttributeError(msg)

        if years is None:
            years = self.get_times_list()

        input_check(tokens=occurrence)
        input_check(tokens=replacement)
        input_check(years=years)

        if isinstance(occurrence, (str, int)):
            occurrence = [occurrence]
        if isinstance(replacement, (str, int)):
            replacement = [replacement]

        occurrence = np.array(self.ensure_ids(occurrence))
        replacement = np.array(self.ensure_ids(replacement))


        pd_dict = self.db.query_tie_context(occurring=occurrence, replacing=replacement, times=years,
                                                   weight_cutoff=weight_cutoff, pos=part_of_speech, return_sentiment=return_sentiment, context_mode=context_mode)

        return {'dyad_context': pd_dict}

    # %% Graph manipulation

    def add_frequencies(self, times: Union[list, int], context: Union[list, str] = None):
        """
        This function adds a field "freq" to the nodes in the conditioned graph, giving the number of occurrences
        of a token.
        """
        if not self.conditioned:
            self.__condition_error(call=inspect.stack()[1][3])

        input_check(years=times)

        self.__add_frequencies(times=times, context=context)

    def to_compositional(self, times: Union[list, int] = None, context: Union[list, str] = None):
        """

        Under the assumption that the network was conditioned in aggregate mode with corresponding
        times and contexts, this function normalizes the ties such that

        A-(substitutes for)-> B
        is normalized by
        P(s|B in s, Context)

        Leading to a measure of
        P(w=A | B, Context)

        Parameters
        ----------
        times
        context

        """

        if not self.conditioned:
            self.__condition_error(call=inspect.stack()[1][3])

        # TODO: Add Compositional Here
        input_check(years=times)
        input_check(tokens=context)

        # Get nodes in network
        node_list = list(self.graph.nodes)
        ids = self.ensure_ids(node_list)

        # Do frequencies exist?
        if self.graph.nodes[node_list[-1]].get('freq') is None:
            self.add_frequencies(times=times, context=context)

        # Normalize per in degree
        for (u, v, wt) in self.graph.edges.data('weight'):
            if self.graph.nodes[v]['freq'] > 0:
                self.graph[u][v]['weight'] = wt / self.graph.nodes[v]['freq']
            else:
                self.graph[u][v]['weight'] = 0

        self.is_compositional = True

        if self.cond_dict['compositional']:
            # Graph was already reversed - update state
            logging.warning(
                "You have invoked compositional mode more than once. This means ties are normalized by frequency^k!")
            pass
        else:
            self.cond_dict['backout'] = True

    def norm_by_total_nr_sequences(self, times: Union[list, int]):
        """

        If comparing values across different times, aggregate replacement ties
        should be normed by the number of sequences.

        This function norms all ties by the number of sequences/1000

        Parameters
        ----------
        time: list of ints or int
            time parameter which to norm by

        Returns
        -------

        """

        if not self.conditioned:
            self.__condition_error(call=inspect.stack()[1][3])

        if not isinstance(times, list):
            if isinstance(times, int):
                times = [times]
            else:
                raise AttributeError("Please provide a list of ints, or an int as time variable")

        query = "MATCH(r: edge) WHERE r.time in " + str(times) + " RETURN count(DISTINCT r.run_index) as nrs"
        try:
            res = self.db.receive_query(query)
        except:
            logging.error("Could not retrieve number of sequences")
            raise
        nrs = [x['nrs'] for x in res][0] / 1000
        logging.info("Normalizing graph by dividing by {} sequences".format(nrs))
        self.graph = renorm_graph(self.graph, nrs)

    def norm_by_total_nr_occurrences(self, times: Union[list, int]):
        """

        If comparing values across different times, aggregate replacement ties
        should be normed.

        This function norms all ties by the number of occurrences/1000

        Parameters
        ----------
        time: list of ints or int
            time parameter which to norm by

        Returns
        -------

        """

        if not self.conditioned:
            self.__condition_error(call=inspect.stack()[1][3])

        if not isinstance(times, list):
            if isinstance(times, int):
                times = [times]
            else:
                raise AttributeError("Please provide a list of ints, or an int as time variable")

        query = "MATCH(r: edge) WHERE r.time in " + str(times) + " RETURN round(sum(r.weight)) as nrs"

        try:
            res = self.db.receive_query(query)
        except:
            logging.error("Could not retrieve number of occurrences")
            raise
        nrs = [x['nrs'] for x in res][0] / 1000
        logging.info("Normalizing graph by dividing by {} sequences".format(nrs))
        self.graph = renorm_graph(self.graph, nrs)

    def sparsify(self, percentage: int = 100):
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

        self.graph = sparsify_graph(self.graph, percentage)

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

        if self.cond_dict['backout']:
            # Graph was already reversed - update state
            self.cond_dict['backout'] = False
        else:
            self.cond_dict['backout'] = True

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

        if self.cond_dict['reverse']:
            # Graph was already reversed - update state
            self.cond_dict['reverse'] = False
        else:
            self.cond_dict['reverse'] = True

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

            sum: sum

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

        if self.cond_dict['symmetric']:
            # Graph was already reversed - update state
            self.cond_dict['symmetric'] = False
        else:
            self.cond_dict['symmetric'] = True

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
            if len(
                    i) != 2:
                raise AssertionError(
                    "Please format a call as (<tokens>,<time>) or (<tokens>,{'start:'<time>, 'end':<time>})")
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
            return self.query_nodes(i, times=year)
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

    def __year_condition(self, years, weight_cutoff=None, context=None, batchsize=None,
                         max_degree: Optional[int] = None):
        """ Condition the entire network over all years """

        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        if not self.conditioned:  # This is the first conditioning
            # Build graph
            self.graph = self.create_empty_graph()

            # TODO Make sure ids are okay here
            # All tokens
            worklist = self.ids
            # Add all tokens to graph
            self.graph.add_nodes_from(worklist)

            # Loop batched over all tokens to condition
            for i in tqdm(range(0, len(worklist), batchsize), leave=False, position=0):
                token_ids = worklist[i:i + batchsize]
                logging.debug(
                    "Conditioning by query batch {} of {} tokens.".format(i, len(token_ids)))
                # Query Neo4j
                self.__add_edges(
                    self.query_nodes(token_ids, context=context, times=years, weight_cutoff=weight_cutoff), max_degree=max_degree)
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


        else:  # Remove conditioning and recondition
            self.decondition()
            self.__year_condition(years=years, weight_cutoff=weight_cutoff, context=context,
                                  max_degree=max_degree)

    def __ego_condition_subset(self, years, token_ids, weight_cutoff=None, depth=None, context=None,
                               batchsize=None, max_degree: Optional[int] = None):

        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        # First, do a year conditioning
        logging.debug("Full year conditioning before ego subsetting.")
        self.__year_condition(years=years, weight_cutoff=weight_cutoff, context=context, batchsize=batchsize,
                              max_degree=max_degree)

        if depth is not None:
            if depth > 0:
                # Create ego graph for each node
                graph_list = []
                if isinstance(token_ids, (list, np.ndarray)):
                    for focal_token in token_ids:
                        temp_graph = nx.generators.ego.ego_graph(self.graph, self.ensure_ids(focal_token), radius=depth,
                                                                 center=True,
                                                                 undirected=False)
                        graph_list.append(temp_graph)
                    # Compose ego graphs
                    self.graph = compose_all(graph_list)
                else:
                    self.graph = nx.generators.ego.ego_graph(self.graph, self.ensure_ids(token_ids), radius=depth,
                                                             center=True, undirected=False)

    def __ego_condition_search(self, years, token_ids, weight_cutoff=None, depth=None, context=None,
                               batchsize=None, max_degree: Optional[int] = None):

        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        if not self.conditioned:  # This is the first conditioning
            # Build graph
            self.graph = self.create_empty_graph()
            self.db.open_session()
            if depth is None:
                logging.debug("Depth is None, but search conditioning is requested. Setting depth to 1.")
                depth = 1
            # Check one level deeper
            or_depth = depth
            depth += 1
            # Create a dict to hold previously queried ids
            prev_queried_ids = []
            # ids to check
            ids_to_check = token_ids
            logging.debug(
                "Start of Depth {} conditioning".format(or_depth))
            while depth > 0:
                if not isinstance(ids_to_check, (list, np.ndarray)):
                    ids_to_check = [ids_to_check]

                # Work from ID list, give error if tokens are not in database
                ids_to_check = self.ensure_ids(ids_to_check)
                # Do not consider already added tokens
                ids_to_check = np.setdiff1d(ids_to_check, prev_queried_ids)
                logging.debug(
                    "Depth {} conditioning: {} new found tokens, where {} already added.".format(depth,
                                                                                                 len(ids_to_check),
                                                                                                 len(prev_queried_ids)))
                # Add ids_to_check to list since they will be queried this iteration
                prev_queried_ids.extend(ids_to_check)
                if isinstance(ids_to_check, (np.ndarray)):
                    ids_to_check = list(ids_to_check)
                ids_to_check.reverse()
                # Add starting nodes
                self.graph.add_nodes_from(ids_to_check)
                for i in tqdm(range(0, len(ids_to_check), batchsize), leave=False, position=0):

                    id_batch = ids_to_check[i:i + batchsize]

                    logging.debug(
                        "Conditioning by query batch {} of {} tokens.".format(i, len(ids_to_check)))
                    # Query Neo4j
                    try:
                        self.__add_edges(
                            self.query_nodes(id_batch, context=context, times=years, weight_cutoff=weight_cutoff), max_degree=max_degree)
                    except:
                        logging.error("Could not condition graph by query method.")
                        raise

                # Delete disconnected nodes
                remove = [node for node, degree in dict(self.graph.degree).items() if degree <= 0]
                remove = list(np.setdiff1d(remove, token_ids))  # Do not remove focal tokens
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
                if isinstance(token_ids, (list, np.ndarray)):
                    self.graph = compose_all([nx.generators.ego.ego_graph(self.graph, x, radius=or_depth, center=True,
                                                                          undirected=False) for x in
                                              self.ensure_ids(token_ids) if
                                              x in self.graph.nodes])
                else:
                    self.graph = nx.generators.ego.ego_graph(self.graph, self.ensure_ids(token_ids), radius=or_depth,
                                                             center=True,
                                                             undirected=False)

        else:  # Remove conditioning and recondition
            self.decondition()
            self.condition(times=years, tokens=token_ids, weight_cutoff=weight_cutoff,
                           depth=depth, context=context, max_degree=max_degree)

        # Continue conditioning

    def __yearly_context_condition(self, times, weight_cutoff=None, batchsize=None, occurrence=False,
                                   max_degree: Optional[int] = None):
        """ Condition the entire network over all years """

        # Same for batchsize
        if batchsize is None:
            batchsize = self.neo_batch_size

        if not self.conditioned:  # This is the first conditioning

            logging.info("Called into yearly conditioning, batch size: {}, cutoff: {}, occurrence: {}".format(batchsize,
                                                                                                              weight_cutoff,
                                                                                                              occurrence))
            # Build graph
            self.graph = self.create_empty_graph()

            # All tokens
            worklist = self.ids
            # Add all tokens to graph
            self.graph.add_nodes_from(worklist)

            # Loop batched over all tokens to condition
            for i in tqdm(range(0, len(worklist), batchsize), leave=False, position=0):
                token_ids = worklist[i:i + batchsize]
                logging.debug(
                    "Conditioning by query batch {} of {} tokens.".format(i, len(token_ids)))
                # Query Neo4j
                self.__add_edges(
                    self.query_context(token_ids, times=times, weight_cutoff=weight_cutoff, occurrence=occurrence),
                    max_degree=max_degree)
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


        else:  # Remove conditioning and recondition
            self.decondition()
            self.__yearly_context_condition(times, weight_cutoff, batchsize, occurrence)

    def __context_ego_conditioning(self, times: Optional[Union[int, list]] = None,
                                   tokens: Optional[Union[int, str, list]] = None,
                                   weight_cutoff: Optional[float] = None, depth: Optional[int] = None,
                                   batchsize: Optional[int] = None, cond_type: Optional[str] = None,
                                   occurrence: Optional[bool] = False, max_degree: Optional[int] = None):

        if not self.conditioned:  # This is the first conditioning
            # Save original depth variable
            or_depth = depth
            if not isinstance(times, (list, np.ndarray)) and times is not None:
                times = [times]
            if cond_type is None:
                if depth is not None:
                    if (depth <= 2 and len(tokens) <= 5):
                        cond_type = "search"
                else:
                    cond_type = "subset"
            logging.info("Context clustering mode: {} and batch size: {}".format(cond_type, batchsize))
            if cond_type == "search":
                # Build graph
                self.graph = self.create_empty_graph()

                # Depth 0 and Depth 1 really mean the same thing here
                if depth is None:
                    depth = 1
                # Check one level deeper

                depth += 1
                # Create a dict to hold previously queried ids
                prev_queried_ids = []
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
                            self.__add_edges(self.query_context(ids=id_batch, times=times, weight_cutoff=weight_cutoff,
                                                                occurrence=occurrence), max_degree=max_degree)
                        except:
                            logging.error("Could not context condition graph by query search method.")

                    # Delete disconnected nodes
                    remove = [node for node, degree in dict(self.graph.degree).items() if degree <= 0]
                    remove = list(np.setdiff1d(remove, tokens))  # Do not remove focal tokens
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
            elif cond_type == "subset":
                self.__yearly_context_condition(times=times, weight_cutoff=weight_cutoff, batchsize=batchsize,
                                                occurrence=occurrence)
            else:
                msg = "Conditioning type {} requested. Please use either search or subset.".format(cond_type)
                logging.debug(msg)
                raise NotImplementedError(msg)

            # Create ego graph for each node and compose
            if or_depth is not None:
                if or_depth > 0:
                    # Create ego graph for each node
                    graph_list = []
                    if isinstance(tokens, (list, np.ndarray)):
                        for focal_token in tokens:
                            temp_graph = nx.generators.ego.ego_graph(self.graph, self.ensure_ids(focal_token),
                                                                     radius=or_depth,
                                                                     center=True,
                                                                     undirected=False)
                            graph_list.append(temp_graph)
                        # Compose ego graphs
                        self.graph = compose_all(graph_list)
                    else:
                        self.graph = nx.generators.ego.ego_graph(self.graph, self.ensure_ids(tokens), radius=or_depth,
                                                                 center=True, undirected=False)


        else:  # Remove conditioning and recondition
            self.decondition()
            self.__context_ego_conditioning(times=times, tokens=tokens, weight_cutoff=weight_cutoff, depth=depth,
                                            batchsize=batchsize, cond_type=cond_type)

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
        self.conditioned = True
        self.filename = self.__create_filename(self.cond_dict)

    # %% Graph abstractions - for now only networkx

    @staticmethod
    def create_empty_graph() -> nx.DiGraph:
        return nx.DiGraph()

    def delete_graph(self):
        self.graph = None

    # %% Internal functions

    def __make_condition_dict(self, tokens: Union[str, int, list], times: Union[int, list], tuple_list: list) -> dict:

        # Set up dict of configuration of the conditioning
        cond_dict = defaultdict(lambda: False)

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

        # Add non "None" or False tuples
        for key, item in tuple_list:
            if item not in (None, False):
                cond_dict.update({key: item})

        return cond_dict

    @staticmethod
    def __create_filename(cond_dict):

        if cond_dict['type'] is not False:
            fn = str(cond_dict['type']) + "-"
        else:
            fn = ""

        if cond_dict['tokens'] is not None:
            if not isinstance(cond_dict['tokens'], list):
                cond_dict['tokens'] = [cond_dict['tokens']]
            fn = fn + "EGO-"
            fn = fn + '-'.join([str(x) for x in cond_dict['tokens']]) + '-'

        if cond_dict['years'] != "ALL":
            if not isinstance(cond_dict['years'], list):
                cond_dict['years'] = [cond_dict['years']]
            fn = fn + 'Y-' + '-'.join([str(x) for x in cond_dict['years']])
        else:
            fn = fn + 'Y-' + "ALL"

        for key in cond_dict.keys():
            if cond_dict[key] is not None and cond_dict[key] is not False:
                if key not in ["type", "tokens", "context", "years"]:
                    fn = fn + '-' + str(key) + str(cond_dict[key])

        if cond_dict['context'] is not False:
            if not isinstance(cond_dict['context'], list):
                cond_dict['context'] = [cond_dict['context']]
            fn = "context-"
            fn = fn + '-'.join([str(x) for x in cond_dict['context']])

        return fn

    @staticmethod
    def __condition_error(call: Optional[str] = None):
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
                call = inspect.stack()[2][3]
            except:
                call = None
        if call is not None:
            msg = "You tried to invoke an operation without conditioning the network. Please use condition() before doing so! The call was: {}".format(
                call)
        else:
            msg = "You tried to invoke an operation without conditioning the network. Please use condition() before doing so!"
        logging.error(msg)
        raise RuntimeError(msg)

    def __prune_by_tokens(self, tokens):
        if not isinstance(tokens, list):
            tokens = [tokens]

        prune_list = [x for x in self.graph.nodes if x not in tokens]
        logging.info("Found {} nodes not in focal list, pruning".format(len(prune_list)))
        self.graph.remove_nodes_from(prune_list)

    def __prune_by_frequency(self, prune_min_frequency=None, times=None, context=None):

        if prune_min_frequency is not None:
            self.__add_frequencies(times=times, context=context)

            # Prune frequencies
            prune_list = [x for x in self.graph.nodes if self.graph.nodes[x]['freq'] < prune_min_frequency + 1]
            logging.info(
                "'Graph has {} nodes. Found {} nodes with frequency {} or less, pruning".format(len(self.graph.nodes),
                                                                                                len(prune_list),
                                                                                                prune_min_frequency))
            self.graph.remove_nodes_from(prune_list)

    def __add_frequencies(self, times=None, context=None):

        # Get nodes in network
        node_list = list(self.graph.nodes)
        ids = self.ensure_ids(node_list)

        # Set default frequency
        nx.set_node_attributes(self.graph, values=0, name="freq")

        # Just in case, translate between node labels in graph and ids
        node_id_dict = {x[0]: x[1] for x in zip(node_list, ids)}
        # Attribute dict
        ret = {node_id_dict[x[0]]: {'freq': x[1]} for x in
               self.db.query_occurrences(ids=ids, times=times, context=context)}
        nx.set_node_attributes(self.graph, ret)

    def __cut_and_norm(self, cutoff: Optional[float] = None, norm: Optional[bool] = False):

        if cutoff is not None:
            cut_edges = [(u, v) for u, v, wt in self.graph.edges.data('weight') if wt <= cutoff]
            self.graph.remove_edges_from(cut_edges)
        if norm:
            # Get in-degrees
            in_deg = dict(self.graph.in_degree(weight='weight'))
            in_deg = pd.Series(list(in_deg.values()), index=list(in_deg.keys()))
            # Normalize per in degree
            for (u, v, wt) in self.graph.edges.data('weight'):
                if in_deg[v] > 0:
                    self.graph[u][v]['weight'] = wt / in_deg[v]

    def __add_edges(self, edges: dict, max_degree: Optional[int] = None):

        if max_degree is not None:
            # Create dataframe which is easier to group and sort
            edge_df = pd.DataFrame(
                [{'ego': int(edge[0]), 'alter': int(edge[1]), 'weight': edge[2]['weight'], 'dicto': edge[2]} for edge in
                 edges])
            edge_df = edge_df.set_index(['ego', 'alter'])
            sel_df = edge_df.loc[
                edge_df.groupby('ego')['weight'].nlargest(max_degree).index.droplevel(0).to_list()].reset_index()
            edges = [(row['ego'], row['alter'], row['dicto']) for index, row in sel_df.iterrows()]
        try:
            self.graph.add_edges_from(edges)
        except:
            logging.error("Could not add edges from query.")
            logging.error("edges: {}".format(edges))
            raise

    # %% Utility functioncs

    def update_dicts(self):
        """Simply update dictionaries"""
        # Update dictionaries
        self.token_id_dict.update(dict(zip(self.tokens, self.ids)))

    def get_token_from_id(self, id):
        """Token of id in data structures used"""
        # id should be int
        if not np.issubdtype(type(id), np.integer):
            raise AssertionError
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
        if not isinstance(token, str):
            raise AssertionError
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
        if tokens is not None:
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
        else:
            return tokens

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
        self.seed = seed

    def export_gefx(self, filename=None, path=None, delete_isolates=True):
        if self.conditioned:
            if filename is None:
                filename = self.filename + ".gexf"
            if path is None:
                path = filename
            else:
                path = path + "/" + filename
                path = check_create_folder(path)
            try:

                # Relabel nodes
                labeldict = dict(
                    zip(self.ids, [self.get_token_from_id(x) for x in self.ids]))
                reverse_dict = dict(
                    zip([self.get_token_from_id(x) for x in self.ids], self.ids))

                cleaned_graph = self.graph.copy()
                cleaned_graph = nx.relabel_nodes(cleaned_graph, labeldict)
                # need to put strings on edges for gefx to write (not sure why)
                # confirm that edge attributes are int
                for n1, n2, d in cleaned_graph.edges(data=True):
                    for att in ['time', 'start', 'end']:
                        d[att] = int(d[att])
                for n, d in cleaned_graph.nodes(data=True):
                    for att in ['freq']:
                        if att in d:
                            d[att] = int(d[att])
                if delete_isolates:
                    isolates = list(nx.isolates(self.graph))
                    logging.debug(
                        "Found {} isolated nodes in graph, deleting.".format(len(isolates)))
                    cleaned_graph.remove_nodes_from(isolates)
                nx.write_gexf(cleaned_graph, path)

            except:
                raise SystemError("Could not save to %s " % path)

    def export_edgelist(self, filename=None, path=None, delete_isolates=True):
        if self.conditioned:
            if filename is None:
                filename = self.filename + ".csv"
            if path is None:
                path = filename
            else:
                path = path + "/" + filename
                path = check_create_folder(path)
            try:

                # Relabel nodes
                labeldict = dict(
                    zip(self.ids, [self.get_token_from_id(x) for x in self.ids]))
                reverse_dict = dict(
                    zip([self.get_token_from_id(x) for x in self.ids], self.ids))

                cleaned_graph = self.graph.copy()
                cleaned_graph = nx.relabel_nodes(cleaned_graph, labeldict)
                # need to put strings on edges for gefx to write (not sure why)
                # confirm that edge attributes are int
                for n1, n2, d in cleaned_graph.edges(data=True):
                    for att in ['time', 'start', 'end']:
                        d[att] = int(d[att])
                for n, d in cleaned_graph.nodes(data=True):
                    for att in ['freq']:
                        if att in d:
                            d[att] = int(d[att])
                if delete_isolates:
                    isolates = list(nx.isolates(self.graph))
                    logging.debug(
                        "Found {} isolated nodes in graph, deleting.".format(len(isolates)))
                    cleaned_graph.remove_nodes_from(isolates)
                nx.write_edgelist(cleaned_graph, path, delimiter=",", data=True)
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
            times = self.get_times_list()

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

    def query_nodes(self, ids, context=None, times=None, weight_cutoff=None, pos=None, return_sentiment=True,
                             context_mode="bidirectional", context_weight=True):
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
        :return: list of tuples (u,occurrences)

        """

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
            return self.db.query_multiple_nodes(ids=ids, context=context, times=times, weight_cutoff=weight_cutoff,
                                                pos=pos, return_sentiment=return_sentiment,
                                                context_mode=context_mode, context_weight=context_weight)
        else:
            return self.db.query_multiple_nodes(ids=ids, times=times, weight_cutoff=weight_cutoff,
                                                pos=pos, return_sentiment=return_sentiment,
                                                context_mode=context_mode, context_weight=context_weight)

    def query_multiple_nodes(self, ids, times=None, weight_cutoff=None, ):
        """
        Old interface
        """
        return self.query_nodes(ids, times, weight_cutoff)

    def query_multiple_nodes_context(self, ids, context, times=None, weight_cutoff=None):
        """
        Old interface
        """
        return self.query_nodes(ids, context, times, weight_cutoff)
