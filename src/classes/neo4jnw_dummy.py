# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:26:32 2020

@author: marquart
"""
import copy
import logging
from collections.abc import MutableSequence
from src.utils.twowaydict import TwoWayDict
import numpy as np


try:
    import networkx as nx
except:
    nx = None

try:
    import igraph as ig
except:
    ig = None


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
        
        self.starttime=starttime
        self.endtime=endtime
        
        self.graph = self.create_empty_graph()
        edgelist = [("chancellor", "president",  {'weight': 0.25, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("king", "president",  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", "president",  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}), 
            ("ceo", "president",  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}), 
            ("father", "president",  {'weight': 0.25, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})]

        x="chancellor"
        edgelist.extend([("president", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("king", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("father", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="king"
        edgelist.extend([("president", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.6, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("chancellor", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("father", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="tyrant"
        edgelist.extend([("president", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("king", x,  {'weight': 0.6, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("chancellor", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("ceo", x,  {'weight': 0.15, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("father", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="ceo"
        edgelist.extend([("president", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("king", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("chancellor", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("father", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="father"
        edgelist.extend([("president", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("king", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("chancellor", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("ceo", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        # verb-noun

        x="judge"
        edgelist.extend([("king", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.25, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("father", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="delegate"
        edgelist.extend([("ceo", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("president", x,  {'weight': 0.5, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("chancellor", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="manage"
        edgelist.extend([("ceo", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("president", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="teach"
        edgelist.extend([("father", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("president", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="rule"
        edgelist.extend([("king", x,  {'weight': 0.6, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        # verb-verb

        x="rule"
        edgelist.extend([("judge", x,  {'weight': 0.8, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("manage", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="teach"
        edgelist.extend([("manage", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("delegate", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])
        
        x="delegate"
        edgelist.extend([("manage", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("teach", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])
 
        x="manage"
        edgelist.extend([("delegate", x,  {'weight': 0.7, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("rule", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="judge"
        edgelist.extend([("rule", x,  {'weight': 0.45, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("manage", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])
        
        # man women
        x="man"
        edgelist.extend([("president", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("king", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("chancellor", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("ceo", x,  {'weight': 0.05, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="woman"
        edgelist.extend([("president", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("tyrant", x,  {'weight': 0.1, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("chancellor", x,  {'weight': 0.4, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1}),
            ("ceo", x,  {'weight': 0.2, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])

        x="woman"
        edgelist.extend([("man", x,  {'weight': 0.6, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])
        x="man"
        edgelist.extend([("woman", x,  {'weight': 0.3, 'time': starttime, 'start': starttime, 'end': endtime, 'occurrences': 1})])
        
        self.graph.add_edges_from(edgelist)
        
        self.update()
        
        # Init parent class
        super().__init__()

    def update(self):
        self.ids=list(self.graph.nodes)
        self.tokens=list(self.graph.nodes)

    def condition(self, years=None, tokens=None, weight_cutoff=None, depth=None, context=None, norm=False,
                  batchsize=10000):
        self.conditioned=True

    def decondition(self):
        return True

    def ensure_ids(self, tokens):
        return tokens



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
            weights = [{'weight': x[2], 'time': self.starttime, 'start': self.starttime, 'end': self.endtime, 'occurrences': 1} for x in value]
            token = map(str, np.repeat(key, len(neighbors)))
        except:
            raise ValueError("Adding requires an iterable over tuples e.g. [(neighbor,time, weight))]")

        # Check if all neighbor tokens present
        assert set(neighbors) < set(self.ids), "ID of node to connect not found. Not in network?"
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
        assert isinstance(token, str), "Please add <token>,<token_id> as str,int"
        self.graph.add_node(token)
        self.update()