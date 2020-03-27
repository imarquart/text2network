from collections.abc import MutableSequence
from torch.utils.data import Dataset
import logging
from NLP.utils.twowaydict import TwoWayDict
import numpy as np
import neo4j
import copy

try:
    import networkx as nx
except:
    nx = None

try:
    import igraph as ig
except:
    ig = None


class neo4j_network(MutableSequence):

    # %% Initialization functions
    def __init__(self, neo4j_creds, batch_size=1000000,graph_type= "networkx", graph_direction="FORWARD", write_before_query=True):
        self.neo4j_connection, self.neo4j_credentials = neo4j_creds
        self.write_before_query = write_before_query
        # Conditioned graph information
        self.graph_type = graph_type
        self.graph = None
        self.conditioned = None
        self.years = []
        self.graph_direction = graph_direction
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

        # Neo4J Internals
        self.neo_queue = []
        self.neo_batch_size = batch_size
        self.connector = neo4j.Connector(self.neo4j_connection, self.neo4j_credentials)
        # Init parent class
        super().__init__()

    # %% Python Interface implementations
    def __delitem__(self, key):
        """
        Deletes a node and all its ties
        :param key: Token or Token id
        :return:
        """
        if isinstance(key, str): key = self.get_id_from_token(key)
        assert key in self.ids, "ID of ego token to connect not found. Not in network?"
        self.remove_token(key)

    def remove(self, key):
        """
        Deletes a node and all its ties
        :param key: Token or Token id
        :return:
        """
        if isinstance(key, str): key = self.get_id_from_token(key)
        assert key in self.ids, "ID of ego token to connect not found. Not in network?"
        self.remove_token(key)

    def __setitem__(self, key, value):
        """
        Set links of node
        :param key: id or token of node.
        :param value: To add links to node: list of tuples [(neighbor,time, weight))] or [(neighbor,time, {'weight':weight}))]. To add node itself, token_id (int)
        :return:
        """
        if isinstance(key, str) and isinstance(value, int):  # Wants to add a node
            self.add_token(value, key)
        else:
            if isinstance(key, str): key = self.get_id_from_token(key)
            assert key in self.ids, "ID of ego token to connect not found. Not in network?"
            try:
                neighbors = [self.get_id_from_token(x[0]) if not isinstance(x[0], int) else x[0] for x in value]
                weights = [{'weight': x[2]} if isinstance(x[2], (int, float)) else x[2] for x in value]
                years = [x[1] for x in value]
                token = map(int, np.repeat(key, len(neighbors)))
            except:
                raise ValueError("Adding requires an iterable over tuples e.g. [(neighbor,time, weight))]")

            # Check if all neighbor tokens present
            assert set(neighbors) < set(self.ids), "ID of node to connect not found. Not in network?"
            ties = zip(token, neighbors, years, weights)

            # TODO Dispatch if graph conditioned

            # Add ties to query
            self.insert_edges_multiple(ties)

    def __getitem__(self, i):
        """
        Retrieve node information with input checking
        :param i: int or list or nodes, or tuple of nodes with timestamp. Format as int YYYYMMDD, or dict with {'start:'<YYYYMMDD>, 'end':<YYYYMMDD>.
        :return: NetworkX compatible node format
        """
        # If so desired, induce a queue write before any query
        if self.write_before_query == True:
            self.write_queue()
        # Are time formats submitted? Handle those and check inputs
        if isinstance(i, tuple):
            assert len(
                i) == 2, "Please format a call as <token>,<YYYYMMDD> or <token>,{'start:'<YYYYMMDD>, 'end':<YYYYMMDD>"
            if not isinstance(i[1], dict):
                assert len(str(i[1])) == 8 and isinstance(i[1],
                                                          int), "Please timestamp as <YYYYMMDD>, or{'start:'<YYYYMMDD>, 'end':<YYYYMMDD>"
            year = i[1]
            i = i[0]
        else:
            year = None
        if isinstance(i, (list, tuple, range)):
            i = [self.get_id_from_token(x) if not isinstance(x, int) else x for x in i]
        elif isinstance(i, str):
            i = [self.get_id_from_token(i)]
        else:
            assert isinstance(i, int), "Please format a call as <token> or <token_id>"
            i = [i]

        # TODO Dispatch in case of graph
        if self.graph is None:
            return self.query_multiple_nodes(i, year)
        else:
            pass

    def __len__(self):
        return len(self.tokens)

    def insert(self, token, token_id):
        """
        Insert a new token
        :param token: Token string
        :param token_id: Token ID
        :return: None
        """
        assert isinstance(token, str) and isinstance(token_id, int), "Please add <token>,<token_id> as str,int"
        self.add_token(token_id, token)

    # %% Setup
    def setup_neo_db(self, tokens, token_ids):
        """
        Creates tokens and token_ids in Neo database. Does not delete existing network!
        :param tokens: list of tokens
        :param token_ids: list of corresponding token IDs
        :return: None
        """
        logging.info("Creating indecies and nodes in Neo4j database.")
        # Create uniqueness constraints
        query = "CREATE CONSTRAINT ON(n:word) ASSERT n.token_id IS UNIQUE"
        self.add_query(query)
        query = "CREATE CONSTRAINT ON(n:word) ASSERT n.token IS UNIQUE"
        self.add_query(query)
        query = "CREATE INDEX ON :edge(time)"
        self.add_query(query)

        # Need to write first because create and structure changes can not be batched
        self.write_queue()
        # Create nodes in neo db
        queries = [''.join(["MERGE (n:word {token_id: ", str(id), ", token: '", tok, "'})"]) for tok, id in
                   zip(tokens, token_ids)]
        self.add_queries(queries)
        self.write_queue()
        self.init_tokens()

    # %% Initializations
    def init_tokens(self):
        # Run neo query to get all nodes
        res = self.connector.run("MATCH (n:word) RETURN n.token_id, n.token")
        # Update results
        self.ids = [x['n.token_id'] for x in res]
        self.tokens = [x['n.token'] for x in res]
        self.update_dicts()

    # %% Neo4J interaction
    # All function that interact with neo are here, dispatched as needed from above
    # TODO: Set internal, allow access only via dispatch

    def add_query(self, query, params=None):
        """
        Add a single query to queue
        :param query: Neo4j query
        :param params: Associates parameters
        :return:
        """
        self.add_queries([query], [params])

    def add_queries(self, query, params=None):
        """
        Add a list of query to queue
        :param query: list - Neo4j queries
        :param params: list - Associates parameters corresponding to queries
        :return:
        """
        assert isinstance(query, list)

        if params is not None:
            assert isinstance(params, list)
            statements = [neo4j.Statement(q, p) for (q, p) in zip(query, params)]
            self.neo_queue.extend(statements)
        else:
            statements = [neo4j.Statement(q) for (q) in query]
            self.neo_queue.extend(statements)

    def write_queue(self):
        # This may or may not be optimized by using parameters and individual queries.
        if len(self.neo_queue) > 0:
            res = self.connector.run_multiple(self.neo_queue, self.neo_batch_size)
            self.neo_queue = []

    def query_node(self, id, times=None):
        """ See query_multiple_nodes"""
        return self.query_multiple_nodes([id], times)

    def query_multiple_nodes(self, ids, times=None):
        """
        Query multiple nodes by ID and over a set of time intervals
        :param ids: list of id's
        :param times: either a number format YYYYMMDD, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :return: list of tuples (u,v,Time,{weight:x})
        """
        # assert isinstance(ids, list) remove checking here, should be done in dispatch functions
        if self.graph_direction == "REVERSE":  # Seek nodes that predict ID nodes hence reverse sender/receiver
            if isinstance(times, dict) or isinstance(times, int):
                if isinstance(times, dict):  # Interval query
                    params = {"ids": ids, "times": times}
                    query = "UNWIND $ids AS id MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word  {token_id:id}) WHERE $times.start <= r.time<= $times.end RETURN b.token_id AS sender,a.token_id AS receiver,r.time AS time,r.weight AS weight"
                elif isinstance(times, int):
                    params = {"ids": ids, "times": times}
                    query = "UNWIND $ids AS id MATCH p=(a:word)-[:onto]->(r:edge {time:$times})-[:onto]->(b:word  {token_id:id}) RETURN b.token_id AS sender,a.token_id AS receiver,r.time AS time,r.weight AS weight"
            else:
                query = "unwind $ids AS id MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word  {token_id:id}) RETURN b.token_id AS sender,a.token_id AS receiver,r.time AS time,r.weight AS weight"
                params = {"ids": ids}
        else:  # Seek nodes that predict nodes
            if isinstance(times, dict) or isinstance(times, int):
                if isinstance(times, dict):  # Interval query
                    params = {"ids": ids, "times": times}
                    query = "UNWIND $ids AS id MATCH p=(a:word  {token_id:id})-[:onto]->(r:edge)-[:onto]->(b:word) WHERE $times.start <= r.time<= $times.end RETURN b.token_id AS receiver,a.token_id AS sender,r.time AS time,r.weight AS weight"
                elif isinstance(times, int):
                    params = {"ids": ids, "times": times}
                    query = "UNWIND $ids AS id MATCH p=(a:word  {token_id:id})-[:onto]->(r:edge {time:$times})-[:onto]->(b:word) RETURN b.token_id AS receiver,a.token_id AS sender,r.time AS time,r.weight AS weight"
            else:
                query = "unwind $ids AS id MATCH p=(a:word  {token_id:id})-[:onto]->(r:edge)-[:onto]->(b:word) RETURN  b.token_id AS receiver,a.token_id AS sender,r.time AS time,r.weight AS weight"
                params = {"ids": ids}

        res = self.connector.run(query, params)
        ties = [(x['sender'], x['receiver'], x['time'], {'weight': x['weight']}) for x in res]
        return ties

    def insert_edges_multiple(self, ties):
        """
        Allows to add ties across nodes
        :param ties: Set of Tuples (u,v,Time,weight)
        :return: None
        """
        # Graph is saved as "PREDICTS", thus we may need to reverse ego/alter here
        if self.graph_direction == "REVERSE":
            params = {"ties": [{"ego": x[1], "alter": x[0], "time": x[2], "weight": x[3]['weight']} for x in ties]}
        else:
            params = {"ties": [{"ego": x[0], "alter": x[1], "time": x[2], "weight": x[3]['weight']} for x in ties]}

        query = "UNWIND $ties AS tie MATCH (a:word {token_id: tie.ego}) MATCH (b:word {token_id: tie.alter}) WITH a,b,tie MERGE (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time})<-[:onto]-(a)"
        self.add_query(query, params)

        # TODO graph dispatch

    def insert_edges(self, ego, ties):
        """
        Add ties from a given ego node u->v
        :param ties: Set of Tuples (v,Time,{"weight":x})
        :return: None
        """
        ties = [(ego, x[0], x[1], x[2]) for x in ties]
        self.insert_edges_multiple(ties)

    def add_token(self, id, token):
        # Update class
        self.tokens.append(token)
        self.ids.append(id)
        self.update_dicts()

        # Update database
        queries = [''.join(["MERGE (n:word {token_id: ", str(id), ", token: '", token, "'})"])]
        self.add_queries(queries)

    def remove_token(self, key):
        self.tokens.remove(self.get_token_from_id(key))
        self.ids.remove(key)
        self.update_dicts()

        # Update database
        queries = [''.join(["MATCH (n:word {token_id: ", str(key), "}) DETACH DELETE n"])]
        self.add_queries(queries)

    # %% Conditoning functions
    def condition(self, years):
        if self.conditioned == False:  # This is the first conditioning
            # Preserve node and token lists
            self.neo_ids = copy.deepcopy(self.ids)
            self.neo_tokens = copy.deepcopy(self.tokens)
            self.neo_token_id_dict = copy.deepcopy(self.token_id_dict)
        else:  # Conditioning on condition
            pass

        # Continue conditioning

    def decondition(self):
        # Reset token lists to original state.
        self.ids = self.neo_ids
        self.tokens = self.neo_tokens
        self.token_id_dict = self.neo_token_id_dict

    # %% Utility functioncs

    def update_dicts(self):
        """Simply update dictionaries"""
        # Update dictionaries
        self.token_id_dict.update(dict(zip(self.tokens, self.ids)))

    def get_index_from_ID(self, id):
        """Index (order) of token in data structures used"""
        # Token has to be string
        assert isinstance(id, int)
        try:
            index = self.id_index_dict[id]
        except:
            raise LookupError("".join(["Index of token with id ", str(id), " missing. Token not in network?"]))
        return index.item()

    def get_token_from_id(self, id):
        """Index (order) of token in data structures used"""
        # Token has to be string
        assert isinstance(id, int)
        try:
            token = self.token_id_dict[id]
        except:
            raise LookupError("".join(["Token with ID ", str(id), " missing. Token not in network?"]))
        return token

    def get_id_from_token(self, token):
        """Index (order) of token in data structures used"""
        # Token has to be string
        assert isinstance(token, str)
        try:
            id = self.token_id_dict[token]
        except:
            raise LookupError("".join(["ID of token ", token, " missing. Token not in network?"]))
        return id
