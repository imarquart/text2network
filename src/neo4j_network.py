import copy
import logging
from collections.abc import MutableSequence

import numpy as np

# import neo4j
import neo4jCon as neo_connector
from src.utils.twowaydict import TwoWayDict

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
    def __init__(self, neo4j_creds, graph_type="networkx", graph_direction="FORWARD", agg_operator="SUM", write_before_query=True,
                 neo_batch_size=None, queue_size=10000, tie_query_limit=100000, tie_creation="UNSAFE"):
        self.neo4j_connection, self.neo4j_credentials = neo4j_creds
        self.write_before_query = write_before_query
        # Conditioned graph information
        self.graph_type = graph_type
        self.graph = None
        self.conditioned = False
        self.years = []
        self.graph_direction = graph_direction
        self.aggregate_operator=agg_operator
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
        # Pick Merge or Create. Create will double ties but Merge becomes very slow for large networks
        if tie_creation=="SAFE":
            self.creation_statement="MERGE"
        else:
            self.creation_statement="CREATE"
        self.neo_queue = []
        self.neo_batch_size = neo_batch_size
        self.queue_size = queue_size
        self.connector = neo_connector.Connector(self.neo4j_connection, self.neo4j_credentials)
        # Init tokens
        self.init_tokens()
        # Init parent class
        super().__init__()

    # %% Python Interface implementations
    def __delitem__(self, key):
        """
        Deletes a node and all its ties
        :param key: Token or Token id
        :return:
        """
        self.remove(key)

    def remove(self, key):
        """
        Deletes a node and all its ties
        :param key: Token or Token id
        :return:
        """
        key = self.ensure_ids(key)
        assert key in self.ids, "ID of ego token to connect not found. Not in network?"
        self.remove_token(key)

    def __setitem__(self, key, value):
        """
        Set links of node
        :param key: id or token of node.
        :param value: To add links to node: list of tuples [(neighbor,time, weight))] or [(neighbor,time, {'weight':weight,'p1':p1,'p2':p2}))]. To add node itself, token_id (int)
        :return:
        """
        if isinstance(key, str) and isinstance(value, int):  # Wants to add a node
            self.add_token(value, key)
        else:
            key=self.ensure_ids(key)
            assert key in self.ids, "ID of ego token to connect not found. Not in network?"
            try:

                neighbors = [ x[0] for x in value]
                neighbors = self.ensure_ids(neighbors)
                weights = [{'weight': x[2]} if isinstance(x[2], (int, float)) else x[2] for x in value]
                years = [x[1] for x in value]
                token = map(int, np.repeat(key, len(neighbors)))
            except:
                raise ValueError("Adding requires an iterable over tuples e.g. [(neighbor,time, weight))]")

            # Check if all neighbor tokens present
            assert set(neighbors) < set(self.ids), "ID of node to connect not found. Not in network?"
            ties = list(zip(token, neighbors, years, weights))

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
        if isinstance(i, (list, tuple, range,str,int)):
            i = self.ensure_ids(i)
        else:
            raise AssertionError("Please format a call as <token> or <token_id>")

        # TODO Dispatch in case of graph
        if self.conditioned==False:
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
        constr = [x['name'] for x in self.connector.run("CALL db.constraints")]
        # Create uniqueness constraints
        if 'id_con' not in constr:
            query = "CREATE CONSTRAINT id_con ON(n:word) ASSERT n.token_id IS UNIQUE"
            self.add_query(query)
        if 'tk_con' not in constr:
            query = "CREATE CONSTRAINT tk_con ON(n:word) ASSERT n.token IS UNIQUE"
            self.add_query(query)
        constr = [x['name'] for x in self.connector.run("CALL db.indexes")]
        if 'timeindex' not in constr:
            query = "CREATE INDEX timeindex FOR (a:edge) ON (a.time)"
            self.add_query(query)
        if 'contimeindex' not in constr:
            query = "CREATE INDEX contimeindex FOR (a:context) ON (a.time)"
            self.add_query(query)
        # Need to write first because create and structure changes can not be batched
        self.non_con_write_queue()
        # Create nodes in neo db
        queries = [''.join(["MERGE (n:word {token_id: ", str(id), ", token: '", tok, "'})"]) for tok, id in
                   zip(tokens, token_ids)]
        self.add_queries(queries)
        self.non_con_write_queue()
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
            statements = [neo_connector.Statement(q, p) for (q, p) in zip(query, params)]
            self.neo_queue.extend(statements)
        else:
            statements = [neo_connector.Statement(q) for (q) in query]
            self.neo_queue.extend(statements)

        # Check for queue size if not conditioned!
        if (len(self.neo_queue) > self.queue_size) and (self.conditioned == False):
            self.write_queue()

    def write_queue(self):
        if len(self.neo_queue) > 0:
            self.connector.run_multiple(self.neo_queue, self.neo_batch_size)



            self.neo_queue = []

    def non_con_write_queue(self):
        self.write_queue()

    def query_node(self, id, times=None,  weight_cutoff=None):
        """ See query_multiple_nodes"""
        return self.query_multiple_nodes([id], times,  weight_cutoff)

    def query_multiple_nodes(self, ids, times=None, weight_cutoff=None):
        """
        Query multiple nodes by ID and over a set of time intervals
        :param ids: list of id's
        :param times: either a number format YYYYMMDD, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :return: list of tuples (u,v,Time,{weight:x})
        """
        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query = ''.join([" WHERE r.weight >=", str(weight_cutoff), " "])
            if isinstance(times, dict):
                where_query =''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        else:
            if isinstance(times, dict):
                where_query="WHERE  $times.start <= r.time<= $times.end "
            else:
                where_query=""
        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}
        # Create query depending on graph direction and whether time variable is queried via where or node property
        if self.graph_direction == "REVERSE":  # Seek alters that predict ID nodes hence reverse sender/receiver
            return_query=''.join([" RETURN b.token_id AS sender,a.token_id AS receiver,count(r.time) AS occurrences,", self.aggregate_operator, "(r.weight) AS agg_weight order by agg_weight"])
            if isinstance(times, int):
                match_query = "UNWIND $ids AS id MATCH p=(a:word)-[:onto]->(r:edge {time:$times})-[:onto]->(b:word  {token_id:id}) "
            else:
                match_query = "UNWIND $ids AS id MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word  {token_id:id}) "
        else:  # Seek ego nodes that predict alters
            return_query = ''.join([" RETURN b.token_id AS receiver,a.token_id AS sender,count(r.time) AS occurrences,",
                                    self.aggregate_operator, "(r.weight) AS agg_weight order by agg_weight"])

            if isinstance(times, int):
                match_query = "UNWIND $ids AS id MATCH p=(a:word  {token_id:id})-[:onto]->(r:edge {time:$times})-[:onto]->(b:word) "
            else:
                match_query = "unwind $ids AS id MATCH p=(a:word  {token_id:id})-[:onto]->(r:edge)-[:onto]->(b:word) "

        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        query = "".join([match_query, where_query, return_query])
        res = self.connector.run(query, params)
        ties = [(x['sender'], x['receiver'], nw_time['m'],
                 {'weight': x['agg_weight'], 't1': nw_time['s'], 't2': nw_time['e'], 'occurences': x['occurrences']})
                for x in res]
        return ties

    def query_multiple_nodes_context(self, ids, times=None, weight_cutoff=None, context_direction="FORWARD"):
        """
        Query the context of multiple nodes
        :param ids: list of id's
        :param times: either a number format YYYYMMDD, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :return: list of tuples (u,v,Time,{weight:x})
        """
        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query = ''.join([" WHERE c.weight >=", str(weight_cutoff), " "])
            if isinstance(times, dict):
                where_query = ''.join([where_query, " AND  $times.start <= c.time<= $times.end "])
        else:
            if isinstance(times, dict):
                where_query = "WHERE  $times.start <= c.time<= $times.end "
            else:
                where_query = ""
        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}
        # Create query depending on graph direction and whether time variable is queried via where or node property
        # REVERSE: (focal)->(context) if (focal)<-(alter) in (context)
        if context_direction == "REVERSE":
            return_query = ''.join(
                [" RETURN b.token_id AS context_token,a.token_id AS focal_token,count(c.time) AS occurrences,",
                 self.aggregate_operator, "(c.weight) AS agg_weight order by agg_weight"])
            if isinstance(times, int):
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})<-[: onto]-(r:edge) - [: conto]->(c:context {time:$times}) - [: conto]->(b:word)"
            else:
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})<-[: onto]-(r:edge) - [: conto]->(c:context) - [: conto]->(b:word)"
        else:  # FORWARD: (focal)->(context) if (focal)->(alter) in (context)
            return_query = ''.join(
                [" RETURN b.token_id AS context_token,a.token_id AS focal_token,count(c.time) AS occurrences,",
                 self.aggregate_operator, "(c.weight) AS agg_weight order by agg_weight"])

            if isinstance(times, int):
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})-[: onto]->(r:edge) - [: conto]->(c:context {time:$times}) - [: conto]->(b:word)"
            else:
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})-[: onto]->(r:edge) - [: conto]->(c:context) - [: conto]->(b:word)"

        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        query = "".join([match_query, where_query, return_query])
        res = self.connector.run(query, params)
        ties = [(x['focal_token'], x['context_token'], nw_time['m'],
                 {'weight': x['agg_weight'], 't1': nw_time['s'], 't2': nw_time['e'], 'occurences': x['occurrences']})
                for x in res]
        return ties

    def insert_edges_context(self, ego, ties, contexts):
        if self.graph_direction == "REVERSE":
            egos = np.array([x[1] for x in ties])
            alters = np.array([x[0] for x in ties])
        else:
            egos = np.array([x[0] for x in ties])
            alters = np.array([x[1] for x in ties])
        con_alters = np.array([x[1] for x in contexts])

        times = np.array([x[2] for x in ties])
        dicts = np.array([x[3] for x in ties])
        con_times = np.array([x[2] for x in contexts])
        con_dicts = np.array([x[3] for x in contexts])

        unique_egos = np.unique(egos)
        if len(unique_egos) == 1:
            ties_formatted = [{"alter": int(x[0]), "time": int(x[1]), "weight": float(x[2]['weight']),
                               "p1": (int(x[2]['p1']) if len(x[2]) > 1 else 0),
                               "p2": (int(x[2]['p2']) if len(x[2]) > 2 else 0)}
                              for x in zip(alters.tolist(), times.tolist(), dicts.tolist())]
            contexts_formatted = [{"alter": int(x[0]), "time": int(x[1]), "weight": float(x[2]['weight']),
                                   "p1": (int(x[2]['p1']) if len(x[2]) > 1 else 0),
                                   "p2": (int(x[2]['p2']) if len(x[2]) > 2 else 0)}
                                  for x in zip(con_alters.tolist(), con_times.tolist(), con_dicts.tolist())]
            params = {"ego": int(egos[0]), "ties": ties_formatted, "contexts": contexts_formatted}
            query = ''.join(
                [" MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties as tie MATCH (b:word {token_id: tie.alter}) ",
                 self.creation_statement,
                 " (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time, p1:tie.p1,p2:tie.p2})<-[:onto]-(a) WITH r UNWIND $contexts as con MATCH (q:word {token_id: con.alter}) WITH r,q,con ",
                 self.creation_statement,
                 " (r)-[:conto]->(c:context {weight:con.weight, time:con.time})-[:conto]->(q)"])
        else:
            logging.error("Batched edge creation with context for multiple egos not supported.")
            raise NotImplementedError

        self.add_query(query, params)

    def insert_edges_multiple(self, ties):
        if self.graph_direction == "REVERSE":
            egos = np.array([x[1] for x in ties])
            alters = np.array([x[0] for x in ties])
        else:
            egos = np.array([x[0] for x in ties])
            alters = np.array([x[1] for x in ties])
        times = np.array([x[2] for x in ties])
        dicts = np.array([x[3] for x in ties])

        unique_egos = np.unique(egos)
        sets = []
        if len(unique_egos) == 1:
            ties_formatted = [{"alter": int(x[0]), "time": int(x[1]), "weight": float(x[2]['weight']),
                               "p1": (int(x[2]['p1']) if len(x[2]) > 1 else 0),
                               "p2": (int(x[2]['p2']) if len(x[2]) > 2 else 0)}
                              for x in zip(alters.tolist(), times.tolist(), dicts.tolist())]
            params = {"ego": int(egos[0]), "ties": ties_formatted}
            query = ''.join([" MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties as tie MATCH (b:word {token_id: tie.alter}) ",self.creation_statement," (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time, p1:tie.p1,p2:tie.p2})<-[:onto]-(a)"])
        else:
            for u_ego in unique_egos:
                mask=egos==u_ego
                subalters=alters[mask]
                subtimes=times[mask]
                subdicts=dicts[mask]
                ties_formatted = [{"alter": int(x[0]), "time": int(x[1]), "weight": float(x[2]['weight']),
                                   "p1": (int(x[2]['p1']) if len(x[2]) > 1 else 0), "p2": (int(x[2]['p2']) if len(x[2]) > 2 else 0)}
                                  for x in zip(subalters.tolist(),subtimes.tolist(),subdicts.tolist())]
                set={'ego':int(u_ego),'ties':ties_formatted}
                sets.append(set)
            params={"sets":sets}
            query = ''.join(["UNWIND $sets as set MATCH (a:word {token_id: set.ego}) WITH a,set UNWIND set.ties as tie MATCH (b:word {token_id: tie.alter}) ",self.creation_statement,"  (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time, p1:tie.p1,p2:tie.p2})<-[:onto]-(a)"])

        self.add_query(query, params)

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
    def condition(self, years, tokens, weight_cutoff=None, depth=None, context=None):
        if not isinstance(tokens, list): tokens = [tokens]
        # Work from ID list
        tokens = self.ensure_ids(tokens)
        if self.conditioned == False:  # This is the first conditioning
            # Preserve node and token lists
            self.neo_ids = copy.deepcopy(self.ids)
            self.neo_tokens = copy.deepcopy(self.tokens)
            self.neo_token_id_dict = copy.deepcopy(self.token_id_dict)
            # Build graph
            self.graph = self.create_empty_graph()
            # Add starting nodes
            self.graph.add_nodes_from(tokens)
            # Query Neo4j
            try:
                self.graph.add_edges_from(self.query_multiple_nodes(tokens, years, weight_cutoff))
            except:
                logging.error("Could not condition graph by query method.")

            # Update IDs and Tokens to reflect conditioning
            new_ids = list(self.graph.nodes)
            self.tokens = [self.get_token_from_id(x) for x in new_ids]
            self.ids = new_ids
            self.update_dicts()

            # Set additional attributes
            att_list = [{"token": x} for x in self.tokens]
            att_dict = dict(list(zip(self.ids, att_list)))
            nx.set_node_attributes(self.graph, att_dict)

            if depth == 1:
                try:
                    self.graph.add_edges_from(self.query_multiple_nodes(new_ids, years, weight_cutoff))
                except:
                    logging.error("Could not condition graph by query method.")
                # Update IDs and Tokens to reflect conditioning
                new_ids = list(self.graph.nodes)
                self.tokens = [self.get_token_from_id(x) for x in new_ids]
                self.ids = new_ids
                self.update_dicts()

                # Set additional attributes
                att_list = [{"token": x} for x in self.tokens]
                att_dict = dict(list(zip(self.ids, att_list)))
                nx.set_node_attributes(self.graph, att_dict)

            # Set conditioning true
            self.conditioned = True

        else:  # Conditioning on condition
            pass

        # Continue conditioning

    def condition_context(self, years, tokens, weight_cutoff=None, depth=None, context_direction="FORWARD"):
        if not isinstance(tokens, list): tokens = [tokens]
        # Work from ID list
        tokens = self.ensure_ids(tokens)
        if self.conditioned == False:  # This is the first conditioning
            # Preserve node and token lists
            self.neo_ids = copy.deepcopy(self.ids)
            self.neo_tokens = copy.deepcopy(self.tokens)
            self.neo_token_id_dict = copy.deepcopy(self.token_id_dict)
            # Build graph
            self.graph = self.create_empty_graph()
            # Add starting nodes
            self.graph.add_nodes_from(tokens)
            # Query Neo4j
            try:
                self.graph.add_edges_from(
                    self.query_multiple_nodes_context(tokens, years, weight_cutoff, context_direction))
            except:
                logging.error("Could not condition graph by query method.")

            # Update IDs and Tokens to reflect conditioning
            new_ids = list(self.graph.nodes)
            self.tokens = [self.get_token_from_id(x) for x in new_ids]
            self.ids = new_ids
            self.update_dicts()

            # Set additional attributes
            att_list = [{"token": x} for x in self.tokens]
            att_dict = dict(list(zip(self.ids, att_list)))
            nx.set_node_attributes(self.graph, att_dict)

            if depth == 1:
                try:
                    self.graph.add_edges_from(
                        self.query_multiple_nodes_context(new_ids, years, weight_cutoff, context_direction))
                except:
                    logging.error("Could not condition graph by query method.")
                # Update IDs and Tokens to reflect conditioning
                new_ids = list(self.graph.nodes)
                self.tokens = [self.get_token_from_id(x) for x in new_ids]
                self.ids = new_ids
                self.update_dicts()

                # Set additional attributes
                att_list = [{"token": x} for x in self.tokens]
                att_dict = dict(list(zip(self.ids, att_list)))
                nx.set_node_attributes(self.graph, att_dict)

            # Set conditioning true
            self.conditioned = True

        else:  # Conditioning on condition
            pass

        # Continue conditioning


    def decondition(self, write=False):
        # Reset token lists to original state.
        self.ids = self.neo_ids
        self.tokens = self.neo_tokens
        self.token_id_dict = self.neo_token_id_dict

        # Decondition
        self.delete_graph()
        self.conditioned=False

        if write==True:
            self.write_queue()
        else:
            self.neo_queue=[]


    # %% Graph abstractions - for now only networkx
    def create_empty_graph(self):

        return nx.MultiDiGraph()

    def delete_graph(self):

        self.graph=None


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

    def ensure_ids(self, tokens):
        """This is just to confirm mixed lists of tokens and ids get converted to ids"""
        if isinstance(tokens, list):
            return [self.get_id_from_token(x) if not isinstance(x, int) else x for x in tokens]
        else:
            if not isinstance(tokens, int):
                return self.get_id_from_token(tokens)
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

    def export_gefx(self, path):
        if self.conditioned == True:
            try:
                # Relabel nodes
                labeldict = dict(zip(self.ids, [self.get_token_from_id(x) for x in self.ids]))
                reverse_dict = dict(zip([self.get_token_from_id(x) for x in self.ids], self.ids))
                self.graph = nx.relabel_nodes(self.graph, labeldict)
                nx.write_gexf(self.graph, path)
                self.graph = nx.relabel_nodes(self.graph, reverse_dict)

            except:
                raise SystemError("Could not save to %s " % path)
