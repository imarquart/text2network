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

    #%% Initialization functions
    def __init__(self,neo4j_creds, graph_type, batch_size=10,graph_direction="PREDICTED"):
        self.neo4j_connection,self.neo4j_credentials = neo4j_creds
        # Conditioned graph information
        self.graph_type = graph_type
        self.graph = None
        self.conditioned = None
        self.years  = []
        self.graph_direction=graph_direction
        # Dictionaries and token/id saved in memory
        self.token_id_dict = TwoWayDict()
        # Since both are numerical, we need to use a single way dict here
        self.id_index_dict = dict()
        self.tokens = []
        self.ids= []
        # Copies to be used during conditioning
        self.neo_token_id_dict = TwoWayDict()
        self.neo_ids = []
        self.neo_tokens = []

        # Neo4J Internals
        self.neo_queue = []
        self.neo_batch_size=batch_size
        self.connector = neo4j.Connector(self.neo4j_connection,self.neo4j_credentials)
        # Init parent class
        super().__init__()

    #%% Python Interface implementations
    def __delitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, i):
        pass

    def __len__(self):
        pass

    def insert(self, token, token_id):
        pass

    #%% Setup
    def setup_neo_db(self,tokens,token_ids):
        """Creates tokens and token_ids in Neo database"""
        logging.info("Creating indecies and nodes in Neo4j database.")
        # Create uniqueness constraints
        query="CREATE CONSTRAINT ON(n:word) ASSERT n.token_id IS UNIQUE"
        self.add_query(query)
        query="CREATE CONSTRAINT ON(n:word) ASSERT n.token IS UNIQUE"
        self.add_query(query)
        # Need to write first because create and structure changes can not be batched
        self.write_queue()
        # Create nodes in neo db
        queries=[''.join(["MERGE (n:word {token_id: ",str(id),", token: '",tok,"'})"]) for tok,id in zip(tokens,token_ids)]
        self.add_queries(queries)
        self.write_queue()


    #%% Initializations
    def init_tokens(self):
        # Run neo query to get all nodes
        res=self.connector.run("MATCH (n:word) RETURN n.token_id, n.token")
        # Update results
        self.ids=[x['n.token_id'] for x in res]
        self.tokens=[x['n.token'] for x in res]
        self.update_dicts()

    #%% Neo4J interaction
    def add_query(self,query,params=None):
        self.add_queries([query],[params])

    def add_queries(self,query,params=None):
        assert isinstance(query,list)

        if params is not None:
            statements=[neo4j.Statement(q,p) for (q,p) in zip(query,params)]
            self.neo_queue.extend(statements)
        else:
            statements = [neo4j.Statement(q) for (q) in query]
            self.neo_queue.extend(statements)

    def write_queue(self):
        # This may or may not be optimized by using parameters and individual queries.
        res=self.connector.run_multiple(self.neo_queue, self.neo_batch_size)
        self.neo_queue=[]

    def query_node(self,id, times=None):
        return self.query_multiple_nodes([id],times)

    def query_multiple_nodes(self, ids, times=None):
        assert isinstance(ids,list)
        if self.graph_direction == "PREDICTED": # Seek nodes that predict ID nodes hence reverse sender/receiver
            if times is not None:
                assert isinstance(times,list)
                params = {"ids": ids, "times":times}
                query="UNWIND times as time UNWIND $ids AS id MATCH p=(a:word)-[r:PREDICTS {time: time}]->(b:word  {token_id:id}) RETURN b.token_id AS sender,a.token_id AS receiver,r.time AS time,r.weight AS weight"
            else:
                query="unwind $ids AS id MATCH p=(a:word)-[r:PREDICTS]->(b:word  {token_id:id}) RETURN b.token_id AS sender,a.token_id AS receiver,r.time AS time,r.weight AS weight"
                params={"ids":ids}
        else: # Seek nodes that predict nodes
            if times is not None:
                assert isinstance(times,list)
                params = {"ids": ids, "times":times}
                query="UNWIND times as time UNWIND $ids AS id MATCH p=(a:word)<-[r:PREDICTS {time: time}]-(b:word  {token_id:id}) RETURN b.token_id AS sender,a.token_id as receiver,r.time AS time,r.weight AS weight"
            else:
                query="unwind $ids AS id MATCH p=(a:word)<-[r:PREDICTS]-(b:word  {token_id:id}) RETURN b.token_id AS sender,a.token_id as receiver,r.time AS time,r.weight AS weight"
                params={"ids":ids}
        res=self.connector.run(query,params)
        ties = [(x['sender'], x['receiver'], x['time'], {'weight': x['weight']}) for x in res]
        return ties

    def insert_edges_query_multiple(self,ties):
        """
        Allows to add ties across nodes
        :param ties: Set of Tuples (u,v,Time,{"weight":x})
        :return:
        """
        # Graph is saved as "PREDICTS", thus we may need to reverse ego/alter here
        if self.graph_direction=="PREDICTED":
            params={"ties": [{"ego":x[1],"alter":x[0],"time":str(x[2]),"weight":x[3]['weight']} for x in ties]}
        else:
            params={"ties": [{"ego":x[0],"alter":x[1],"time":str(x[2]),"weight":x[3]['weight']} for x in ties]}

        query="UNWIND $ties AS tie MATCH (a:word {token_id: tie.ego}) MATCH (b:word {token_id: tie.alter}) WITH a,b,tie MERGE (a)-[:PREDICTS {weight:tie.weight, time:date(tie.time)}]->(b)"
        self.add_query(query,params)

    def insert_edges_query(self,ego,ties):
        """
        Add ties from a given ego node u->v
        TODO: Refactor into above function
        :param ties: Set of Tuples (v,Time,{"weight":x})
        :return:
        """
        # Graph is saved as "PREDICTS", thus we may need to reverse ego/alter here
        if self.graph_direction=="PREDICTED":
            params = {"ties": [{"alter": x[0], "time": str(x[1]), "weight": x[2]['weight']} for x in ties], "ego": ego}
            query = "MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties AS tie MATCH (b:word {token_id: tie.alter}) WITH a,b,tie MERGE (a)<-[:PREDICTS {weight:tie.weight, time:date(tie.time)}]-(b)"
        else:
            params={"ties": [{"alter":x[0],"time":str(x[1]), "weight": x[2]['weight']} for x in ties], "ego":ego}
            query="MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties AS tie MATCH (b:word {token_id: tie.alter}) WITH a,b,tie MERGE (a)-[:PREDICTS {weight:tie.weight, time:date(tie.time)}]->(b)"
        self.add_query(query,params)



    #%% Conditoning functions
    def condition(self,years):
        if self.conditioned == False: # This is the first conditioning
            # Preserve node and token lists
            self.neo_ids = copy.deepcopy(self.ids)
            self.neo_tokens = copy.deepcopy(self.tokens)
            self.neo_token_id_dict = copy.deepcopy(self.token_id_dict)
        else: # Conditioning on condition
            pass

        # Continue conditioning

    def decondition(self):
        # Reset token lists to original state.
        self.ids = self.neo_ids
        self.tokens = self.neo_tokens
        self.token_id_dict = self.neo_token_id_dict

    #%% Utility functioncs

    def update_dicts(self):
        """Simply update dictionaries"""
        # Update dictionaries
        self.token_id_dict.update(dict(zip(self.tokens,self.ids)))

