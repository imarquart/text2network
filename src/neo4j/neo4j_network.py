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
    def setup_neo_db(self,tokens,token_ids,years):
        # Create uniqueness constraints
        query=["CREATE CONSTRAINT ON(n:word) ASSERT n.token_id IS UNIQUE"]
        self.add_query(query)
        query=["CREATE CONSTRAINT ON(n:word) ASSERT n.token IS UNIQUE"]
        self.add_query(query)
        # Need to write first because create and structure changes can not be batched
        self.write_queue()
        # Create nodes in neo db
        queries=[''.join(["CREATE (n:word {token_id: ",str(id),", token: '",tok,"'})"]) for tok,id in zip(tokens,token_ids)]
        self.add_query(queries)
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
        if params is not None:
            self.neo_queue.extend(neo4j.Statement(query,params))
        else:
            self.neo_queue.extend(neo4j.Statement(query))

    def write_queue(self):
        # This may or may not be optimized by using parameters and individual queries.
        res=self.connector.run_multiple(self.neo_queue, self.neo_batch_size)
        self.neo_queue=[]

    def query_node(self,id, years=None):
        return self.query_multiple_nodes([id],years)

    def query_multiple_nodes(self, ids, years=None):
        assert isinstance(ids,list)
        if years is not None:
            assert isinstance(years,list)
            params = {"ids": ids, "years":years}
            query="UNWIND $years as year UNWIND $ids AS id MATCH p=(a:word)-[r:PREDICTS {year: year}]->(b:word  {token_id:id}) RETURN b.token_id,a.token_id,r.year AS year,r.weight AS weight"
        else:
            query="unwind $ids AS id MATCH p=(a:word)-[r:PREDICTS]->(b:word  {token_id:id}) RETURN b.token_id,a.token_id,r.year AS year,r.weight AS weight"
            params={"ids":ids}

        res=self.connector.run(query,params)
        if self.graph_direction=="PREDICTED":
            ties=[(x['b.token_id'], x['a.token_id'], x['year'], {'weight': x['weight']}) for x in res]
        else:
            ties = [(x['a.token_id'], x['b.token_id'], x['year'], {'weight': x['weight']}) for x in res]
        return ties

    def insert_edges_query_multiple(self,ties):

        # Graph is saved as "PREDICTS", thus we may need to reverse ego/alter here
        if self.graph_direction=="PREDICTED":
            params={"ties": [{"ego":x[1],"alter":x[0],"year":x[2],"weight":x[3]['weight']} for x in ties]}
        else:
            params={"ties": [{"ego":x[0],"alter":x[1],"year":x[2],"weight":x[3]['weight']} for x in ties]}

        query="UNWIND $ties AS tie MATCH (a:word {token_id: tie.ego}) MATCH (b:word {token_id: tie.alter}) WITH a,b,tie MERGE (a)-[:PREDICTS {weight:tie.weight, year:tie.year}]->(b)"
        self.add_query(query,params)

    def insert_edges_query(self,ego,ties):
        # Graph is saved as "PREDICTS", thus we may need to reverse ego/alter here
        if self.graph_direction=="PREDICTED":
            params = {"ties": [{"alter": x[0], "year": x[1], "weight": x[2]['weight']} for x in ties], "ego": ego}
            query = "MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties AS tie MATCH (b:word {token_id: tie.alter}) WITH a,b,tie MERGE (a)<-[:PREDICTS {weight:tie.weight, year:tie.year}]-(b)"
        else:
            params={"ties": [{"alter":x[0],"year":x[1],"weight":x[2]['weight']} for x in ties], "ego":ego}
            query="MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties AS tie MATCH (b:word {token_id: tie.alter}) WITH a,b,tie MERGE (a)-[:PREDICTS {weight:tie.weight, year:tie.year}]->(b)"
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






class neo4j_network_old(MutableSequence):


    def init_test_graph(self):
        """Just add a test graph from neo data"""
        self.graph=nx.MultiDiGraph()
        self.graph.add_nodes_from(self.neo_ids)
        self.graph_tokens=self.neo_tokens
        self.graph_ids = list(self.graph.nodes)
        self.update_graph_dicts()

    def init_neo4j_tokens(self):
        """Init neo4j"""
        self.neo_tokens = np.array(["Car", "House", "Window"])
        self.neo_ids = np.array([3,4,5])
        self.update_neo_dicts()

    #%% Utility functions

    def get_index_from_token(self,token):
        """Index (order) of token in data structures used"""
        # Token has to be string
        assert isinstance(token, str)
        if self.graph is None: # No graph initialized
            index=np.flatnonzero(self.neo_tokens == token)
            if index.size == 0:
                index=False
        else: # graph initialized, use our dicts
            try:
                id=self.token_id_dict[token]
                index=self.id_index_dict[id]
            except:
                index=False
        return index.item()

    def get_id_from_token(self,token):
        """Get the token id"""
        # Token has to be string
        assert isinstance(token, str)
        if self.graph is None: # No graph initialized
            try:
                id=self.neo_token_id_dict[token]
            except:
                id=False
        else: # graph initialized, use our dicts
            try:
                id=self.token_id_dict[token]
            except:
                # token not in graph, infer from neo4j
                try:
                    id = self.neo_token_id_dict[token]
                except:
                    id = False
        return id

    def get_token_from_id(self,id):
        """Get the token assigned to id"""
        # id has to be integer
        assert isinstance(id, int)
        if self.graph is None: # No graph initialized
            try:
                token=self.neo_token_id_dict[id]
            except:
                token="UNK"
        else: # graph initialized, use our dicts
            try:
                token=self.token_id_dict[id]
            except:
                # Failed to find in graph, infer from neo4j
                try:
                    token = self.neo_token_id_dict[id]
                except:
                    token = "UNK"
        return token

    def add_neo_queue(self,add):
        self.neo_queue.append(add)

    #%% Delete / Update functions

    def write_neo4j_query(self):
        # TODO: Write query
        pass

    def update_graph_dicts(self):
        """Use graph properties to update dicts"""
        if self.graph is not None:
            indecies = list(range(0, len(self.graph_ids)))
            # Update dictionaries
            self.token_id_dict.update(dict(zip(self.graph_tokens,self.graph_ids)))
            self.id_index_dict.update(dict(zip(self.graph_ids,indecies)))

    def update_neo_dicts(self):
        """Simply update dictionaries"""
        # Update dictionaries
        self.neo_token_id_dict.update(dict(zip(self.neo_tokens,self.neo_ids)))

    #%% Adding Functions

    def __setitem__(self, key, value):
        """Interpret as adding ties in the graph"""
        # We work with token id's here
        if isinstance(key, str): key = self.get_id_from_token(key)
        try:
            neighbors,wy = zip(*value)
            weights, years = zip(*wy)
            neighbors=[self.get_id_from_token(x) if not isinstance(x,int) else x for x in neighbors]
            neighbors=np.array(neighbors)
            weights=np.array(weights)
            years = np.array(years)
            token = np.repeat(key,len(neighbors))
        except:
            raise ValueError("Adding requires an iterable over tuples e.g. [(neighbor,(weight,year))]")

        for y in np.unique(years):
            mask=years==y
            self.add_ties(token[mask],neighbors[mask],weights[mask], years[mask][0])

    def add_ties(self, token_ids,neighbors,weights,year):
        """This function adds ties, checking if nodes need to be added"""
        if self.graph is not None:
            add_ids=token_ids[~np.isin(token_ids,self.graph_ids)]
            add_neighbors = neighbors[~np.isin(neighbors, self.graph_ids)]
            add_ids=np.union1d(add_ids,add_neighbors)
            if add_ids.size > 0:
                add_token = np.array([self.get_token_from_id(np.int(x)) for x in add_ids],dtype=np.str)
                self.insert(add_token,add_ids)
            data=[{"weight":x} for x in weights]
            edges=zip(token_ids,neighbors,np.repeat(year,len(neighbors)),data)
            self.graph.add_edges_from(edges)
        # Add to neo queue
        self.add_neo_queue("Add ties")

    def insert(self, token, token_id):
        """Basic function to add a token and its token id"""
        if self.graph is not None:
            self.graph.add_nodes_from(token_id)
            self.graph_ids=np.append(self.graph_ids,token_id)
            self.graph_tokens=np.append(self.graph_tokens,token)
            self.update_graph_dicts()
        ## Add to neo queue
        # First add internally, then update queue
        self.neo_tokens=np.append(self.neo_tokens,token)
        self.neo_ids = np.append(self.neo_ids, token_id)
        self.update_neo_dicts()
        self.add_neo_queue("Add nodes")

    #%% Query functions

    def __getitem__(self, i):
        """Main function to return token information"""
        # We work with token id's here
        if isinstance(i,str): i = self.get_id_from_token(i)
        if self.graph is not None: # Graph initialized
            return self.graph[i]
        else: # Use neo4j
            # Query neo4j
            return self.neo_token_id_dict[i]

    def __len__(self):
        # Since all tokens are accessible via query, return the number of tokens in DB
        if self.graph is None:
            return len(self.neo_tokens)
        else:
            return len(self.graph_tokens)


asdf=neo4j_network(None, None)