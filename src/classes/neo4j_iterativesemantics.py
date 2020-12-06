from src.classes.neo4jnw import neo4j_network
import networkx as nx
import scipy as sp
import itertools
import logging
import numpy as np

class neo4jnw_semantic_iterator():

    def __init__(self, neo4nw, weight_cutoff=None,norm_ties=False, logging_level=logging.NOTSET):
        # Set logging level
        logging.disable(logging_level)
        # Create network
        self.neo4nw=neo4nw
        self.norm_ties=norm_ties
        self.weight_cutoff=weight_cutoff

    def condition_dispatch(self,focal_tokens, years, context=None, alter_tokens=None):

        # Input checks
        assert isinstance(years, dict) or isinstance(years, int), "Parameter years must be int or interval dict {'start':int,'end':int}"
        assert isinstance(focal_tokens, list) or isinstance(focal_tokens, int) or isinstance(focal_tokens, str), "Token parameter should be string, int or list."

        # Do conditioning

    def check_conditioning(self):
        if self.neo4nw.conditioned == True:
            return True
        else:
            logging.warning("Operation on graph requested, but network is not conditioned")
            return False

    def backout_measure(self,focal_tokens,nodelist=None):

        # Check conditioning
        assert self.check_conditioning()==True
        # Get Scipy sparse matrix
        if nodelist == None:
            G=nx.to_scipy_sparse_matrix(self.neo4nw.graph)
            n=len(self.neo4nw.graph.nodes)
            orig_index = nodelist
        else:
            G=nx.to_scipy_sparse_matrix(self.neo4nw.graph, nodelist = nodelist)
            n=len(nodelist)
            orig_nodes=np.array(self.neo4nw.graph.nodes)
            orig_index = np.where(orig_nodes==np.array(nodelist))[0]



        inv_Laplacian = sp.sparse.inv(sp.sparse.eye(n,n)-G)
        focal_rows=inv_Laplacian[]