# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 01:45:25 2020

@author: marquart
"""
import networkx as nx
import scipy as sp
import itertools
import logging
import numpy as np
from scipy.sparse.linalg import inv


def backout_measure(graph, nodelist=None, decay=None, method="invert", stopping=25):
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
    
    Returns
    -------
    Graph with modified ties.

    """
    # Get Scipy sparse matrix
    if nodelist is None:
        G = nx.to_scipy_sparse_matrix(graph,   format="csc")
        n = len(graph.nodes)
        nodelist=np.array(graph.nodes)
    else:
        G = nx.to_scipy_sparse_matrix(graph, nodelist=nodelist,   format="csc")
        n = len(nodelist)

    if decay is None:
        eigenvalues, eigenvectors = sp.sparse.linalg.eigs(G)
        sp_rad = 1.5*np.max(np.abs(eigenvalues))
        decay = 1/sp_rad

    if method == "invert":
        inv_ties = inv(sp.sparse.eye(n, n, format='csc')-decay*G)
        inv_ties.setdiag(np.zeros(n))
        inv_ties = inv_ties.tocsr()
    elif method =="series":
        inv_ties=sp.sparse.csc_matrix((n,n))
        for t in range(0,stopping):
            inv_ties=inv_ties+np.power(decay,t)*G**t
        inv_ties.setdiag(np.zeros(n))
    else:
        raise NotImplementedError("Method is either invert or series.")
        
    inv_ties.eliminate_zeros()
    new_graph = nx.from_scipy_sparse_matrix(inv_ties)
    idx=list(new_graph.nodes)
    idxdict=dict(zip(idx,nodelist))
    new_graph = nx.relabel_nodes(new_graph, idxdict)
    
    return new_graph
    

