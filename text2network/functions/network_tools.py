import itertools
from typing import Union

import networkx as nx
import scipy as sp
import numpy as np

from text2network.functions.rowvec_tools import cutoff_percentage


def make_reverse(graph):
    """
    Reverses a networkx graph

    Parameters
    ----------
    graph : networkx graph
        A directed, weighted graph.

    Returns
    -------
    graph : networkx graph
        graph with modified edges.

    """

    if nx.is_directed(graph):
        return graph.reverse()
    else:
        return graph

def make_symmetric(graph, technique="avg-sym"):
    """
    Make a networkx graph symmetric

    Parameters
    ----------
    graph : networkx graph
        A directed, weighted graph.
    technique : TYPE, optional
        transpose: Transpose and average adjacency matrix. Note: Loses other edge parameters!
        min-sym: Retain minimum direction i.e. no tie if zero / unidirectional.
        max-sym: Retain maximum direction; tie exists even if unidirectional.
        avg-sym: Average ties. 
        min-sym-avg: Average ties if link is bidirectional, otherwise no tie.
        The default is "avg-sym".

    Returns
    -------
    graph : networkx graph
        graph with modified edges.

    """
    if technique == "transpose":
        M = nx.to_scipy_sparse_matrix(graph)
        nodes_list = list(graph.nodes)
        M = (M + M.T) / 2 - sp.sparse.diags(M.diagonal(), dtype=int)
        graph = nx.convert_matrix.from_scipy_sparse_matrix(M)
        mapping = dict(zip(range(0, len(nodes_list)), nodes_list))
        new_graph = nx.relabel_nodes(graph, mapping)
    elif technique == "min-sym-avg":
        new_graph = graph.to_undirected()
        nodepairs = itertools.combinations(list(graph.nodes), r=2)
        for u, v in nodepairs:
            if graph.has_edge(u, v) and graph.has_edge(v, u):
                avg_weight = (
                                     graph.edges[u, v]['weight'] + graph.edges[v, u]['weight']) / 2
                new_graph[u][v]['weight'] = avg_weight
            else:
                if graph.has_edge(u, v) or graph.has_edge(v, u):
                    new_graph.remove_edge(u, v)
    elif technique == "min-sym":
        new_graph = graph.to_undirected()
        nodepairs = itertools.combinations(list(graph.nodes), r=2)
        for u, v in nodepairs:
            if graph.has_edge(u, v) and graph.has_edge(v, u):
                min_weight = min(
                    graph.edges[u, v]['weight'], graph.edges[v, u]['weight'])
                new_graph[u][v]['weight'] = min_weight
            else:
                if graph.has_edge(u, v) or graph.has_edge(v, u):
                    new_graph.remove_edge(u, v)
    elif technique == "max-sym":
        new_graph = graph.to_undirected()
        nodepairs = itertools.combinations(list(graph.nodes), r=2)
        for u, v in nodepairs:
            if graph.has_edge(u, v) and graph.has_edge(v, u):
                max_weight = max(
                    graph.edges[u, v]['weight'], graph.edges[v, u]['weight'])
                new_graph[u][v]['weight'] = max_weight
            else:
                if graph.has_edge(u, v) or graph.has_edge(v, u):
                    new_graph.remove_edge(u, v)
    elif technique == "avg-sym":
        new_graph = graph.to_undirected()
        nodepairs = itertools.combinations(list(graph.nodes), r=2)
        for u, v in nodepairs:
            if graph.has_edge(u, v) or graph.has_edge(v, u):
                wt = 0

                if graph.has_edge(u, v):
                    wt = wt + graph.edges[u, v]['weight']

                if graph.has_edge(v, u):
                    wt = wt + graph.edges[v, u]['weight']

                wt = wt / 2
                new_graph[u][v]['weight'] = wt
    elif technique == "sum":
        new_graph = graph.to_undirected()
        nodepairs = itertools.combinations(list(graph.nodes), r=2)
        for u, v in nodepairs:
            if graph.has_edge(u, v) or graph.has_edge(v, u):
                wt = 0

                if graph.has_edge(u, v):
                    wt = wt + graph.edges[u, v]['weight']

                if graph.has_edge(v, u):
                    wt = wt + graph.edges[v, u]['weight']

                new_graph[u][v]['weight'] = wt
    else:
        raise AttributeError("Method parameter not recognized")

    return new_graph


def merge_nodes(graph, u, v, method="sum"):
    new_graph = graph.copy()
    # Remove both nodes
    new_graph.remove_nodes_from([u])

    in_u = [(x, z['weight'])
            for (x, y, z) in graph.in_edges(u, data=True) if not x == v]
    out_u = [(y, z['weight'])
             for (x, y, z) in graph.out_edges(u, data=True) if not y == v]

    # Merge in-edges
    for (x, z) in in_u:
        if new_graph.has_edge(x, v):
            new_graph[x][v]['weight'] = z + \
                                        new_graph.get_edge_data(x, v)['weight']
        else:
            new_graph.add_edge(x, v, weight=z)

    for (x, z) in out_u:
        if new_graph.has_edge(v, x):
            new_graph[v][x]['weight'] = z + \
                                        new_graph.get_edge_data(v, x)['weight']
        else:
            new_graph.add_edge(v, x, weight=z)
    # Mean
    if method == "average":
        for (x, y, wt) in graph.in_edges(v, data=True):
            graph[x][y]['weight'] = wt / 2
        for (x, y, wt) in graph.out_edges(v, data=True):
            graph[x][y]['weight'] = wt / 2

    return new_graph


def renorm_graph(graph:Union[nx.DiGraph,nx.Graph], norm:Union[float,int]=1):
    """

    Simply divides all ties in the graph by the given number

    Parameters
    ----------
    graph
    norm

    Returns
    -------
    normed graph
    """

    for u, v, a in graph.edges(data=True):
        graph[u][v]['weight']=a['weight']/norm

    return graph


def sparsify_graph(graph:Union[nx.DiGraph,nx.Graph], percentage:int=99):
    """
    Sparsify graph as follows:
    For each node, keep *percentage* of aggregate tie weights of outgoing ties.

    Parameters
    ----------
    graph
    percentage

    Returns
    -------
    sparsified graph
    """
    for v in graph.nodes:
        peers=np.array([z[1] for z in graph.out_edges(v, data="weight")])
        weights = np.array([z[2] for z in graph.out_edges(v, data="weight")])
        if len(peers)>0:
            weights=cutoff_percentage(weights,percentage)
            for u,wt in zip(peers,weights):
                if wt>0:
                    graph[v][u]['weight']=wt
                else:
                    graph.remove_edge(v,u)
    return graph






def plural_elimination(graph, method="sum"):
    candidates = [x for x in graph.nodes if x[-1] == 's']
    plurals = [x for x in candidates if x[:-1] in graph.nodes]
    pairs = [(x, x[:-1]) for x in plurals]

    for (u, v) in pairs:
        graph = merge_nodes(graph, u, v, method=method)

    return graph
