import networkx as nx
import numpy as np
import itertools
import logging

def load_graph(year, cfg, sum_folder, network_type):
    data_folder = ''.join([cfg.data_folder, '/', str(year)])
    network_file = ''.join([data_folder, sum_folder, '/', network_type, '.gexf'])
    logging.info("Loading %s" % network_file)
    return nx.read_gexf(network_file)

def merge_nodes(graph,u,v,method="sum"):

    new_graph=graph.copy()
    # Remove both nodes
    new_graph.remove_nodes_from([u])

    in_u=[(x,z['weight']) for (x,y,z) in graph.in_edges(u,data=True) if not x==v]
    out_u=[(y,z['weight']) for (x,y,z) in graph.out_edges(u,data=True) if not y==v]

    # Merge in-edges
    for (x,z) in in_u:
        if new_graph.has_edge(x,v):
            new_graph[x][v]['weight'] = z + new_graph.get_edge_data(x,v)['weight']
        else:
            new_graph.add_edge(x,v, weight=z)

    for (x, z) in out_u:
        if new_graph.has_edge(v,x):
            new_graph[v][x]['weight'] = z + new_graph.get_edge_data(v,x)['weight']
        else:
            new_graph.add_edge(v,x, weight=z)
    # Mean
    if method=="average":
        for (x, y, wt) in graph.in_edges(v,data=True): graph[x][y]['weight'] = wt / 2
        for (x, y, wt) in graph.out_edges(v, data=True): graph[x][y]['weight'] = wt / 2

    return new_graph

def plural_elimination(graph, method="sum"):

    candidates=[x for x in graph.nodes if x[-1]=='s']
    plurals=[x for x in candidates if x[:-1] in graph.nodes]
    pairs=[(x,x[:-1]) for x in plurals]

    for (u,v) in pairs:
        graph=merge_nodes(graph, u,v,method=method)

    return graph



# TODO: Comment
def graph_merge(graph_list, average_links=True, method=None, merge_mode=None):
    """
    Merges a list of graphs under the condition that the nodes are the same.

    :param graph_list:
    :param average_links:
    :param method: "majority" - whether to prune links less than 0.5
    :param merge_mode: None - uses adjacency matrices and depends on node order, safe uses networkx logic
    :return:
    """

    if merge_mode=="safe":
        graph=graph_list[0].copy()
        for i in range(1,len(graph_list)):
            for (u, v, wt) in graph_list[i].edges.data('weight'):
                if graph.has_edge(u,v):
                    graph[u][v]['weight']=wt+graph.get_edge_data(u,v)['weight']
                else:
                    graph.add_edge(u,v,weight=wt)
        # Mean
        if average_links==True:
            for (u, v, wt) in graph.edges.data('weight'): graph[u][v]['weight']=wt/len(graph_list)
    else:
        nodes_list = list(graph_list[0].nodes)
        graph_pairs = itertools.combinations(graph_list, r=2)
        for g, h in graph_pairs:
            assert list(g.nodes) == list(h.nodes), "Graphs must have same nodes sets!"
        # Add adjacency matrices
        for i,g in enumerate(graph_list):
            if i == 0:
                A = nx.to_scipy_sparse_matrix(g)
            else:
                A = A + nx.to_scipy_sparse_matrix(g)

        if average_links==True:
            # Average
            A = A / len(graph_list)

        if method=="majority":
            A[A < 0.5] = 0

        A.eliminate_zeros()
        graph = nx.convert_matrix.from_scipy_sparse_matrix(A)
        mapping = dict(zip(range(0, len(nodes_list)), nodes_list))
        graph = nx.relabel_nodes(graph, mapping)
    return graph

