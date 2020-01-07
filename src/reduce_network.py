import networkx as nx
import scipy as sp
from networkx.readwrite.gexf import read_gexf,write_gexf
from networkx.readwrite.gml import write_gml
from NLP.utils.rowvec_tools import graph_merge
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_ego_network_mem(graph,focal_node,limit=15,plot_title="Ego Network",save_folder=None, plot_screen=False):

    neighbors = graph[focal_node]
    edge_weights = [x['weight'] for x in neighbors.values()]
    edge_sort = np.argsort(-np.array(edge_weights))
    neighbors = [x for x in neighbors]

    edge_weights = np.array(edge_weights)
    neighbors = np.array(neighbors)
    edge_weights = edge_weights[edge_sort]
    neighbors = neighbors[edge_sort]

    limit=min(limit,len(neighbors))

    neighbors = neighbors[0:limit]
    edge_weights = edge_weights[0:limit]

    step = 2 * np.pi / len(edge_weights)
    coords = [np.array([0.7 * np.cos(t * step - 0.5 * np.pi), 0.7 * np.sin(t * step + 0.5 * np.pi)]) for t in
              range(0, len(edge_weights))]
    coords_label = [np.array([0.85 * np.cos(t * step - 0.5 * np.pi), 0.85 * np.sin(t * step + 0.5 * np.pi)]) for t in
                    range(0, len(edge_weights))]

    neighbors = np.append(neighbors, focal_node)
    coords.append(np.array([0, 0]))
    coords_label.append(np.array([0, 0.1]))

    pos = dict(zip(neighbors, coords))
    pos_label = dict(zip(neighbors, coords_label))

    G = nx.ego_graph(graph, focal_node)
    G = nx.subgraph(graph, neighbors)

    node_sizes = [v * 90 for v in edge_weights]
    node_sizes.append(max(node_sizes))

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    weights=np.array(weights)

    fig, ax = plt.subplots()

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, nodelist=neighbors, alpha=0.15, node_color="blue",
                           with_labels=False)
    nx.draw_networkx_edges(G, pos, alpha=0.8,width=3, edge_color=weights, edge_cmap=plt.cm.Blues)

    nx.draw_networkx_labels(G, pos_label)
    fig.set_size_inches(10, 10)
    plot_margin = 0.3

    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - plot_margin,
              y1 + plot_margin))
    plt.title(plot_title)
    if plot_screen==True:
        plt.show()
    if save_folder is not None:
        plt.savefig(save_folder)

    plt.close()


def draw_ego_network(network_file,focal_node,limit=15,plot_title="Ego Network",save_folder=None, plot_screen=False):

    graph=read_gexf(network_file)

    neighbors = graph[focal_node]
    edge_weights = [x['weight'] for x in neighbors.values()]
    edge_sort = np.argsort(-np.array(edge_weights))
    neighbors = [x for x in neighbors]

    edge_weights = np.array(edge_weights)
    neighbors = np.array(neighbors)
    edge_weights = edge_weights[edge_sort]
    neighbors = neighbors[edge_sort]

    neighbors = neighbors[0:limit]
    edge_weights = edge_weights[0:limit]

    step = 2 * np.pi / len(edge_weights)
    coords = [np.array([0.7 * np.cos(t * step - 0.5 * np.pi), 0.7 * np.sin(t * step + 0.5 * np.pi)]) for t in
              range(0, len(edge_weights))]
    coords_label = [np.array([0.85 * np.cos(t * step - 0.5 * np.pi), 0.85 * np.sin(t * step + 0.5 * np.pi)]) for t in
                    range(0, len(edge_weights))]

    neighbors = np.append(neighbors, focal_node)
    coords.append(np.array([0, 0]))
    coords_label.append(np.array([0, 0.1]))

    pos = dict(zip(neighbors, coords))
    pos_label = dict(zip(neighbors, coords_label))

    G = nx.subgraph(graph, neighbors)

    node_sizes = [v * 90 for v in edge_weights]
    node_sizes.append(graph.out_degree[focal_node]*3)

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    weights=np.array(weights)

    fig, ax = plt.subplots()

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, nodelist=neighbors, alpha=0.15, node_color="blue",
                           with_labels=False)
    nx.draw_networkx_edges(G, pos, alpha=0.8,width=3, edge_color=weights, edge_cmap=plt.cm.Blues)

    nx.draw_networkx_labels(G, pos_label)
    fig.set_size_inches(10, 10)
    plot_margin = 0.3

    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin,
              x1 + plot_margin,
              y0 - plot_margin,
              y1 + plot_margin))
    plt.title(plot_title)
    if plot_screen==True:
        plt.show()
    if save_folder is not None:
        plt.savefig(save_folder)

    plt.close()

def moving_avg_networks(years,cfg,ma_order=3,network_type="Rgraph-Sum",average_links=True):
    graphs={}
    for year in years:
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        network_file = ''.join([data_folder, cfg.sums_folder,'/', network_type, '.gexf'])
        graph = nx.read_gexf(network_file)
        graphs.update({year: graph})

    for year in years:
        lb=max(years[0],year-ma_order+1)
        graph_range=range(lb,year+1)
        merge_list=[]
        for i in graph_range:
            merge_list.append(graphs[i])
        merged_graph=graph_merge(merge_list,average_links=average_links)
        # Save yearly graph
        data_folder = ''.join([cfg.data_folder, '/', str(year)])
        network_file = ''.join([data_folder, cfg.ma_folder, '/', network_type, '.gexf'])
        graph = nx.write_gexf(merged_graph,network_file)


def reduce_network(network_file,reverse=True,method="sum",save_folder=None):

    network=read_gexf(network_file)
    if reverse==True:
        network=network.reverse()

    if method=="sum":
        nodes=list(network.nodes)
        A = nx.to_scipy_sparse_matrix(network)
        graph=nx.convert_matrix.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
        token_map = {v: nodes[v] for v in graph.nodes}
        graph = nx.relabel_nodes(graph, token_map)

    if save_folder is not None:
        logging.info("Saving graph to %s" % save_folder)
        network = write_gexf(graph, save_folder)

    return graph

