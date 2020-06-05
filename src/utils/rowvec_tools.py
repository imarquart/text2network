import numpy as np
from src.utils.delwords import create_stopword_strings


def inverse_edge_weight(u, v, d):
    edge_wt = d.get('weight', 1)
    if edge_wt > 0.01:
        return 1 / edge_wt
    else:
        return 1000000


def prune_network_edges(graph, edge_weight=0.2):
    remove = [(a,b) for a,b,c in graph.edges.data() if c['weight'] <= edge_weight]
    remove_nodes = [x for x in graph.nodes if x.isdigit() == True]
    delwords=list(create_stopword_strings())
    remove_nodes = list(np.union1d(delwords, remove_nodes))
    try:
        graph.remove_edges_from(remove)
        graph.remove_nodes_from(remove_nodes)
        return graph
    except:
        graph_c=graph.copy()
        graph_c.remove_edges_from(remove)
        graph_c.remove_nodes_from(remove_nodes)
        return graph_c


def add_to_networks(graph, context_graph, attention_graph, replacement, context, context_att,token=0, cutoff_percent=80, max_degree=100, pos=0, sequence_id=0):
    # Create Adjacency List for Replacement Dist

    cutoff_number, cutoff_probability = calculate_cutoffs(replacement, method="percent",
                                                         percent=cutoff_percent,max_degree=max_degree)
    graph.add_weighted_edges_from(
        get_weighted_edgelist(token, replacement, cutoff_number, cutoff_probability), 'weight',
        seq_id=sequence_id, pos=pos)



    #Create Adjacency List for Context Dist
    cutoff_number, cutoff_probability = calculate_cutoffs(context, method="percent",
                                                         percent=cutoff_percent-15)
    context_graph.add_weighted_edges_from(
        get_weighted_edgelist(token, context, cutoff_number-15, cutoff_probability), 'weight',
       seq_id=sequence_id, pos=pos)

    # DISABLED
    # Create Adjacency List for Attention Dist
    #cutoff_number, cutoff_probability = calculate_cutoffs(context_att, method="percent",
    #                                                      percent=cutoff_percent,max_degree=max_degree)
    #attention_graph.add_weighted_edges_from(
    #    get_weighted_edgelist(token, context_att, cutoff_number, cutoff_probability), 'weight',
    #    seq_id=sequence_id, pos=pos)

    return graph, context_graph,attention_graph

def calculate_cutoffs(x, method="mean", percent=100, max_degree=100,min_cut=0.0005):
    """
    Different methods to calculate cutoff probability and number.

    :param x: Contextual vector
    :param method: mean: Only accept entries above the mean; percent: Take the k biggest elements that explain X% of mass.
    :return: cutoff_number and probability
    """
    if method == "mean":
        cutoff_probability = max(np.mean(x), min_cut)
        cutoff_number = max(np.int(len(x) / 100), 100)
    elif method == "percent":
        sortx = np.sort(x)[::-1]
        cum_sum = np.cumsum(sortx)
        cutoff = cum_sum[-1] * percent/100
        cutoff_number = np.where(cum_sum >= cutoff)[0][0]
        cutoff_probability = min_cut
    else:
        cutoff_probability = min_cut
        cutoff_number = 0

    return min(cutoff_number,max_degree), cutoff_probability


def get_weighted_edgelist(token, x, cutoff_number=100, cutoff_probability=0):
    """
    Sort probability distribution to get the most likely neighbor nodes.
    Return a networksx weighted edge list for a given focal token as node.

    :param token: Numerical, token which to add
    :param x: Probability distribution
    :param cutoff_number: Number of neighbor token to consider. Not used if 0.
    :param cutoff_probability: Lowest probability to consider. Not used if 0.
    :return: List of tuples compatible with networkx
    """
    # Get the most pertinent words
    if cutoff_number > 0:
        neighbors = np.argsort(-x)[:cutoff_number]
    else:
        neighbors = np.argsort(-x)[:]

    # Cutoff probability (zeros)
    if len(neighbors > 0):
        if cutoff_probability>0:
            neighbors = neighbors[x[neighbors] > cutoff_probability]

        weights = x[neighbors]
        # edgelist = [(token, x) for x in neighbors]
    return [(token, x[0], x[1]) for x in list(zip(neighbors, weights))]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x - np.max(x))
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def simple_norm(x):
    """Just want to start at zero and sum to 1, without norming anything else"""
    x_org=x
    x = x - np.min(x, axis=-1)
    if np.sum(x, axis=-1) > 0:
        return x / np.sum(x, axis=-1)
    else:
        return x_org


