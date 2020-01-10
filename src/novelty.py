import networkx as nx
from NLP.utils.network_tools import load_graph
from NLP.utils.rowvec_tools import prune_network_edges
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

def entropy_network(focal_token, year, cfg, graph_type, save_folder=None):
    multigraph = load_graph(year, cfg, cfg.nw_folder, graph_type)
    # Prune nodes not needed
    multigraph=prune_network_edges(multigraph,0)
    # We constrain ourselves to the ego graph, given density that's still
    graph=nx.DiGraph()
    # half of all nodes with a radius of 2
    multigraph=nx.ego_graph(multigraph,focal_token,radius=2).copy()
    # We need to loop over all nodes, sadly
    logging.info("Building Entropy Graph by cycling over all nodes.")
    for i_token in tqdm(multigraph.nodes):
        links = [[{"token": token, "seq": x['seq_id'], 'pos': x['pos'], 'weight': x['weight']} for x in
                  multigraph[i_token][token].values()] for token in multigraph[i_token]]
        # I don't know how to do this faster
        links2 = []
        for x in links: links2.extend(x)
        if len(links2) > 0:
            links = pd.DataFrame(links2)
            entropy = links.groupby(['seq', 'pos']).transform(lambda x: -((x / x.sum()) * np.log((x / x.sum()))).sum())[
                'weight']
            links['entropy'] = entropy
            means = links.groupby('token',as_index=False).mean()

            tuple=[(x[0],i_token,{'weight': x[1]}) for x in means[['token','entropy']].values]
            graph.add_edges_from(tuple)

    del multigraph
    if save_folder is not None:
        logging.info("Saving graph to %s" % save_folder)
        graph = prune_network_edges(graph, 0.001)
        network = nx.write_gexf(graph, save_folder)


def novelty_test(focal_token, year, cfg, graph_type, save_folder=None):
    multigraph = load_graph(year, cfg, cfg.nw_folder, graph_type)
    links = [[{"token": token, "seq": x['seq_id'], 'pos': x['pos'], 'weight': x['weight']} for x in
              multigraph[focal_token][token].values()] for token in multigraph[focal_token]]
    links2 = []
    for x in links: links2.extend(x)
    links = pd.DataFrame(links2)
    weight_sum=links.groupby(['seq','pos']).transform(lambda x: x.sum())['weight']
    links_tr=links.groupby(['seq','pos']).transform(lambda x: (x / x.sum()))
    entropy=links.groupby(['seq','pos']).transform(lambda x: -((x/x.sum()) * np.log((x/x.sum()))).sum())['weight']

    weight_sum=pd.to_numeric(weight_sum, errors='coerce')
    seq_length = links.groupby(['seq', 'pos']).transform(lambda x: (x.count()))['token']
    links['norm_weight']=links_tr
    links['total_mass'] = weight_sum
    links['seq_length'] = seq_length
    links['entropy']=entropy
    means=links.groupby('token').mean()


    token_g=links.groupby(['token'])
    occurrences=token_g.size()
    means=links.groupby('token').mean().sort_values('norm_weight', ascending=False)
    means['nr_pred']=occurrences
    sums=token_g.sum().sort_values('weight', ascending=False)
    sums['nr_pred']=occurrences


    filename = ''.join([cfg.cluster_xls, '/', focal_token, '_', graph_type, '_noveltyBeReplacedM.csv'])
    means.to_csv(filename)

    filename = ''.join([cfg.cluster_xls, '/', focal_token, '_', graph_type, '_noveltyBeReplacedS.csv'])
    sums.to_csv(filename)


    unique_sequences = links['seq'].unique()
    y = [len(links[links['seq'] == x]) for x in unique_sequences]

    multigraph = multigraph.reverse()
    neighbor_list = list(nx.neighbors(multigraph, focal_token))
    neighbor_list.append(focal_token)
    multigraph1 = nx.subgraph(multigraph, neighbor_list).copy()
    multigraph2 = nx.ego_graph(multigraph, focal_token, 1).copy()
