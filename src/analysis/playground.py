# %% Config

import networkx as nx
import numpy as np
import pandas as pd

from config.config import configuration
from src.neo4j_network import neo4j_network

cfg = configuration()

db_uri = "http://localhost:7474"
db_pwd = ('neo4j', 'esmt')
neo_creds = (db_uri, db_pwd)
neograph = neo4j_network(neo_creds, graph_direction="REVERSE")
neograph2 = neo4j_network(neo_creds, graph_direction="FORWARD")

focal_tokens = ["leader"]

for focal_token in focal_tokens:
    # %% One step ego network
    neograph.condition(2019, [focal_token], weight_cutoff=None, depth=0)
    neograph2.condition(2019, [focal_token], weight_cutoff=None, depth=0)
    nodes_fw = list(neograph.graph.nodes).copy()
    nodes_bw = list(neograph2.graph.nodes).copy()
    nodes = np.union1d(nodes_fw, nodes_bw)
    newgraph = nx.Graph()
    newgraph.add_nodes_from(nodes)
    neograph.decondition()
    neograph2.decondition()
    neograph.condition(2019, [focal_token], weight_cutoff=None, depth=1)
    neograph2.condition(2019, [focal_token], weight_cutoff=None, depth=1)
    for token_id in nodes:
        idx = token_id
        if idx in nodes_fw:
            adj = neograph.graph[idx].copy()
        else:
            adj = []
        if idx in nodes_bw:
            adj2 = neograph2.graph[idx].copy()
        else:
            adj2 = []

        for alter in np.setdiff1d(nodes, idx):
            if alter in adj:
                w1 = adj[alter][2019]['weight']
            else:
                w1 = 0
            if alter in adj2:
                w2 = adj2[alter][2019]['weight']
            else:
                w2 = 0
            w = (w1 + w2) / 2
            if w > 0:
                newgraph.add_edge(idx, alter, weight=w)
                newgraph[alter][idx]['weight'] = w
    a = list(newgraph.nodes)
    labeldict = dict(zip(a, [neograph.get_token_from_id(int(x)) for x in a]))
    labeldict_rev = dict(zip([neograph.get_token_from_id(int(x)) for x in a], a))
    newgraph = nx.relabel_nodes(newgraph, labeldict)
    nx.write_gexf(newgraph, "".join([cfg.nw_folder, "/", focal_token, "_", "0-depth.gexf"]))
    newgraph = nx.relabel_nodes(newgraph, labeldict_rev)
    # %% Normed version of graph
    newdigraph = nx.DiGraph()
    newdigraph.add_nodes_from(newgraph.nodes)
    for idx in newgraph.nodes:
        adj = newgraph[idx]
        weights = np.array([adj[x]['weight'] for x in adj])
        norm_weights = weights / np.sum(weights)
        alters = [x for x in adj]
        for i, a in enumerate(alters):
            newdigraph.add_edge(idx, a, weight=norm_weights[i])
    a = list(newdigraph.nodes)
    labeldict = dict(zip(a, [neograph.get_token_from_id(int(x)) for x in a]))
    labeldict_rev = dict(zip([neograph.get_token_from_id(int(x)) for x in a], a))
    newdigraph = nx.relabel_nodes(newdigraph, labeldict)
    nx.write_gexf(newdigraph, "".join([cfg.nw_folder, "/", focal_token, "_", "0-depth-normed.gexf"]))
    newdigraph = nx.relabel_nodes(newdigraph, labeldict_rev)

    # %% Two step ego network
    neograph.decondition()
    neograph2.decondition()
    neograph = neo4j_network(neo_creds, graph_direction="REVERSE")
    neograph2 = neo4j_network(neo_creds, graph_direction="FORWARD")
    neograph.condition(1, [focal_token], weight_cutoff=0.1, depth=1)
    neograph2.condition(1, [focal_token], weight_cutoff=0.1, depth=1)
    nodes = np.union1d(neograph.graph.nodes, neograph2.graph.nodes)
    newgraph = nx.Graph()
    newgraph.add_nodes_from(nodes)
    for token_id in nodes:
        idx = token_id
        if idx in neograph.graph.nodes:
            adj = neograph.graph[idx].copy()
        else:
            adj = []
        if idx in neograph2.graph.nodes:
            adj2 = neograph2.graph[idx].copy()
        else:
            adj2 = []
        token_id1 = [x for x in adj]
        token_id2 = [x for x in adj2]
        ints = np.intersect1d(token_id1, token_id2)
        for alter in ints:
            w1 = adj[alter][1]['weight']
            w2 = adj2[alter][1]['weight']
            w = (w1 + w2) / 2
            newgraph.add_edge(idx, alter, weight=w)
            newgraph[alter][idx]['weight'] = w
        missing = np.setdiff1d(token_id2, token_id1)
        single = np.setdiff1d(token_id1, token_id2)
        for alter in missing:
            w1 = 0
            w2 = adj2[alter][1]['weight']
            w = (w1 + w2) / 2
            newgraph.add_edge(token_id, alter, weight=w)
            newgraph[alter][idx]['weight'] = w
        for alter in single:
            w1 = 0
            w2 = adj[alter][1]['weight']
            w = (w1 + w2) / 2
            newgraph.add_edge(token_id, alter, weight=w)
            newgraph[alter][idx]['weight'] = w
    a = list(newgraph.nodes)
    labeldict = dict(zip(a, [neograph.get_token_from_id(int(x)) for x in a]))
    labeldict_rev = dict(zip([neograph.get_token_from_id(int(x)) for x in a], a))
    newgraph = nx.relabel_nodes(newgraph, labeldict)
    nx.write_gexf(newgraph, "".join([cfg.nw_folder, "/", focal_token, "_", "1-depth.gexf"]))
    newgraph = nx.relabel_nodes(newgraph, labeldict_rev)
    # %% Normed version of 2-step graph
    newdigraph = nx.DiGraph()
    newdigraph.add_nodes_from(newgraph.nodes)
    for idx in newgraph.nodes:
        adj = newgraph[idx]
        weights = np.array([adj[x]['weight'] for x in adj])
        norm_weights = weights / np.sum(weights)
        alters = [x for x in adj]
        for i, a in enumerate(alters):
            newdigraph.add_edge(idx, a, weight=norm_weights[i])
    a = list(newdigraph.nodes)
    labeldict = dict(zip(a, [neograph.get_token_from_id(int(x)) for x in a]))
    labeldict_rev = dict(zip([neograph.get_token_from_id(int(x)) for x in a], a))
    newdigraph = nx.relabel_nodes(newdigraph, labeldict)
    nx.write_gexf(newdigraph, "".join([cfg.nw_folder, "/", focal_token, "_", "1-depth-normed.gexf"]))
    newdigraph = nx.relabel_nodes(newdigraph, labeldict_rev)
    neograph.decondition()
    neograph2.decondition()

    # %% CONTEXT
    # %% One step ego network
    neograph.condition_context(1, [focal_token], weight_cutoff=0.1, depth=0)
    neograph2.condition_context(1, [focal_token], weight_cutoff=0.1, depth=0, context_direction="REVERSE")
    nodes_fw = list(neograph.graph.nodes).copy()
    nodes_bw = list(neograph2.graph.nodes).copy()
    nodes = np.union1d(nodes_fw, nodes_bw)
    newgraph = nx.Graph()
    newgraph.add_nodes_from(nodes)
    neograph.decondition()
    neograph2.decondition()
    neograph.condition_context(1, [focal_token], weight_cutoff=0.1, depth=1)
    neograph2.condition_context(1, [focal_token], weight_cutoff=0.1, depth=1, context_direction="REVERSE")
    for token_id in nodes:
        idx = token_id
        if idx in nodes_fw:
            adj = neograph.graph[idx].copy()
        else:
            adj = []
        if idx in nodes_bw:
            adj2 = neograph2.graph[idx].copy()
        else:
            adj2 = []

        for alter in np.setdiff1d(nodes, idx):
            if alter in adj:
                w1 = adj[alter][1]['weight']
            else:
                w1 = 0
            if alter in adj2:
                w2 = adj2[alter][1]['weight']
            else:
                w2 = 0
            w = (w1 + w2) / 2
            if w > 0:
                newgraph.add_edge(idx, alter, weight=w)
                newgraph[alter][idx]['weight'] = w
    a = list(newgraph.nodes)
    labeldict = dict(zip(a, [neograph.get_token_from_id(int(x)) for x in a]))
    labeldict_rev = dict(zip([neograph.get_token_from_id(int(x)) for x in a], a))
    newgraph = nx.relabel_nodes(newgraph, labeldict)
    nx.write_gexf(newgraph, "".join([cfg.nw_folder, "/", focal_token, "_", "c0-depth.gexf"]))
    newgraph = nx.relabel_nodes(newgraph, labeldict_rev)
    # %% Normed version of graph
    newdigraph = nx.DiGraph()
    newdigraph.add_nodes_from(newgraph.nodes)
    for idx in newgraph.nodes:
        adj = newgraph[idx]
        weights = np.array([adj[x]['weight'] for x in adj])
        norm_weights = weights / np.sum(weights)
        alters = [x for x in adj]
        for i, a in enumerate(alters):
            newdigraph.add_edge(idx, a, weight=norm_weights[i])
    a = list(newdigraph.nodes)
    labeldict = dict(zip(a, [neograph.get_token_from_id(int(x)) for x in a]))
    labeldict_rev = dict(zip([neograph.get_token_from_id(int(x)) for x in a], a))
    newdigraph = nx.relabel_nodes(newdigraph, labeldict)
    nx.write_gexf(newdigraph, "".join([cfg.nw_folder, "/", focal_token, "_", "c0-depth-normed.gexf"]))
    newdigraph = nx.relabel_nodes(newdigraph, labeldict_rev)
    neograph.decondition()
    neograph2.decondition()

id_player = neograph.ensure_ids(w2)
id_coach = neograph.ensure_ids(w3)

id_list = [id_leader, id_player, id_coach]
nb_df = {}
for idx in id_list:
    node_dict = neograph.graph[idx]
    neighbors = list(node_dict)
    tp = [(n, neograph.ensure_tokens(n), node_dict[n][1]['weight']) for n in neighbors]
    df = pd.DataFrame(tp, columns=['id', 'token', 'proximity']).sort_values('proximity', ascending=False)
    # df.set_index("id", inplace=True)
    # df.set_index("token", inplace=True)
    df['nprox'] = df.proximity / sum(df.proximity)
    nb_df.update({idx: df})

# overlap
overlap_leader_player = pd.merge(nb_df[id_leader], nb_df[id_player], on=["id", "token"],
                                 suffixes=("leader", "player"), how="outer")
overlap_leader_coach = pd.merge(nb_df[id_leader], nb_df[id_coach], on=["id", "token"], suffixes=("leader", "coach"),
                                how="outer")
overlap_leader_player = overlap_leader_player.fillna(0)
overlap_leader_coach = overlap_leader_coach.fillna(0)
hel_leader_player = np.sum(
    np.sqrt(np.array(np.multiply(overlap_leader_player.nproxleader, overlap_leader_player.nproxplayer))))
hel_leader_coach = np.sum(
    np.sqrt(np.array(np.multiply(overlap_leader_coach.nproxleader, overlap_leader_coach.nproxcoach))))

year_list.append(year)
hel_c_list.append(hel_leader_coach)
hel_p_list.append(hel_leader_player)

overlap_leader_coach = pd.merge(nb_df[id_leader], nb_df[id_coach], on=["id", "token"], suffixes=("leader", "coach"))

neograph.decondition()

results = pd.DataFrame({'year': year_list, 'hel_coach': hel_c_list, 'hel_player': hel_p_list})
results.to_excel("E:/NLP/cluster_xls/coca.xlsx", sheet_name='distances')

db_uri = "http://localhost:7474"
db_pwd = ('neo4j', 'nlp')
neo_creds = (db_uri, db_pwd)
neograph = neo4j_network(neo_creds, graph_direction="REVERSE")

y_start = int(''.join([str(years[0]), "0101"]))
y_end = int(''.join([str(years[-1]), "0101"]))
year_int = {'start': y_start, 'end': y_end}
year_var = int((year_int['end'] + year_int['start']) / 2)
w1 = 'leader'
w2 = 'ceo'
w3 = 'coach'
neograph.condition(year_int, [w1, w2, w3], weight_cutoff=0.001)
id_leader = neograph.ensure_ids(w1)
id_player = neograph.ensure_ids(w2)
id_coach = neograph.ensure_ids(w3)

id_list = [id_leader, id_player, id_coach]
nb_df = {}
for idx in id_list:
    node_dict = neograph.graph[idx]
    neighbors = list(node_dict)
    tp = [(n, neograph.ensure_tokens(n), node_dict[n][year_var]['weight']) for n in neighbors]
    df = pd.DataFrame(tp, columns=['id', 'token', 'proximity']).sort_values('proximity', ascending=False)
    # df.set_index("id", inplace=True)
    # df.set_index("token", inplace=True)
    df['nprox'] = df.proximity / sum(df.proximity)
    nb_df.update({idx: df})
neograph.decondition()

coach1 = nb_df[id_coach].copy()
coach2 = nb_df[id_coach].copy()
leader = nb_df[id_leader].copy()

coach1.to_excel("E:/NLP/cluster_xls/coach_hbr.xlsx", sheet_name='distances')
leader.to_excel("E:/NLP/cluster_xls/leader_hbr.xlsx", sheet_name='distances')
