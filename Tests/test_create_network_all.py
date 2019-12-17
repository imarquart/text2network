import os
# os.chdir(('/home/ingo/PhD/BERT-NLP'))

import time
import networkx as nx
from NLP.src.create_network_all import create_network_all
from NLP.utils.load_bert import get_bert_and_tokenizer

cwd = os.getcwd()
data_path = os.path.join(cwd, 'NLP/data/')
database = os.path.join(cwd, 'NLP/data/tensor_db_context_element.h5')
modelpath = os.path.join(cwd, 'NLP/models')
MAX_SEQ_LENGTH = 30
tokenizer, _ = get_bert_and_tokenizer(modelpath)
start_token = "manager"

batch_size = [1, 10, 100]

for bs in batch_size:
    print("#############")
    print("BATCH SIZE %i, Single access" % bs)
    start_time = time.time()
    graph, context_graph = create_network_all(database, tokenizer, start_token, nr_clusters=2, batch_size=bs,
                                              dset_method="single")
    print("--- %s seconds ---" % (time.time() - start_time))

token_map = {v: k for k, v in tokenizer.vocab.items()}

# Label nodes by token
graph = nx.relabel_nodes(graph, token_map)
# Take edge subgraph: Delete non-needed nodes
graph = graph.edge_subgraph(graph.edges)
graph_path = os.path.join(data_path, "".join([start_token, '_graph.gexf']))
nx.write_gexf(graph, graph_path)

# Label nodes by token
context_graph = nx.relabel_nodes(context_graph, token_map)
# Take edge subgraph: Delete non-needed nodes
context_graph = context_graph.edge_subgraph(context_graph.edges)
graph_path = os.path.join(data_path, "".join([start_token, '_Cgraph.gexf']))
nx.write_gexf(context_graph, graph_path)
