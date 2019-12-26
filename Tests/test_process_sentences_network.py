import os
import time
from NLP.src.process_sentences_network import process_sentences_network
from NLP.utils.load_bert import get_bert_and_tokenizer_local
import networkx as nx
import tables
# os.chdir('/home/ingo/PhD/BERT-NLP/BERTNLP')
import subprocess

MAX_SEQ_LENGTH = 30
batch_size = 36
method = "context_element"
cwd = os.getcwd()
text_db = os.path.join(cwd, 'NLP/data/texts.h5')
modelpath = os.path.join(cwd, 'NLP/models')
tensor_path = ''.join(['NLP/data'])
tensor_path = os.path.join(cwd, tensor_path)
cutoff_percent = 80

if __name__ == '__main__':
    start_time = time.time()

    tokenizer, bert = get_bert_and_tokenizer_local(modelpath)
    DICT_SIZE = tokenizer.vocab_size

    graph, context_graph = process_sentences_network(tokenizer, bert, text_db, MAX_SEQ_LENGTH, DICT_SIZE, batch_size,
                                                     nr_workers=0,
                                                     method=method, cutoff_percent=cutoff_percent)

    print("Relabeling and saving graphs")
    token_map = {v: k for k, v in tokenizer.vocab.items()}

    # Label nodes by token
    graph = nx.relabel_nodes(graph, token_map)
    # Take edge subgraph: Delete non-needed nodes
    graph = graph.edge_subgraph(graph.edges)
    graph_path = os.path.join(tensor_path, "".join([method,'_graph.gexf']))
    nx.write_gexf(graph, graph_path)

    # Label nodes by token
    context_graph = nx.relabel_nodes(context_graph, token_map)
    # Take edge subgraph: Delete non-needed nodes
    context_graph = context_graph.edge_subgraph(context_graph.edges)
    graph_path = os.path.join(tensor_path, "".join([method,'_Cgraph.gexf']))
    nx.write_gexf(context_graph, graph_path)
    print("Total Time: %s seconds" % (time.time() - start_time))
