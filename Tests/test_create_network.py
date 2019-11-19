
import os
import time
import networkx as nx
from NLP.Experiments.create_network import create_network
from NLP.utils.load_bert import get_bert_and_tokenizer

start_time = time.time()
cwd= os.getcwd()
data_path=os.path.join(cwd, 'NLP/data/')
database=os.path.join(cwd,'NLP/data/tensors.h5')
modelpath=os.path.join(cwd,'NLP/models')
MAX_SEQ_LENGTH=30
tokenizer, _ = get_bert_and_tokenizer(modelpath)
start_token="manager"

graphs=create_network(database,tokenizer,start_token,4)

#tokenizer.convert_ids_to_tokens(range)

token_map={v: k for k, v in tokenizer.vocab.items()}


for idx,graph in enumerate(graphs):
    graph=nx.relabel_nodes(graph,token_map)
    graph_path=os.path.join(data_path,"".join([start_token,"_graph_",str(idx),'.gexf']))
    nx.write_gexf(graph,graph_path)

print("--- %s seconds ---" % (time.time() - start_time))