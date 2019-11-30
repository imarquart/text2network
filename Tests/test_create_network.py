
import os
#os.chdir(('/home/ingo/PhD/BERT-NLP'))

import time
import networkx as nx
from NLP.Experiments.create_network import create_network
from NLP.utils.load_bert import get_bert_and_tokenizer


cwd= os.getcwd()
data_path=os.path.join(cwd, 'NLP/data/')
database=os.path.join(cwd,'NLP/data/tensor_db_conelem.h5')
modelpath=os.path.join(cwd,'NLP/models')
MAX_SEQ_LENGTH=30
tokenizer, _ = get_bert_and_tokenizer(modelpath)
start_token="manager"




#print("#############")
#print("BATCH SIZE 1")
#start_time = time.time()
#graphs=create_network(database,tokenizer,start_token,nr_clusters=2,batch_size=1)
#print("--- %s seconds ---" % (time.time() - start_time))

print("#############")
print("BATCH SIZE 5")
start_time = time.time()
graphs,context_graphs=create_network(database,tokenizer,start_token,nr_clusters=2,batch_size=5)
print("--- %s seconds ---" % (time.time() - start_time))


#tokenizer.convert_ids_to_tokens(range)

token_map={v: k for k, v in tokenizer.vocab.items()}


for idx,graph in enumerate(graphs):
    # Label nodes by token
    graph=nx.relabel_nodes(graph,token_map)
    # Take edge subgraph: Delete non-needed nodes
    graph=graph.edge_subgraph(graph.edges)
    graph_path=os.path.join(data_path,"".join([start_token,"_graph_",str(idx),'.gexf']))
    nx.write_gexf(graph,graph_path)

for idx,graph in enumerate(context_graphs):
    # Label nodes by token
    graph=nx.relabel_nodes(graph,token_map)
    # Take edge subgraph: Delete non-needed nodes
    graph=graph.edge_subgraph(graph.edges)
    graph_path=os.path.join(data_path,"".join([start_token,"_Cgraph_",str(idx),'.gexf']))
    nx.write_gexf(graph,graph_path)
