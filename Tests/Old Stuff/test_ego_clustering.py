
import os
#os.chdir(('/home/ingo/PhD/BERT-NLP'))

import time
import networkx as nx
from NLP.src.ego_clustering import create_clusters
from NLP.utils.load_bert import get_bert_and_tokenizer


cwd= os.getcwd()
data_path=os.path.join(cwd, 'NLP/data/')
#database=os.path.join(cwd,'NLP/data/tensor_db_attention.h5')
database=os.path.join(cwd,'NLP/data/tensor_db_context_element.h5')

modelpath=os.path.join(cwd,'NLP/models')
MAX_SEQ_LENGTH=30
tokenizer, _ = get_bert_and_tokenizer(modelpath)
start_token="leader"
nr_clusters=2

create_clusters(database,tokenizer,start_token,nr_clusters=nr_clusters,cutoff=0.9999)