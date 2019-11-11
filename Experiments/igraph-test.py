import igraph
import numpy as np
import tables
import os
from utils.load_bert import get_bert_and_tokenizer

os.chdir('/home/ingo/PhD/BERT-NLP/BERTNLP')
cwd= os.getcwd()
database=os.path.join(cwd,'data/tensor_db.h5')
modelpath=os.path.join(cwd,'models')

try:
    data_file = tables.open_file(database, mode="r", title="Data File")
except:
    print("ERROR")

try:
    token_table = data_file.root.token_data.table
except:
    print("ERROR")

tokenizer, bert = get_bert_and_tokenizer(modelpath)


nodes=np.unique(token_table.col('token_id'))
nr_nodes=nodes.shape[0]
g = igraph.Graph()
g.add_vertices(nr_nodes)

stopwords=[',', '.', 'and', '-', 'the','##d', '...', 'that', 'to', 'as', 'for', '"', 'in', "'", 'a', 'of', 'only', ':', 'so', 'all', 'one', 'it', 'then', 'also', 'with', 'but', 'by', 'on', 'just', 'like', 'again', ';', 'more', 'this', 'not', 'is', 'there', 'was', 'even', 'still', 'after', 'here', 'later', '!', 'over', 'from', 'i', 'or', '?', 'at', 'first', '##s', 'while', ')', 'before', 'when', 'once', 'too',  'out', 'yet', 'because', 'some', 'though', 'had',  'instead', 'always', '(', 'well', 'back', 'tonight', 'since', 'about', 'through', 'will', 'them', 'left', 'often', 'what', 'ever', 'until',   'sometimes', 'if', 'however', 'finally', 'another', 'somehow', 'everything', 'further', 'really', 'last', 'an', '/', 'rather','s',  'may', 'be', 'each', 'thus', 'almost', 'where', 'anyway', 'their', 'has',  'something',  'already', 'within', 'any', 'indeed', '##a', '[UNK]', '~',  'every', 'meanwhile', 'would', '##e', 'have','nevertheless', 'which','how', '1', 'are', 'either', 'along', 'thereafter',  'otherwise', 'did',  'quite', 'these', 'can', '2', 'its', 'merely', 'actually', 'certainly',  '3', 'else','upon', 'except',  'those',  'especially',  'therefore','beside',   'apparently', 'besides', 'third', 'whilst',  '*', 'although', 'were','likewise', 'mainly', 'four', 'seven', 'into',  'm',  ']', 'than', 't', 'surely', '|',  '#',   'till', '##ly',  '_',  'al']
stopwords_ids=tokenizer.convert_tokens_to_ids(stopwords)
for token in nodes:
    query="".join(['token_id==',str(token)])
    rows=token_table.read_where(query)
    nr_rows=len(rows)
    context_dists=np.stack([x[0] for x in rows],axis=0).squeeze()
    context_dists[context_dists <= 0] = 0
    context_dists[:, stopwords_ids] = 0
    sm_context_dists=sum(context_dists)/nr_rows
    spred = np.argsort(-sm_context_dists)[:50]
    asdf=tokenizer.convert_ids_to_tokens(spred)

    own_dists=np.stack([x[1] for x in rows],axis=0).squeeze()
    own_dists[own_dists <= 0] = 0
    own_dists[:, stopwords_ids] = 0
    sm_own_dists=sum(own_dists)/nr_rows
    spred = np.argsort(-sm_own_dists)[:50]
    asdf2=tokenizer.convert_ids_to_tokens(spred)

data_file.close()