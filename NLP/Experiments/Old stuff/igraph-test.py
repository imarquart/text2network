
import torch
import numpy as np
import tables
import networkx as nx
import os
from utils.load_bert import get_bert_and_tokenizer
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from numpy import inf
from sklearn.cluster import KMeans

def hel_divergence(p,q):
    diff=np.sum( np.sqrt( p*q),axis=1 )
    diff[diff == 0] = 0.00001
    return np.sqrt( 1-diff )

def bc_divergence(p,q):
    diff=np.sum( np.sqrt( p*q),axis=1 )
    diff[diff == 0] = 0.00001
    return - np.log( diff )

def kl_divergence(p, q):
    index=(p != 0).squeeze()
    mino=np.min(q[q>0])/10
    return np.sum(-p[:,index]*np.log(np.where(q[:,index]/p[:,index]!=0,q[:,index]/p[:,index],mino)),axis=1)

def softmax_sparse(x):
    """Compute softmax values for each sets of scores in x."""

    # Throw away negative scores
    x[x <= 0] = 0
    # Apply softmax
    x = np.exp(x - np.max(x))
    x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    # Sparsify: Set mins to zero
    maxmin = np.max(np.min(x, axis=1))
    x[x <= maxmin] = 0
    # Re-normalize
    return normalize(x,norm='l1',copy=False)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.exp(x - np.max(x))
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def simple_norm(x):
    """Just want to start at zero and sum to 1, without norming anything else"""
    x=x-np.min(x,axis=-1)
    return x/np.sum(x,axis=-1)


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

all_nodes=np.unique(token_table.col('token_id'))
all_nodes_str=[str(x) for x in all_nodes]
nr_nodes=all_nodes.shape[0]



g = nx.DiGraph()
g.add_nodes_from(range(1,tokenizer.vocab_size))

alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
alphabet_ids=tokenizer.convert_tokens_to_ids(alphabet)
numbers = ['1','2','3','4','5','6','7','8','9','10','one','two','three','four']
numbers_ids=tokenizer.convert_tokens_to_ids(numbers)
pronouns=['we', 'us','my','yourself','you','me','he','her','his','him','she','they','their','them','me','myself','himself','herself','themselves']
pronouns_ids=tokenizer.convert_tokens_to_ids(pronouns)
stopwords=['&',',', '.', 'and', '-', 'the','##d', '...', 'that', 'to', 'as', 'for', '"', 'in', "'", 'a', 'of', 'only', ':', 'so', 'all', 'one', 'it', 'then', 'also', 'with', 'but', 'by', 'on', 'just', 'like', 'again', ';', 'more', 'this', 'not', 'is', 'there', 'was', 'even', 'still', 'after', 'here', 'later', '!', 'over', 'from', 'i', 'or', '?', 'at', 'first', '##s', 'while', ')', 'before', 'when', 'once', 'too',  'out', 'yet', 'because', 'some', 'though', 'had',  'instead', 'always', '(', 'well', 'back', 'tonight', 'since', 'about', 'through', 'will', 'them', 'left', 'often', 'what', 'ever', 'until',   'sometimes', 'if', 'however', 'finally', 'another', 'somehow', 'everything', 'further', 'really', 'last', 'an', '/', 'rather','s',  'may', 'be', 'each', 'thus', 'almost', 'where', 'anyway', 'their', 'has',  'something',  'already', 'within', 'any', 'indeed', '##a', '[UNK]', '~',  'every', 'meanwhile', 'would', '##e', 'have','nevertheless', 'which','how', '1', 'are', 'either', 'along', 'thereafter',  'otherwise', 'did',  'quite', 'these', 'can', '2', 'its', 'merely', 'actually', 'certainly',  '3', 'else','upon', 'except',  'those',  'especially',  'therefore','beside',   'apparently', 'besides', 'third', 'whilst',  '*', 'although', 'were','likewise', 'mainly', 'four', 'seven', 'into',  'm',  ']', 'than', 't', 'surely', '|',  '#',   'till', '##ly',  '_',  'al']
stopwords_ids=tokenizer.convert_tokens_to_ids(stopwords)

delwords=np.union1d(stopwords_ids,pronouns_ids)
delwords=np.union1d(delwords,alphabet_ids)
delwords=np.union1d(delwords,numbers_ids)
nodes=np.setdiff1d(all_nodes, delwords)

query=['industry','jones','economy','ceo', 'company','business']

query=['coach','player','university', 'game', 'star','club','season','series']

query_id=tokenizer.convert_tokens_to_ids(query)
query_dist=np.zeros([1,30522])
query_dist[:,query_id]=100
#query_dist=softmax(query_dist)
query_dist=normalize(query_dist,norm='l1',copy=False)

for token in tqdm(nodes):
    token=3208
    query="".join(['token_id==',str(token)])
    rows=token_table.read_where(query)
    nr_rows=len(rows)

    context_dists=np.stack([x[0] for x in rows],axis=0).squeeze()
    if nr_rows == 1:
        context_dists=np.reshape(context_dists, (-1, context_dists.shape[0]))
    context_dists[:, token] = np.min(context_dists)
    context_dists[:, delwords] = np.min(context_dists)
    #context_dists=normalize(context_dists,norm='l1',copy=False)
    context_dists=softmax(context_dists)

    from sklearn.cluster import KMeans
    from sklearn.cluster import SpectralClustering
    spcluster=SpectralClustering(n_clusters=4,assign_labels = "discretize",random_state = 0).fit(context_dists)

    kmeans = KMeans(n_clusters=4, random_state=0).fit(context_dists)
    centroids=kmeans.cluster_centers_

    outlier=context_dists[kmeans.labels_==3,:]
    inlier=context_dists[spcluster.labels_==0,:]

    outlierdists=np.sum(outlier,axis=0)/outlier.shape[0]
    spred = np.argsort(-context)[:25]
    outlier_list=tokenizer.convert_ids_to_tokens(spred)

    inlierdists=np.sum(inlier,axis=0)/inlier.shape[0]
    spred = np.argsort(-inlierdists)[:25]
    inlier_list=tokenizer.convert_ids_to_tokens(spred)


    hel_divergence(context_dists[0], context_dists)
    #kl_weights=kl_divergence(query_dist,context_dists)
    #kl_weights=1/np.where(kl_weights!=0,kl_weights,100)*100
    #kl_weights=kl_weights / np.sum(kl_weights)
    #kl_weights=softmax(kl_weights)
    #kl_weights=np.reshape(kl_weights,(-1,1))
    #sm_context_dists=np.sum(kl_weights*context_dists,axis=0)
    #spred = np.argsort(-sm_context_dists)[:50]
    #asdf=tokenizer.convert_ids_to_tokens(spred)

    own_dists=np.stack([x[1] for x in rows],axis=0).squeeze()
    if nr_rows == 1:
        own_dists=np.reshape(own_dists, (-1, own_dists.shape[0]))
    own_dists[:, delwords] = np.min(own_dists)
    own_dists[:, token]=np.min(own_dists)
    #own_dists[own_dists <= 0] = 0
    own_dists=softmax(own_dists)

    rep_outlier=own_dists[spcluster.labels_==1,:]
    rep_inlier=own_dists[spcluster.labels_==0,:]

    outlierdists=np.sum(rep_outlier,axis=0)/rep_outlier.shape[0]
    spred = np.argsort(-outlierdists)[:25]
    rep_outlier_list=tokenizer.convert_ids_to_tokens(spred)

    inlierdists=np.sum(rep_inlier,axis=0)/rep_inlier.shape[0]
    neighbors = np.argsort(-inlierdists)[:25]
    rep_inlier_list=tokenizer.convert_ids_to_tokens(neighbors)

    weights=inlierdists[neighbors]
    edgelist = [(token,x) for x in neighbors]
    weighted_edgelist = [(token,x[0],x[1]) for x in list(zip(neighbors,weights))]

    g.add_weighted_edges_from(weighted_edgelist)

    sg = nx.edge_subgraph(g, edgelist)
    sg=g
    sg.remove_nodes_from(nx.isolates(sg.to_undirected()))

    graphs = sorted(list(nx.connected_components(g.to_undirected())))

    #kmeans = KMeans(n_clusters=2, random_state=0).fit(own_dists)
    #centroids=kmeans.cluster_centers_
    #sm_own_dists=np.sum(kl_weights*own_dists,axis=0)
    #spred = np.argsort(-sm_own_dists)[:50]
    #asdf2=tokenizer.convert_ids_to_tokens(spred)

data_file.close()