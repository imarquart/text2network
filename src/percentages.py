import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cluster import KMeans
import hdbscan
from scipy.spatial.distance import cosine, pdist,squareform
import pandas as pd
import glob
import xlsxwriter
import os, time, sys
import logging
from shutil import copyfile
from sys import exit
from NLP.src.text_processing.preprocess_files_HBR import preprocess_files_HBR
from NLP.config.config import configuration
from NLP.src.process_sentences_network import process_sentences_network
from NLP.utils.load_bert import get_bert_and_tokenizer
from NLP.utils.network_tools import load_graph, make_symmetric, graph_merge,load_graph_overall
from NLP.src.run_bert import bert_args, run_bert
from NLP.src.dynamic_clustering import dynamic_clustering, louvain_cluster, overall_clustering, \
    overall_onelevel_clustering
from NLP.src.centrality_measures import dynamic_centralities, raw_ego_network
import networkx as nx
from NLP.src.draw_networks import draw_ego_network_mem, draw_ego_network
import itertools
import community

data=pd.read_csv("E:/NLP/cluster_xls/cluster_assignments.csv",delimiter=";",header=None)
dcluster=data.iloc[:,0]
dname=data.iloc[:,1]
name_dict=dict(zip(dcluster,dname))

data=pd.read_csv("E:/NLP/cluster_xls/centralities/WCleader_Rgraph-Sum-Rev_order3_cut0_clusters.csv")

data['Cname']=""
unique_clusters=np.unique(data['Cluster'].values)
years=np.unique(data['Year'].values)
measures=np.unique(data['Measure'].values)

for cl in unique_clusters:
    data.loc[data["Cluster"]==cl,"Cname"]=name_dict[cl]

data = data[data.Cname != "isolate"]

unique_cnames=np.unique(data['Cname'].values)


framedict={}
for measure in measures:
    subdata=data[data['Measure']==measure][['Year','Cname','WAvg','SAvg','Missing']]
    gr=subdata.groupby('Year').transform(lambda x: x/x.sum())[['WAvg','SAvg']]
    subdata['pWAvg']=gr['WAvg']
    subdata['pSAvg']=gr['SAvg']

    mlist=["WAvg","SAvg","pWAvg","pSAvg"]
    iterables=[unique_cnames,mlist]
    id=pd.MultiIndex.from_product(iterables, names=['cluster', 'measure'])
    rows=len(unique_cnames)*len(mlist)
    cols=len(years)
    piv_table=pd.DataFrame(np.ones([rows,cols]),index=id,columns=years)
    for cl in unique_cnames:
        for m in mlist:
            fill=subdata[subdata['Cname']==cl][m].values
            piv_table.loc[cl,m]=fill

    framedict.update({measure:piv_table.copy(deep=True)})

with pd.ExcelWriter('E:/NLP/cluster_xls/centralities/PWCleader_Rgraph-Sum-Rev_order3_cut0_clusters.xlsx') as writer:
    for measure in measures:
        framedict[measure].to_excel(writer, sheet_name=measure,merge_cells=False)
