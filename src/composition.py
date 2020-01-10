import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cluster import KMeans
import hdbscan

csv=pd.read_csv("E:/NLP/cluster_xls/centralities/Book4.csv",delimiter=";")
tokens=csv.iloc[:-1,0].to_numpy()
tokens=[np.str(x) for x in tokens]
years=pd.to_numeric(csv.columns[1:]).to_numpy()
data=csv.iloc[0:-1,1:]
long_data=data.transpose()
long_data.apply(pd.to_numeric)
np_data=long_data.to_numpy()


#clusterer = KMeans(n_clusters=nr_clusters).fit(trunc_dist)
# #cluster_selection_method="leaf",
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric='l2', prediction_data=True).fit(
    np_data)
labels=clusterer.labels_
print(labels)
means=[]
for cluster in np.unique(clusterer.labels_):
    vec=np_data[labels==cluster,:]
    avg_vec=np.mean(vec,axis=0)
    years_vec=years[labels==cluster]
    means.append(pd.Series(avg_vec,index=tokens))

mean_frame=pd.DataFrame(means)
year_frame=pd.DataFrame(labels,index=years)

mean_frame.to_csv("E:/NLP/cluster_xls/centralities/mean_frame.csv")
year_frame.to_csv("E:/NLP/cluster_xls/centralities/year_frame.csv")
