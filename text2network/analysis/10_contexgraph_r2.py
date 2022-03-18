import os
import pickle
from time import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import KernelPCA
from text2network.classes.neo4jnw import neo4j_network
from text2network.functions.graph_clustering import consensus_louvain
from text2network.utils.file_helpers import check_folder
from text2network.utils.logging_helpers import setup_logger
os.environ['NUMEXPR_MAX_THREADS'] = '16'


def get_convex_hull(Y, set_points, starting_point):
    # Leftmost point
    convex_hull = []

    convex_hull.append(Y.loc[starting_point, :])

    ending = False
    i = 0
    while ending == False:
        curpoint = convex_hull[i].copy()
        endpoint = Y.loc[set_points[0], :]
        curset = set_points.copy()
        # curset.remove(curpoint.name)
        for c in curset:
            # print("Checking {}".format(c))
            candidate = Y.loc[c, :]
            Orin = (endpoint.x - curpoint.x) * (candidate.y - curpoint.y) - (candidate.x - curpoint.x) * (
                    endpoint.y - curpoint.y)
            # print(angle2-angle1 )
            if (Orin > 0) or (endpoint.name == curpoint.name):
                endpoint = candidate
        convex_hull.append(endpoint.copy())
        # set_points.remove(endpoint.name)
        if endpoint.name == convex_hull[0].name or i >= len(set_points) + 1:
            ending = True
        else:
            i = i + 1

    return convex_hull


def get_filename(csv_path, main_folder, focal_token, cutoff, tfidf, context_mode, contextual_relations,
                 postcut, keep_top_k, depth, max_degree=None, algo=None, level=None, rs=None, tf=None, sub_mode=None):
    output_path = check_folder(csv_path)
    output_path = check_folder(csv_path + "/" + main_folder + "/")
    output_path = check_folder(
        "".join([output_path, "/", str(focal_token), "_cut", str(int(cutoff * 100)), "_tfidf",
                 str(tfidf is not None), "_cm", str(context_mode), "/"]))
    output_path = check_folder("".join(
        [output_path, "/", "conRel", str(contextual_relations), "_postcut", str(int(postcut * 100)), "/"]))
    output_path = check_folder("".join(
        [output_path, "/", "keeptopk", str(keep_top_k), "_keeponlyt_", str(depth == 0), "/"]))
    if max_degree is not None and algo is not None:
        output_path = check_folder("".join(
            [output_path, "/", "md", str(max_degree), "_algo", str(algo.__name__), "/"]))
    if sub_mode is not None:
        output_path = check_folder("".join(
            [output_path, "/", "submode", str(sub_mode), "/"]))
    if level is not None:
        output_path = check_folder("".join(
            [output_path, "/", "lev", str(level), "/"]))
    if tf is not None:
        output_path = check_folder("".join([output_path, "/" + "tf_", str(tf) + "/"]))
    filename = "".join(
        [output_path, "/",
         "rs", str(rs)])
    return filename, output_path


# Set a configuration path
configuration_path = 'config/analyses/LeaderSenBert40.ini'
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_folder(configuration_path))
config.read(check_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "7_embedding.py")

semantic_network = neo4j_network(config)


# Filename loading
years = list(range(1980, 2021))
focal_token = "leader"

main_folder = "profile_relationships2_bidirectional_ADJ_VERB_NOUN_ADV_ADP"

filename, picdir = get_filename(config['Paths']['csv_outputs'], main_folder,
                                focal_token=focal_token, cutoff=0.1, tfidf="weight",
                                context_mode="bidirectional", contextual_relations=True,
                                postcut=0.01, keep_top_k=1000, depth=0,
                                max_degree=100, algo=consensus_louvain, level=10, rs=100, tf="weight", sub_mode="bidirectional")
picdir = "\\".join(picdir.split("\\")[0:-4])


# %% SEttings
focal_substitutes = focal_token
focal_occurrences = None
cluster_subset = ["business", "people", "team", "good", "organization", "ceo", "better", "company", "make", "global",
                  "work", "help", "industry", "person", "lead", "different", "level", "need", "strong", "leadership"][0:14]

sel_alter2 = None
sel_alter_list = []
#sel_alter_list.append(None)
sel_alter_list.append(["ceo", "president", "founder", "successor", "chairman"])
sel_alter_list.append(["manager", "executive", "pioneer", "follower", "champion"])
sel_alter_list.append(["boss", "supervisor", "father", "subordinate", "superior"])
sel_alter_list.append(["supervisor","father","boss",   "subordinate", "superior","manager", "executive", "pioneer", "follower", "champion"])
sel_alter_list.append(["father","boss", "supervisor",  "subordinate", "superior","manager", "executive", "pioneer", "follower", "champion","ceo", "president", "founder", "successor", "chairman"])
sel_alter_list.append(["chairman","ceo", "president", "founder", "successor", "boss", "supervisor", "father", "subordinate", "superior"])

year_list = [1980, 2021]
moving_average = (20, 20)

imagemethod = "imshow"
imagemethod = "contour"
im_int_method = "gaussian"
grid_method = "linear"
npts = 200
int_level = 18
ngridx = 8
ngridy = ngridx
scale_percent = 0.1

# Embedding Settings
top_n = 6
emb_top_n = 30
nr_tokens = 500000000
convex_proximity = False

handle_leader = "drop"
#handle_leader = "switch"
#handle_leader = None

picdir = picdir + "\\hl_" + str(handle_leader) + "_"

# %% Get Cluster Data

if not (isinstance(focal_substitutes, list) or focal_substitutes is None):
    focal_substitutes = [focal_substitutes]
if not (isinstance(focal_occurrences, list) or focal_occurrences is None):
    focal_occurrences = [focal_occurrences]

checkname = filename + "_CLdf" + ".xlsx"
pname = filename + "_CLdict_tk.p"
df_clusters = pd.read_excel(checkname)
cldict = pickle.load(open(pname, "rb"))

# Get Adjacency matrix and drop columns which may be cluster names
X = df_clusters.iloc[:, 1:-7]
X.index = X.columns.to_list()
for dcol in ["year", "type", "pos", "sub", "occ"]:
    if dcol in X.columns:
        X = X.drop(columns=dcol)
    if dcol in X.index:
        X = X.drop(dcol)
cl_name = X.columns.to_list()
assert all(X.columns == X.index)

if cluster_subset is not None:
    X = X[X.index.isin(cluster_subset)]
    X = X.loc[:, X.columns.isin(cluster_subset)]
cl_name = X.index
# X=X.div(X.sum(axis=1), axis=0)
# X.iloc[:,:]=(X.to_numpy()+X.to_numpy().T)/2
xx = X.to_numpy()
sorted_row_idx = np.argsort(xx, axis=1)[:, 0:-top_n]
col_idx = np.arange(xx.shape[0])[:, None]
xx[col_idx, sorted_row_idx] = 0
X = pd.DataFrame(xx)
# X=X/np.max(X.max())
color = list(range(0, len(X.index)))
X.index = cl_name

#%% Embedding
nodes = list(X.index)
kernel_pca = KernelPCA(remove_zero_eig=False,
                       n_components=2, kernel="precomputed", random_state=100)
X.index = nodes
X.columns = nodes
Y = kernel_pca.fit_transform(X)
pos = Y.copy()

# Create Network Representation
ids = list(range(0, len(X.index)))
index_dict = dict(zip(ids, nodes))
node_dict = dict(zip(nodes, ids))
X.index = ids
X.columns = ids
G = nx.from_pandas_adjacency(X)

Y = pd.DataFrame(Y)
X.index = nodes
X.columns = nodes
Y.index = nodes
Y["color"] = [item / np.max(color) for item in color]
Y.columns = ["x", "y", "color"]

# %% Landscape
# Get all datapoints
ylist = []
for year in years:
    checkname = filename + "REGDF" + str(year) + ".xlsx"
    df = pd.read_excel(checkname)
    df["tYear"] = year
    ylist.append(df.copy())
df = pd.concat(ylist)

# Replace Leader Occurrences with
if handle_leader == "drop":
    df=df.loc[~(df["occ"] == "leader"),:].copy()
elif handle_leader == "switch":
    df.loc[df["occ"] == "leader", "occ"] = df.loc[df["occ"] == "leader", "sub"]
else:
    pass

Xall = df.loc[:,cl_name]
# X2=X2.div(X2.sum(axis=1), axis=0)
Xall.loc[:,"color"] = df.rweight / np.max(df.rweight)
Xall["alter"] = semantic_network.ensure_tokens(df.occ)
Xall["year"] = df.tYear
Xall = Xall.replace([np.inf, -np.inf], np.nan).dropna(how="any")
Xall = Xall.sort_values(by="color", ascending=False).iloc[0:nr_tokens, :]
# Drop rows with no connections
Xall = Xall.iloc[np.where(Xall[cl_name].sum(axis=1) > 0)[0], :]


xx = Xall[cl_name].to_numpy()
sorted_row_idx = np.argsort(xx, axis=1)[:, 0:-emb_top_n]
col_idx = np.arange(xx.shape[0])[:, None]
xx[col_idx, sorted_row_idx] = 0
xx = xx / np.sum(xx, axis=1, keepdims=True)
Y2 = np.matmul(xx, Y[["x", "y"]].to_numpy())
Y2 = pd.DataFrame(Y2)

#Y["color"] = [item / np.max(color) for item in color]
Y2.index = Xall.index
Y2["color"] = Xall.loc[:, "color"]
Y2["alter"] = Xall.loc[:, "alter"]
Y2["year"] = Xall.year
Y.columns = ["x", "y", "color"]
Y2.columns = ["x", "y", "color", "alter", "year"]
t1 = time()
Y2 = Y2.dropna()
print(np.max(Y2, axis=0))
print(np.min(Y2, axis=0))
sel_alter = ["leader"]
window=5
for sel_alter in sel_alter_list:
    difflist=[]
    for t1 in years[window:]:
        t0=t1-window

        Y2all = Y2.loc[(Y2.year >= t0) & (Y2.year <=t1),:].copy()

        dmean=Y2all.loc[:,["x","y"]].to_numpy()-np.mean(Y2all[["x","y"]]).to_numpy()
        dmeansq=np.square(dmean)
        weights = Y2all["color"] / Y2all["color"].sum()
        Y2all.loc[:,["x","y"]]=dmeansq
        Y2all.loc[:,"color"] = Y2all.loc[:,"color"] / Y2all.loc[:,"color"].sum()

        Y2all.loc[:,["x","y"]]=np.multiply(dmeansq,weights.to_numpy().reshape(-1,1))

        sd_all = Y2all[["x","y"]].sum()
        sd_cluster=Y2all.loc[Y2all.alter.isin(sel_alter),:].copy()
        sd_cluster=sd_cluster[["x","y"]].sum()

        diff = np.sum(sd_cluster)/np.sum(sd_all)
        print("{}-{}: Summed Diff = {}".format(t0,t1,diff))
        difflist.append(diff)

    diffs=pd.DataFrame(difflist)
    diffs.index=years[window:]
    diffs.plot()
    plt.title("Explained variation by cluster "+ str(sel_alter[0]) + "\n for "+ str(window) +"-year period")
    ftitle = picdir + "TS_diff"+ str(window) +"_" + str(sel_alter[0]) + ".png"
    plt.savefig(ftitle)
    diffs.to_excel(ftitle + ".xlsx")

for sel_alter in sel_alter_list:
    sumproblist=[]
    difflist=[]
    for t1 in years[window:]:
        t0=t1-window

        Y2all = Y2.loc[(Y2.year >= t0) & (Y2.year <=t1),:].copy()

        dmean=Y2all.loc[:,["x","y"]].to_numpy()-np.mean(Y2all[["x","y"]]).to_numpy()
        dmeansq=np.square(dmean)
        weights = Y2all["color"] / Y2all["color"].sum()
        Y2all.loc[:,["x","y"]]=dmeansq
        #Y2all.loc[:,"color"] = Y2all.loc[:,"color"] / Y2all.loc[:,"color"].sum()

        Y2all.loc[:,["x","y"]]=np.multiply(dmeansq,weights.to_numpy().reshape(-1,1))

        sd_all = Y2all[["x","y"]].sum()
        df_cluster=Y2all.loc[Y2all.alter.isin(sel_alter),:].copy()
        sum_prob = df_cluster["color"].sum()
        sd_cluster=df_cluster[["x","y"]].sum()

        diff = np.sum(sd_cluster)/np.sum(sd_all)

        diff = diff / sum_prob

        print("{}-{}-{}: len={}, Summed Diff = {}".format(sel_alter[0],t0,t1,len(df_cluster),diff))
        difflist.append(diff)
        sumproblist.append(sum_prob)

    probs=pd.DataFrame(sumproblist)
    probs.index=years[window:]
    probs.plot()
    plt.title("Probability sum for cluster "+ str(sel_alter[0]) + "\n for "+ str(window) +"-year period")
    ftitle = picdir + "prob_sum_"+ str(window) +"_" + str(sel_alter[0]) + ".png"
    plt.savefig(ftitle)
    probs.to_excel(ftitle + ".xlsx")
    plt.close()



for sel_alter in sel_alter_list:
    sumproblist=[]
    difflist=[]
    for t1 in years[window:]:
        t0=t1-window

        Y2all = Y2.loc[(Y2.year >= t0) & (Y2.year <=t1),:].copy()

        dmean=Y2all.loc[:,["x","y"]].to_numpy()-np.mean(Y2all[["x","y"]]).to_numpy()
        dmeansq=np.square(dmean)
        weights = Y2all["color"] / Y2all["color"].sum()
        Y2all.loc[:,["x","y"]]=dmeansq
        #Y2all.loc[:,"color"] = Y2all.loc[:,"color"] / Y2all.loc[:,"color"].sum()

        Y2all.loc[:,["x","y"]]=np.multiply(dmeansq,weights.to_numpy().reshape(-1,1))

        sd_all = Y2all[["x","y"]].sum()
        df_cluster=Y2all.loc[Y2all.alter.isin(sel_alter),:].copy()
        sum_prob = df_cluster["color"].sum() / Y2all["color"].sum()
        sd_cluster=df_cluster[["x","y"]].sum()

        diff = np.sum(sd_cluster)/np.sum(sd_all)

        diff = diff / (100*sum_prob)

        print("{}-{}-{}: len={}, Summed Diff = {}".format(sel_alter[0],t0,t1,len(df_cluster),diff))
        difflist.append(diff)
        sumproblist.append(sum_prob)

    diffs=pd.DataFrame(difflist)
    diffs.index=years[window:]
    diffs.plot()

    plt.title("Explained variation by cluster "+ str(sel_alter[0]) + "\n for "+ str(window) +"-year period")
    ftitle = picdir + "ratio_normed_TS_diff"+ str(window) +"_" + str(sel_alter[0]) + ".png"
    plt.savefig(ftitle)
    diffs.to_excel(ftitle + ".xlsx")
    plt.close()
    probs=pd.DataFrame(sumproblist)
    probs.index=years[window:]
    probs.plot()
    plt.title("Probability sum for cluster "+ str(sel_alter[0]) + "\n for "+ str(window) +"-year period")
    ftitle = picdir + "ratio_prob_sum_"+ str(window) +"_" + str(sel_alter[0]) + ".png"
    plt.savefig(ftitle)
    probs.to_excel(ftitle + ".xlsx")
    plt.close()




for t1 in [2000,2021]:
    problist= {}
    ratiolist={}
    for sel_alter in sel_alter_list:
        # Total variance (all years)
        # Explained variance (all years)

        t0 = t1-20

        Y2all = Y2.loc[(Y2.year >= t0) & (Y2.year <= t1), :].copy()
        dmean=Y2all.loc[:,["x","y"]].to_numpy()-np.mean(Y2all[["x","y"]]).to_numpy()
        dmeansq=np.square(dmean)
        weights = Y2all["color"] / Y2all["color"].sum()
        Y2all.loc[:,["x","y"]]=dmeansq
        #Y2all.loc[:,"color"] = Y2all.loc[:,"color"] / Y2all.loc[:,"color"].sum()
        Y2all.loc[:,["x","y"]]=np.multiply(dmeansq,weights.to_numpy().reshape(-1,1))
        sd_all = Y2all[["x","y"]].sum()
        df_cluster=Y2all.loc[Y2all.alter.isin(sel_alter),:].copy()
        sum_prob = df_cluster["color"].sum() / Y2all["color"].sum()
        sd_cluster=df_cluster[["x","y"]].sum()

        diff = np.sum(sd_cluster)/np.sum(sd_all)

        diff_ratio = diff / (sum_prob)

        problist[sel_alter[0]]=diff
        ratiolist[sel_alter[0]] = diff_ratio
    print(problist)
    print(ratiolist)
    yad=pd.DataFrame(problist.values(), index=problist.keys())
    yad.plot(kind="bar")
    ftitle = picdir + "summed_exp_var"+ str(window) + "t1"+ str(t1) + ".png"
    plt.savefig(ftitle)
    yad.to_excel(ftitle + ".xlsx")
    plt.close()

    yad=pd.DataFrame(ratiolist.values(), index=ratiolist.keys())
    yad.plot(kind="bar")
    ftitle = picdir + "ratio_summed_exp_var"+ str(window) + "t1"+ str(t1) + ".png"
    plt.savefig(ftitle)
    yad.to_excel(ftitle + ".xlsx")
    plt.close()

