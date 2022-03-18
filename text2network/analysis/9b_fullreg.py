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

import statsmodels.api as sm
from statsmodels.formula.api import ols

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
cluster_subset = None

sel_alter2 = None
sel_alter_list = []
#sel_alter_list.append(None)
sel_alter_list.append(["ceo", "president", "founder", "successor", "chairman"])
sel_alter_list.append(["manager", "executive", "pioneer", "follower", "champion"])
sel_alter_list.append(["boss", "supervisor", "father", "subordinate", "superior"])

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

window=5

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


if years is None:
    # Get all datapoints
    checkname = filename + "REGDF_allY.xlsx"
    df = pd.read_excel(checkname)
else:
    ylist = []
    for year in years:
        checkname = filename + "REGDF" + str(year) + ".xlsx"
        df = pd.read_excel(checkname)
        df["tYear"] = year
        ylist.append(df.copy())
    df = pd.concat(ylist)


ff = df[df["occ"].isin(["leader"])]
ff=ff[["tYear","Tsent","Tsub"]].groupby("tYear").mean()
ff["Tsent"].plot()
plt.title("Sentiment of leader being substituted \n n={}".format(len(df[df["occ"].isin(["leader"])])))
ftitle = picdir + "TS_sent_leader_occ.png"
plt.savefig(ftitle)
plt.close()

plt.show()

ff = df[df["sub"].isin(["leader"])]
ff=ff[["tYear","Tsent","Tsub"]].groupby("tYear").mean()
ff["Tsent"].plot()
plt.title("Sentiment of leader as substitute \n n={}".format(len(df[df["sub"].isin(["leader"])])))
ftitle = picdir + "TS_sent_leader_sub.png"
plt.savefig(ftitle)
plt.show()


#%% Conditional R2

print(len(df))
ff = df[df["sub"].isin(["leader"])]
print(len(ff))
X2 = ff[cl_name].copy()
X2["prob"] = ff.rweight / np.max(ff.rweight)
X2["alter"] = semantic_network.ensure_tokens(ff.occ)
X2["year"] = ff.tYear
X2 = X2.replace([np.inf, -np.inf], np.nan).dropna(how="any")
# Drop rows with no connections
X2 = X2.iloc[np.where(X2[cl_name].sum(axis=1) > 0)[0], :]
Y = X2.prob
X = X2[cl_name]
cl_name_f = list(cl_name)
cl_name_f[np.where(np.array(cl_name) == "global")[0][0]] = "gglobal"
X.columns = cl_name_f
X = X / 40
X = X / 100
X.div(Y, axis=0)
X = sm.add_constant(X)
X["year"] = X2.year
X["y"] = Y
f1 = "y~" + "+".join([x for x in cl_name_f])
r2yl = []
for year in years[window:]:
    yrange = np.arange(year - window, year + 1)
    model = ols(formula=f1, data=X[X.year.isin(yrange)])
    results = model.fit(cov_type='HC1', )
    r2yl.append(results.rsquared)
ydf=pd.Series(r2yl)
ydf.index=years[window:]
ydf.plot()
plt.title("Expl. Variation of Leader substituting by context. \n Conditional on Leader being a substitute")
ftitle = picdir + "TS_R2cond_leader"+"window_"+str(window)+".png"
plt.savefig(ftitle)
ydf.to_excel(ftitle+".xlsx")
plt.close()

#%% Absolute R2





print((df.rweight.sum()))
ff = df.copy()
ff.loc[~ff["sub"].isin(["leader"]),"rweight"]=0
print((ff.rweight.sum()))
X2 = ff[cl_name].copy()
X2["prob"] = ff.rweight / np.max(ff.rweight)
X2["alter"] = semantic_network.ensure_tokens(ff.occ)
X2["year"] = ff.tYear
X2 = X2.replace([np.inf, -np.inf], np.nan).dropna(how="any")
# Drop rows with no connections
X2 = X2.iloc[np.where(X2[cl_name].sum(axis=1) > 0)[0], :]
Y = X2.prob
X = X2[cl_name]
cl_name_f = list(cl_name)
cl_name_f[np.where(np.array(cl_name) == "global")[0][0]] = "gglobal"
X.columns = cl_name_f
X = X / 40
X = X / 100
X.div(Y, axis=0)
X = sm.add_constant(X)
X["year"] = X2.year
X["y"] = Y
f1 = "y~" + "+".join([x for x in cl_name_f])
r2yl = []
for year in years[window:]:
    yrange = np.arange(year - window, year + 1)
    model = ols(formula=f1, data=X[X.year.isin(yrange)])
    results = model.fit(cov_type='HC1', )
    r2yl.append(results.rsquared)
ydf=pd.Series(r2yl)
ydf.index=years[window:]
ydf.plot()
plt.title("Expl. Variation of Leader substituting by context. \n For all occurrences")
ftitle = picdir + "TS_R2_leader"+"window_"+str(window)+".png"
plt.savefig(ftitle)
ydf.to_excel(ftitle+".xlsx")
plt.close()


for sel_alter in sel_alter_list:

    X2.loc[~X2.alter.isin(sel_alter),"prob"]=0
    X2a=X2[X2.alter.isin(sel_alter)]

    dfs = df[df["occ"].isin(sel_alter)]
    ff = dfs[["tYear", "Tsent", "Tsub"]].groupby("tYear").mean()
    ff["Tsent"].plot()
    plt.title("Sentiment of leader substituting \n for {}, \n n={}".format(sel_alter, len(dfs)))
    ftitle = picdir + "TS_sent_leader_sub_"+str(sel_alter[0])+".png"
    plt.savefig(ftitle)
    plt.close()

    dfs = df[df["sub"].isin(sel_alter)]
    ff = dfs[["tYear", "Tsent", "Tsub"]].groupby("tYear").mean()
    ff["Tsent"].plot()
    plt.title("Sentiment of leader being substituted \n by {}, \n n={}".format(sel_alter, len(dfs)))
    ftitle = picdir + "TS_sent_leader_occ_"+str(sel_alter[0])+".png"
    plt.savefig(ftitle)

    plt.close()



    print((df.rweight.sum()))
    ff = df.copy()
    ff.loc[~ff["sub"].isin(["leader"]), "rweight"] = 0
    print((ff.rweight.sum()))

    ff.loc[~ff["occ"].isin(sel_alter), "rweight"] = 0
    print((ff.rweight.sum()))

    X2 = ff[cl_name].copy()
    X2["prob"] = ff.rweight / np.max(ff.rweight)
    X2["alter"] = semantic_network.ensure_tokens(ff.occ)
    X2["year"] = ff.tYear
    X2 = X2.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    # Drop rows with no connections
    X2 = X2.iloc[np.where(X2[cl_name].sum(axis=1) > 0)[0], :]
    Y = X2.prob
    X = X2[cl_name]
    cl_name_f = list(cl_name)
    cl_name_f[np.where(np.array(cl_name) == "global")[0][0]] = "gglobal"
    X.columns = cl_name_f
    X = X / 40
    X = X / 100
    X.div(Y, axis=0)
    X = sm.add_constant(X)
    X["year"] = X2.year
    X["y"] = Y
    f1 = "y~" + "+".join([x for x in cl_name_f])
    r2yl = []
    for year in years[window:]:
        yrange = np.arange(year-window,year+1)
        model = ols(formula=f1, data=X[X.year.isin(yrange)])
        results = model.fit(cov_type='HC1', )
        r2yl.append(results.rsquared)
    ydf = pd.Series(r2yl)
    ydf.index = years[window:]
    ydf.plot()
    plt.title("Expl. Variation of Leader substituting \n for {} \n by context. For all occurrences".format(sel_alter))
    ftitle = picdir + "TS_R2_leader_sub_"+str(sel_alter[0])+"window_"+str(window)+".png"
    plt.savefig(ftitle)
    ydf.to_excel(ftitle + ".xlsx")
    plt.close()

    if handle_leader == "drop":
        df = df.loc[~(df["occ"] == "leader"), :].copy()
    elif handle_leader == "switch":
        df.loc[df["occ"] == "leader", "occ"] = df.loc[df["occ"] == "leader", "sub"]
    else:
        pass

    print((df.rweight.sum()))
    ff = df.copy()
    ff = ff[ff["occ"].isin(sel_alter)]
    X2 = ff[cl_name].copy()
    X2["prob"] = ff.rweight / np.max(ff.rweight)
    X2["alter"] = semantic_network.ensure_tokens(ff.occ)
    X2["year"] = ff.tYear
    X2 = X2.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    # Drop rows with no connections
    X2 = X2.iloc[np.where(X2[cl_name].sum(axis=1) > 0)[0], :]


    cclusterl = ["business", "people", "team", "good", "organization", "ceo", "better", "company", "make",
                      "global",
                      "work", "help", "industry", "person", "lead", "different", "level", "need", "strong",
                      "leadership"][0:14]
    clusterll = [["better", "people", "good", "help", "work","team", "person","ceo","make"], ["organization","company","business","industry","global"]]
    clusterll = [["better", "people", "good", "help", "work","team", "person","ceo","make","organization",], ["company","business","industry","global"]]

    for ccluster in clusterll:
        F=X2.copy()
        other_cluster = list(np.setdiff1d(cclusterl,cl_name))
        F["joint"]=F[ccluster].sum(axis=1)*F["prob"]
        F["Total"] = F[cl_name].sum(axis=1) * F["prob"]

        FF=F.groupby("year").sum()

        joint_prob_normed=FF["joint"]/FF["Total"]
        joint_prob = FF["joint"] / FF["prob"]
        plt.plot(joint_prob)
        plt.title("Joint probability of \n {} \n and  {} context cluster".format(sel_alter[0],ccluster))
        ftitle = picdir + "TS_JointProb_" + str(sel_alter[0]) + "_with_"+ str(ccluster[0]) + ".png"
        plt.savefig(ftitle)
        joint_prob.to_excel(ftitle + ".xlsx")
        plt.close()
        plt.plot(joint_prob_normed)
        plt.title("Joint Normed probability of \n {} \n and  {} context cluster".format(sel_alter[0],ccluster))
        ftitle = picdir + "TS_N_JointProb_" + str(sel_alter[0]) + "_with_"+ str(ccluster[0]) + ".png"
        plt.savefig(ftitle)
        joint_prob_normed.to_excel(ftitle + ".xlsx")
        plt.close()