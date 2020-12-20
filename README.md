# Text2Network

## Introduction

Text2Network is a python package to generate "semantic networks" from arbitrary text sources using a (deep neural) PyTorch transformer model (e.g. BERT).
Details for this procedure are available in the following research paper: (Coming soon)

Text2Network allows you to understand direct and higher-order linguistic relationships in your texts. Since these relationships are extracted directly from a sophisticated transformer model, they are more powerful than prior approaches. For example, once a text is represented as network, you can query each potential relation between words conditional on a certain context, such as other words that may appear in a sentence. You can also aggregate across arbitrary time periods or text pieces. The network is constructed in such a way that each conditioning operation leads to proper probability measures that are directly interpretable without further nuisance parameters and refer, correctly, to the language use of the subset in question - be it a given context, or a subset of the corpus.
Using network techniques, such as centralities, distances and community structures, you can then analyze global structure of language as used in the text corpus.

The semantic network also represents all the information about linugistic dependencies captured by the deep neural network. As such, it is useful to understand both the text source, as well as the model. Note, however, that this procedure is inferential and does not provide a good input for downstream tasks.

## Components

Text2Networks includes four python objects, one for preprocessing, one for training the language model, one for inference and one class that directly represents the semantic network and all operations on it.

The objects correspond to the four steps necessary to analyze a corpus:

1. Pre-processing: Texts are ingested, cleaned and saved in a hdfs database. Associated metadata is created, this notably includes a time variable and a number of freely chosen parameters.

2. Training of the DNN: The researcher determines which combination of parameters constitutes a unit of analysis (for example newspaper-year-issue) and passes this hierarchy to the trainer class, which trains the required number of DNN models. Please see the research paper why this is required to capture linguistic relations in the text corpus correctly.

3. Processing: Using the trained language models, the each word in the corpus is analyzed in its context. For each such occurrence, ties representing probabilistic relations are created. Here, we use a Neo4j graph database to save this network.

4. Analysis: The semantic_network class directly encapsulates the graph database and conditioning operations thereon. Networks created here can either be analyzed directly, or exported in gexf format.

## Working with the semantic network

### TODO Check if still works

Once initialized, the semantic network class represents and interface to the Neo4j graph database. It acts similar to a networkx graph and can query individual nodes directly. For example, to return a networkx-compatible list of ties for the token "president", you can use

```python
semantic_network=sem_network(config)

semantic_network['president']
```

This will query the neo4j network directly.

### Conditioning

If you are interested in analyzing more than a single node, it is a good idea to condition the graph. Conditioning the graph will query relevant ties from Neo4j and construct an in-memory networkx graph that is used until the deconditioning function is called. Conditioning the graph norms ensures that the probabilities implied by the network are correctly conditioned on context and time-frame.

For example, to derive the 2-step ego network for the token "president", across a set of years, conditional on sentences with the context "USA" or "China", do

```python
semantic_network.condition(years=[1992,2005], ego_nw_tokens="president", depth=2,weight_cutoff=0.05, context=['USA','China'])
```

To release the conditioning

```python
semantic_network.decondition()
```

### Transforming

Several transformations are available and can be called directly on a conditioned network. For example

```python
semantic_network.to_symmetric()

semantic_network.to_backout()
```

### Measures


In addition, we provide a host of analysis functions and formatting options. For example, printing the centralities of the terms "president","tyrant","man" and "woman" from a pandas dataframe can be accomplished in two lines:

```python
semantic_network=sem_network(config)

centralities=semantic_network.centralities(focal_tokens=['president','tyrant','man', 'woman'], types=["PageRank"])

print(pd_format(centralities))
```

where centralities computes centralities of different kind (here: PageRank) and pd_format transforms the output into a pandas DataFrame for easy printing and saving.

If you instead want to print the centralities in a 2-step ego network with "tyrant" as its center, considering texts written in the year 2002, just write

```python
semantic_network=sem_network(config)

centralities=semantic_network.centralities(focal_tokens=['president','tyrant','man', 'woman'], years=[2002], ego_nw_tokens="tyrant", depth=2)

print(pd_format(centralities))
```
Or, if you want to do the same but restrict inference to contexts in which "USA" is likely to occur, do

```python
semantic_network=sem_network(config)

centralities=semantic_network.centralities(focal_tokens=['president','tyrant','man', 'woman'], ego_nw_tokens="tyrant", depth=2, context=["USA"])

print(pd_format(centralities))
```

Each time, the created network and its measures will be based on the correctly conditioned probability measures that the language model generates.

If the network is already conditioned, this will be taken into account

```python
semantic_network.condition(years=[1992,2005], ego_nw_tokens="president", depth=2,weight_cutoff=0.05, context=['USA','China'])

centralities=semantic_network.centralities(focal_tokens=['president','tyrant','man', 'woman']))

print(pd_format(centralities))
```

### Clustering

Semantic networks cluster hierarchically into linguistic components. This is usually where syntactic and grammatical boundaries appear.
Text2Network offers several tools to derive these clusters. By default, louvain-clustering from the community package is used, however, you can add any callable.

Clusters are handled as dictionary container. They contain the relevant subgraph, names, measures (such as centralities) and metadata.
By default, the cluster function takes care of handling these details.
For example, to cluster the semantic network, call

```python
semantic_network=sem_network(config)

clusters=semantic_network.cluster(levels=1)
```

The clusters variable is a list of cluster containers, including the base graph amended by cluster identifiers for each node, as well as the subgraphs implied by the clustering algorithm.

If the network is not already conditioned, you can also pass the usual parameters

```python
semantic_network=sem_network(config)

clusters=semantic_network.cluster(levels=1,  ego_nw_tokens="tyrant", depth=2, context=["USA"])
```

You can automatically apply measures of choice on each cluster, which are saved as list of dictionaries in the cluster container

```python
clusters=semantic_network.cluster(levels=1, to_measure=[proximity,centrality])

print(pd_format(clusters[0]['measures']))
```

Clusters keep track of their level and parent cluster, such that hierarchies become apparent

```python
levels=2
clusters=semantic_network.cluster(levels=levels)

for cl in clusters:
    print("Name: {}, Level: {}, Parent: {}, Nodes: {}".format(cl['name'],cl['level'],cl['parent'],cl['graph'].nodes))
```

```
Name: base, Level: 0, Parent: , Nodes: ['chancellor', 'president', 'king', 'tyrant', 'ceo', 'father', 'judge', 'delegate', 'manage', 'teach', 'rule', 'man', 'woman']
Name: base-0, Level: 1, Parent: base, Nodes: ['father', 'woman', 'king', 'chancellor', 'man', 'tyrant']
Name: base-1, Level: 1, Parent: base, Nodes: ['manage', 'president', 'teach', 'delegate', 'ceo']
Name: base-2, Level: 1, Parent: base, Nodes: ['rule', 'judge']
Name: base-0-0, Level: 2, Parent: base-0, Nodes: ['father', 'king', 'chancellor', 'tyrant']
Name: base-0-1, Level: 2, Parent: base-0, Nodes: ['man', 'woman']
Name: base-1-0, Level: 2, Parent: base-1, Nodes: ['manage', 'teach', 'delegate']
Name: base-1-1, Level: 2, Parent: base-1, Nodes: ['ceo', 'president']
Name: base-2-0, Level: 2, Parent: base-2, Nodes: ['rule', 'judge']
```

Of course, you can also apply clustering to networkx graphs yourself. 
For example, assume you have previously conditioned your network
```python
semantic_network.condition(years=[1992,2005], weight_cutoff=0.05, context=['USA','China'])
```

Then, you can package the graph into a cluster container

```python
packaged_graph=return_cluster(semantic_network.graph,name="Test",parent="",level=0,measures=[],metadata={'years':[1992,2005], 'context':['USA','China']})
```

and run the clustering function. This will return the base cluster, amended by measures and node identifiers, all well as a list of the subgraphs of the given clusters.
In contrast to the cluster function of the semantic network class, the base cluster is returned as separate entity of a tuple.

```python
base_cluster,subgraph_clusters=cluster_graph(packaged_graph, to_measure=[proximity, centrality],algorithm=louvain_cluster)
```

Note that the cluster function of the semantic network can condition if necessary, such that the above is equivalent to

```python
clusters=semantic_network.cluster(levels=1, name="Test", to_measure=[proximity,centrality], metadata={'years':[1992,2005], 'context':['USA','China']}, years=[1992,2005], weight_cutoff=0.05, context=['USA','China'])
```
