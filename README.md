# Text2Network

## Introduction

Text2Network is a python package to generate "semantic networks" from arbitrary text sources using a (deep neural) PyTorch transformer model (e.g. BERT).
Details for this procedure are available in the following research paper: (Coming soon)

Text2Network allows you to understand direct and higher-order linguistic relationships in your texts. Since these relationships are extracted directly from a sophisticated transformer model, they are more powerful than prior approaches. For example, once a text is represented as network, you can query each potential relation between words conditional on a certain context, such as other words that may appear in a sentence. You can also aggregate across arbitrary time periods or text pieces. The network is constructed in such a way that each conditioning operation leads to proper probability measures that are directly interpretable without further nuisance parameters and refer, correctly, to the language use of the subset in question - be it a given context, or a subset of the corpus.
Using network techniques, such as centralities, distances and community structures, you can then analyze global structure of language as used in the text corpus.

The semantic network also represents all the information about linugistic dependencies captured by the deep neural network. As such, it is useful to understand both the text source, as well as the model. Note, however, that this procedure is inferential and does not provide a good input for downstream tasks.

 In addition, we provide a host of analysis functions and formatting options. For example, printing the centralities of the terms "president","tyrant","man" and "woman" from a pandas dataframe can be accomplished in two lines:

``````
semantic_network=sem_network(config)``

print(pd_format(semantic_network.centralities(focal_tokens=['president','tyrant','man', 'woman'])))
``````

If you instead want to print the centralities in a 2-step ego network with "tyrant" as its center, considering texts written in the year 2002, just write

``````
semantic_network=sem_network(config)

print(pd_format(semantic_network.centralities(focal_tokens=['president','tyrant','man', 'woman'], years=[2002], ego_nw_tokens="tyrant", depth=2)))
``````
Or, if you want to do the same but restrict inference to contexts in which "USA" is likely to occur, do

``````
semantic_network=sem_network(config)

print(pd_format(semantic_network.centralities(focal_tokens=['president','tyrant','man', 'woman'], ego_nw_tokens="tyrant", depth=2, context=["USA"])))
``````

Each time, the created network and its measures will be based on the correctly conditioned probability measures that the language model generates.

If you want to repeat the analysis in a symmetric network, using 

## Components

Text2Networks includes four python objects, one for preprocessing, one for training the language model, one for inference and one class that directly represents the semantic network and all operations on it.

The objects correspond to the four steps necessary to analyze a corpus:

1. Pre-processing: Texts are ingested, cleaned and saved in a hdfs database. Associated metadata is created, this notably includes a time variable and a number of freely chosen parameters.

2. Training of the DNN: The researcher determines which combination of parameters constitutes a unit of analysis (for example newspaper-year-issue) and passes this hierarchy to the trainer class, which trains the required number of DNN models. Please see the research paper why this is required to capture linguistic relations in the text corpus correctly.

3. Processing: Using the trained language models, the each word in the corpus is analyzed in its context. For each such occurrence, ties representing probabilistic relations are created. Here, we use a Neo4j graph database to save this network.

4. Analysis: The semantic_network class directly encapsulates the graph database and conditioning operations thereon. Networks created here can either be analyzed directly, or exported in gexf format.
