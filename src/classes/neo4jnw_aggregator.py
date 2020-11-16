from src.classes.neo4jnw import neo4j_network
import networkx as nx
import itertools
import logging
import numpy as np

class neo4jnw_aggregator():

    def __init__(self, neo4nw, weight_cutoff=None,norm_ties=False, logging_level=logging.NOTSET):
        # Set logging level
        logging.disable(logging_level)
        # Create network
        self.neo4nw=neo4nw
        self.norm_ties=norm_ties
        self.weight_cutoff=weight_cutoff

    def proximities(self,focal_tokens, years, context=None, alter_tokens=None):
        """
        Calculate proximities for given tokens.
        :param focal_tokens: List of tokens of interest. If not provided, centralities for all tokens will be returned.
        :param years: Given year, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :param context: List of tokens that need to appear in the context distribution of a tie
        :return: Dictionary of form {token_id:{alter_id: proximity}}
        """
        # Input checks
        assert isinstance(years, dict) or isinstance(years, int), "Parameter years must be int or interval dict {'start':int,'end':int}"
        assert isinstance(focal_tokens, list) or isinstance(focal_tokens, int) or isinstance(focal_tokens, str), "Token parameter should be string, int or list."
        proximity_dict={}
        for token in focal_tokens:
            logging.debug("Conditioning year(s) {} with focus on tokens {}".format(years, tokens))
            self.neo4nw.condition(years, token_ids=[token], weight_cutoff=self.weight_cutoff, depth=1, context=None, norm=self.norm_ties)
            # Get list of alter token ids, either those found in network, or those specified by user
            if alter_tokens==None:
                token_ids=self.neo4nw.ids
            else:
                token_ids=self.neo4nw.ensure_ids(alter_tokens)
            logging.debug("token_ids {}".format(token_ids))
            if isinstance(token_ids,int):
                token_ids=[token_ids]
            
            # Extract 
            neighbors = self.neo4nw.graph[token]
            n_keys=list(neighbors.keys())
            # Choose only relevant alters
            n_keys=np.intersect1d(token_ids,n_keys)
            neighbors=neighbors[n_keys]

            # Extract edge weights and sort by weight
            edge_weights = [x['weight'] for x in neighbors.values()]
            edge_sort = np.argsort(-np.array(edge_weights))
            neighbors = [x for x in neighbors]
            edge_weights = np.array(edge_weights)
            neighbors = np.array(neighbors)
            edge_weights = edge_weights[edge_sort]
            neighbors = neighbors[edge_sort]

            tie_dict=dict(zip(neighbors,edge_weights))
            proximity_dict.update({token:tie_dict})
        return proximity_dict
     
    def yearly_centralities(self, year_list, focal_tokens=None, ego_nw_tokens=None, depth=1, types=["PageRank"]):
        """
        Calculate year by year centrality.
        :param year_list: List of years for which to calculate centrality
        :param focal_tokens: List of focal tokens of interest
        :param ego_nw_tokens: List of tokens for an ego-network if desired
        :param depth: Maximal path length for ego network
        :param types: Types of centrality to calculate
        :return:
        """
        cent_year={}
        assert isinstance(year_list, list), "Please provide list of years."
        for year in year_list:
            cent_measures=self.centrality(tokens=focal_tokens, years=year,ego_nw_tokens=ego_nw_tokens, depth=depth, types=types)
            cent_year.update({year:cent_measures})
            

        return cent_year

    def centrality(self,focal_tokens=None,years=None, ego_nw_tokens=None, depth=1, types=["PageRank","normedPageRank"]):
        """
        Calculate centralities for given tokens over an aggregate of given years
        :param focal_tokens: List of tokens of interest. If not provided, centralities for all tokens will be returned.
        :param years: Given year, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :param ego_nw_tokens: List of tokens for an ego-network if desired
        :param depth: Maximal path length for ego network
        :param types: Types of centrality to calculate
        :return: Dictionary of dictionary, form {measure:{token_id: centrality}}
        """
        if years is not None:
            assert isinstance(years, dict) or isinstance(years, int), "Parameter years must be int or interval dict {'start':int,'end':int}"
        if focal_tokens is not None:
            assert isinstance(focal_tokens, list) or isinstance(focal_tokens, int) or isinstance(focal_tokens, str), "Token parameter should be string, int or list."
        if isinstance(types, str):
            types=[types]
        elif not isinstance(types, list):
            logging.error("Centrality types must be list")
            raise ValueError("Centrality types must be list")
        # Condition either overall, or via ego network
        if ego_nw_tokens==None:
            logging.debug("Conditioning year(s) {} with focus on tokens {}".format(years, focal_tokens))
            self.neo4nw.condition(years, token_ids=None, weight_cutoff=self.weight_cutoff, depth=None, context=None, norm=self.norm_ties)
            logging.debug("Finished conditioning, {} nodes and {} edges in graph".format(len(self.neo4nw.graph.nodes), len(self.neo4nw.graph.edges)))

        else:
            logging.debug("Conditioning ego-network for {} tokens with depth {}, for year(s) {} with focus on tokens {}".format(len(ego_nw_tokens), depth, years, focal_tokens))
            logging.debug("Conditioning ego-network for {} tokens with depth {}, for year(s) {} with focus on tokens {}".format(len(ego_nw_tokens), depth, years, focal_tokens))
            self.neo4nw.condition(years, token_ids=ego_nw_tokens, weight_cutoff=self.weight_cutoff, depth=depth, context=None, norm=self.norm_ties)
            logging.debug("Finished ego conditioning, {} nodes and {} edges in graph".format(len(self.neo4nw.graph.nodes), len(self.neo4nw.graph.edges)))

        # Get list of token ids
        logging.debug("Tokens {}".format(focal_tokens))
        if focal_tokens==None:
            token_ids=self.neo4nw.ids
        else:
            token_ids=self.neo4nw.ensure_ids(focal_tokens)
        logging.debug("token_ids {}".format(token_ids))
        if isinstance(token_ids,int):
            token_ids=[token_ids]
        measures={}
        for measure in types:
        # PageRank centrality
            centralities=self.compute_centrality(self.neo4nw.graph, measure)
            for node in token_ids:
                if node in centralities.keys():
                    centralities.update({node: centralities[node]})
                else:
                    centralities.update({node: 0})
            measures.update({measure:centralities})


        # Decondition
        self.neo4nw.decondition()
        return measures

    def symmetrize_graph(self, method="avg"):
        if self.neo4nw.conditioned==False:
            raise AttributeError("Graph must be conditioned to symmetrize it!")
        if method=="avg":
            # First use networkx to translate edges randomly to undirected
            new_graph = neo4j_network.graph.to_undirected()
            # Now redo weights correctly
            nodepairs = itertools.combinations(list(neo4j_network.graph), r=2)
            for u, v in nodepairs:
                if neo4j_network.graph.has_edge(u, v) or neo4j_network.graph.has_edge(v, u):
                    wt = 0
                    if neo4j_network.graph.has_edge(u, v):
                        wt = wt + neo4j_network.graph.edges[u, v]['weight']
                    if neo4j_network.graph.has_edge(v, u):
                        wt = wt + neo4j_network.graph.edges[v, u]['weight']
                    wt = wt / 2
                    new_graph[u][v]['weight'] = wt
            neo4j_network.graph = new_graph
        else:
            raise AttributeError("Method parameter not recognized")

    def compute_centrality(self, graph, measure):
        if measure=="PageRank":
            # PageRank centrality
            try:
                centralities = nx.pagerank_scipy(self.neo4nw.graph, weight='weight')
                logging.debug(
                    "Calculated {} PageRank centralities".format(
                        len(centralities)))

            except:
                logging.error("Could not calculate Page Rank centralities")
                raise Exception("Could not calculate Page Rank centralities")
        elif measure=="normedPageRank":
            # PageRank centrality
            try:
                centralities = nx.pagerank_scipy(self.neo4nw.graph, weight='weight')
                logging.debug(
                    "Calculated {} normalized PageRank centralities".format(
                        len(centralities)))
                centvec=np.array(list(centralities.values()))
                normconst=np.sum(centvec)
                nr_n=len(centvec)
                
                centvec=(centvec/normconst)*nr_n
                logging.debug("For N={}, sum of changed vector is {}".format(nr_n,np.sum(centvec)))
                centralities=dict(zip(centralities.keys(),centvec))

            except:
                logging.error("Could not calculate normed Page Rank centralities")
                raise Exception("Could not calculate normed Page Rank centralities")
        else:
            raise AttributeError("Centrality measure {} not found in list".format(measure))

        return centralities