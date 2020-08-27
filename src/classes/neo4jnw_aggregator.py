from src.classes.neo4jnw import neo4j_network
import pandas as pd
import networkx as nx
import itertools

class neo4jnw_aggregator():

    def __init__(self, neo4j_creds, weight_cutoff=None,norm_ties=False):
        # Create network
        self.neo4nw=neo4j_network(neo4j_creds)
        self.norm_ties=norm_ties
        self.weight_cutoff=weight_cutoff


    def yearly_centralities(self, year_list, tokens=None, ego_nw_tokens=None, depth=1, types=["PageRank"]):
        """
        Calculate year by year centrality.
        :param year_list: List of years for which to calculate centrality
        :param tokens: List of tokens of interest
        :param ego_nw_tokens: List of tokens for an ego-network if desired
        :param depth: Maximal path length for ego network
        :param types: Types of centrality to calculate
        :return:
        """
        cent_year={}
        assert isinstance(year_list, list), "Please provide list of years."
        for year in year_list:
            cent_measures=self.centrality(tokens=tokens, years=year,ego_nw_tokens=ego_nw_tokens, depth=depth, types=types)
            cent_year.update({year:cent_measures})

        return cent_year

    def centrality(self,tokens=None,years=None, ego_nw_tokens=None, depth=1, types=["PageRank"]):
        """
        Calculate centralities for given tokens over an aggregate of given years
        :param tokens: List of tokens of interest
        :param years: Given year, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :param ego_nw_tokens: List of tokens for an ego-network if desired
        :param depth: Maximal path length for ego network
        :param types: Types of centrality to calculate
        :return: Dictionary of dictionary, form {measure:{token_id: centrality}}
        """
        if years is not None:
            assert isinstance(years, dict) or isinstance(years, int), "Parameter years must be int or interval dict {'start':int,'end':int}"
        if tokens is not None:
            assert isinstance(tokens, list) or isinstance(tokens, int) or isinstance(tokens, str), "Token parameter should be string, int or list."

        # Condition either overall, or via ego network
        if ego_nw_tokens==None:
            self.neo4nw.condition(years, token_ids=None, weight_cutoff=self.weight_cutoff, depth=None, context=None, norm=self.norm_ties)
        else:
            self.neo4nw.condition(years, token_ids=ego_nw_tokens, weight_cutoff=self.weight_cutoff, depth=depth, context=None, norm=self.norm_ties)

        # Get list of token ids
        if tokens==None:
            token_ids=self.neo4nw.ids
        else:
            token_ids=self.neo4nw.ensure_ids(tokens)
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
            except:
                raise Exception("Could not calculate Page Rank centralities")
        else:
            raise AttributeError("Centrality measure not found in list")

        return centralities