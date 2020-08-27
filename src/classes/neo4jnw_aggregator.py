from src.classes.neo4jnw import neo4j_network
import pandas as pd
import networkx as nx
import itertools

class neo4jnw_aggregator():

    def __init__(self, neo4j_creds, norm_ties=False):
        # Create network
        self.neo4nw=neo4j_network(neo4j_creds)
        self.norm_ties=norm_ties


    def centralities(self,tokens=None,years=None):
        """
        Calculate centralities for given tokens and given years
        :param tokens: List of tokens
        :param years: Given year, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :return: Pandas data frame with sorted centralities
        """
        if years is not None:
            assert isinstance(years, dict) or isinstance(years, int), "Parameter years must be int or interval dict {'start':int,'end':int}"
        if tokens is not None:
            assert isinstance(tokens, list) or isinstance(tokens, int) or isinstance(tokens, str), "Token parameter should be string, int or list."

        return

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