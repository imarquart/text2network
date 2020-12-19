import copy
import logging
from collections.abc import MutableSequence
from src.utils.twowaydict import TwoWayDict
import numpy as np
# import neo4j
import neo4jCon as neo_connector


class neo4j_database():
    def __init__(self, neo4j_creds, agg_operator="SUM",
                 write_before_query=True,
                 neo_batch_size=10000, queue_size=100000, tie_query_limit=100000, tie_creation="UNSAFE",
                 logging_level=logging.NOTSET):
        # Set logging level
        logging.disable(logging_level)

        self.neo4j_connection, self.neo4j_credentials = neo4j_creds
        self.write_before_query = write_before_query

        # Neo4J Internals
        # Pick Merge or Create. Create will double ties but Merge becomes very slow for large networks
        if tie_creation == "SAFE":
            self.creation_statement = "MERGE"
        else:
            self.creation_statement = "CREATE"
        self.neo_queue = []
        self.neo_batch_size = neo_batch_size
        self.queue_size = queue_size
        self.aggregate_operator = agg_operator
        self.connector = neo_connector.Connector(self.neo4j_connection, self.neo4j_credentials)
        # Init tokens
        self.init_tokens()
        # Init parent class
        super().__init__()

    # %% Setup
    def setup_neo_db(self, tokens, token_ids):
        """
        Creates tokens and token_ids in Neo database. Does not delete existing network!
        :param tokens: list of tokens
        :param token_ids: list of corresponding token IDs
        :return: None
        """
        logging.debug("Creating indecies and nodes in Neo4j database.")
        constr = [x['name'] for x in self.connector.run("CALL db.constraints")]
        # Create uniqueness constraints
        logging.debug("Creating constraints in Neo4j database.")
        if 'id_con' not in constr:
            query = "CREATE CONSTRAINT id_con ON(n:word) ASSERT n.token_id IS UNIQUE"
            self.add_query(query)
        if 'tk_con' not in constr:
            query = "CREATE CONSTRAINT tk_con ON(n:word) ASSERT n.token IS UNIQUE"
            self.add_query(query)
        constr = [x['name'] for x in self.connector.run("CALL db.indexes")]
        if 'timeindex' not in constr:
            query = "CREATE INDEX timeindex FOR (a:edge) ON (a.time)"
            self.add_query(query)
        if 'contimeindex' not in constr:
            query = "CREATE INDEX contimeindex FOR (a:context) ON (a.time)"
            self.add_query(query)
        # Need to write first because create and structure changes can not be batched
        self.non_con_write_queue()
        # Create nodes in neo db
        queries = [''.join(["MERGE (n:word {token_id: ", str(id), ", token: '", tok, "'})"]) for tok, id in
                   zip(tokens, token_ids)]
        self.add_queries(queries)
        self.non_con_write_queue()

    def clean_database(self, time=None, del_limit=1000000):
        # DEBUG
        nr_nodes = self.connector.run("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
        nr_context = self.connector.run("MATCH (n:context) RETURN count(*) AS nodes")[0]['nodes']
        logging.info("Before cleaning: Network has %i edge-nodes and %i context-nodes" % (nr_nodes, nr_context))

        if time is not None:
            # Delete previous edges
            node_query = ''.join(
                ["MATCH (p:edge {time:", str(time), "}) WITH p LIMIT ", str(del_limit), " DETACH DELETE p"])
            # Delete previous context edges
            context_query = ''.join(
                ["MATCH (p:context {time:", str(time), "})  WITH p LIMIT ", str(del_limit), "  DETACH DELETE p"])
        else:
            # Delete previous edges
            node_query = ''.join(
                ["MATCH (p:edge)  WITH p LIMIT ", str(del_limit), " DETACH DELETE p"])
            # Delete previous context edges
            context_query = ''.join(
                ["MATCH (p:context)  WITH p LIMIT ", str(del_limit), " DETACH DELETE p"])

        while nr_nodes > 0:
            # Delete edge nodes
            self.connector.run(node_query)
            nr_nodes = self.connector.run("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
            logging.info("Network has %i edge-nodes and %i context-nodes" % (nr_nodes, nr_context))
        while nr_context > 0:
            # Delete context nodes
            self.connector.run(context_query)
            nr_context = self.connector.run("MATCH (n:context) RETURN count(*) AS nodes")[0]['nodes']
            logging.info("Network has %i edge-nodes and %i context-nodes" % (nr_nodes, nr_context))

        # DEBUG
        nr_nodes = self.connector.run("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
        nr_context = self.connector.run("MATCH (n:context) RETURN count(*) AS nodes")[0]['nodes']
        logging.info("After cleaning: Network has %i nodes and %i ties" % (nr_nodes, nr_context))

    # %% Initializations
    def init_tokens(self):
        """
        Gets all tokens and token_ids in the database
        and sets up two-way dicts
        :return: ids,tokens
        """
        logging.debug("Querying tokens and filling data structure.")
        # Run neo query to get all nodes
        res = self.connector.run("MATCH (n:word) RETURN n.token_id, n.token")
        # Update results
        ids = [x['n.token_id'] for x in res]
        tokens = [x['n.token'] for x in res]
        return ids, tokens

    # %% Query Functions
    def query_multiple_nodes(self, ids, times=None, weight_cutoff=None, norm_ties=True):
        """
        Query multiple nodes by ID and over a set of time intervals
        :param ids: list of id's
        :param times: either a number format YYYYMMDD, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :param weight_cutoff: float in 0,1
        :return: list of tuples (u,v,Time,{weight:x})
        """
        logging.debug("Querying {} nodes in Neo4j database.".format(len(ids)))
        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query = ''.join([" WHERE r.weight >=", str(weight_cutoff), " "])
            if isinstance(times, dict):
                where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        else:
            if isinstance(times, dict):
                where_query = "WHERE  $times.start <= r.time<= $times.end "
            else:
                where_query = ""
        # Create query depending on graph direction and whether time variable is queried via where or node property
        # By default, a->b when ego->is_replaced_by->b
        # Given an id, we query b:id(sender)<-a(receiver)
        # This gives all ties where b -predicts-> a
        return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver,count(r.pos) AS occurrences,",
                                self.aggregate_operator, "(r.weight) AS agg_weight order by receiver"])

        if isinstance(times, int):
            match_query = "UNWIND $ids AS id MATCH p=(a:word)-[:onto]->(r:edge {time:$times})-[:onto]->(b:word {token_id:id}) "
        else:
            match_query = "unwind $ids AS id MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word {token_id:id}) "

        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        query = "".join([match_query, where_query, return_query])
        res = self.connector.run(query, params)

        tie_weights = np.array([x['agg_weight'] for x in res])
        senders = [x['sender'] for x in res]
        receivers = [x['receiver'] for x in res]
        occurrences = [x['occurrences'] for x in res]
        # Normalization
        # Ties should be normalized by the number of occurrences of the receiver
        if norm_ties == True:
            norms = dict(self.query_occurrences(receivers, times, weight_cutoff))
            for i, token in enumerate(receivers):
                tie_weights[i] = tie_weights[i] / norms[token]

        ties = [
            (x[0], x[1],
             {'weight': x[2], 'time': nw_time['m'], 'start': nw_time['s'], 'end': nw_time['e'], 'occurrences': x[3]})
            for x in zip(senders, receivers, tie_weights, occurrences)]

        return ties

    def query_multiple_nodes_in_context(self, ids, context, times=None, weight_cutoff=None, norm_ties=True):
        """
        Query multiple nodes by ID and over a set of time intervals
        Each replacement must occur within a context-element distribution including at least one
        contextual token in context list
        :param ids: list of id's
        :param context: list of context ids
        :param times: either a number format YYYYMMDD, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :param weight_cutoff: float in 0,1
        :return: list of tuples (u,v,Time,{weight:x})
        """
        logging.debug("Querying {} nodes in Neo4j database.".format(len(ids)))

        # Create context query
        context_where = ' ALL(r in nodes(p) WHERE size([(r) - [: conto]->(:context) - [: conto]->(e:word) WHERE e.token_id IN $clist | e]) > 0 OR(r: word))'

        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query = ''.join([" WHERE r.weight >=", str(weight_cutoff), " AND "])
            if isinstance(times, dict):
                where_query = ''.join([where_query, " $times.start <= r.time<= $times.end "])
        else:
            if isinstance(times, dict):
                where_query = "WHERE  $times.start <= r.time<= $times.end AND "
            else:
                where_query = "WHERE "

        # Join where and context query
        where_query = ' '.join([where_query, context_where])

        # Create query depending on graph direction and whether time variable is queried via where or node property
        # By default, a->b when ego->is_replaced_by->b
        # Given an id, we query b:id(sender)<-a(receiver)
        # This gives all ties where b -predicts-> a
        return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver,count(r.pos) AS occurrences,",
                                self.aggregate_operator, "(r.weight) AS agg_weight order by receiver"])

        if isinstance(times, int):
            match_query = "UNWIND $ids AS id MATCH p=(a:word)-[:onto]->(r:edge {time:$times})-[:onto]->(b:word {token_id:id}) "
        else:
            match_query = "unwind $ids AS id MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word {token_id:id}) "

        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int):
            params = {"ids": ids, "times": times, "clist": context}
        else:
            params = {"ids": ids, "clist": context}

        query = "".join([match_query, where_query, return_query])
        res = self.connector.run(query, params)

        tie_weights = np.array([x['agg_weight'] for x in res])
        senders = [x['sender'] for x in res]
        receivers = [x['receiver'] for x in res]
        occurrences = [x['occurrences'] for x in res]
        # Normalization
        # Ties should be normalized by the number of occurrences of the receiver
        if norm_ties == True:
            norms = dict(self.query_occurrences_in_context(receivers, context, times, weight_cutoff))
            for i, token in enumerate(receivers):
                tie_weights[i] = tie_weights[i] / norms[token]

        ties = [
            (x[0], x[1],
             {'weight': x[2], 'time': nw_time['m'], 'start': nw_time['s'], 'end': nw_time['e'], 'occurrences': x[3]})
            for x in zip(senders, receivers, tie_weights, occurrences)]

        return ties

    def query_occurrences(self, ids, times=None, weight_cutoff=None):
        """
        Query multiple nodes by ID and over a set of time intervals, return distinct occurrences
        :param ids: list of id's
        :param times: either a number format YYYY, or an interval dict {"start":YYYY,"end":YYYY}
        :param weight_cutoff: float in 0,1
        :return: list of tuples (u,occurrences)
        """
        logging.debug("Querying {} node occurrences for normalization".format(len(ids)))
        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        # If times is a dict, we want an interval and hence where query
        if weight_cutoff is not None:
            where_query = ''.join([" WHERE r.weight >=", str(weight_cutoff), " "])
            if isinstance(times, dict):
                where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        else:
            if isinstance(times, dict):
                where_query = "WHERE  $times.start <= r.time<= $times.end "
            else:
                where_query = ""
        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        return_query = ''.join([
                                   " WITH a.token_id AS idx, r.seq_id AS sequence_id ,(r.time) as year, count(DISTINCT(r.pos)) as pos_count RETURN idx, sum(pos_count) AS occurrences order by idx"])

        if isinstance(times, int):
            match_query = "UNWIND $ids AS id MATCH p=(a:word  {token_id:id})-[:onto]->(r:edge {time:$times})-[:onto]->(b:word) "
        else:
            match_query = "unwind $ids AS id MATCH p=(a:word  {token_id:id})-[:onto]->(r:edge)-[:onto]->(b:word) "

        query = "".join([match_query, where_query, return_query])
        res = self.connector.run(query, params)

        ties = [(x['idx'], x['occurrences']) for x in res]

        return ties

    def query_occurrences_in_context(self, ids, context, times=None, weight_cutoff=None):
        """
        Query multiple nodes by ID and over a set of time intervals, return distinct occurrences
        under the condition that elements of context are present in the context element distribution of
        this occurrence
        :param ids: list of ids
        :param context: list of ids
        :param times: either a number format YYYY, or an interval dict {"start":YYYY,"end":YYYY}
        :param weight_cutoff: float in 0,1
        :return: list of tuples (u,occurrences)
        """

        logging.debug("Querying {} node occurrences for normalization".format(len(ids)))

        # Create context query
        context_where = ' ALL(r in nodes(p) WHERE size([(r) - [: conto]->(:context) - [: conto]->(e:word) WHERE e.token_id IN $clist | e]) > 0 OR(r: word))'

        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        # If times is a dict, we want an interval and hence where query
        if weight_cutoff is not None:
            where_query = ''.join([" WHERE r.weight >=", str(weight_cutoff), " AND "])
            if isinstance(times, dict):
                where_query = ''.join([where_query, " $times.start <= r.time<= $times.end AND "])
        else:
            if isinstance(times, dict):
                where_query = "WHERE  $times.start <= r.time<= $times.end AND "
            else:
                # Always need a WHERE query for context
                where_query = "WHERE "
        # Join where and context query
        where_query = ' '.join([where_query, context_where])

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int):
            params = {"ids": ids, "times": times, "clist": context}
        else:
            params = {"ids": ids, "clist": context}

        # return_query = ''.join([" WITH a.token_id AS idx, r.seq_id AS sequence_id ,(r.time) as year, count(DISTINCT(r.pos)) as pos_count RETURN idx, sum(pos_count) AS occurrences order by idx"])
        return_query = " WITH s.token_id as idx, count(r.pos) as pos_count, r.seq_id as seq_id return idx, sum(pos_count) as occurrences order by idx"
        if isinstance(times, int):
            match_query = "UNWIND $ids AS id MATCH p=(s:word  {token_id:id})-[:onto]->(r:edge {time:$times})-[:onto]->(v:word) "
        else:
            match_query = "unwind $ids AS id MATCH p=(s:word  {token_id:id})-[:onto]->(r:edge)-[:onto]->(v:word) "

        query = "".join([match_query, where_query, return_query])
        logging.debug(query)
        res = self.connector.run(query, params)

        ties = [(x['idx'], x['occurrences']) for x in res]

        return ties

    def query_context_element(self, ids, times=None, weight_cutoff=None, disable_normalization=False,
                              replacement_ce=False, ):
        """
        Queries the aggregated context element distribution for tokens given by ids.
        P(c|t € s) where t is the focal token appearing in s, and c is another random token appearing in s.
        Note that stopwords are ignored.
        :param ids: ids of focal token
        :param times: int or interval dict {"start":YYYY,"end":YYYY}
        :param weight_cutoff: list of tuples (u,v,Time,{weight:x, further parameters})
        :param disable_normalization: Do not normalize context element distribution
        :param replacement_ce: Query the context in which focal token replaces another token (reverse direction)
        :return:
        """
        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query = ''.join([" WHERE c.weight >=", str(weight_cutoff), " "])
            if isinstance(times, dict):
                where_query = ''.join([where_query, " AND  $times.start <= c.time<= $times.end "])
        else:
            if isinstance(times, dict):
                where_query = "WHERE  $times.start <= c.time<= $times.end "
            else:
                where_query = ""
        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        # Format return query
        return_query = ''.join(
            [" RETURN b.token_id AS context_token,a.token_id AS focal_token,count(c.time) AS occurrences,",
             self.aggregate_operator, "(c.weight) AS agg_weight order by agg_weight"])
        if replacement_ce == False:
            if isinstance(times, int):
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})-[: onto]->(r:edge) - [: conto]->(c:context {time:$times}) - [: conto]->(b:word)"
            else:
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})-[: onto]->(r:edge) - [: conto]->(c:context) - [: conto]->(b:word)"
        else:
            # Reverse direction
            if isinstance(times, int):
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})<-[: onto]-(r:edge) - [: conto]->(c:context {time:$times}) - [: conto]->(b:word)"
            else:
                match_query = "UNWIND $ids AS id MATCH p=(a: word {token_id:id})<-[: onto]-(r:edge) - [: conto]->(c:context) - [: conto]->(b:word)"
        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        query = "".join([match_query, where_query, return_query])
        res = self.connector.run(query, params)
        focal_tokens = np.array([x['focal_token'] for x in res])
        context_tokens = np.array([x['context_token'] for x in res])
        weights = np.array([x['agg_weight'] for x in res])
        occurrences = np.array([x['occurrences'] for x in res])
        # Normalize context element
        if disable_normalization == False:
            for focal_token in np.unique(focal_tokens):
                mask = focal_tokens == focal_token
                weight_sum = np.sum(weights[mask])
                weights[mask] = weights[mask] / weight_sum

        ties = [(x[0], x[1], nw_time['m'],
                 {'weight': x[2], 't1': nw_time['s'], 't2': nw_time['e'], 'occurrences': x[3]})
                for x in zip(focal_tokens, context_tokens, weights, occurrences)]
        return ties

    # %% Insert functions
    def insert_edges_context(self, ego, ties, contexts, logging_level=logging.DEBUG):
        if logging_level is not None:
            logging.disable(logging_level)
        logging.debug("Insert {} ego nodes with {} ties".format(ego, len(ties)))
        # Tie direction matters
        # Ego by default is the focal token to be replaced. Normal insertion points the link accordingly.
        # Hence, a->b is an instance of b replacing a!
        # Contextual ties always point toward the context word!

        egos = np.array([x[0] for x in ties])
        alters = np.array([x[1] for x in ties])
        con_alters = np.array([x[1] for x in contexts])

        times = np.array([x[2] for x in ties])
        dicts = np.array([x[3] for x in ties])
        con_times = np.array([x[2] for x in contexts])
        con_dicts = np.array([x[3] for x in contexts])

        unique_egos = np.unique(egos)
        if len(unique_egos) == 1:
            ties_formatted = [{"alter": int(x[0]), "time": int(x[1]), "weight": float(x[2]['weight']),
                               "seq_id": int(x[2]['seq_id']),
                               "pos": int(x[2]['pos']),
                               "run_index": int(x[2]['run_index']),
                               "p1": ((x[2]['p1']) if len(x[2]) > 4 else 0),
                               "p2": ((x[2]['p2']) if len(x[2]) > 5 else 0),
                               "p3": ((x[2]['p3']) if len(x[2]) > 6 else 0),
                               "p4": ((x[2]['p4']) if len(x[2]) > 7 else 0), }
                              for x in zip(alters.tolist(), times.tolist(), dicts.tolist())]
            contexts_formatted = [{"alter": int(x[0]), "time": int(x[1]), "weight": float(x[2]['weight']),
                                   "seq_id": int(x[2]['seq_id'] if len(x[2]) > 1 else 0),
                                   "pos": int(x[2]['pos'] if len(x[2]) > 2 else 0),
                                   "run_index": int(x[2]['run_index'] if len(x[2]) > 3 else 0),
                                   "p1": ((x[2]['p1']) if len(x[2]) > 4 else 0),
                                   "p2": ((x[2]['p2']) if len(x[2]) > 5 else 0),
                                   "p3": ((x[2]['p3']) if len(x[2]) > 6 else 0),
                                   "p4": ((x[2]['p4']) if len(x[2]) > 7 else 0), }
                                  for x in zip(con_alters.tolist(), con_times.tolist(), con_dicts.tolist())]
            params = {"ego": int(egos[0]), "ties": ties_formatted, "contexts": contexts_formatted}

            # Select order of parameters
            p1 = np.array([str(x['p1']) if (len(x)>4) else "0" for x in dicts ])
            p2 = np.array([str(x['p2']) if (len(x)>5) else "0" for x in dicts ])
            p3 = np.array([str(x['p3']) if (len(x)>6) else "0" for x in dicts ])
            p4 = np.array([str(x['p4']) if (len(x)>7) else "0" for x in dicts ])
            # Select order of context parameters
            cseq_id = np.array([x['seq_id'] if len(x)>1 else "0" for x in con_dicts ], dtype=np.str)
            cpos = np.array([x['pos'] if len(x)>2 else "0" for x in con_dicts ], dtype=np.str)
            crun_index = np.array([x['run_index'] if len(x)>3 else "0" for x in con_dicts ], dtype=np.str)
            cp1 =  np.array([str(x['p1']) if (len(x)>4) else "0" for x in con_dicts ])
            cp2 =  np.array([str(x['p2']) if (len(x)>5) else "0" for x in con_dicts ])
            cp3 =  np.array([str(x['p3']) if (len(x)>6) else "0" for x in con_dicts ])
            cp4 = np.array([str(x['p4'] )if (len(x)>7) else "0" for x in con_dicts ])


            # Build parameter string
            parameter_string=""
            if not all(p1 == "0") and not all(p1==''):
                parameter_string=parameter_string+", p1:tie.p1"
            if not all(p2 == "0") and not all( p2==''):
                parameter_string=parameter_string+", p2:tie.p2"
            if not all(p3 == "0") and not all(p3==''):
                parameter_string=parameter_string+", p3:tie.p3"
            if not all(p4 == "0") and not all( p4==''):
                parameter_string=parameter_string+", p4:tie.p4 "


            cparameter_string = ""
            if not all(cseq_id == "0") and not all(cseq_id==''):
                cparameter_string = cparameter_string + ", seq_id:con.seq_id"
            if not all(cpos == "0" ) and not all(cpos==''):
                cparameter_string = cparameter_string + ", pos:con.pos"
            if not all(crun_index == "0") and not all(crun_index==''):
                cparameter_string = cparameter_string + ", run_index:pos.run_index"
            if not all(cp1 == "0") and not all(cp1==''):
                cparameter_string = cparameter_string + ", p1:con.p1"
            if not all(cp2 == "0" ) and not all( cp2==''):
                cparameter_string = cparameter_string + ", p2:con.p2"
            if not all(cp3 == "0" ) and not all(cp3==''):
                cparameter_string = cparameter_string + ", p3:con.p3"
            if not all(cp4 == "0" ) and not all(cp4==''):
                cparameter_string = cparameter_string + ", p4:con.p4"


            query = ''.join(
                [" MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties as tie MATCH (b:word {token_id: tie.alter}) ",
                 self.creation_statement,
                 " (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time, seq_id:tie.seq_id,pos:tie.pos, run_index:tie.run_index ",parameter_string, "})<-[:onto]-(a) WITH r UNWIND $contexts as con MATCH (q:word {token_id: con.alter}) WITH r,q,con ",
                 self.creation_statement,
                 " (r)-[:conto]->(c:context {weight:con.weight, time:con.time ", cparameter_string, "})-[:conto]->(q)"])
        else:
            logging.error("Batched edge creation with context for multiple egos not supported.")
            raise NotImplementedError

        self.add_query(query, params)

    # %% Neo4J interaction
    # All function that interact with neo are here, dispatched as needed from above

    def add_query(self, query, params=None):
        """
        Add a single query to queue
        :param query: Neo4j query
        :param params: Associates parameters
        :return:
        """
        self.add_queries([query], [params])

    def add_queries(self, query, params=None):
        """
        Add a list of query to queue
        :param query: list - Neo4j queries
        :param params: list - Associates parameters corresponding to queries
        :return:
        """
        assert isinstance(query, list)

        if params is not None:
            assert isinstance(params, list)
            statements = [neo_connector.Statement(q, p) for (q, p) in zip(query, params)]
            self.neo_queue.extend(statements)
        else:
            statements = [neo_connector.Statement(q) for (q) in query]
            self.neo_queue.extend(statements)

        # Check for queue size if not conditioned!
        if (len(self.neo_queue) > self.queue_size) and (self.conditioned == False):
            self.write_queue()

    def write_queue(self):
        """
        If called will run queries in the queue and empty it.
        :return:
        """
        if len(self.neo_queue) > 0:
            ret = self.connector.run_multiple(self.neo_queue, self.neo_batch_size)
            logging.debug(ret)

            self.neo_queue = []

    def non_con_write_queue(self):
        """
        Utility function to write queue without concurrency (depreciated currently)
        :return:
        """
        self.write_queue()
