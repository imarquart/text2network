import asyncio
import logging
from typing import Union

import numpy as np


try:
    from neo4j import GraphDatabase
except:
    GraphDatabase = None


class neo4j_database():
    def __init__(self, neo4j_creds, agg_operator="SUM",
                 write_before_query=True,
                 neo_batch_size=10000, queue_size=100000, tie_query_limit=100000, tie_creation="UNSAFE",
                 context_tie_creation="SAFE",
                 logging_level=logging.NOTSET, connection_type="bolt", cache_yearly_occurrences=False,
                 consume_type=None):
        # Set logging level
        # logging.disable(logging_level)

        self.neo4j_connection, self.neo4j_credentials = neo4j_creds
        self.write_before_query = write_before_query
        oldlevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(30)
        # Set up Neo4j driver
        if connection_type == "bolt" and GraphDatabase is not None:
            self.driver = GraphDatabase.driver(self.neo4j_connection, auth=self.neo4j_credentials)
            self.connection_type = "bolt"
            self.read_session = None

        else:  # Fallback custom HTTP connector for Neo4j <= 4.02
            raise NotImplementedError("HTTP connection is no longer supported!")
        logging.getLogger().setLevel(oldlevel)
        # Neo4J Internals
        # Pick Merge or Create. Create will double ties but Merge becomes very slow for large networks
        if tie_creation == "SAFE":
            self.creation_statement = "MERGE"
        else:
            self.creation_statement = "CREATE"

        if context_tie_creation == "SAFE":
            self.context_creation_statement = "MERGE"
        else:
            self.context_creation_statement = "CREATE"
        self.neo_queue = []
        self.neo_batch_size = neo_batch_size
        self.queue_size = queue_size
        self.aggregate_operator = agg_operator

        self.consume_type = consume_type

        # Init tokens in the database, required since order etc. may be different
        self.db_ids, self.db_tokens = self.init_tokens()
        self.db_ids = np.array(self.db_ids)
        self.db_tokens = np.array(self.db_tokens)
        self.db_id_dict = {}

        # Occurrence Cache
        self.cache_yearly_occurrences = cache_yearly_occurrences
        self.occ_cache = {}
        # Init parent class
        super().__init__()

    # %% Setup

    def check_create_tokenid_dict(self, tokens, token_ids, fail_on_missing=False):
        tokens = np.array(tokens)
        self.db_tokens = np.array(self.db_tokens)
        token_ids = np.array(token_ids)
        db_ids = np.array(self.db_ids)
        if len(self.db_ids) > 0:
            new_ids = [np.where(self.db_tokens == k)[0][0] for k in tokens if k in self.db_tokens]
        else:
            new_ids = []
        if not len(new_ids) == len(token_ids):
            missing_ids = np.array(list(np.setdiff1d(token_ids, token_ids[new_ids])))
            missing_positions = np.where(token_ids == missing_ids)[0]
            missing_tokens = tokens[missing_positions]

            msg = "Token ID translation failed. {} Tokens missing in database: {}".format(len(missing_tokens),
                                                                                          missing_tokens)
            if fail_on_missing:
                logging.error(msg)
                raise ValueError(msg)

            added_ids = list(range(len(db_ids), len(db_ids) + len(missing_tokens)))

            db_id_dict = {x[0]: x[1] for x in zip(token_ids[new_ids], new_ids)}
            db_id_dict2 = {x[0]: x[1] for x in zip(missing_ids, added_ids)}
            db_id_dict.update(db_id_dict2)
            self.db_id_dict = db_id_dict
            return new_ids, missing_tokens, added_ids
        else:
            # Update
            self.db_id_dict = {x[0]: x[1] for x in zip(token_ids, new_ids)}
            return new_ids, [], []

    def reset_dictionary(self):
        """
        Usually, external token_ids differ from database ids. If the class is initialized with a dictionary,
        a translation is kept.

        This function resets this translation such that token ids correspond to the database.
        """
        self.db_id_dict = {x[0]: x[1] for x in zip(self.db_ids, self.db_ids)}

    def translate_token_ids(self, ids):
        try:
            new_ids = [self.db_id_dict[x] for x in ids]
        except:
            msg = "Could not translate {} token ids: {}".format(len(ids), ids)
            logging.error(msg)
            raise ValueError(msg)
        return new_ids

    def setup_neo_db(self, tokens, token_ids):
        """
        Creates tokens and token_ids in Neo database. Does not delete existing network!
        :param tokens: list of tokens
        :param token_ids: list of corresponding token IDs
        :return: None
        """
        logging.debug("Creating indecies and nodes in Neo4j database.")
        constr = [x['name'] for x in self.receive_query("CALL db.constraints")]
        # Create uniqueness constraints
        logging.debug("Creating constraints in Neo4j database.")
        if 'id_con' not in constr:
            query = "CREATE CONSTRAINT id_con ON(n:word) ASSERT n.token_id IS UNIQUE"
            self.add_query(query)
        if 'tk_con' not in constr:
            query = "CREATE CONSTRAINT tk_con ON(n:word) ASSERT n.token IS UNIQUE"
            self.add_query(query)
        constr = [x['name'] for x in self.receive_query("CALL db.indexes")]
        if 'timeindex' not in constr:
            query = "CREATE INDEX timeindex FOR (a:edge) ON (a.time)"
            self.add_query(query)
        if 'contimeindex' not in constr:
            query = "CREATE INDEX contimeindex FOR (a:context) ON (a.time)"
            self.add_query(query)
        if 'runidxindex' not in constr:
            query = "CREATE INDEX runidxindex FOR (a:edge) ON (a.run_index)"
            self.add_query(query)
        if 'conrunidxindex' not in constr:
            query = "CREATE INDEX conrunidxindex FOR (a:context) ON (a.run_index)"
            self.add_query(query)
        if 'posedgeindex' not in constr:
            query = "CREATE INDEX posedgeindex FOR (a:edge) ON (a.pos)"
            self.add_query(query)
        if 'posconindex' not in constr:
            query = "CREATE INDEX posconindex FOR (a:context) ON (a.pos)"
            self.add_query(query)
        # Need to write first because create and structure changes can not be batched
        self.non_con_write_queue()
        # Create nodes in neo db
        # Get rid of signs that can not be used
        tokens = [x.translate(x.maketrans({"\"": '#e1#', "'": '#e2#', "\\": '#e3#'})) for x in tokens]
        token_ids, missing_tokens, missing_ids = self.check_create_tokenid_dict(tokens, token_ids)
        if len(missing_tokens) > 0:
            queries = [''.join(["MERGE (n:word {token_id: ", str(id), ", token: '", tok, "'})"]) for tok, id in
                       zip(missing_tokens, missing_ids)]
            self.add_queries(queries)
            self.non_con_write_queue()
        self.db_ids, self.db_tokens = self.init_tokens()

    def clean_database(self, time=None, del_limit=1000000):
        # DEBUG
        nr_nodes = self.receive_query("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
        nr_context = self.receive_query("MATCH (n:context) RETURN count(*) AS nodes")[0]['nodes']
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
            self.add_query(node_query, run=True)
            nr_nodes = self.receive_query("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
            logging.info("Network has %i edge-nodes and %i context-nodes" % (nr_nodes, nr_context))
        while nr_context > 0:
            # Delete context nodes
            self.add_query(context_query, run=True)
            nr_context = self.receive_query("MATCH (n:context) RETURN count(*) AS nodes")[0]['nodes']
            logging.info("Network has %i edge-nodes and %i context-nodes" % (nr_nodes, nr_context))

        # DEBUG
        nr_nodes = self.receive_query("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
        nr_context = self.receive_query("MATCH (n:context) RETURN count(*) AS nodes")[0]['nodes']
        logging.info("After cleaning: Network has %i nodes and %i ties" % (nr_nodes, nr_context))

    # %% Initializations
    def init_tokens(self):
        """
        Gets all tokens and token_ids in the database
        and sets up two-way dicts
        :return: ids,tokens
        """
        logging.debug("Init tokens: Querying tokens and filling data structure.")
        # Run neo query to get all nodes
        res = self.receive_query("MATCH (n:word) RETURN n.token_id, n.token")
        # Update results
        ids = [x['n.token_id'] for x in res]
        tokens = [x['n.token'] for x in res]
        return ids, tokens

    def prune_database(self):
        """
        Deletes all disconnected nodes
        Returns
        -------

        """
        logging.debug("Pruning disconnected tokens in database.")
        self.add_query("MATCH (n) WHERE size((n)--())=0 DELETE (n)", run=True)

    # %% Query Functions

    def query_tie_context(self, occurring, replacing, times=None, weight_cutoff=None):
        """

        // dyadic context
        Match p=(x:word {token:"leader"})-[:onto]->(r:edge)-[:onto]->(v {token:"coach"})
        WITH  DISTINCT(r.run_index) as ridx, collect(DISTINCT r.pos) as rpos,sum(r.weight) as rweight MATCH (t:word)<-[:onto]-(q:edge {run_index:ridx})<-[:onto]-(x:word) WHERE NOT q.pos in rpos
        WITH t.token as idx, q.weight as qweight, rweight, rpos, q.pos as qpos  Return DISTINCT(idx) as idx, COUNT(qpos) as nr, sum(qweight*rweight) as weight, sum(qweight) order by weight DESC

        Parameters
        ----------
        occurring
        replacing
        times
        weight_cutoff

        Returns
        -------

        """
        logging.debug("Querying tie between {}->replacing->{}.".format(replacing, occurring))

        # Optimization for time
        if isinstance(times, list):
            if len(times) == 1:
                times = int(times[0])

        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        elif isinstance(times, list):
            sort_times = np.sort(times)
            nw_time = {"s": sort_times[0], "e": sort_times[-1], "m": int((sort_times[0] + sort_times[-1]) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int) or isinstance(times, list):
            params = {"occurring": occurring, "replacing": replacing, "times": times}
        else:
            params = {"occurring": occurring, "replacing": replacing, }

        where_query_1 = "WHERE TRUE "
        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query_1 = ''.join([where_query_1, " AND r.weight >=", str(weight_cutoff), " "])
        if isinstance(times, dict):
            where_query_1 = ''.join([where_query_1, " AND  $times.start <= r.time<= $times.end "])
        elif isinstance(times, list):
            where_query_1 = ''.join([where_query_1, " AND  r.time in $times "])

        where_query_2 = "  WHERE NOT q.pos in rpos AND Not x.token_id  in [$occurring,$replacing] AND Not t.token_id  in [$occurring,$replacing] "
        # Allow cutoff value of (non-aggregated) weights
        if weight_cutoff is not None:
            where_query_2 = ''.join([where_query_2, " AND q.weight >=", str(weight_cutoff), " "])

        if isinstance(times, int):
            match_query = "".join(["Match p=(s:word {token_id:$occurring})-[:onto]->(r:edge {time:", str(times),
                                   "})-[:onto]->(v:word {token_id:$replacing})  "])
        else:
            match_query = "Match p=(s:word {token_id:$occurring})-[:onto]->(r:edge)-[:onto]->(v:word {token_id:$replacing}) "

        match_query_2 = " WITH  DISTINCT(r.run_index) as ridx, collect(DISTINCT r.pos) as rpos,sum(r.weight) as rweight MATCH (t:word)<-[:onto]-(q:edge {run_index:ridx})<-[:onto]-(x:word) "

        return_query = "WITH t.token_id as idx, q.weight as weight, rweight  Return DISTINCT(idx) as idx, sum(weight*rweight) as weight order by idx"

        query = "".join([match_query, where_query_1, match_query_2, where_query_2, return_query])
        logging.debug("Tie Query: {}".format(query))
        res = self.receive_query(query, params)

        weights = np.array([x['weight'] for x in res])
        idx = np.array([x['idx'] for x in res])
        weights = weights / np.sum(weights)

        return idx, weights

    def query_context_of_node(self, ids, times=None, weight_cutoff=None, occurrence=False):
        """

        //Context from Replacement
        Match p=(r:edge)-[:onto]->(v:word)  WHERE v.token in ["leader"]  AND  r.time in [1995]
        WITH DISTINCT(r.run_index) as ridx, collect(DISTINCT r.pos) as rpos,sum(r.weight) as rweight, v.token as ego
        MATCH (t: word) < -[: onto]-  (q:edge {run_index:ridx}) WHERE not q.pos in rpos AND NOT t.token = ego
        WITH t.token as idx, q.weight as qweight, rweight, ego
        Return DISTINCT(idx) as alter, ego, sum(qweight*rweight) as weight, sum(qweight) as qw, sum(rweight) as rw order by weight DESC

        Parameters
        ----------
        ids
        times
        weight_cutoff
        occurrence

        Returns
        -------

        """
        # New where query
        where_query = " WHERE v.token_id in $ids"

        # Optimization for time
        if isinstance(times, list):
            if len(times) == 1:
                times = int(times[0])

        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query = ''.join([where_query, " AND r.weight >=", str(weight_cutoff), " "])
        if isinstance(times, dict):
            where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        elif isinstance(times, list):
            where_query = ''.join([where_query, " AND  r.time in $times "])
        elif isinstance(times, int):
            where_query = ''.join([where_query, " AND  r.time = $times "])

        if occurrence:
            match_query_1 = "Match p=(r:edge)<-[:onto]-(v:word) "
        else:
            match_query_1 = "Match p=(r:edge)-[:onto]->(v:word) "

        if occurrence:
            match_query_2 = "WITH DISTINCT(r.run_index) as ridx, collect(DISTINCT r.pos) as rpos,sum(r.weight) as rweight, v.token_id as ego  MATCH (t: word)  -[: onto]->(q:edge {run_index:ridx}) "
        else:
            # CHANGE NOTE, deleted  < -[: onto]-(x:word)
            match_query_2 = "WITH DISTINCT(r.run_index) as ridx, collect(DISTINCT r.pos) as rpos,sum(r.weight) as rweight, v.token_id as ego  MATCH (t: word) < -[: onto]-  (q:edge {run_index:ridx}) "

        where_query_2 = "WHERE not q.pos in rpos AND NOT t.token_id = ego "
        if weight_cutoff is not None:
            where_query_2 = ''.join([where_query_2, " AND q.weight >=", str(weight_cutoff), " "])

        with_query = "WITH t.token_id as idx, q.weight as qweight, rweight, ego "
        # CHANGE NOTE, changed sum(qweight) * sum(rweight)
        return_query = "Return DISTINCT(idx) as alter, ego, sum(qweight*rweight) as weight order by ego"
        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        elif isinstance(times, list):
            sort_times = np.sort(times)
            nw_time = {"s": sort_times[0], "e": sort_times[-1], "m": int((sort_times[0] + sort_times[-1]) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int) or isinstance(times, list):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        query = "".join([match_query_1, where_query, match_query_2, where_query_2, with_query, return_query])
        logging.debug("Tie Query: {}".format(query))
        res = self.receive_query(query, params)

        tie_weights = np.array([x['weight'] for x in res], dtype=float)
        egos = np.array([x['ego'] for x in res])
        alters = np.array([x['alter'] for x in res])

        unique_egos = np.unique(egos)
        # Normalize
        for ego in unique_egos:
            mask = egos == ego
            sum_weights = np.sum(tie_weights[mask])
            tie_weights[mask] = tie_weights[mask] / sum_weights

        ties = [(x[0], x[1], {'weight': x[2], 'time': nw_time['m'], 'start': nw_time['s'], 'end': nw_time['e']}) for x
                in zip(egos, alters, tie_weights)]

        return ties

    def query_multiple_nodes(self, ids, times=None, weight_cutoff=None, context=None, norm_ties=False,
                             context_mode="bidirectional", context_weight=True, mode="new"):
        if (mode=="new" or context_weight) and norm_ties and context is not None:
            logging.warning("Compositional mode only supported for contextual queries without context weights. Weights are derived by occurrences.")
            context_weight=False
            mode="old"
        if mode=="old":
            logging.debug("Query Dispatch OLD")
            return self.query_multiple_nodes1(ids=ids,times=times,weight_cutoff=weight_cutoff, context=context,norm_ties=norm_ties, context_mode=context_mode, context_weight=context_weight)
        else:
            return self.query_multiple_nodes2(ids=ids, times=times, weight_cutoff=weight_cutoff, context=context,
                                              norm_ties=norm_ties, context_mode=context_mode,
                                              context_weight=context_weight)


    def query_multiple_nodes1(self, ids, times=None, weight_cutoff=None, context=None, norm_ties=False,
                             context_mode="bidirectional", context_weight=True):
        """
        Query multiple nodes by ID and over a set of time intervals


        Parameters
        ----------
        :param ids: list of id's
        :param times: either a number format YYYYMMDD, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :param weight_cutoff: float in 0,1
        :param norm_ties: Compositional mode
        :return: list of tuples (u,v,Time,{weight:x})
        """
        logging.debug("Querying {} nodes in Neo4j database.".format(len(ids)))

        # Optimization for time
        if isinstance(times, list):
            if len(times) == 1:
                times = int(times[0])

        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        elif isinstance(times, list):
            sort_times = np.sort(times)
            nw_time = {"s": sort_times[0], "e": sort_times[-1], "m": int((sort_times[0] + sort_times[-1]) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int) or isinstance(times, list):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        if context is not None:
            params.update({'contexts': context})

        # QUERY CREATION
        where_query = " WHERE b.token_id in $ids "

        # Context if desired
        if context is not None:
            c_where_query = " WHERE  e.token_id IN $contexts "
        else:
            c_where_query = ""

        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            where_query = ''.join([where_query, " AND r.weight >=", str(weight_cutoff), " "])
            if context is not None:
                c_where_query = ''.join([c_where_query, " AND q.weight >=", str(weight_cutoff), " "])
        if isinstance(times, dict):
            where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
            if context is not None:
                c_where_query = ''.join([c_where_query, " AND  $times.start <= q.time<= $times.end "])
        elif isinstance(times, list):
            where_query = ''.join([where_query, " AND  r.time in $times "])
            if context is not None:
                c_where_query = ''.join([c_where_query, " AND  q.time in $times "])

        # Create query depending on graph direction and whether time variable is queried via where or node property
        # By default, a->b when ego->is_replaced_by->b
        # Given an id, we query b:id(sender)<-a(receiver)
        # This gives all ties where b -replaces-> a
        # Enable this for occurrences
        # return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver,count(r.pos) AS occurrences,",
        #                        self.aggregate_operator, "(r.weight) AS agg_weight order by receiver"])
        if context_weight and context is not None:
            return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver, ",
                                    self.aggregate_operator, "(r.weight*qwd) AS agg_weight order by receiver"])
        else:
            return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver, ",
                                    self.aggregate_operator, "(r.weight) AS agg_weight order by receiver"])
        if isinstance(times, int):
            if context is not None:
                match_query = "".join(
                    ["MATCH p=(a:word)-[:onto]->(r:edge {time:", str(times), ", run_index:ridx})-[:onto]->(b:word) "])
                if context_mode == "bidirectional":
                    c_match = "".join(["MATCH (q:edge {time:", str(times), "}) - [:onto]-(e:word) "])
                else:
                    c_match = "".join(["MATCH (q:edge {time:", str(times), "}) - [:onto]->(e:word) "])
            else:
                match_query = "".join(["MATCH p=(a:word)-[:onto]->(r:edge {time:", str(times), "})-[:onto]->(b:word) "])
                c_match = ""
        else:
            if context is not None:
                match_query = "MATCH p=(a:word)-[:onto]->(r:edge {run_index:ridx})-[:onto]->(b:word) "
                if context_mode == "bidirectional":
                    c_match = "".join(["MATCH (q:edge) - [:onto]-(e:word) "])
                else:
                    c_match = "".join(["MATCH (q:edge) - [:onto]->(e:word) "])
            else:
                match_query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) "
                c_match = ""

        if context is not None:
            if context_weight:
                c_with = "WITH DISTINCT q.run_index as ridx, SUM(q.weight) as qwd "
            else:
                c_with = "WITH DISTINCT q.run_index as ridx "
        else:
            c_with = ""

        c_query = "".join([c_match, c_where_query, c_with])

        query = "".join([c_query, match_query, where_query, return_query])
        logging.debug("Tie Query: {}".format(query))
        res = self.receive_query(query, params)
        # Normalization
        # Ties should be normalized by the number of occurrences of the receiver
        if norm_ties:
            unique_receivers = [int(x) for x in np.unique([y['receiver'] for y in res])]
            norms = dict(self.query_occurrences(unique_receivers, times, weight_cutoff, context))
            ties = [(x['sender'], x['receiver'],
                     {'weight': np.float((100 * x['agg_weight'] / norms[x['receiver']]) if norms[x['receiver']] else 0),
                      'time': nw_time['m'], 'start': nw_time['s'], 'end': nw_time['e']}) for
                    x in res]
        else:
            ties = [(x['sender'], x['receiver'],
                     {'weight': np.float(x['agg_weight']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e']}) for
                    x in res]

        return ties

    def query_multiple_nodes2(self, ids, times=None, weight_cutoff=None, context=None, norm_ties=False,
                              context_mode="bidirectional", context_weight=True):
        """
        Query multiple nodes by ID and over a set of time intervals


        // Neo4j example query replicated here WITH CONTEXT
        MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word)
        WHERE b.token in ["zuckerberg"]
        With a,r,b
        MATCH (z:word)- [:onto]-  (q:edge {run_index:r.run_index}) - [:onto]-(e:word) WHERE q.pos<>r.pos AND  e.token IN ["facebook"]
        With
        r.pos as rpos, r.run_index as ridx, b.token AS sender,a.token AS receiver,
        CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as cweight, head(collect(r.weight)) AS rweight
        WITH DISTINCT(rpos) as rp, ridx, sender, receiver,  cweight,rweight,cweight*rweight as weight
        RETURN sender, receiver, sum(cweight) as cw, sum(rweight) as rw, sum(weight) as agg_weight order by agg_weight DESC

        // Neo4j example query replicated here WITHOUT CONTEXT
        MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word)
        WHERE b.token in ["zuckerberg"]
        With
        r.pos as rpos, r.run_index as ridx, b.token AS sender,a.token AS receiver, head(collect(r.weight)) AS rweight
        RETURN sender, receiver, 0 as cw, sum(rweight) as rw, sum(rweight) as agg_weight order by agg_weight DESC
        Parameters
        ----------
        :param ids: list of id's
        :param times: either a number format YYYYMMDD, or an interval dict {"start":YYYYMMDD,"end":YYYYMMDD}
        :param weight_cutoff: float in 0,1
        :param norm_ties: Compositional mode
        :return: list of tuples (u,v,Time,{weight:x})
        """
        logging.debug("Querying {} nodes in Neo4j database.".format(len(ids)))

        # Get rid of integer time to make query easier
        if isinstance(times, int):
            times = [times]

        # Format time to set for network
        if isinstance(times, int):
            nw_time = {"s": times, "e": times, "m": times}
        elif isinstance(times, dict):
            nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        elif isinstance(times, list):
            sort_times = np.sort(times)
            nw_time = {"s": sort_times[0], "e": sort_times[-1], "m": int((sort_times[0] + sort_times[-1]) / 2)}
        else:
            nw_time = {"s": 0, "e": 0, "m": 0}

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int) or isinstance(times, list):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        if context is not None:
            params.update({'contexts': context})

        # QUERY CREATION

        ### MATCH QUERIES
        match_query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) "

        if context is not None:
            if context_mode == "bidirectional":
                c_match = "".join(["MATCH (q:edge {run_index:r.run_index}) - [:onto]-(e:word) "])
            else:
                c_match = "".join(["MATCH (q:edge {run_index:r.run_index}) - [:onto]->(e:word) "])
        else:
            c_match = " "

        ### WITH QUERIES

        if context is not None:
            with_query = "WITH a,r,b "
            #c_with = " WITH r.pos as rpos, r.run_index as ridx, b.token_id AS sender,a.token_id AS receiver, CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as cweight, head(collect(r.weight)) AS rweight WITH DISTINCT(rpos) as rp, ridx, sender, receiver,  cweight,rweight,cweight*rweight as weight "
            c_with = " WITH r.pos as rpos, r.run_index as ridx, b.token_id AS sender,a.token_id AS receiver, CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as cweight, head(collect(r.weight)) AS rweight WITH rpos as rp, ridx, sender, receiver,  cweight,rweight,cweight*rweight as weight "
        else:
            with_query = " "
            c_with = " WITH r.pos as rpos, r.run_index as ridx, b.token_id AS sender,a.token_id AS receiver, head(collect(r.weight)) AS rweight"

        ### WHERE QUERIES
        where_query = " WHERE b.token_id in $ids "

        # Context if desired
        if context is not None:
            c_where_query = " WHERE e.token_id IN $contexts AND q.pos<>r.pos "
        else:
            c_where_query = ""

        # Allow cutoff value of (non-aggregated) weights and set up time-interval query
        if weight_cutoff is not None:
            if weight_cutoff <= 1e-07:
                weight_cutoff = None
        if weight_cutoff is not None:
            where_query = ''.join([where_query, " AND r.weight >=", str(weight_cutoff), " "])
            if context is not None:
                c_where_query = ''.join([c_where_query, " AND q.weight >=", str(weight_cutoff), " "])
        if isinstance(times, dict):
            where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        elif isinstance(times, list):
            where_query = ''.join([where_query, " AND  r.time in $times "])

        ### RETURN QUERIES

        # Create query depending on graph direction and whether time variable is queried via where or node property
        # By default, a->b when ego->is_replaced_by->b
        # Given an id, we query b:id(sender)<-a(receiver)
        # This gives all ties where b -replaces-> a
        # Enable this for occurrences
        # return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver,count(r.pos) AS occurrences,",
        #                        self.aggregate_operator, "(r.weight) AS agg_weight order by receiver"])
        if context_weight and context is not None:
            if context_weight:
                return_query = ''.join([" RETURN sender, receiver, ",
                                        self.aggregate_operator, "(weight) as agg_weight order by receiver"])
            else:
                return_query = ''.join([" RETURN sender, receiver, ",
                                        self.aggregate_operator, "(rweight) as agg_weight order by receiver"])
            #return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver, ",
            #                        self.aggregate_operator, "(r.weight*qwd) AS agg_weight order by receiver"])
        else:
            #return_query = ''.join([" RETURN b.token_id AS sender,a.token_id AS receiver, ",
            #                        self.aggregate_operator, "(r.weight) AS agg_weight order by receiver"])
            return_query = ''.join([" RETURN sender, receiver, ", self.aggregate_operator,"(rweight) as agg_weight order by receiver"])

        query = "".join([match_query, where_query, with_query, c_match, c_where_query, c_with, return_query])
        logging.debug("Tie Query: {}".format(query))
        res = self.receive_query(query, params)
        # Normalization
        # Ties should be normalized by the number of occurrences of the receiver
        if norm_ties:
            unique_receivers = [int(x) for x in np.unique([y['receiver'] for y in res])]
            norms = dict(self.query_occurrences(unique_receivers, times, weight_cutoff, context))
            ties = [(x['sender'], x['receiver'],
                     {'weight': np.float((100 * x['agg_weight'] / norms[x['receiver']]) if norms[x['receiver']] else 0),
                      'time': nw_time['m'], 'start': nw_time['s'], 'end': nw_time['e']}) for
                    x in res]
        else:
            ties = [(x['sender'], x['receiver'],
                     {'weight': np.float(x['agg_weight']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e']}) for
                    x in res]

        return ties

    def query_occurrences(self, ids, times=None, weight_cutoff=None, context=None):
        """
        Query multiple nodes by ID and over a set of time intervals, return distinct occurrences
        :param ids: list of id's
        :param times: either a number format YYYY, or an interval dict {"start":YYYY,"end":YYYY}
        :param weight_cutoff: float in 0,1
        :return: list of tuples (u,occurrences)
        """
        logging.debug("Querying {} node occurrences for normalization".format(len(ids)))

        # Optimization for time
        if isinstance(times, list):
            if len(times) == 1:
                times = int(times[0])

        # Create params with or without time
        if isinstance(times, dict) or isinstance(times, int) or isinstance(times, list):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        if context is not None:
            params.update({'contexts': context})

        # Check if occurrences are cached for this year
        if self.cache_yearly_occurrences and isinstance(times, int) and context is None:
            res = self.get_year_cache(times)
        else:
            res = False

        if not res:

            # Check if we want to cache yearly occurrences, in which case we query all
            if self.cache_yearly_occurrences and isinstance(times, int) and context is None:
                logging.debug("Yearly occurrences requested, cache will be filled.")
                where_query = " WHERE TRUE "
            else:
                where_query = " WHERE a.token_id in $ids "

            # Context if desired
            if context is not None:
                c_where_query = " WHERE  e.token_id IN $contexts "
            else:
                c_where_query = ""

            # Allow cutoff value of (non-aggregated) weights and set up time-interval query
            if weight_cutoff is not None:
                where_query = ''.join([where_query, " AND r.weight >=", str(weight_cutoff), " "])
                if context is not None:
                    c_where_query = ''.join([c_where_query, " AND q.weight >=", str(weight_cutoff), " "])
            if isinstance(times, dict):
                where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
                if context is not None:
                    c_where_query = ''.join([c_where_query, " AND  $times.start <= q.time<= $times.end "])
            elif isinstance(times, list):
                where_query = ''.join([where_query, " AND  r.time in $times "])
                if context is not None:
                    c_where_query = ''.join([c_where_query, " AND  q.time in $times "])

            return_query = ''.join([
                "RETURN a.token_id AS idx, round(sum(r.weight)) as occurrences order by idx"])

            if context is not None:
                c_with = "WITH DISTINCT q.run_index as ridx "
            else:
                c_with = ""

            if isinstance(times, int):
                if context is not None:
                    match_query = "MATCH p=(a:word)-[:onto]->(r:edge {time:$times, run_index:ridx}) "
                    c_match = "MATCH (q:edge {time:", str(times), "}) - [:onto]->(e:word) "
                else:
                    match_query = "MATCH p=(a:word)-[:onto]->(r:edge {time:$times}) "
                    c_match = ""
            else:
                if context is not None:
                    match_query = "MATCH p=(a:word)-[:onto]->(r:edge {run_index:ridx}) "
                    c_match = "MATCH (q:edge {time:", str(times), "}) - [:onto]->(e:word) "
                else:
                    match_query = "MATCH p=(a:word)-[:onto]->(r:edge) "
                    c_match = ""

            c_query = "".join([c_match, c_where_query, c_with])
            query = "".join([c_query, match_query, where_query, return_query])

            logging.debug("Occurrence Query: {}".format(query))
            res = self.receive_query(query, params)

            if self.cache_yearly_occurrences is False or context is not None:
                ties = [(x['idx'], x['occurrences']) for x in res]
            else:
                ties = [(x['idx'], x['occurrences']) for x in res if x['idx'] in ids]
                # Update yearly cache:
                if isinstance(times, int):
                    if self.cache_yearly_occurrences:
                        self.add_year_cache(times, res)

        else:
            logging.debug("Cached Occurrences found for {}".format(times))
            ties = [(x['idx'], x['occurrences']) for x in res if x['idx'] in ids]

        return ties


    # %% Insert functions
    def insert_edges_context(self, ego, ties, contexts, logging_level=logging.DEBUG, no_context=False):
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

        # token translation
        egos = np.array(self.translate_token_ids(egos))
        alters = np.array(self.translate_token_ids(alters))
        con_alters = np.array(self.translate_token_ids(con_alters))

        times = np.array([x[2] for x in ties])
        dicts = np.array([x[3] for x in ties])
        con_times = np.array([x[2] for x in contexts])
        con_dicts = np.array([x[3] for x in contexts])

        # Delte just to make sure translation is taken
        del ties, contexts

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
                                   "seq_id": int(x[2]['seq_id'] if len(x[2]) > 2 else 0),
                                   "pos": int(x[2]['pos'] if len(x[2]) > 3 else 0),
                                   "run_index": int(x[2]['run_index'] if len(x[2]) > 1 else 0),
                                   "p1": ((x[2]['p1']) if len(x[2]) > 4 else 0),
                                   "p2": ((x[2]['p2']) if len(x[2]) > 5 else 0),
                                   "p3": ((x[2]['p3']) if len(x[2]) > 6 else 0),
                                   "p4": ((x[2]['p4']) if len(x[2]) > 7 else 0), }
                                  for x in zip(con_alters.tolist(), con_times.tolist(), con_dicts.tolist())]
            params = {"ego": int(egos[0]), "ties": ties_formatted, "contexts": contexts_formatted}

            # Select order of parameters
            p1 = np.array([str(x['p1']) if (len(x) > 4) else "0" for x in dicts])
            p2 = np.array([str(x['p2']) if (len(x) > 5) else "0" for x in dicts])
            p3 = np.array([str(x['p3']) if (len(x) > 6) else "0" for x in dicts])
            p4 = np.array([str(x['p4']) if (len(x) > 7) else "0" for x in dicts])
            # Select order of context parameters
            cseq_id = np.array([x['seq_id'] if len(x) > 2 else "0" for x in con_dicts], dtype=np.str)
            cpos = np.array([x['pos'] if len(x) > 3 else "0" for x in con_dicts], dtype=np.str)
            crun_index = np.array([x['run_index'] if len(x) > 1 else "0" for x in con_dicts], dtype=np.str)
            cp1 = np.array([str(x['p1']) if (len(x) > 4) else "0" for x in con_dicts])
            cp2 = np.array([str(x['p2']) if (len(x) > 5) else "0" for x in con_dicts])
            cp3 = np.array([str(x['p3']) if (len(x) > 6) else "0" for x in con_dicts])
            cp4 = np.array([str(x['p4']) if (len(x) > 7) else "0" for x in con_dicts])

            # Build parameter string
            parameter_string = ""
            if not all(p1 == "0") and not all(p1 == ''):
                parameter_string = parameter_string + ", p1:tie.p1"
            if not all(p2 == "0") and not all(p2 == ''):
                parameter_string = parameter_string + ", p2:tie.p2"
            if not all(p3 == "0") and not all(p3 == ''):
                parameter_string = parameter_string + ", p3:tie.p3"
            if not all(p4 == "0") and not all(p4 == ''):
                parameter_string = parameter_string + ", p4:tie.p4 "

            cparameter_string = ""
            if not all(cseq_id == "0") and not all(cseq_id == ''):
                cparameter_string = cparameter_string + ", seq_id:con.seq_id"
            if not all(cpos == "0") and not all(cpos == ''):
                cparameter_string = cparameter_string + ", pos:con.pos"
            if not all(crun_index == "0") and not all(crun_index == ''):
                cparameter_string = cparameter_string + ", run_index:con.run_index"
            if not all(cp1 == "0") and not all(cp1 == ''):
                cparameter_string = cparameter_string + ", p1:con.p1"
            if not all(cp2 == "0") and not all(cp2 == ''):
                cparameter_string = cparameter_string + ", p2:con.p2"
            if not all(cp3 == "0") and not all(cp3 == ''):
                cparameter_string = cparameter_string + ", p3:con.p3"
            if not all(cp4 == "0") and not all(cp4 == ''):
                cparameter_string = cparameter_string + ", p4:con.p4"

            if not no_context:
                query = ''.join(
                    [
                        " MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties as tie MATCH (b:word {token_id: tie.alter}) ",
                        self.creation_statement,
                        " (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time, seq_id:tie.seq_id,pos:tie.pos, run_index:tie.run_index ",
                        parameter_string,
                        "})<-[:onto]-(a) WITH r UNWIND $contexts as con MATCH (q:word {token_id: con.alter}) WITH r,q,con MERGE (c:context {weight:con.weight, time:con.time ",
                        cparameter_string, "})-[:conto]->(q) WITH r,c ",
                        self.context_creation_statement,
                        " (r)-[:conto]->(c)"])
            else:
                query = ''.join(
                    [
                        " MATCH (a:word {token_id: $ego}) WITH a UNWIND $ties as tie MATCH (b:word {token_id: tie.alter}) ",
                        self.creation_statement,
                        " (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time, seq_id:tie.seq_id,pos:tie.pos, run_index:tie.run_index ",
                        parameter_string, "})<-[:onto]-(a)"])
        else:
            logging.error("Batched edge creation with context for multiple egos not supported.")
            raise NotImplementedError

        self.add_query(query, params)

    # %% Cache

    def get_year_cache(self, year: int) -> Union[list, bool]:
        """
        If there are occurrences cached for a given year, get those

        Parameters
        ----------
        year: int
            The year

        Returns
        -------
            list of dicts of form {idx:x, occurrences:y}
            if not existing, return False
        """

        if self.cache_yearly_occurrences:
            if year in list(self.occ_cache.keys()):
                return self.occ_cache[year]
            else:
                return False
        else:
            return False

    def add_year_cache(self, year: int, res: list):
        """
        Update year cache of occurrences

        Parameters
        ----------
        year: int
        res : list
            List of dicts of form: {idx: , occurrences: }

        Returns
        -------

        """
        if self.cache_yearly_occurrences:
            self.occ_cache[year] = res

    # %% Neo4J interaction
    # All function that interact with neo are here, dispatched as needed from above

    def add_query(self, query, params=None, run=False):
        """
        Add a single query to queue
        :param query: Neo4j query
        :param params: Associates parameters
        :return:

        Parameters
        ----------
        run
        """
        if params is not None:
            self.add_queries([query], [params])
        else:
            self.add_queries([query], None)

        if run:
            self.write_queue()

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
            statements = [{'statement': p, 'parameters':q} for (q, p) in zip(query, params)]
            self.neo_queue.extend(statements)
        else:
            statements = [{'statement': q} for (q) in query]
            self.neo_queue.extend(statements)

        # Check for queue size if not conditioned!
        if (len(self.neo_queue) > self.queue_size):
            self.write_queue()

    def write_queue(self):
        """
        If called will run queries in the queue and empty it.
        :return:
        """
        if len(self.neo_queue) > 0:
            if self.connection_type == "bolt":
                with self.driver.session() as session:
                    with session.begin_transaction() as tx:
                        for statement in self.neo_queue:
                            if 'parameters' in statement:
                                tx.run(statement['statement'], statement['parameters'])
                            else:
                                tx.run(statement['statement'])
                        tx.commit()
                        tx.close()
            else:
                raise NotImplementedError("HTTP connector no longer supported")

            self.neo_queue = []

    def non_con_write_queue(self):
        """
        Utility function to write queue immediately
        :return:
        """
        self.write_queue()

    @staticmethod
    def consume_result(tx, query, params=None, consume_type=None):
        result = tx.run(query, params)
        if consume_type == "data":
        #if True:
            return result.data()
        else:
            return [dict(x) for x in result]

    def open_session(self, fetch_size=50):
        logging.debug("Opening Session!")
        self.read_session=self.driver.session(fetch_size=fetch_size)

    def close_session(self):
        self.read_session.close()
        self.read_session=None

    def receive_query(self, query, params=None):
        clean_up=False
        if self.read_session is None:
            logging.debug("Session was closed, opening temporary one")
            self.open_session()
            clean_up=True
        oldlevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(30)
        #res = self.read_session.read_transaction(self.consume_result, query, params, self.consume_type)
        result=self.read_session.run(query,params)
        res=[dict(x) for x in result]
        logging.getLogger().setLevel(oldlevel)
        if clean_up:
            self.close_session()
        return res

    def close(self):
        if self.read_session is not None:
            self.read_session.close()

        if self.connection_type == "bolt":
            self.driver.close()
