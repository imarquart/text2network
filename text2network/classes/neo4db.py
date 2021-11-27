# TODO: Redo Compositional ties
# TODO: Take out context element fragments

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
        if 'runidxindex' not in constr:
            query = "CREATE INDEX runidxindex FOR (a:edge) ON (a.run_index)"
            self.add_query(query)
        if 'posedgeindex' not in constr:
            query = "CREATE INDEX posedgeindex FOR (a:edge) ON (a.pos)"
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
        logging.info("Before cleaning: Network has %i edge-nodes ", (nr_nodes))

        if time is not None:
            # Delete previous edges
            node_query = ''.join(
                ["MATCH (p:edge {time:", str(time), "}) WITH p LIMIT ", str(del_limit), " DETACH DELETE p"])
        else:
            # Delete previous edges
            node_query = ''.join(
                ["MATCH (p:edge)  WITH p LIMIT ", str(del_limit), " DETACH DELETE p"])

        while nr_nodes > 0:
            # Delete edge nodes
            self.add_query(node_query, run=True)
            nr_nodes = self.receive_query("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
            logging.info("Network has %i edge-nodes", (nr_nodes))

        # DEBUG
        nr_nodes = self.receive_query("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
        logging.info("After cleaning: Network has %i nodes and %i ties", (nr_nodes))

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

    def query_tie_context(self, occurring, replacing, times=None, pos=None, scale=100, context_mode="bidirectional", return_sentiment=True, weight_cutoff=None):
        """
        This returns contextual words, weighted by


        // Dyadic Context (Bidirectional)
        MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token in ["leader"] and a.token in ["ceo"] and a.token_id <> b.token_id
        With a,r,b
        Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos
        WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b
        Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]-(e:word) WHERE q.pos<>r.pos
        WITH a,r,b,q,e, seq_length
        MATCH (q)-[:pos]-(pos:part_of_speech) WHERE pos.part_of_speech in ["VERB"]
        With
        r.pos as rpos, r.run_index as ridx, b.token AS substitute,a.token AS occurrence,CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as cweight, head(collect(r.weight)) AS rweight, e.token as context, head(collect(r.sentiment)) AS sentiment, head(collect(r.subjectivity)) AS subjectivity, seq_length
        WITH  ridx, context, substitute, occurrence, CASE WHEN sum(cweight)>1.0 THEN 1 else sum(cweight) END as cweight,CASE WHEN sum(rweight)>1.0 THEN 1 else sum(rweight) END as rweight,avg(sentiment) as sentiment, avg(subjectivity) as subjectivity, seq_length
        WITH ridx, context, substitute, occurrence, 100*cweight*rweight/seq_length as weight, sentiment, subjectivity, rweight, cweight
        RETURN context,  sum(weight) as weight, collect(DISTINCT(substitute)) as substitute, collect(DISTINCT(occurrence)) as occurrence,sum(rweight) as subst_weight, sum(cweight) as context_weight, avg(sentiment) as sentiment, avg(subjectivity) as subjectivity ORDER by weight DESC
        Parameters
        ----------
        occurring
        replacing
        times
        weight_cutoff

        Returns
        -------

        """
        logging.debug("Querying tie between {}->replacing->{} at {}.".format(replacing, occurring,times))

        if weight_cutoff is not None:
            if weight_cutoff <= 1e-07:
                weight_cutoff=None


        if occurring is not None:
            if isinstance(occurring, int):
                occurring = [occurring]
            if isinstance(occurring, np.ndarray):
                occurring = occurring.tolist()
        if replacing is not None:
            if isinstance(replacing, int):
                replacing = [replacing]
            if isinstance(replacing, np.ndarray):
                occurring = replacing.tolist()

        # Get rid of integer time to make query easier
        if isinstance(times, int):
            times = [times]

        if isinstance(pos, str):
            pos = [pos]

        params={}
        # Create params with or without time
        if isinstance(times, (dict, int, list)):
            params = {"times": times}
        if occurring is not None:
            params["id_occ"]=occurring
        if replacing is not None:
            params["id_repl"]=replacing
        # QUERY CREATION

        ### MATCH QUERY 1: TIES
        match_query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) "
        where_query = " WHERE b.token_id <> a.token_id "
        if replacing is not None:
            where_query = ''.join([where_query," and b.token_id in $id_repl "])
        if occurring is not None:
            where_query = ''.join([where_query, " and a.token_id in $id_occ "])

        if weight_cutoff is not None:
            where_query = ''.join([where_query, " AND r.weight >=", str(weight_cutoff), " "])
        if isinstance(times, dict):
            where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        elif isinstance(times, list):
            where_query = ''.join([where_query, " AND  r.time in $times "])

        # MATCH QUERY: Sequence Length
        sqlength_match= " With a,r,b Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos "

        ### MATCH QUERY 2: CONTEXT
        if context_mode == "bidirectional":
            c_match = "".join([" WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b  MATCH (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]-(e:word) "])
        elif context_mode == "occuring":
            c_match = "".join([" WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b MATCH (r)-[:seq]-(s:sequence)-[:seq]-(q:edge)<- [:onto]-(e:word) "])
        else:
            c_match = "".join([" WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b MATCH (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]->(e:word) "])

        c_where = " WHERE q.pos<>r.pos "
        if weight_cutoff is not None:
            c_where = ''.join([c_where, " AND q.weight >=", str(weight_cutoff), " "])


        ### MATCH QUERY 3: PART OF SPEECH
        if pos is not None:
            pos_match = " WITH a,r,b,q,e,seq_length MATCH (q)-[:pos]-(pos:part_of_speech) WHERE pos.part_of_speech in "
            pos_vector = "["+",".join(["'"+str(x)+"'" for x in pos])+"] "
            pos_match = " ".join([pos_match, pos_vector])
        else:
            pos_match = " "

        ### FINAL MATCH
        match = " ".join([match_query,where_query, sqlength_match,  c_match, c_where, pos_match])

        ### AGGREGATION QUERY
        part1 = "WITH r.pos as rpos, r.run_index as ridx, b.token_id AS substitute,a.token_id AS occurrence,CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as cweight, head(collect(r.weight)) AS rweight, e.token_id as context, head(collect(r.sentiment)) AS sentiment, head(collect(r.subjectivity)) AS subjectivity, seq_length "
        part2 = "WITH  ridx, context, substitute, occurrence, CASE WHEN sum(cweight)>1.0 THEN 1 else sum(cweight) END as cweight,CASE WHEN sum(rweight)>1.0 THEN 1 else sum(rweight) END as rweight,avg(sentiment) as sentiment, avg(subjectivity) as subjectivity,seq_length "
        part3 = "WITH ridx, context, substitute, occurrence, {}*cweight*rweight/seq_length as weight, sentiment, subjectivity, rweight, cweight ".format(scale)

        agg = " ".join([part1, part2, part3])

        # RETURN QUERY
        return_query = "RETURN context, collect(distinct(substitute)) as substitute, collect(distinct(occurrence)) as occurrence, sum(weight) as weight "
        if return_sentiment:
            return_query = return_query+", avg(sentiment) as sentiment, avg(subjectivity) as subjectivity "
        return_query = return_query+" order by substitute"

        query = "".join([match, agg, return_query])
        #logging.debug("Tie Context Query: {}".format(query))
        res = self.receive_query(query, params)

        if pos is not None:
            pos = "-".join([str(x) for x in pos])
        else:
            pos ="None"


        if return_sentiment:
            ret = [{'substitute': x['substitute'], 'occurrence': x['occurrence'], 'idx': x['context'], 'weight': x['weight'], 'sentiment': x['sentiment'],'subjectivity': x['subjectivity'],'pos': pos}
                   for x in res]
        else:
            ret=[{'substitute': x['substitute'], 'occurrence': x['occurrence'], 'idx': x['context'], 'weight': x['weight'], 'pos': pos} for x in res]

        return ret

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
        if isinstance(times, (dict, int, list)):
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

    def query_substitution_in_dyadic_context(self, ids, occurring=None, replacing=None,  times=None, scale=100,weight_cutoff=None, return_sentiment=True):
        """
        // PY:query_substitution_in_dyadic_context
        MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token in ["leader"] and a.token in ["ceo"] and a.token_id <> b.token_id and r.time in [1990,1991,1992]
        With a,r,b
        Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos
        WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b
        Match (r)-[:seq]-(s:sequence)<-[:seq]-(q:edge) WHERE q.pos<>r.pos
        WITH a,r,b,q,seq_length
        MATCH (e:word) -[:onto]->(q)-[:onto]->(f:word) WHERE f.token in ["make"]
        WITH
        r.run_index as ridx, f.token as sub, e.token as occ, b.token AS rep_dyad,a.token AS occ_dyad,100*sum(q.weight)*sum(r.weight)/seq_length as weight
        RETURN
        sub,occ,sum(weight) as weight,[collect(distinct(rep_dyad)),collect(distinct(occ_dyad))] as dyad
        ORDER by occ DESC

        Parameters
        ----------
        return_sentiment
        ids
        occurring
        replacing
        times
        weight_cutoff

        Returns
        -------

        """
        logging.debug("Querying tie between {}->replacing->{}.".format(replacing, occurring))

        if weight_cutoff is not None:
            logging.warning("Weight cutoff {}".format(weight_cutoff))
            if weight_cutoff <= 1e-07:
                weight_cutoff = None

        if isinstance(ids, int):
            ids = [ids]

        if occurring is not None:
            if isinstance(occurring, int):
                occurring = [occurring]
            if isinstance(occurring, np.ndarray):
                occurring = occurring.tolist()
        if replacing is not None:
            if isinstance(replacing, int):
                replacing = [replacing]
            if isinstance(replacing, np.ndarray):
                occurring = replacing.tolist()

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
        if isinstance(times, (dict, int, list)):
            params = {"idx": ids, "times": times}
        else:
            params = {"idx": ids}
        if occurring is not None:
            params["id_occ"]=occurring
        if replacing is not None:
            params["id_repl"]=replacing

        # QUERY CREATION

        ### MATCH QUERY 1: TIES
        match_query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) "
        where_query = " WHERE b.token_id <> a.token_id "
        if replacing is not None:
            where_query = ''.join([where_query," and b.token_id in $id_repl "])
        if occurring is not None:
            where_query = ''.join([where_query, " and a.token_id in $id_occ "])
        if weight_cutoff is not None:
            where_query = ''.join([where_query, " AND r.weight >=", str(weight_cutoff), " "])
        if isinstance(times, dict):
            where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        elif isinstance(times, list):
            where_query = ''.join([where_query, " AND  r.time in $times "])

        # MATCH QUERY 2: Sequence Length
        sqlength_match= " WITH a,r,b Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos "

        ### MATCH QUERY 2: OCCURRING TOKEN
        c_match = "".join([" WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b  MATCH  (r)-[:seq]-(s:sequence)<-[:seq]-(q:edge) "])
        c_where = " WHERE q.pos<>r.pos "
        if weight_cutoff is not None:
            c_where = ''.join([c_where, " AND q.weight >=", str(weight_cutoff), " "])

        ### MATCH QUERY 3: Replacing ties
        pos_match = " WITH a,r,b,q,seq_length MATCH (e:word) -[:onto]->(q)-[:onto]->(f:word) WHERE f.token_id in $idx and e.token_id <> f.token_id "

        ### FINAL MATCH
        match = " ".join([match_query,where_query, sqlength_match,  c_match, c_where, pos_match])

        ### AGGREGATION QUERY
        agg = " WITH r.run_index as ridx, f.token_id as sub, e.token_id as occ, b.token_id AS rep_dyad,a.token_id AS occ_dyad,sum(q.weight)*sum(r.weight)/seq_length*{} as weight, avg(r.sentiment) as sentiment, avg(r.subjectivity) as subjectivity ".format(scale)

        # RETURN QUERY
        return_query = "RETURN sub,occ,[collect(distinct(rep_dyad)),collect(distinct(occ_dyad))] as dyad ,sum(weight) as weight "
        if return_sentiment:
            return_query = return_query + ", avg(sentiment) as sentiment, avg(subjectivity) as subjectivity "
        return_query = return_query + " order by occ"

        query = "".join([match, agg, return_query])
        logging.debug("Tie Context Query: {}".format(query))
        res = self.receive_query(query, params)

        if return_sentiment:
            ties = [(x['sub'], x['occ'],
                     {'weight': np.float(x['weight']), 'dyad':str(x['dyad']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e'], 'sentiment': np.float(x['sentiment']), 'subjectivity':  np.float(x['subjectivity'])}) for
                    x in res]
        else:
            ties = [(x['sub'], x['occ'],
                     {'weight': np.float(x['weight']), 'dyad':str(x['dyad']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e']}) for
                    x in res]

        return ties

    def query_context_in_dyadic_context(self, ids, occurring=None, replacing=None,  times=None, scale=100,weight_cutoff=None, return_sentiment=True):
        """
        // PY: query_context_in_dyadic_context
        MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token in ["leader"]  and a.token_id <> b.token_id and r.time in [1990,1991,1992]
        With a,r,b
        Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos
        WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b
        Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos
        WITH a,r,b,q,seq_length
        MATCH (f:word) -[:onto]-(q)-[:seq]-(s:sequence)-[:seq]-(t:edge)-[:onto]-(e:word) WHERE f.token in ["make"] and f.token <> e.token
        WITH
        r.run_index as ridx, f.token as occ1, e.token as occ2, b.token AS rep_dyad,a.token AS occ_dyad,
        CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as qweight,
        CASE WHEN sum(t.weight)>1.0 THEN 1 else sum(t.weight) END as tweight,
        head(collect(r.weight)) as rweight,seq_length
        RETURN
         occ1, occ2,100*sum(qweight*tweight*rweight)/seq_length as weight,[collect(distinct(rep_dyad)),collect(distinct(occ_dyad))] as dyad order by weight DESC

        Parameters
        ----------
        return_sentiment
        ids
        occurring
        replacing
        times
        weight_cutoff

        Returns
        -------

        """
        logging.debug("Querying tie between {}->replacing->{}.".format(replacing, occurring))

        if weight_cutoff is not None:
            logging.warning("Weight cutoff {}".format(weight_cutoff))
            if weight_cutoff <= 1e-07:
                weight_cutoff = None

        if isinstance(ids, int):
            ids = [ids]

        if occurring is not None:
            if isinstance(occurring, int):
                occurring = [occurring]
            if isinstance(occurring, np.ndarray):
                occurring = occurring.tolist()
        if replacing is not None:
            if isinstance(replacing, int):
                replacing = [replacing]
            if isinstance(replacing, np.ndarray):
                occurring = replacing.tolist()

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
        if isinstance(times, (dict, int, list)):
            params = {"idx": ids, "times": times}
        else:
            params = {"idx": ids}
        if occurring is not None:
            params["id_occ"]=occurring
        if replacing is not None:
            params["id_repl"]=replacing

        # QUERY CREATION

        ### MATCH QUERY 1: TIES
        match_query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) "
        where_query = " WHERE b.token_id <> a.token_id "
        if replacing is not None:
            where_query = ''.join([where_query," and b.token_id in $id_repl "])
        if occurring is not None:
            where_query = ''.join([where_query, " and a.token_id in $id_occ "])
        if weight_cutoff is not None:
            where_query = ''.join([where_query, " AND r.weight >=", str(weight_cutoff), " "])
        if isinstance(times, dict):
            where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
        elif isinstance(times, list):
            where_query = ''.join([where_query, " AND  r.time in $times "])

        # MATCH QUERY 2: Sequence Length
        sqlength_match= " WITH a,r,b Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos "

        ### MATCH QUERY 2: OCCURRING TOKEN
        c_match = "".join([" WITH count(DISTINCT([q.pos,q.run_index])) as seq_length, a,r,b,s  MATCH  (r)-[:seq]-(s)-[:seq]-(q:edge) "])
        c_where = " WHERE q.pos<>r.pos "
        if weight_cutoff is not None:
            c_where = ''.join([c_where, " AND q.weight >=", str(weight_cutoff), " "])

        ### MATCH QUERY 3: Replacing ties
        pos_with = " WITH a,r,b,q,seq_length,s "

        pos_match = "MATCH (f: word) -[: onto]-(q) - [: seq]-(s) - [: seq]-(t:edge) - [: onto]-(e:word) "
        pos_where = " WHERE f.token_id <> e.token_id and f.token_id in $idx "

        ### FINAL MATCH
        match = " ".join([match_query, where_query,sqlength_match, c_match, c_where, pos_with, pos_match, pos_where])

        ### AGGREGATION QUERY
        agg = " WITH r.run_index as ridx, f.token_id as sub, e.token_id as occ, b.token_id AS rep_dyad,a.token_id AS occ_dyad,avg(r.sentiment) as sentiment, avg(r.subjectivity) as subjectivity,  CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as qweight, CASE WHEN sum(t.weight)>1.0 THEN 1 else sum(t.weight) END as tweight, head(collect(r.weight)) as rweight,seq_length "

        # RETURN QUERY
        return_query = "RETURN sub, occ,{}*sum(qweight*tweight*rweight)/seq_length as weight,[collect(distinct(rep_dyad)),collect(distinct(occ_dyad))] as dyad ".format(scale)
        if return_sentiment:
            return_query = return_query + ", avg(sentiment) as sentiment, avg(subjectivity) as subjectivity "
        return_query = return_query + " order by occ"

        query = "".join([match, agg, return_query])
        logging.debug("Tie Context Query: {}".format(query))
        res = self.receive_query(query, params)

        if return_sentiment:
            ties = [(x['sub'], x['occ'],
                     {'weight': np.float(x['weight']), 'dyad':str(x['dyad']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e'], 'sentiment': np.float(x['sentiment']), 'subjectivity':  np.float(x['subjectivity'])}) for
                    x in res]
        else:
            ties = [(x['sub'], x['occ'],
                     {'weight': np.float(x['weight']), 'dyad':str(x['dyad']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e']}) for
                    x in res]

        return ties

    def query_multiple_nodes(self, ids, times=None, weight_cutoff=None, context=None, pos=None, return_sentiment=True,
                             context_mode="bidirectional", context_weight=True):
        """
        Query multiple nodes by ID and over a set of time intervals


        // Neo4j example query replicated here WITH CONTEXT
        PROFILE
        MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) WHERE b.token in ["president","leader"] and a.token_id <> b.token_id
        With a,r,b
        Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]-(e:word) WHERE q.pos<>r.pos AND  e.token IN ["democracy",
        "leadership","company"]
        With
        r.pos as rpos, r.run_index as ridx, b.token AS sender,a.token AS receiver,CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as cweight, head(collect(r.weight)) AS rweight
        WITH sender, receiver, rweight,cweight*rweight as weight
        RETURN sender, receiver, sum(rweight) as rw, sum(weight) as agg_weight order by agg_weight DESC

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
        :param context: Tokens which are to appear in the context of the substitution
        :param context_mode: Choose "occurring" if contextual token should occur, "substitution" if it should appear in
            a substitution distribution, or "bidirectional" if either
        :param pos: String/List indicating the Part Of Speech
        :param return_sentiment: Return sentiment and objectivity scores
        :return: list of tuples (u,v,Time,{weight:x})
        """
        logging.debug("Querying {} nodes in Neo4j database.".format(len(ids)))

        # Get rid of integer time to make query easier
        if isinstance(times, int):
            times = [times]

        if isinstance(pos, str):
            pos = [pos]

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
        if isinstance(times, (dict, int, list)):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        if context is not None:
            params.update({'contexts': context})

        # QUERY CREATION

        ### MATCH QUERIES
        match_query = "MATCH p=(a:word)-[:onto]->(r:edge)-[:onto]->(b:word) "

        if pos is not None:
            pos_match = " WTIH a,r,b MATCH (r)-[:pos]-(pos:part_of_speech) WHERE pos.part_of_speech in "
            pos_vector = "["+",".join(["'"+str(x)+"'" for x in pos])+"] "
            pos_match = " ".join([pos_match, pos_vector])
            match_query = match_query + pos_match


        if context is not None:
            if context_mode == "bidirectional":
                c_match = "".join(["MATCH (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]-(e:word) "])
            elif context_mode == "occuring":
                c_match = "".join(["MATCH (r)-[:seq]-(s:sequence)-[:seq]-(q:edge)<- [:onto]-(e:word) "])
            else:
                c_match = "".join(["MATCH (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]->(e:word) "])
        else:
            c_match = " "

        ### WITH QUERIES

        if context is not None:
            with_query = "WITH a,r,b "
            c_with = " WITH r.pos as rpos, r.run_index as ridx, b.token_id AS sender,a.token_id AS receiver, CASE WHEN sum(q.weight)>1.0 THEN 1 else sum(q.weight) END as cweight, head(collect(r.weight)) AS rweight,  head(collect(r.sentiment)) as sentiment,  head(collect(r.subjectivity)) as subjectivity WITH sender, receiver, sentiment, subjectivity,  rweight,cweight*rweight as weight "
        else:
            with_query = " "
            c_with = " WITH b.token_id AS sender,a.token_id AS receiver, r.weight AS rweight, r.subjectivity as subjectivity, r.sentiment as sentiment "

        ### WHERE QUERIES
        # Change 5.11.2021: Added " and b.token_id <> a.token_id "
        where_query = " WHERE b.token_id in $ids and b.token_id <> a.token_id "


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
        if context_weight and context is not None:
            if context_weight:
                return_query = ''.join([" RETURN sender, receiver, ",
                                        self.aggregate_operator, "(weight) as agg_weight "])
            else:
                return_query = ''.join([" RETURN sender, receiver, ",
                                        self.aggregate_operator, "(rweight) as agg_weight "])
        else:

            return_query = ''.join(
                [" RETURN sender, receiver, ", self.aggregate_operator, "(rweight) as agg_weight "])

        if return_sentiment:
            return_query = return_query+", avg(sentiment) as sentiment, avg(subjectivity) as subjectivity "
        return_query = return_query+" order by receiver"

        query = "".join([match_query, where_query, with_query, c_match, c_where_query, c_with, return_query])
        logging.debug("Tie Query: {}".format(query))
        res = self.receive_query(query, params)

        if pos is not None:
            pos = "-".join([str(x) for x in pos])
        else:
            pos ="None"

        if return_sentiment:
            ties = [(x['sender'], x['receiver'],
                     {'weight': np.float(x['agg_weight']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e'], 'pos':pos,  'sentiment': np.float(x['sentiment']), 'subjectivity':  np.float(x['subjectivity'])}) for
                    x in res]
        else:
            ties = [(x['sender'], x['receiver'],
                     {'weight': np.float(x['agg_weight']), 'time': nw_time['m'], 'start': nw_time['s'],
                      'end': nw_time['e'], 'pos':pos}) for
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
        logging.debug("Querying {} node occurrences".format(len(ids)))

        # Optimization for time
        if isinstance(times, int):
            times = [times]
        elif not isinstance(times, (dict, list)):
            AttributeError("Occurrence Query: Times variable must be dict, list or integer!")

        # Create params with or without time
        if isinstance(times, (dict, list)):
            params = {"ids": ids, "times": times}
        else:
            params = {"ids": ids}

        if context is not None:
            params.update({'contexts': context})

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

        # Timing
        if isinstance(times, dict):
            where_query = ''.join([where_query, " AND  $times.start <= r.time<= $times.end "])
            if context is not None:
                c_where_query = ''.join([c_where_query, " AND  $times.start <= q.time<= $times.end "])
        elif isinstance(times, list):
            where_query = ''.join([where_query, " AND  r.time in $times "])
            if context is not None:
                c_where_query = ''.join([c_where_query, " AND  q.time in $times "])

        # return_query = ''.join([
        #    "RETURN a.token_id AS idx, sum(r.weight) as occurrences order by idx"])
        # WARNING I changed to round here
        return_query = ''.join([
            "RETURN a.token_id AS idx, round(sum(r.weight)) as occurrences order by idx"])

        if context is not None:
            c_with = " WITH DISTINCT q.run_index as ridx "
        else:
            c_with = ""

        if context is not None:
            match_query = "MATCH p=(a:word)-[:onto]->(r:edge {run_index:ridx}) "
            c_match = "MATCH (q:edge) - [:onto]->(e:word) "
        else:
            match_query = "MATCH p=(a:word)-[:onto]->(r:edge) "
            c_match = ""

        c_query = "".join([c_match, c_where_query, c_with])
        query = "".join([c_query, match_query, where_query, return_query])

        logging.debug("Occurrence Query: {}".format(query))
        res = self.receive_query(query, params)

        ties = [(x['idx'], x['occurrences']) for x in res]

        return ties

    # %% Insert functions
    def insert_edges(self, ego, ties):

        logging.debug("Insert {} ego nodes with {} ties".format(ego, len(ties)))
        # Tie direction matters
        # Ego by default is the focal token to be replaced. Normal insertion points the link accordingly.
        # Hence, a->b is an instance of b replacing a!
        # Contextual ties always point toward the context word!

        egos = np.array([x[0] for x in ties])
        alters = np.array([x[1] for x in ties])

        # token translation
        egos = np.array(self.translate_token_ids(egos))
        alters = np.array(self.translate_token_ids(alters))

        times = np.array([x[2] for x in ties])
        dicts = np.array([x[3] for x in ties])

        # Delte just to make sure translation is taken
        del ties

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

            params = {"ego": int(egos[0]), "ties": ties_formatted}

            # Select order of parameters
            p1 = np.array([str(x['p1']) if (len(x) > 4) else "0" for x in dicts])
            p2 = np.array([str(x['p2']) if (len(x) > 5) else "0" for x in dicts])
            p3 = np.array([str(x['p3']) if (len(x) > 6) else "0" for x in dicts])
            p4 = np.array([str(x['p4']) if (len(x) > 7) else "0" for x in dicts])

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

            # Build Query

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
        if not isinstance(query, list):
            raise AssertionError

        if params is not None:
            if not isinstance(params, list):
                raise AssertionError
            statements = [{'statement': p, 'parameters': q} for (q, p) in zip(query, params)]
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
            # if True:
            return result.data()
        else:
            return [dict(x) for x in result]

    def open_session(self, fetch_size=50):
        logging.debug("Opening Session!")
        self.read_session = self.driver.session(fetch_size=fetch_size)

    def close_session(self):
        self.read_session.close()
        self.read_session = None

    def receive_query(self, query, params=None):
        clean_up = False
        if self.read_session is None:
            logging.debug("Session was closed, opening temporary one")
            self.open_session()
            clean_up = True
        oldlevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(30)
        result = self.read_session.run(query, params)
        res = [dict(x) for x in result]
        logging.getLogger().setLevel(oldlevel)
        if clean_up:
            self.close_session()
        return res

    def close(self):
        if self.read_session is not None:
            self.read_session.close()

        if self.connection_type == "bolt":
            self.driver.close()
