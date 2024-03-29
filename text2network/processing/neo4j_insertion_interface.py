import random
from typing import Union

import numpy as np
import logging

try:
    from neo4j import GraphDatabase
except:
    GraphDatabase = None


class Neo4j_Insertion_Interface():

    # %% Initialization functions
    def __init__(self, config=None, neo4j_creds=None,  agg_operator="SUM",
                 write_before_query=True,
                 neo_batch_size=None, queue_size=100000,  tie_creation="UNSAFE",
                 logging_level=None, connection_type=None, consume_type=None, seed=100):
        # Fill parameters from configuration file
        if logging_level is not None:
            self.logging_level = logging_level
        else:
            if config is not None:
                self.logging_level = config['General'].getint('logging_level')
            else:
                msg = "Please provide valid logging level."
                logging.error(msg)
                raise AttributeError(msg)
        # Set logging level
        logging.disable(self.logging_level)

        if neo_batch_size is not None:
            self.neo_batch_size = neo_batch_size
        else:
            if config is not None:
                self.neo_batch_size = int(config['General']['neo_batch_size'])
            else:
                msg = "Please provide valid neo_batch_size."
                logging.error(msg)
                raise AttributeError(msg)

        if connection_type is not None:
            self.connection_type = connection_type
        else:
            if config is not None:
                self.connection_type = config['NeoConfig']['protocol']
            else:
                msg = "Please provide valid protocol."
                logging.error(msg)
                raise AttributeError(msg)

        if neo4j_creds is not None:
            self.neo4j_creds = neo4j_creds
        else:
            if config is not None:
                if self.connection_type == "http":
                    self.neo4j_creds = (
                        config['NeoConfig']["http_uri"], (config['NeoConfig']["db_db"], config['NeoConfig']["db_pwd"]))
                else:
                    self.neo4j_creds = (
                        config['NeoConfig']["db_uri"], (config['NeoConfig']["db_db"], config['NeoConfig']["db_pwd"]))
            else:
                msg = "Please provide valid neo4j_creds."
                logging.error(msg)
                raise AttributeError(msg)

        self.seed=seed
        self.set_random_seed(seed)

        self.neo4j_connection, self.neo4j_credentials = self.neo4j_creds
        self.write_before_query = write_before_query
        oldlevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(30)
        # Set up Neo4j driver
        if self.connection_type == "bolt" and GraphDatabase is not None:
            self.driver = GraphDatabase.driver(self.neo4j_connection, auth=self.neo4j_credentials)
            self.connection_type = "bolt"
            self.neo_session = None

        else:  # Fallback custom HTTP connector for Neo4j <= 4.02
            raise NotImplementedError("HTTP connection is no longer supported!")
        logging.getLogger().setLevel(oldlevel)
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

        self.consume_type = consume_type

        # Init tokens in the database, required since order etc. may be different
        self.db_ids, self.db_tokens = self.get_neo_tokens_and_ids()
        self.db_ids = np.array(self.db_ids)
        self.db_tokens = np.array(self.db_tokens)

        # Dictionary between tokenizer and database
        self.db_id_dict = {}
        self.reset_dictionary()

        # Tokenizer tokens and id's, if needed
        self.tokenizer_tokens = []
        self.tokenizer_ids = []

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.seed=seed

   # %% Setup, Initializations

    @staticmethod
    def check_create_tokenid_dict(tokenizer_tokens, tokenizer_ids, db_tokens, db_ids, fail_on_missing=False, debug=False):
        """
        This function creates a dictionary that translates tokens and ids from an external tokenizer
        with the ones found in the database.
        It further checks which tokens might be missing in the database, and adds them with new
        ids after the existing ones.

        Parameters
        ----------
        tokenizer_tokens
            List or array of tokens
        tokenizer_token_ids
            List or array of token ids
        fail_on_missing
            If True, function will throw error if there are missing tokens in the database

        Returns
        -------

        """
        tokenizer_tokens = np.array(tokenizer_tokens)
        tokenizer_ids = np.array(tokenizer_ids)
        if len(tokenizer_tokens) != len(tokenizer_ids):
            raise AssertionError(logging.error("Tokenizer token and id arrays MUST be same length!"))

        # Find overlapping and missing tokens / ids
        if len(db_ids) > 0:
            overlapping_index = [np.where(tokenizer_tokens == k)[0][0] for k in tokenizer_tokens if k in db_tokens]
            missing_index = [np.where(tokenizer_tokens == k)[0][0] for k in tokenizer_tokens if k not in db_tokens]
            overlapping_tokenizer_tokens = tokenizer_tokens[overlapping_index]
            missing_tokenizer_tokens = tokenizer_tokens[missing_index]
            overlapping_tokenizer_ids = tokenizer_ids[overlapping_index]
            missing_tokenizer_ids = tokenizer_ids[missing_index]
        else:
            overlapping_tokenizer_tokens = []
            missing_tokenizer_tokens = tokenizer_tokens
            overlapping_tokenizer_ids = []
            missing_tokenizer_ids = tokenizer_ids

        # Check whether we have overlapping tokens
        if len(overlapping_tokenizer_tokens) > 0:
            # Get the index of the overlapping tokens
            db_idx=[np.where(db_tokens == k)[0][0] for k in overlapping_tokenizer_tokens]
            # Get the corresponding ids
            overlapping_db_ids = db_ids[db_idx]
            # we have correspondence between overlapping_token_ids <-> overlapping_db_ids
            overlapping_db_id_dict = {x[0]: x[1] for x in zip(overlapping_tokenizer_ids, overlapping_db_ids)}
        else:
            overlapping_db_id_dict = {}

        # Check whether there are missing tokens
        if len(missing_tokenizer_tokens) > 0:
            # We create new Ids based on the maximum id previously in the database
            # This should work if the ids are consecutive, but also if they are not
            if len(db_ids)>0:
                added_ids = list(range(np.max(db_ids)+1, np.max(db_ids)+1 + len(missing_tokenizer_tokens)))
            else:
                added_ids = list(range(0, len(missing_tokenizer_tokens)))
            # Create missing dictionary
            if len(added_ids) != len(missing_tokenizer_ids):
                raise AssertionError(logging.error("Internal error: Length of newly defined ids for missing tokens does not equal array of tokenizer ids!"))
            missing_db_id_dict = {x[0]: x[1] for x in zip(missing_tokenizer_ids, added_ids)}
        else:
            missing_db_id_dict = {}
            added_ids=[]

        # Unite both DBs
        db_id_dict=overlapping_db_id_dict
        db_id_dict.update(missing_db_id_dict)
        all_tokens = np.array(list(db_tokens)+list(missing_tokenizer_tokens))
        db_ids = np.array(list(db_ids)+list(added_ids))

        # Since we do this once per init and since this is so important,
        # We now manually check whether each token has the right ID and so on
        for x in overlapping_tokenizer_ids:
            tokenizer_tk = tokenizer_tokens[np.where(tokenizer_ids == x)[0][0]]
            db_id = db_id_dict[x]
            db_tk = all_tokens[np.where(db_ids == db_id)[0][0]]
            if debug:
                logging.debug(
                    "tokenizer-db translation for tokenizer token {}, (id: {}), assigned to database token {}, (id: {})".format(
                        tokenizer_tk, x, db_tk, db_id))
            if tokenizer_tk != db_tk:
                raise AssertionError("Mismatch when creating tokenizer-db translation for tokenizer token {}, (id: {}), assigned to database token {}, (id: {})".format(tokenizer_tk,x,db_tk,db_id))
        for x in missing_tokenizer_ids:
            tokenizer_tk = tokenizer_tokens[np.where(tokenizer_ids == x)[0][0]]
            db_id = db_id_dict[x]
            if tokenizer_tk in list(db_tokens):
                raise AssertionError("Mismatch when creating tokenizer-db translation, assigned tokenizer token {}, (id: {}) with new id {}".format(tokenizer_tk,x,db_id))
            if debug:
                logging.debug("Assigned tokenizer token {}, (id: {}) with new id {} as missing, but found in database".format(tokenizer_tk,x,db_id))


        return db_id_dict, db_ids, missing_tokenizer_tokens, added_ids

    def setup_neo_db(self, tokens, token_ids):
        """
        Creates tokens and token_ids in Neo database. Does not delete existing network!
        :param tokens: list of tokens
        :param token_ids: list of corresponding token IDs
        :return: None
        """
        # Get rid of signs that can not be used
        tokens = [x.translate(x.maketrans({"\"": '#e1#', "'": '#e2#', "\\": '#e3#'})) for x in tokens]
        # Retain tokenizer assignments
        self.tokenizer_ids=np.array(token_ids).copy()
        self.tokens=np.array(tokens).copy()

        # Check that database is setup
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
        if 'seq_runindex_con' not in constr:
            query = "CREATE CONSTRAINT seq_runindex_con ON(n:sequence) ASSERT n.run_index IS UNIQUE"
            self.add_query(query)
        if 'pos_con' not in constr:
            query = "CREATE CONSTRAINT pos_con ON(n:part_of_speech) ASSERT n.part_of_speech IS UNIQUE"
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
        # Check if tokens are missing or if ids differ. Get new ids, create translation and get the missing tokens
        db_id_dict,token_ids, missing_tokens, missing_ids = self.check_create_tokenid_dict(tokenizer_tokens=tokens, tokenizer_ids=token_ids,db_ids=self.db_ids,db_tokens=self.db_tokens)
        self.db_id_dict=db_id_dict
        # Add the missing tokens with their new ids to the database
        if len(missing_tokens) > 0:
            queries = [''.join(["MERGE (n:word {token_id: ", str(id), ", token: '", tok, "'})"]) for tok, id in
                       zip(missing_tokens, missing_ids)]
            self.add_queries(queries)
            self.non_con_write_queue()
        # Just to be sure, again query tokens and ids
        self.db_ids, self.db_tokens = self.get_neo_tokens_and_ids()
        self.db_ids = np.array(self.db_ids)
        self.db_tokens = np.array(self.db_tokens)

    def delete_database(self, time=None, del_limit=10000):
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

        #Prune Sequence
        self.add_query("MATCH (n:sequence) WHERE size((n)--())=0 DELETE (n)", run=True)
        self.add_query("MATCH (n:part_of_speech) WHERE size((n)--())=0 DELETE (n)", run=True)

        nr_nodes = self.receive_query("MATCH (n:edge) RETURN count(n) AS nodes")[0]['nodes']
        logging.info("After cleaning: Network has %i nodes ties", (nr_nodes))

    def get_neo_tokens_and_ids(self):
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

    # %% Tokens and IDS

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

    def get_token_from_tokenizer_id(self,idx:Union[int, np.integer]):
        """
        Given an ID from the tokenizer, return the corresponding token
        Parameters
        ----------
        idx: integer
            ID (from tokenizer)

        Returns
        -------
        token: str
        """
        if not isinstance(idx, (int, np.integer)):
            raise AssertionError("Token IDs need to be of integer type.")
        try:
            db_id=self.db_id_dict[idx]
            pos = np.where(self.db_ids == db_id)[0][0]
            token=self.db_tokens[pos]
        except: # Not in translation dictionary, try via token
            try:
                logging.warning("Warning, tokenizer ID {} has no translation to any database ID. Returning tokenizer token.".format(idx))
                pos=np.where(self.tokenizer_ids==idx)[0][0]
                token=self.tokenizer_tokens[pos]
            except:
                logging.error("Provided ID not assigned to token in database or tokenizer!")
                raise
        return token

    def get_token_from_db_id(self,idx:Union[int, np.integer])->str:
        """

        Parameters
        ----------
        idx: Union[int, np.integer]
            ID from database

        Returns
        -------
        token: str
        """
        if not isinstance(idx, (int, np.integer)):
            raise AssertionError("Token IDs need to be of integer type.")
        try:
            pos = np.where(self.db_ids == idx)[0][0]
            token=self.db_tokens[pos]
        except:
            logging.error("Provided ID {} not assigned to token in database!".format(idx))
            raise
        return token

    def get_db_id_from_token(self,token:Union[str, np.character])->int:
        if not isinstance(token, (str, np.character)):
            raise AssertionError("Tokens need to be string or char type.")
        try:
            pos = np.where(self.db_tokens == token)[0][0]
            idx=self.db_ids[pos]
        except: # some error, try to do it via translation
            try:
                pos = np.where(self.tokenizer_tokens == token)[0][0]
                tok_idx = self.tokenizer_ids[pos]
                idx = self.db_id_dict[tok_idx]
                logging.warning("Warning, Token {} not found in database. Returning via tokenizer data and translation dictionary. Please check consistency!".format(token))
            except:
                logging.error("Provided Token {} not assigned to token in database!".format(token))
                raise
        return idx

    def get_tokenizer_id_from_token(self,token):
        if not isinstance(token, (str, np.character)):
            raise AssertionError("Tokens need to be string or char type.")
        try:
            pos = np.where(self.tokenizer_tokens == token)[0][0]
            idx=self.tokenizer_ids[pos]
        except:
            logging.error("Provided Token {} not assigned to tokenizer!".format(token))
            raise
        return idx

    def ensure_db_ids(self,tokens):
        """This is just to confirm mixed lists of tokens and ids get converted to ids"""
        if isinstance(tokens, (list, tuple, np.ndarray)):
            # Transform strings to corresponding IDs
            tokens = [self.get_db_id_from_token(x) if not np.issubdtype(
                type(x), np.integer) else x for x in tokens]
            # Make sure np arrays get transformed to int lists
            return [int(x) if not isinstance(x, int) else x for x in tokens]
        else:
            if not np.issubdtype(type(tokens), np.integer):
                return self.get_db_id_from_token(tokens)
            else:
                return int(tokens)

    def ensure_tokenizer_ids(self, tokens):
        """This is just to confirm mixed lists of tokens and ids get converted to ids (tokenizer)"""
        if isinstance(tokens, (list, tuple, np.ndarray)):
            # Transform strings to corresponding IDs
            tokens = [self.get_tokenizer_id_from_token(x) if not np.issubdtype(
                type(x), np.integer) else x for x in tokens]
            # Make sure np arrays get transformed to int lists
            return [int(x) if not isinstance(x, int) else x for x in tokens]
        else:
            if not np.issubdtype(type(tokens), np.integer):
                return self.get_tokenizer_id_from_token(tokens)
            else:
                return int(tokens)

    def ensure_tokens(self,ids, tokenizer_ids=False):
        """This is just to confirm mixed lists of tokens and ids get converted to ids"""
        if isinstance(ids, list):
            if tokenizer_ids:
                return [self.get_token_from_tokenizer_id(x) if not isinstance(x, str) else x for x in ids]
            else:
                return [self.get_token_from_db_id(x) if not isinstance(x, str) else x for x in ids]
        else:
            if not isinstance(ids, str):
                if tokenizer_ids:
                    return self.get_token_from_tokenizer_id(ids)
                else:
                    return self.get_token_from_tokenizer_id(ids)
            else:
                return ids

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
                               "part_of_speech": x[2]['part_of_speech'],
                               "sentiment": x[2]['sentiment'],
                               "subjectivity": x[2]['subjectivity'],
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
                    " (b)<-[:onto]-(r:edge {weight:tie.weight, time:tie.time, seq_id:tie.seq_id,pos:tie.pos, run_index:tie.run_index, part_of_speech:tie.part_of_speech, sentiment:tie.sentiment, subjectivity:tie.subjectivity ",
                    parameter_string, "})<-[:onto]-(a) "])
            query2 = ''.join([" WITH r, tie MERGE (h:sequence {run_index:tie.run_index,seq_id:tie.seq_id}) WITH r, tie, h CREATE (r)-[:seq]->(h) WITH r, tie  MERGE (f:part_of_speech {part_of_speech:tie.part_of_speech}) WITH r, tie, f CREATE (r)-[:pos]->(f)"])
            #query2 = ''.join([" WITH r, tie MERGE (r)-[:seq]->(h:sequence {seq_id:tie.run_index}) WITH r, tie MERGE (r)-[:seq]->(f:part_of_speech {part_of_speech:tie.part_of_speech})"])

            query = ''.join([query,query2])
        else:
            logging.error("Batched edge creation with context for multiple egos not supported.")
            raise NotImplementedError

        self.add_query(query, params)

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

    def add_queries(self, query, params=None, run=False):
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
            statements = [{'statement': p, 'parameters':q} for (p, q) in zip(query, params)]
            self.neo_queue.extend(statements)
        else:
            statements = [{'statement': q} for (q) in query]
            self.neo_queue.extend(statements)

        # Check for queue size if not conditioned!
        if (len(self.neo_queue) > self.queue_size):
            self.write_queue()
        if run:
            self.write_queue()

    def write_queue(self):
        """
        If called will run queries in the queue and empty it.
        :return:
        """
        if len(self.neo_queue) > 0:
            if self.connection_type == "bolt":
                if self.neo_session is None:
                    logging.debug("Session was closed, opening temporary one")
                    self.open_session()
                    clean_up = True
                else:
                    clean_up = False
                with self.neo_session.begin_transaction() as tx:
                    oldlevel = logging.getLogger().getEffectiveLevel()
                    logging.getLogger().setLevel(30)
                    for statement in self.neo_queue:
                        if 'parameters' in statement:
                            tx.run(statement['statement'], statement['parameters'])
                        else:
                            tx.run(statement['statement'])
                    tx.commit()
                    tx.close()
                    logging.getLogger().setLevel(oldlevel)
                if clean_up:
                    self.close_session()
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
        try:
            self.neo_session=self.driver.session(fetch_size=fetch_size)
        except:
            self.error("Could not open Neo Session")
            raise

    def close_session(self):
        try:
            self.neo_session.close()
            self.neo_session=None
        except:
            logging.warning("Tried closing neo session, was unable to. Continuing for now.")
            self.neo_session=None

    def receive_query(self, query, params=None):
        clean_up=False
        if self.neo_session is None:
            logging.debug("Session was closed, opening temporary one")
            self.open_session()
            clean_up=True
        oldlevel = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(30)
        result=self.neo_session.run(query,params)
        res=[dict(x) for x in result]
        logging.getLogger().setLevel(oldlevel)
        if clean_up:
            self.close_session()
        return res

    def close(self):
        if self.neo_session is not None:
            self.neo_session.close()

        if self.connection_type == "bolt":
            self.driver.close()
