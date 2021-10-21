import pytest

import os

import pytest

from text2network.classes.neo4jnw import neo4j_network
from text2network.utils.file_helpers import check_create_folder
from text2network.preprocessing.nw_preprocessor import nw_preprocessor
from text2network.training.bert_trainer import bert_trainer
from text2network.utils.hash_file import hash_string, check_step, complete_step
from text2network.utils.load_bert import get_only_tokenizer
from text2network.utils.logging_helpers import setup_logger
from text2network.processing.neo4j_insertion_interface import Neo4j_Insertion_Interface
from pathlib import Path
import logging


def get_token_list():
    tokens = pf(["manager", "leader", "boss", "company", "team", "employee"])
    token_ids = [1, 2, 3, 4, 5, 6]
    return tokens,token_ids

def pf(tokens):
    """
    Prefixes tokens with t_ such that there are no overwrites possible in a production database
    """
    for i, token in enumerate(tokens):
        if token[0:2] != "t_":
            tokens[i]="t_"+tokens[i]
    return tokens

@pytest.fixture(scope="session")
def test_config():
    """
    Creates a configuration file based on the test config.
    """
    # Set a configuration path
    configuration_path = 'Tests/config/test.ini'

    # Load Configuration file
    import configparser

    config = configparser.ConfigParser()
    print(check_create_folder(configuration_path))
    config.read(check_create_folder(configuration_path))
    # Setup logging
    setup_logger(config['Paths']['log'], config['General']['logging_level'], "Tests")

    return config

@pytest.fixture(scope="session")
def test_db(test_config):
    """
    Provides a db.hd5 file with the example sentences.
    If not available, then such a file is created by the preprocessor
    """
    db_file = check_create_folder(test_config['Paths']['database'], True)
    text_folders = check_create_folder(test_config['Paths']['import_folder'], False)
    db_file = Path(db_file)
    if db_file.is_file():
        db_file.unlink()
    preprocessor=nw_preprocessor(test_config)
    preprocessor.preprocess_folders(text_folders, overwrite=True)
    yield db_file
    db_file.unlink()


@pytest.fixture
def get_bert_trainer(test_config):
    """
    Get a BERT
    """
    return bert_trainer(test_config)

@pytest.fixture
def get_training_tokenizer_folder(get_bert_trainer):
    token_folder = os.path.join(get_bert_trainer.trained_folder, "tokenizer")
    hash = hash_string(token_folder, hash_factory="md5")
    if check_step(token_folder, hash):
        logging.info("Pre-populated tokenizer found. Using!")
        tokenizer = get_only_tokenizer(token_folder)
        bert = None
        logging.info("Loaded Tokenizer vocabulary {} items.".format(len(tokenizer)))
    else:
        logging.info("Pre-Loading Data and Populating tokenizers")
        queries = [x[0] for x in get_bert_trainer.uniques["query_filename"]]
        tokenizer, missing_tokens, total_tokens = get_bert_trainer.get_consistent_vocabulary(queries)
        tokenizer.save_pretrained(token_folder)
        hash = hash_string(token_folder, hash_factory="md5")
        complete_step(token_folder, hash)
    return token_folder

@pytest.fixture(scope="module")
def Neo4j_Interface(test_config):
    """
    Creates a insertion interface, also checking that the database is empty
    """
    neo4j_interface = Neo4j_Insertion_Interface(test_config)

    # Hard check if databse is empty
    query = "MATCH p=(a:word) RETURN count(p)  as nr_nodes"
    res=neo4j_interface.receive_query(query)[0]
    if res["nr_nodes"] > 0:
        raise ImportError("The provided Neo4j database is not empty. For security reasons, testing is disallowed on non-empty databases!")
    yield neo4j_interface
    tokens,token_ids = get_token_list()
    for token in tokens:
        qry = "".join(["Match (r:word {token:'", token, "'})-[:onto]->(t) DETACH DELETE t"])
        neo4j_interface.add_query(qry, run=True)
        qry = "".join(["Match (r:word {token:'", token, "'}) DETACH DELETE r"])
        neo4j_interface.add_query(qry, run=True)

@pytest.fixture(scope="module")
def Neo4j_Network(test_config):
    """
    Creates a insertion interface, also checking that the database is empty
    """
    neo4j_interface = neo4j_network(test_config)
    yield neo4j_interface
    # cleanup here



@pytest.fixture
def setup_clean_db_example(Neo4j_Interface):
    tokens,token_ids = get_token_list()

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.db.reset_dictionary()

    yield Neo4j_Interface

    for token in tokens:
        qry = "".join(["Match (r:word {token:'",token,"'})-[:onto]->(t) DETACH DELETE t"])
        Neo4j_Interface.add_query(qry, run=True)
        qry = "".join(["Match (r:word {token:'", token, "'}) DETACH DELETE r"])
        Neo4j_Interface.db.add_query(qry, run=True)


def test_cleanup(semantic_network):

    print("Clearing Database")
    tokens = ["t_manager", "t_leader", "t_boss", "t_company", "t_team", "t_employee"]

    for token in tokens:
        qry = "".join(["Match (r:word {token:'",token,"'})-[:onto]->(t) DETACH DELETE t"])
        semantic_network.db.add_query(qry, run=True)
        qry = "".join(["Match (r:word {token:'", token, "'}) DETACH DELETE r"])
        semantic_network.db.add_query(qry, run=True)
