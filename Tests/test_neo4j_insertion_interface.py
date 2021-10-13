import pytest
import numpy as np

from Tests.test_setups import get_token_list


@pytest.mark.usefixtures("Neo4j_Interface")
def test_node_additions(Neo4j_Interface):
    tokens,token_ids = get_token_list()

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.reset_dictionary()


@pytest.mark.usefixtures("Neo4j_Interface")
def test_id_additions(Neo4j_Interface):
    tokens,token_ids = get_token_list()

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.reset_dictionary()


@pytest.mark.usefixtures("Neo4j_Interface")
def test_ensure_db_id(Neo4j_Interface):
    tokens,token_ids = get_token_list()

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.reset_dictionary()

    assert Neo4j_Interface.ensure_db_ids("t_manager")==1
    assert Neo4j_Interface.ensure_db_ids("t_company")==4


@pytest.mark.usefixtures("Neo4j_Interface")
def test_ensure_tokenizer_id(Neo4j_Interface):
    tokens,token_ids = get_token_list()

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.reset_dictionary()


    token_ids = [9, 10, 11, 12, 13, 14]

    Neo4j_Interface.setup_neo_db(tokens, token_ids)

    assert Neo4j_Interface.ensure_tokenizer_ids("t_manager") == 9
    assert Neo4j_Interface.ensure_tokenizer_ids("t_manager") == 12

@pytest.mark.usefixtures("Neo4j_Interface")
def test_id_translation(Neo4j_Interface):
    tokens,token_ids = get_token_list()

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.reset_dictionary()

    token_ids = [9, 10, 11, 12, 13, 14]

    Neo4j_Interface.setup_neo_db(tokens, token_ids)

    assert  Neo4j_Interface.translate_token_ids([9,11,14]) == Neo4j_Interface.ensure_db_ids(["t_manager", "t_boss",  "t_employee"])