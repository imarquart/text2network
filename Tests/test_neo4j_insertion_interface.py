import pytest
import numpy as np


@pytest.mark.usefixtures("Neo4j_Interface")
def test_node_additions(Neo4j_Interface):
    tokens = ["t_manager", "t_leader", "t_boss", "t_company", "t_team", "t_employee"]
    token_ids = [1, 2, 3, 4, 5, 6]

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.reset_dictionary()


@pytest.mark.usefixtures("Neo4j_Interface")
def test_id_additions(Neo4j_Interface):
    tokens = ["t_manager", "t_leader", "t_boss", "t_company", "t_team", "t_employee"]
    token_ids = [1, 2, 3, 4, 5, 6]

    Neo4j_Interface.setup_neo_db(tokens, token_ids)
    Neo4j_Interface.reset_dictionary()

