from unittest import TestCase
from NLP.src.neo4j import neo4j_network

class Testneo4j_network(TestCase):

    def setUp(self) -> None:
        db_uri="http://localhost:7474"
        db_pwd=('neo4j','nlp')
        graph_type="networkx"
        tokens=["car","house","dog","cat"]
        token_ids=[10,11,12,13]
        add_string=[('house', (1, 2000)), ('dog', (0.6, 2000))]
        add_string_new=[('house', (1, 2000)), ('bird', (0.5, 2000))]

        neo4nw=neo4j_network((db_uri,db_pwd),graph_type)



    def test_initialize_connection(self):
        pass

    def test_init_tokens(self):
        pass

    def test_get_token_from_db(self):
        pass

    def test_get_token_from_memory(self):
        pass

    def test_confirm_stack_write(selfs):

    def test_add_token(self):
        pass

    def test_update_token(self):
        pass

    def test_add_link_token(self):
        pass

    def test_add_link_id(self):
        pass

    def test_get_link_token(self):
        pass

    def test_get_link_id(self):
        pass

    def test_add_link_new_token(self):
        pass

    def test_get_links_missing_ego(self):
        pass