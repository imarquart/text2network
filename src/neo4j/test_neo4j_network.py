from unittest import TestCase
from NLP.src.neo4j.neo4j_network import neo4j_network

class Testneo4j_network(TestCase):

    def setUp(self) -> None:
        db_uri="http://localhost:7474"
        db_pwd=('neo4j','nlp')
        graph_type="networkx"
        self.tokens=["car","house","dog","cat"]
        self.token_ids=[10,11,12,13]
        self.ego_id=11
        self.ego_token="house"
        self.add_string=[(11, 12, 20000101, {'weight': 0.5}), (11, 13, 20000101, {'weight': 0.5}), (12, 13, 20000101, {'weight': 0.5})]
        self.add_string_ego=[(12, 20000101, {'weight': 0.5}), (13, 20000101, {'weight': 0.5})]
        self.add_string_ego_missing=[(20, 20000101, {'weight': 0.5}), (13, 20000101, {'weight': 0.5})]
        self.add_string_ego_missing2=[('elephant', 20000101, {'weight': 0.5}), ('cat', 20000101, {'weight': 0.5})]
        self.add_string_ego_nodict=[(12, 20000101, 0.5), (13, 20000101, 0.5)]
        self.add_string_ego_names=[('dog', 20000101, {'weight': 0.5}), ('cat', 20000101, {'weight': 0.5})]
        self.add_string_timing = [(12, 20000101, {'weight': 0.5}), (13, 20010101, {'weight': 0.5}),(10, 20020101, {'weight': 0.5})]


        self.neo4nw=neo4j_network((db_uri,db_pwd),graph_type)
        self.setup_network()

    def setup_network(self):
        """Delete network and re-create it """
        # Run deletion query
        query="MATCH (n) DETACH DELETE n"
        self.neo4nw.connector.run(query)
        # Setup network
        self.neo4nw.setup_neo_db(self.tokens,self.token_ids)
        self.neo4nw.init_tokens()



    def test_setup(self):
        # Test
        query = "MATCH (n) RETURN n.token_id AS id"
        res = self.neo4nw.connector.run(query)
        id_list = [x['id'] for x in res]
        self.assertTrue(set(id_list) == set(self.token_ids))

    def test_init_tokens(self):
        self.neo4nw.init_tokens()
        self.assertTrue(set(self.neo4nw.ids) == set(self.token_ids))
        self.assertTrue(set(self.neo4nw.tokens) == set(self.tokens))

    def test_add_token(self):
        pass

    def test_update_token(self):
        pass

    def test_missing_token(self):
        """Try add a link from or to token without ID/Token information"""
        self.assertRaises(AssertionError,self.neo4nw.__setitem__,self.ego_id, self.add_string_ego_missing)
        self.assertRaises(ValueError,self.neo4nw.__setitem__,self.ego_id, self.add_string_ego_missing2)
        self.assertRaises(LookupError,self.neo4nw.__setitem__,"Elephant", self.add_string_ego)
        self.assertRaises(AssertionError,self.neo4nw.__setitem__,22, self.add_string_ego)

    def test_add_get_link_token(self):
        """Handle dispatch correctly if token names and not ID's are supplied"""
        self.neo4nw[self.ego_token]=self.add_string_ego_names
        self.neo4nw.write_queue()
        res = self.neo4nw.query_node(self.ego_id)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res), set(comp))

    def test_add_get_dispatch(self):
        """Add two ties towards ego node, using dict format"""
        self.neo4nw[self.ego_id]=self.add_string_ego
        self.neo4nw.write_queue()
        res = self.neo4nw.query_node(self.ego_id)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res), set(comp))


    def test_get_formats(self):
        """Tests the different ways to query nodes"""
        # Add some nodes
        self.neo4nw[self.ego_id]=self.add_string_ego
        self.neo4nw.write_queue()
        yy=self.add_string_ego[0][1]

        comp = [(self.ego_id, x[0], x[2]['weight']) for x in self.add_string_ego]

        # Check normal get
        res=self.neo4nw[self.ego_id]
        print("res=self.neo4nw[self.ego_id] ")
        print(res)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        self.assertEqual(set(res), set(comp))
        # Check get with date
        res=self.neo4nw[self.ego_id,yy]
        print("res=self.neo4nw[self.ego_id,yy] ")
        print(res)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        self.assertEqual(set(res), set(comp))
        # Check get with date range
        res=self.neo4nw[self.ego_id,{'start':yy-10,'end':yy+10}]
        print("res=self.neo4nw[self.ego_id,{'start':yy-10,'end':yy+10}] ")
        print(res)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        self.assertEqual(set(res), set(comp))
        # Check get with token range
        res=self.neo4nw[[self.ego_id-1,self.ego_id,self.ego_id+1],yy]
        print("res=self.neo4nw[[self.ego_id-1,self.ego_id,self.ego_id+1],yy] ")
        print(res)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        self.assertEqual(set(res), set(comp))
        # Check get with list of strings
        res = self.neo4nw[self.tokens, yy]
        print("res = self.neo4nw[self.tokens, yy] ")
        print(res)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        self.assertEqual(set(res), set(comp))

    def test_add_get_dispatch_nodict(self):
        """Add two ties towards ego node, using triplet format"""

        self.neo4nw[self.ego_id]=self.add_string_ego_nodict
        self.neo4nw.write_queue()
        res = self.neo4nw.query_node(self.ego_id)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res), set(comp))

    def test_add_get_link_id(self):
        """We add a list of edges between 3 nodes and check it is returned correctly (same format)"""
        # Since ties are saved as "PREDICTS", this includes reversal aka
        # here self.ego_id=11 is ego, but in neo4j ties go towards 11
        self.neo4nw.insert_edges_query_multiple(self.add_string)
        self.neo4nw.write_queue()
        res=self.neo4nw.query_node(self.ego_id)
        res=[(x[0],x[1],x[3]['weight']) for x in res]
        comp=[(self.ego_id,x[0],x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res),set(comp))

    def test_add_get_link_id_ego(self):
        """Add two ties towards ego node"""
        # Again, input states ties going away from self.ego_id but
        # network should save as going toward self.ego_id
        # and reverse when giving return query
        self.neo4nw.insert_edges_query(self.ego_id,self.add_string_ego)
        self.neo4nw.write_queue()
        res=self.neo4nw.query_node(self.ego_id)
        res=[(x[0],x[1],x[3]['weight']) for x in res]
        comp=[(self.ego_id,x[0],x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res),set(comp))

    def test_add_get_link_id_ego_reverse(self):
        """Add links via ego, reverse direction"""
        # Now direction of class and neo4j are equal
        self.setup_network()
        self.neo4nw.graph_direction="PREDICTS"

        self.neo4nw.insert_edges_query(self.ego_id,self.add_string_ego)
        self.neo4nw.write_queue()

        res=self.neo4nw.query_node(self.ego_id)
        res=[(x[0],x[1],x[3]['weight']) for x in res]
        comp=[(self.ego_id,x[0],x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res),set(comp))

    def test_add_get_link_id_reverse(self):
        """Add and retrieve list of links"""
        # Again this is the direction as given in neo4j
        self.neo4nw.graph_direction = "PREDICTS"
        self.neo4nw.insert_edges_query(self.ego_id,self.add_string_ego)
        self.neo4nw.write_queue()
        res=self.neo4nw.query_node(self.ego_id)
        res=[(x[0],x[1],x[3]['weight']) for x in res]
        comp=[(self.ego_id,x[0],x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res),set(comp))

    def test_query_time_interval(self):
        """Add some links across time then query back some of them"""
        self.neo4nw.insert_edges_query(self.ego_id, self.add_string_timing)
        self.neo4nw.write_queue()
        times={"start":20000101,"end":20010101}
        res = self.neo4nw.query_node(self.ego_id,times)

        res=[(x[0],x[1],x[2],x[3]['weight']) for x in res]
        comp=[(self.ego_id,x[0],x[1],x[2]['weight']) for x in self.add_string_timing[0:2]]
        self.assertEqual(set(res),set(comp))

    def test_query_time(self):
        """Add some links across time then query back one time point"""
        self.neo4nw.insert_edges_query(self.ego_id, self.add_string_timing)
        self.neo4nw.write_queue()
        times = 20010101
        res = self.neo4nw.query_node(self.ego_id, times)

        res = [(x[0], x[1], x[2], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], x[1], x[2]['weight']) for x in self.add_string_timing[1:2]]
        self.assertEqual(set(res), set(comp))

    def test_add_link_new_token(self):
        pass

    def test_get_links_missing_ego(self):
        pass

#if __name__ == '__main__':
#    unittest.main()