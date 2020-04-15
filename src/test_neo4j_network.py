from unittest import TestCase
from NLP.src.neo4j_network import neo4j_network


class Testneo4j_network(TestCase):

    def setUp(self) -> None:
        db_uri = "http://localhost:7474"
        db_pwd = ('neo4j', 'nlp')
        graph_type = "networkx"
        self.tokens = ["car", "house", "dog", "cat"]
        self.token_ids = [10, 11, 12, 13]
        self.ego_id = 11
        self.ego_token = "house"

        self.add_string_ego = [(12, 20000101, {'weight': 0.2, 'p1': 15}), (13, 20020101, {'weight': 0.5})]


        self.neo4nw = neo4j_network((db_uri, db_pwd))
        self.setup_network()

    def setup_network(self):
        """Delete network and re-create it """
        # Run deletion query
        query = "MATCH (n) DETACH DELETE n"
        self.neo4nw.connector.run(query)
        # Setup network
        self.neo4nw.setup_neo_db(self.tokens, self.token_ids)
        self.neo4nw.init_tokens()

    def test_setup(self):
        """Confirm that setup worked"""
        query = "MATCH (n) RETURN n.token_id AS id"
        res = self.neo4nw.connector.run(query)
        id_list = [x['id'] for x in res]
        self.assertTrue(set(id_list) == set(self.token_ids))

    def test_init_tokens(self):
        """Confirm that init token loads correct ids"""
        self.neo4nw.init_tokens()
        self.assertTrue(set(self.neo4nw.ids) == set(self.token_ids))
        self.assertTrue(set(self.neo4nw.tokens) == set(self.tokens))

    def test_delete_token(self):
        """Delete token and its associated ties"""
        self.neo4nw.add_token(15, "newtoken")
        self.neo4nw.write_queue()
        # Also add some ties
        self.neo4nw[15] = self.add_string_ego
        self.neo4nw.remove_token(15)
        self.neo4nw.write_queue()
        self.assertEqual(self.neo4nw[15], [])

    def test_add_token(self):
        """Add a token via the two ways possible. Check it exists"""
        self.neo4nw.add_token(15, "newtoken")
        self.neo4nw.write_queue()
        self.assertEqual(self.neo4nw[15], [])
        self.assertEqual(self.neo4nw[15], self.neo4nw['newtoken'])

        self.neo4nw['newtoken2'] = 16
        self.neo4nw.write_queue()
        self.assertEqual(self.neo4nw[16], [])
        self.assertEqual(self.neo4nw[16], self.neo4nw['newtoken2'])

    def test_update_token(self):
        pass

    def test_missing_token(self):
        """Try add a link from or to token without ID/Token information"""
        self.assertRaises(LookupError, self.neo4nw.__setitem__, "Elephant", self.add_string_ego)
        self.assertRaises(AssertionError, self.neo4nw.__setitem__, 22, self.add_string_ego)

    def test_adding_and_querying_ego(self):
        """Test query options using ego method. Also test both dispatch options and timing returns"""

        # Write query string
        self.neo4nw[self.ego_id] = self.add_string_ego
        self.neo4nw.write_queue()

        # Test whether output of dispatch and direct query coincide
        res = self.neo4nw.query_node(self.ego_id)
        res_dispatch = self.neo4nw[self.ego_id]
        self.assertEqual(res, res_dispatch)
        # Test whether non-timing query is correct
        # Prepare expected return value
        res = [(x[0], x[1], x[2], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], 0, x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res),set(comp))

        # Test whether simple-timing query is correct
        times = self.add_string_ego[0][1]
        res_dispatch = self.neo4nw[(self.ego_id, times)]
        res = self.neo4nw.query_node(self.ego_id, times)
        # Test that both dispatch methods are equal
        self.assertEqual(res, res_dispatch)
        res = [(x[0], x[1], x[2], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], x[1], x[2]['weight']) for x in self.add_string_ego  if x[1]== times]
        self.assertEqual(set(res), set(comp))

        # Test whether interval-timing query is correct
        times = {"start": self.add_string_ego[0][1], "end": self.add_string_ego[1][1]}
        # Start, end and middle value
        nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        res_dispatch = self.neo4nw[(self.ego_id, times)]
        res = self.neo4nw.query_node(self.ego_id, times)
        # Test that both dispatch methods are equal
        self.assertEqual(res, res_dispatch)
        res = [(x[0], x[1], x[2], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0],nw_time['m'], x[2]['weight']) for x in self.add_string_ego]
        self.assertEqual(set(res), set(comp))

    def test_query_cutoff(self):
        """Test query options using ego method. Also test both dispatch options and timing returns"""

        query = [(12, 20000101, {'weight': 0.2, 'p1': 15}), (12, 20010101, {'weight': 0.2, 'p1': 22}),(13, 20000101, {'weight': 0.6, 'p1': 15}), (13, 20010101, {'weight': 0.6, 'p1': 22})]
        # Write query string
        self.neo4nw[self.ego_id] = query
        self.neo4nw.write_queue()

        # Test whether output of dispatch and direct query coincide
        res = self.neo4nw.query_node(self.ego_id)
        # Test whether both links are returned
        # Prepare expected return value
        res = [(x[0], x[1], x[2], x[3]['weight']) for x in res]
        self.assertTrue(len(res)==2)

        # Test whether output of dispatch and direct query coincide
        res = self.neo4nw.query_node(self.ego_id, weight_cutoff=0.5)
        # Test whether both links are returned
        # Prepare expected return value
        res = [(x[0], x[1], x[2], x[3]['weight']) for x in res]
        self.assertTrue(len(res)==1)


    def test_query_aggregation(self):
        """Add several links over time and test whether the aggregation is correct"""

        query = [(12, 20000101, {'weight': 0.2, 'p1': 15}),(12, 20010101, {'weight': 0.5, 'p1': 22}),(12, 20020101, {'weight': 0.9, 'p1': 24})]
        self.neo4nw[self.ego_id]=query
        self.neo4nw.write_queue()

        # Create two intervals
        times = {"start": query[0][1], "end": query[1][1]}
        nw_time = {"s": times['start'], "e": times['end'], "m": int((times['end'] + times['start']) / 2)}
        w1=query[0][2]['weight']+query[1][2]['weight']
        times2 = {"start": query[0][1], "end": query[2][1]}
        nw_time2 = {"s": times2['start'], "e": times2['end'], "m": int((times2['end'] + times2['start']) / 2)}
        w2=query[0][2]['weight']+query[1][2]['weight']+query[2][2]['weight']

        # Short interval
        res = self.neo4nw.query_node(self.ego_id, times)
        res = [(x[0], x[1], x[2], x[3]['weight'],x[3]['occurences'],x[3]['t1'],x[3]['t2']) for x in res]
        # Format expected return
        comp = [(self.ego_id, query[0][0], nw_time['m'], w1 ,2,nw_time['s'],nw_time['e'])]

        # Long interval
        res = self.neo4nw.query_node(self.ego_id, times2)
        res = [(x[0], x[1], x[2], x[3]['weight'],x[3]['occurences'],x[3]['t1'],x[3]['t2']) for x in res]
        # Format expected return
        comp = [(self.ego_id, query[0][0], nw_time2['m'], w2 ,3,nw_time2['s'],nw_time2['e'])]

        self.assertEqual(set(res), set(comp))

    def test_confirm_reversion_mechanism(self):
        """Add links via ego, reverse direction"""

        # Now direction of class and neo4j are equal
        self.setup_network()

        # We add ego->alter in the database
        self.neo4nw[self.ego_id] = self.add_string_ego
        self.neo4nw.write_queue()

        res = self.neo4nw.query_node(self.ego_id)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], x[2]['weight']) for x in self.add_string_ego]
        # Expect to get ego->alter assignment back
        self.assertEqual(set(res), set(comp))

        # Reverse the network
        # Now, a query for ego should give links alter->ego
        # of which there are none
        self.neo4nw.graph_direction = "REVERSE"
        res = self.neo4nw.query_node(self.ego_id)
        self.assertTrue(len(res)==0)

        # Add links in reverse mode
        self.setup_network()
        self.neo4nw.graph_direction = "REVERSE"
        # Given as ego->alter but saved as alter->ego
        self.neo4nw[self.ego_id] = self.add_string_ego
        self.neo4nw.write_queue()

        # Querying in reverse mode should give correct results
        res = self.neo4nw.query_node(self.ego_id)
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        comp = [(self.ego_id, x[0], x[2]['weight']) for x in self.add_string_ego]
        # Expect to get ego->alter assignment back
        self.assertEqual(set(res), set(comp))

        # Switch back to forward mode and assert that reverse nodes are returned
        # Above we added alter->ego links
        # There should be no ego->alter links
        self.neo4nw.graph_direction = "FORWARD"
        res = self.neo4nw.query_node(self.ego_id)
        self.assertTrue(len(res)==0)

        # But querying a alter link should give one result
        res = self.neo4nw.query_node(self.add_string_ego[0][0])
        res = [(x[0], x[1], x[3]['weight']) for x in res]
        comp = [(self.add_string_ego[0][0], self.ego_id, self.add_string_ego[0][2]['weight'])]
        self.assertEqual(set(res), set(comp))


    def test_conditioning(self):
        """This tests conditioning of the network, with time intervals and with cutoff weights"""

        # Write query string
        self.neo4nw[self.ego_id] = self.add_string_ego
        self.neo4nw.write_queue()
        times = {"start": self.add_string_ego[0][1], "end": self.add_string_ego[-1][1]}
        expected_nodes=[x[0] for x in self.add_string_ego]
        expected_nodes.append(self.ego_id)

        # Now condition on ego node
        self.neo4nw.condition(times,self.ego_id)
        self.assertEqual(set(self.neo4nw.graph.nodes), set(expected_nodes))
        self.assertEqual(set(self.neo4nw.ids), set(expected_nodes))

        # Decondition
        self.neo4nw.decondition()
        self.assertTrue(self.neo4nw.graph == None)
        self.assertEqual(set(self.neo4nw.ids), set(self.token_ids))


        # Condition instead on Token string
        self.neo4nw.condition(times,self.ego_token)
        self.assertEqual(set(self.neo4nw.graph.nodes), set(expected_nodes))
        self.assertEqual(set(self.neo4nw.ids), set(expected_nodes))



        # Condition by time, expecting only first node returned
        # Decondition
        self.neo4nw.decondition()
        times=self.add_string_ego[0][1]
        expected_nodes = [x[0] for x in self.add_string_ego if x[1]==times]
        expected_nodes.append(self.ego_id)
        self.neo4nw.condition(times,self.ego_id)
        self.assertEqual(set(self.neo4nw.graph.nodes), set(expected_nodes))
        self.assertEqual(set(self.neo4nw.ids), set(expected_nodes))


        # Condition by weight
        self.neo4nw.decondition()
        times = {"start": self.add_string_ego[0][1], "end": self.add_string_ego[-1][1]}
        expected_nodes = [x[0] for x in self.add_string_ego if x[2]['weight'] >= 0.5]
        expected_nodes.append(self.ego_id)
        self.neo4nw.condition(times, self.ego_id, weight_cutoff=0.5)
        self.assertEqual(set(self.neo4nw.graph.nodes), set(expected_nodes))
        self.assertEqual(set(self.neo4nw.ids), set(expected_nodes))







# if __name__ == '__main__':
#    unittest.main()
