import unittest
from src.classes.neo4j_processor import neo4j_processor
import glob
import logging
import os
import time
import configparser
import json
from src.classes.neo4jnw import neo4j_network
from src.classes.neo4db import neo4j_database
import numpy as np


class Processor_Test(unittest.TestCase):

    def setUp(self):
        # Load Configuration file
        config = configparser.ConfigParser()
        config.read('D:/NLP/InSpeech/BERTNLP/config/config.ini')
        logging_level = config['General'].getint('logging_level')

        # Set up logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging_level)

        db_uri = "http://localhost:7474"
        db_pwd = ('neo4j', 'nlp')
        neo_creds = (db_uri, db_pwd)

        test = neo4j_database(neo_creds)
        self.neograph = neo4j_network(neo_creds)

        self.processor = neo4j_processor(config['Paths']['trained_berts'], self.neograph,
                                    config['Preprocessing'].getint('max_seq_length'), config['Processing'],
                                    text_db=config['Paths']['database'],
                                    split_hierarchy=json.loads(config.get('General', 'split_hierarchy')))

    def test_norm(self):

        # Regular normalization
        x=np.array([1,2,3])
        expected_x=np.array([1/6,2/6,3/6])
        normed_x=self.processor.norm(x,min_zero=False)
        self.assertTrue((normed_x==expected_x).all(), msg="Regular normalization. x: {} - expected: {}".format(normed_x,expected_x))
        self.assertEqual(np.sum(normed_x), 1, msg="Regular normalization. sum(x): {} - expected: {}".format(np.sum(normed_x),1))

        # Zero normalization
        x=np.array([0,1,2,3,0])
        expected_x=np.array([0,1/6,2/6,3/6,0])
        normed_x=self.processor.norm(x,min_zero=False)
        self.assertTrue((normed_x==expected_x).all(), msg="Zero normalization. x: {} - expected: {}".format(normed_x,expected_x))
        self.assertEqual(np.sum(normed_x), 1, msg="Zero normalization. sum(x): {} - expected: {}".format(np.sum(normed_x), 1))

        # Fractional normalization
        x=np.array([0,1/2,1/4,0])
        expected_x=np.array([0,2/3,1/3,0])
        normed_x=self.processor.norm(x,min_zero=False)
        self.assertTrue((normed_x==expected_x).all(), msg="Fractional normalization. x: {} - expected: {}".format(normed_x,expected_x))
        self.assertEqual(np.sum(normed_x), 1, msg="Fractional normalization. sum(x): {} - expected: {}".format(np.sum(normed_x), 1))

        # List normalization
        x=[1,2,3]
        expected_x=np.array([1/6,2/6,3/6])
        normed_x=self.processor.norm(x,min_zero=False)
        self.assertTrue((normed_x==expected_x).all(), msg="List normalization. x: {} - expected: {}".format(normed_x,expected_x))
        self.assertEqual(np.sum(normed_x), 1, msg="List normalization. sum(x): {} - expected: {}".format(np.sum(normed_x), 1))


        # Delete Minimum normalization
        x=np.array([0,1/2,1/4,1/4])
        expected_x=np.array([0,1,0,0])
        normed_x=self.processor.norm(x,min_zero=True)
        self.assertTrue((normed_x==expected_x).all(), msg="Delete Minimum normalization. x: {} - expected: {}".format(normed_x,expected_x))
        self.assertEqual(np.sum(normed_x), 1, msg="Delete Minimum normalization. sum(x): {} - expected: {}".format(np.sum(normed_x), 1))

    def test_calculate_cutoffs(self):

        # Vector of ties of length 5, with degree 4 since one is zero
        x = np.array([0, 0.5, 0.25, 0.15, 0.1])

        sortx = np.sort(x)[::-1]
        max_degree=1000

        # No cutoffs
        percent=100
        expected_degree=len(x[x>0])
        expected_cutoff=0.1
        cutoff_degree, cut_prob = self.processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
        self.assertTrue(cutoff_degree == expected_degree,
                        msg="No cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree))
        self.assertTrue(cut_prob <= expected_cutoff,
                        msg="No cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff))
        implied_mass=np.sum(sortx[0:cutoff_degree])
        self.assertTrue(implied_mass >= percent/100,
                        msg="No cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent))

        # 50% cutoff
        percent=50
        expected_degree=1
        expected_cutoff=0.5
        cutoff_degree, cut_prob = self.processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
        self.assertTrue(cutoff_degree == expected_degree,
                        msg="50%  cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree))
        self.assertTrue(cut_prob <= expected_cutoff,
                        msg="50%  cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff))
        implied_mass=np.sum(sortx[0:cutoff_degree])
        self.assertTrue(implied_mass >= percent/100,
                        msg="50%  cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent))

        # 75% cutoff
        percent=75
        expected_degree=2
        expected_cutoff=0.25
        cutoff_degree, cut_prob = self.processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
        self.assertTrue(cutoff_degree == expected_degree,
                        msg="75% cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree))
        self.assertTrue(cut_prob <= expected_cutoff,
                        msg="75% cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff))
        implied_mass=np.sum(sortx[0:cutoff_degree])
        self.assertTrue(implied_mass >= percent/100,
                        msg="75% cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent))


        # Mean cutoff
        xmean=np.mean(x)
        expected_degree=2
        expected_cutoff=0.2
        cutoff_degree, cut_prob = self.processor.calculate_cutoffs(x, method="mean", percent=percent,
                                                                   max_degree=max_degree, min_cut=0)
        self.assertTrue(cutoff_degree == expected_degree,
                        msg="Mean cutoffs. cutoff_degree: {} - expected: {}".format(cutoff_degree, expected_degree))
        self.assertTrue(cut_prob <= expected_cutoff,
                        msg="Mean cutoffs. cut_prob: {} - expected: {}".format(cut_prob, expected_cutoff))
        implied_mass=np.sum(sortx[0:cutoff_degree])
        self.assertTrue(implied_mass >= percent/100,
                        msg="Mean cutoffs. implied_mass: {} - expected: {}".format(implied_mass*100, percent))

    def test_get_weighted_edgelist(self):

        # Vector of ties of length 5, with degree 4 since one is zero
        x = np.array([0, 0.5, 0.25, 0.15, 0.1])
        max_degree=5

        # 100 Percent
        percent=100
        cutoff_degree, cut_prob = self.processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
        print("Degree: {}, Probability: {}".format(cutoff_degree,cut_prob))
        ties=self.processor.get_weighted_edgelist(token=100, x=x, time=1995, cutoff_number=cutoff_degree, cutoff_probability=cut_prob, seq_id=100, pos=1, p1="p1",
                              p2="p2", p3="p3", p4="p4", max_degree=max_degree)
        self.assertTrue(len(ties) == cutoff_degree,
                        msg="{}% cutoffs. Returned ties: {} - expected: {}".format(percent, len(ties),cutoff_degree))
        print(ties)

        # 75 Percent
        percent=75
        cutoff_degree, cut_prob = self.processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
        print("Degree: {}, Probability: {}".format(cutoff_degree,cut_prob))
        ties=self.processor.get_weighted_edgelist(token=100, x=x, time=1995, cutoff_number=cutoff_degree, cutoff_probability=cut_prob, seq_id=100, pos=1, p1="p1",
                              p2="p2", p3="p3", p4="p4", max_degree=max_degree)
        self.assertTrue(len(ties) == cutoff_degree,
                        msg="{}% cutoffs. Returned ties: {} - expected: {}".format(percent, len(ties),cutoff_degree))
        print(ties)


        # 25 Percent
        percent=25
        cutoff_degree, cut_prob = self.processor.calculate_cutoffs(x, method="percent", percent=percent, max_degree=max_degree, min_cut=0)
        print("Degree: {}, Probability: {}".format(cutoff_degree,cut_prob))
        ties=self.processor.get_weighted_edgelist(token=100, x=x, time=1995, cutoff_number=cutoff_degree, cutoff_probability=cut_prob, seq_id=100, pos=1, p1="p1",
                              p2="p2", p3="p3", p4="p4", max_degree=max_degree)
        self.assertTrue(len(ties) == cutoff_degree,
                        msg="{}% cutoffs. Returned ties: {} - expected: {}".format(percent, len(ties),cutoff_degree))
        print(ties)

if __name__ == '__main__':
    unittest.main()
