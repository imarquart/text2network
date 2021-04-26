import unittest
from src.classes.nw_processor import nw_processor
import glob
import logging
import os
import time
import configparser
import json
from src.classes.neo4jnw import neo4j_network
from src.classes.neo4db import neo4j_database
from Tests.test_setups import test_setup, test_cleanup
import numpy as np


class neo4nw_test(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        test_cleanup(self.neograph)
        pass

    @classmethod
    def setUpClass(self):
        self.neograph, config = test_setup()

    def test_dyad_context(self):

        self.neograph.decondition()
        self.neograph.set_norm_ties(True)
        prox = self.neograph.proximities()
        prox = self.neograph.pd_format(prox)[0]
        assert all(np.sum(prox, axis=0) == 100)
        context = self.neograph.get_dyad_context(
            [("t_manager", "t_leader"), ("t_leader", "t_manager"), ("t_boss", "t_leader"), ("t_leader", "t_boss"),
             ("t_manager", "t_boss")])
        prox = self.neograph.pd_format(context)[0]
        assert not (any((np.sum(prox, axis=0) != 0).values & (np.sum(prox, axis=0) != 1).values))

    def test_condition(self):

        self.neograph.set_norm_ties(False)
        self.neograph.decondition()
        self.neograph.condition()
        assert len(self.neograph.graph) == 6

        self.neograph.decondition()
        assert self.neograph.graph == None
        assert self.neograph.conditioned == False

        self.neograph.decondition()
        self.neograph.condition(tokens=['t_manager'], depth=1)
        assert len(self.neograph.graph) == 2

        self.neograph.decondition()
        self.neograph.condition(tokens=['t_leader'], depth=1)
        assert len(self.neograph.graph) == 3

        self.neograph.decondition()
        self.neograph.condition(tokens=['t_manager'], depth=2)
        assert len(self.neograph.graph) == 3

        self.neograph.decondition()
        self.neograph.condition(tokens=['t_manager', 't_company'], depth=6)
        assert len(self.neograph.graph) == 5

        self.neograph.condition(tokens=['t_leader', 't_company', 't_team', 't_employee', 't_manager'], depth=6)
        assert len(self.neograph.graph) == 6

    def test_proximities(self):

        self.neograph.set_norm_ties(False)
        self.neograph.decondition()
        prox = self.neograph.proximities('t_leader')
        prox = self.neograph.pd_format(prox)[0]
        assert prox.loc["t_leader", "t_manager"] == 1
        assert prox.loc["t_leader", "t_boss"] == 1
        del prox
        prox = self.neograph.proximities('t_manager')
        prox = self.neograph.pd_format(prox)[0]
        assert prox.loc["t_manager", "t_leader",] == 1.5
        del prox

        self.neograph.set_norm_ties(True)
        prox = self.neograph.proximities('t_leader')
        prox = self.neograph.pd_format(prox)[0]
        assert prox.loc["t_leader", "t_manager"] == 100
        assert prox.loc["t_leader", "t_boss"] == 100
        del prox

        prox = self.neograph.proximities('t_manager')
        prox = self.neograph.pd_format(prox)[0]
        assert prox.loc["t_manager", "t_leader",] == 75
        del prox

        self.neograph.decondition()
        self.neograph.set_norm_ties(False)
        self.neograph.condition()
        prox = self.neograph.proximities('t_leader')
        prox = self.neograph.pd_format(prox)[0]
        assert prox.loc["t_leader", "t_manager"] == 1
        assert prox.loc["t_leader", "t_boss"] == 1
        self.neograph.to_symmetric()
        prox = self.neograph.proximities('t_leader')
        prox = self.neograph.pd_format(prox)[0]
        assert prox.loc["t_leader", "t_manager"] == 1.25

    def test___getitem__(self):

        out = self.neograph['t_leader']
        for i, x in enumerate(out):
            out[i] = (self.neograph.ensure_tokens(x[0]), self.neograph.ensure_tokens(x[1]), x[2])
            x = out[i]
            if x[1] == "t_manager":
                assert x[2]['weight'] == 1.0
            if x[1] == "t_boss":
                assert x[2]['weight'] == 1.0

        out = self.neograph['t_manager']
        for i, x in enumerate(out):
            out[i] = (self.neograph.ensure_tokens(x[0]), self.neograph.ensure_tokens(x[1]), x[2])
            x = out[i]
            if x[1] == "t_leader":
                assert x[2]['weight'] == 1.5
            if x[1] == "t_boss":
                assert x[2]['weight'] == 0

        out = self.neograph['t_boss']
        for i, x in enumerate(out):
            out[i] = (self.neograph.ensure_tokens(x[0]), self.neograph.ensure_tokens(x[1]), x[2])
            x = out[i]
            if x[1] == "t_leader":
                assert x[2]['weight'] == 0.5
            if x[1] == "t_manager":
                assert x[2]['weight'] == 0

        self.neograph.set_norm_ties(True)
        out = self.neograph['t_leader']
        for i, x in enumerate(out):
            out[i] = (self.neograph.ensure_tokens(x[0]), self.neograph.ensure_tokens(x[1]), x[2])
            x = out[i]
            if x[1] == "t_manager":
                assert x[2]['weight'] == 1.0 * 100
            if x[1] == "t_boss":
                assert x[2]['weight'] == 1.0 * 100

        out = self.neograph['t_manager']
        for i, x in enumerate(out):
            out[i] = (self.neograph.ensure_tokens(x[0]), self.neograph.ensure_tokens(x[1]), x[2])
            x = out[i]
            if x[1] == "t_leader":
                assert x[2]['weight'] == 0.75 * 100
            if x[1] == "t_boss":
                assert x[2]['weight'] == 0 * 100

        out = self.neograph['t_boss']
        for i, x in enumerate(out):
            out[i] = (self.neograph.ensure_tokens(x[0]), self.neograph.ensure_tokens(x[1]), x[2])
            x = out[i]
            if x[1] == "t_leader":
                assert x[2]['weight'] == 0.25 * 100
            if x[1] == "t_manager":
                assert x[2]['weight'] == 0 * 100


if __name__ == '__main__':
    unittest.main()
