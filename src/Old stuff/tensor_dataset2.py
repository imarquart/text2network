# TODO: Comment

import torch
import tables
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numbers


class tensor_dataset(Dataset):
    def __init__(self, data_path, method=0):
        self.data_path = data_path
        self.method = method
        # Init data as empty
        self.data = None
        # Open database connection, and table
        self.data, self.table = self.open_db_con()

        # Get token list
        try:
            tbl_token_list = self.table.root.token_list
        except:
            ImportError("Could not load token list.")

        self.nitems = tbl_token_list.nrows
        # Save a list of nodes
        self.nodes = np.unique(tbl_token_list.col('idx'))
        self.nodes = np.sort(self.nodes)

        self.table.close()

    def open_db_con(self):
        try:
            table = tables.open_file(self.data_path, mode="r")
            group = self.data
            # Check if data is not initialized
            if (self.data is None):
                group = table.root.token_data
            # Check if data has been closed
            elif (self.data._v_isopen == False):
                group = table.root.token_data

        except:
            raise FileNotFoundError("Could not read token table from database.")

        return group, table

    def __del__(self):
        self.close()

    def close(self):
        self.table.close()

    def tokenid_to_index(self, token_id):
        idx = np.flatnonzero(self.nodes == token_id)
        return idx

    def __getitem__(self, index):

        # Transform all index requests to lists to query
        if torch.is_tensor(index):
            index = index.to_list(index)
        if isinstance(index, slice):
            index = list(range(index.start or 0, index.stop or self.nitems, index.step or 1))
        if isinstance(index, numbers.Integral):
            index = np.array([index])
        if isinstance(index, list) or isinstance(index, tuple):
            index = np.array(index)

        if self.table.isopen == 0:
            self.data, self.table = self.open_db_con()

        chunk = self.nodes[index]
        limits = [chunk[0], chunk[-1]]
        rows = []

        # Open group of token tables
        try:
            group = self.table.root.token_data
        except:
            ImportError("Could not open root group of token data.")

        # %% Method "Single": Query each token separately
        context_dists = []
        own_dists = []
        token_idx = []
        for node in chunk:
            token_name = "".join(["tk_", str(node)])
            try:
                token_table = group.__getitem__(token_name)
            except:
                ImportError("Could not find token information on %i" % node)

            row = token_table[:]
            nr_rows = len(row)

            own_dist = row['own_dist'].squeeze()
            context_dist = row['context_dist'].squeeze()
            token_id = np.repeat([node], nr_rows)

            # Control for 1-sequence results with zero first dimension
            if len(row) == 1:
                context_dist = np.reshape(context_dist, (-1, context_dist.shape[0]))
                own_dist = np.reshape(own_dist, (-1, own_dist.shape[0]))

            own_dists.append(own_dist)
            context_dists.append(context_dist)
            token_idx.append(token_id)

        nr_rows = len(token_idx[0])
        if nr_rows == 0:
            raise AssertionError("Database error: Token information missing")
        own_dists = np.concatenate(own_dists, axis=0)
        context_dists = np.concatenate(context_dists, axis=0)
        token_idx = np.concatenate(token_idx, axis=0)

        return chunk, token_idx, own_dists, context_dists


def tensor_dataset_collate_batchsample(batch):
    """
    This function collates batches of text_dataset, if a batch is
    determined by the batch_sampler and just needs to be passed
    """
    return batch
