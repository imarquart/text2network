# TODO: Comment

import torch
import tables
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numbers


class tensor_dataset(Dataset):
    def __init__(self, data_path, method=0 ):
        self.data_path = data_path
        self.method=method
        # Init data as empty
        self.data = None
        # Open database connection, and table
        self.data, self.table = self.open_db_con()

        self.nitems = self.data.nrows
        # Save a list of nodes
        self.nodes = np.unique(self.data.col('token_id'))

        self.table.close()



    def open_db_con(self):
        try:
            table = tables.open_file(self.data_path, mode="r")
            token_table=self.data
            # Check if data is not initialized
            if (self.data is None):
                token_table = table.root.token_data.table
            # Check if data has been closed
            elif (self.data._v_isopen == False):
                token_table = table.root.token_data.table

        except:
            raise FileNotFoundError("Could not read token table from database.")

        return token_table,table

    def __del__(self):
        self.close()

    def close(self):
        self.table.close()

    def __getitem__(self, index):

        # Transform all index requests to lists to query
        if torch.is_tensor(index):
            index=index.to_list(index)
        if isinstance(index, slice):
            index=list(range(index.start or 0, index.stop or self.nitems, index.step or 1))
        if isinstance(index, numbers.Integral):
            index = [int(index)]
        if type(index) is not list:
             index = [index]

        if self.table.isopen==0:
            self.data, self.table = self.open_db_con()

        chunk = self.nodes[index]
        limits = [chunk[0], chunk[-1]]
        rows=[]


        if self.method == "single":
            # %% Method "Single": Query each token separately
            context_dists=[]
            own_dists=[]
            token_idx=[]
            for node in chunk:
                query = "".join(['(token_id==', str(node), ')'])
                row = self.data.read_where(query)
                own_dists.append(row['own_dist'].squeeze())
                context_dists.append(row['own_dist'].squeeze())
                token_idx.append(row['token_id'].squeeze())

            nr_rows = len(token_idx)
            if nr_rows == 0:
                raise AssertionError("Database error: Token information missing")
            own_dists=np.concatenate(own_dists, axis=0)
            context_dists = np.concatenate(context_dists, axis=0)
            token_idx = np.concatenate(token_idx, axis=0)


        else:
            # %% Method 0: Range query (sorted)
            query = "".join(['(token_id>=', str(limits[0]), ') & (token_id<=', str(limits[1]), ')'])
            rows = self.data.read_where(query)
            nr_rows = len(rows)
            if nr_rows == 0:
                raise AssertionError("Database error: Token information missing")
            context_dists = np.stack([x[0] for x in rows], axis=0).squeeze()
            own_dists = np.stack([x[1] for x in rows], axis=0).squeeze()
            token_idx = np.stack([x['token_id'] for x in rows], axis=0).squeeze()

        if nr_rows == 1:
            context_dists = np.reshape(context_dists, (-1, context_dists.shape[0]))
            own_dists = np.reshape(own_dists, (-1, own_dists.shape[0]))

        return chunk,token_idx,own_dists,context_dists


def text_dataset_collate_batchsample(batch):
    """
    This function collates batches of text_dataset, if a batch is
    determined by the batch_sampler and just needs to be passed
    """
    return batch