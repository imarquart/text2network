
# TODO: Comment
# TODO: Add parameters


import torch
import tables
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numbers
import logging


class text_dataset_subset(Dataset):
    def __init__(self, data_path, subset_tokens, tokenizer=None, fixed_seq_length=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.fixed_seq_length = fixed_seq_length

        logging.info("Attempting to open %s" % self.data_path)
        # Open table and read data completely
        try:
            self.tables = tables.open_file(self.data_path, mode="r")
        except:
            ConnectionError("Could not open %s" % self.data_path)

        data_table = self.tables.root.textdata.table
        data_iterator = data_table.iterrows()
        # Retain any sequence including focal terms from the subset
        items = [(x['run_index'], x['text'].decode("utf-8")) for x in data_iterator if
                 any(ext in x['text'].decode("utf-8") for ext in subset_tokens)]

        # Init data as empty
        self.data = items
        self.nitems = len(items)

        self.tables.close()

    def __del__(self):
        self.close()

    def close(self):
        self.tables.close()

    def __getitem__(self, index):
        """
        This function can be queried both with a single index i, and a range or list i:j, or i,j,k...

        It returns a tuple with three elements
        1. nxlength tensor of padded tokenized sequences of fixed length including special tokens
        2. list of length of sequences NOT including special tokens
        3. list of tokens, not including special tokens
        """

        if torch.is_tensor(index):
            index = index.to_list(index)

        if isinstance(index, slice):
            index = list(range(index.start or 0, index.stop or self.nitems, index.step or 1))

        if isinstance(index, numbers.Integral):
            index = [int(index)]

        if type(index) is not list:
            index = [index]

        # Get text
        items=[self.data[i] for i in index]
        # If a Tokenizer is set, we will tokenize and return pytorch tensor (padded)
        list_tokens = []
        token_id = []
        sequence_ids = []
        for item in items:
            text = item[1]
            seq_id = item[0]
            indexed_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            # If we wanted fixed size, not just padded to max, we need to cut
            if self.fixed_seq_length is not None:
                indexed_tokens = indexed_tokens[:self.fixed_seq_length]
            indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)
            inputs = torch.tensor([indexed_tokens], requires_grad=False)
            inputs = inputs.squeeze(dim=0).squeeze()
            list_tokens.append(inputs)
            # Also save list of tokens
            token_id.append(inputs[1:-1])
            sequence_ids.append(seq_id)

        # If fixed size, add a sequence of desired length to fix padding
        # In case all other sequences are of smaller length
        # We then use the automatic padding and afterwards delete the padding sequence
        if self.fixed_seq_length is not None:
            # Add padding sequence
            list_tokens.append(torch.zeros([self.fixed_seq_length + 2], dtype=torch.int))
            token_tensor = pad_sequence(list_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            # Detele padding sequence
            token_tensor = token_tensor[:-1, :]
        else:
            token_tensor = pad_sequence(list_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Expand index vector to a 1-dim vector indexing each token by sequence-id
        lengths = ([x.shape[0] for x in token_id])
        indexvec = torch.repeat_interleave(torch.as_tensor(sequence_ids), torch.as_tensor(lengths))

        # Stack token-ids into vector of same length
        token_id = torch.cat([x for x in token_id])

        # Here will need custom collate fn
        return token_tensor, indexvec, token_id

    def __len__(self):
        return self.nitems


class text_dataset(Dataset):
    def __init__(self, data_path, tokenizer=None, fixed_seq_length=None, maxn=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.fixed_seq_length = fixed_seq_length

        # We open table once to get #items, but close it again
        # Bc multithreading requires opening within getitem
        self.tables = tables.open_file(self.data_path, mode="r")
        if maxn is not None:
            self.nitems=min(self.tables.root.textdata.table.nrows,maxn)
        else:
            self.nitems = self.tables.root.textdata.table.nrows
        self.tables.close()

        # Init data as empty
        self.data = None

    def __del__(self):
        self.close()

    def close(self):
        self.tables.close()

    def __getitem__(self, index):
        """
        This function can be queried both with a single index i, and a range or list i:j, or i,j,k...

        It returns a tuple with three elements
        1. nxlength tensor of padded tokenized sequences of fixed length including special tokens
        2. list of length of sequences NOT including special tokens
        3. list of tokens, not including special tokens
        """

        if torch.is_tensor(index):
            index = index.to_list(index)

        if isinstance(index, slice):
            index = list(range(index.start or 0, index.stop or self.nitems, index.step or 1))

        if isinstance(index, numbers.Integral):
            index = [int(index)]

        if type(index) is not list:
            index = [index]

        # Open table if necessary
        if self.tables.isopen == 0:
            self.tables = tables.open_file(self.data_path, mode="r")

        # Check if data is not initialized
        if (self.data is None):
            self.data = self.tables.root.textdata.table

        # Check if data has been closed
        if (self.data._v_isopen == False):
            self.data = self.tables.root.textdata.table

        # Get text
        item = self.data.read_coordinates(index, field='text')
        # Because of pyTables, we have to encode.
        item = [x.decode("utf-8") for x in item]

        # Get year
        year_vec = torch.tensor(self.data.read_coordinates(index, field='year'), requires_grad=False)
        p1= self.data.read_coordinates(index, field='p1')
        p2= self.data.read_coordinates(index, field='p2')
        p3= self.data.read_coordinates(index, field='p3')
        p4= self.data.read_coordinates(index, field='p4')
        p1 = torch.tensor([x.decode("utf-8") for x in p1], requires_grad=False)
        p2 = torch.tensor([x.decode("utf-8") for x in p2], requires_grad=False)
        p3 = torch.tensor([x.decode("utf-8") for x in p3], requires_grad=False)
        p4 = torch.tensor([x.decode("utf-8") for x in p4], requires_grad=False)

        # If a Tokenizer is set, we will tokenize and return pytorch tensor (padded)
        if self.tokenizer is not None:
            list_tokens = []
            token_id = []
            for text in item:
                indexed_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                # If we wanted fixed size, not just padded to max, we need to cut
                if self.fixed_seq_length is not None:
                    indexed_tokens = indexed_tokens[:self.fixed_seq_length]
                indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)
                inputs = torch.tensor([indexed_tokens], requires_grad=False)
                inputs = inputs.squeeze(dim=0).squeeze()
                list_tokens.append(inputs)
                # Also save list of tokens
                token_id.append(inputs[1:-1])

            # If fixed size, add a sequence of desired length to fix padding
            if self.fixed_seq_length is not None:
                # Add padding sequence
                list_tokens.append(torch.zeros([self.fixed_seq_length + 2], dtype=torch.int))
                item = pad_sequence(list_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                # Detele padding sequence
                item = item[:-1, :]
            else:
                item = pad_sequence(list_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)

            # Expand index vector to a 1-dim vector indexing each token by sequence-id
            lengths = ([x.shape[0] for x in token_id])
            indexvec = torch.repeat_interleave(torch.as_tensor(index), torch.as_tensor(lengths))

            # Stack token-ids into vector of same length
            token_id = torch.cat([x for x in token_id])

            # Here will need custom collate fn
            return item, indexvec, token_id, year_vec, p1,p2,p3,p4
        else:
            return index, item

    def __len__(self):
        return self.nitems


def text_dataset_collate_randomsample(batch):
    """
    This function collates batches of text_dataset, if a batch is a
    list or tuple of [dataset[i],dataset[j]] and so forth

    In particular, this is to be used with the pyTorch RANDOM dataloader, who does exactly this
    """

    tokens = torch.cat([x[2] for x in batch])
    indexvec = torch.cat([torch.as_tensor(x[1]) for x in batch])
    elem = batch[0][0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for batch_el in batch for x in batch_el[0]])
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    return torch.stack([x[0].squeeze() for x in batch], 0), indexvec, tokens


def text_dataset_collate_batchsample(batch):
    """
    This function collates batches of text_dataset, if a batch is
    determined by the batch_sampler and just needs to be passed
    """
    return batch
