import torch
import tables
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numbers

#TODO: Comment

class text_dataset_simple(Dataset):
    def __init__(self, data_path, tokenizer=None, fixed_seq_length=None ):
        self.data_path = data_path
        self.tokenizer=tokenizer
        self.fixed_seq_length=fixed_seq_length

        # We open table once to get #items, but close it again
        # Bc multithreading requires opening within getitem
        self.tables = tables.open_file(self.data_path, mode="r")
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
            index=index.to_list(index)

        if isinstance(index, slice):
            index=list(range(index.start or 0, index.stop or self.nitems, index.step or 1))

        if isinstance(index, numbers.Integral):
            index = [int(index)]

        if type(index) is not list:
             index = [index]

        # Open table if necessary
        if self.tables.isopen==0:
            self.tables = tables.open_file(self.data_path, mode="r")

        # Check if data is not initialized
        if (self.data is None):
            self.data = self.tables.root.textdata.table

        # Check if data has been closed
        if (self.data._v_isopen==False):
            self.data = self.tables.root.textdata.table

        # Get text
        item = self.data.read_coordinates(index,field='text')
        # Because of pyTables, we have to encode.
        item = [x.decode("utf-8") for x in item]

        # If a Tokenizer is set, we will tokenize and return pytorch tensor (padded)
        if self.tokenizer is not None:
            list_tokens = []
            token_id = []
            for text in item:
                indexed_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                # If we wanted fixed size, not just padded to max, we need to cut
                if self.fixed_seq_length is not None:
                    indexed_tokens=indexed_tokens[:self.fixed_seq_length]
                indexed_tokens=self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)
                inputs = torch.tensor([indexed_tokens], requires_grad=False)
                inputs = inputs.squeeze(dim=0).squeeze()
                list_tokens.append(inputs)
                # Also save list of tokens
                token_id.append(inputs[1:-1])

            # If fixed size, add a sequence of desired length to fix padding
            if self.fixed_seq_length is not None:
                # Add padding sequence
                list_tokens.append(torch.zeros([self.fixed_seq_length+2], dtype=torch.int))
                item = pad_sequence(list_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                # Detele padding sequence
                item=item[:-1,:]
            else:
                item = pad_sequence(list_tokens, batch_first=True, padding_value=self.tokenizer.pad_token_id)

            # Here will need custom collate fn
            return item
        else:
            return item

    def __len__(self):
        return self.nitems



