
import numpy as np
import torch
import tables
import pandas as pd
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numbers
import logging
import os
import pickle
import nltk
from nltk.tag import pos_tag, map_tag
from tqdm.auto import tqdm


try:
    from textblob import TextBlob
    textblob_available=True
except:
    TextBlob=None
    textblob_available=False

try:
    #nltk.download('vader_lexicon')
    # Initialize the VADER sentiment analyzer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_available=True
except:
    vader_analyzer = None
    vader_available=False

from text2network.utils.load_bert import get_full_vocabulary




class query_dataset(Dataset):
    def __init__(self, data_path, tokenizer=None, fixed_seq_length=None, maxn=None, query=None,
                 logging_level=logging.DEBUG, pos=True, sentiment=True):
        # TODO: Redo out of memory if necessary
        # TODO: Add maxn option
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.fixed_seq_length = fixed_seq_length
        self.query = query
        self.logging_level = logging_level
        self.pos=pos
        self.sentiment=sentiment
        logging.disable(logging_level)
        logging.info("Creating features from database file at %s", self.data_path)

        self.tables = tables.open_file(self.data_path, mode="r")
        self.data = self.tables.root.textdata.table

        items = self.data.read_where(self.query)

        # Get data
        items_text = items['text']
        items_year = items['year']
        items_seqid = items['seq_id']  # ?
        items_runindex = items['run_index']
        items_p1 = items['p1']
        items_p2 = items['p2']
        items_p3 = items['p3']
        items_p4 = items['p4']

        # Because of pyTables, we have to encode.
        items_text = [x.decode("utf-8") for x in items_text]
        items_p1 = [x.decode("utf-8") for x in items_p1]
        items_p2 = [x.decode("utf-8") for x in items_p2]
        items_p3 = [x.decode("utf-8") for x in items_p3]
        items_p4 = [x.decode("utf-8") for x in items_p4]

        self.data = pd.DataFrame(
            {"year": items_year, "seq_id": items_seqid, "run_index": items_runindex, "text": items_text, "p1": items_p1, "p2": items_p2,
             "p3": items_p3, "p4": items_p4})
        self.tables.close()

        # Delete very short sequences
        length = self.data.text.apply(len)
        index=length >= 3
        self.data=self.data.loc[index,:]
        self.nitems = len(self.data.text)

        # Setup unique words ID masking
        logging.info("Setting up unique words")
        all_tokens=[tokenizer.tokenize(x) for x in items_text]
        all_ids=[tokenizer.convert_tokens_to_ids(x) for x in all_tokens]
        all_ids=list(set(sum(all_ids,[])))
        # ID mask is 1 for tokens that do not appear in the text
        self.id_mask=torch.ones(len(tokenizer))
        self.id_mask[all_ids]=0
        self.id_mask=np.where(self.id_mask==1)[0]

    def __getitem__(self, index):
        """
        Pulls items from data and returns a batch
        Given a fixed_seq_length of n
        and an index query of k elements
        the main tensor is of shape k,n+2 with 2 added delimiting tokens
        :param index: Can be integer, list of integer or a range
        :return: One batch for data ingest
                    token_input_vec: k,n+2 tensor with fixed length sequences
                    token_id_vec: 1,k*n tensor giving a sequence of all token_ids in batch
                    seq_vec, run_index: 1,k*n tensor giving the sequence id (in db) and run_index for each token in batch
                    index_vec: 1,k*n tensor giving the index (in dataset) for each token in batch
                    year_vec: 1,k*n tensor, giving the year for each token in batch
                    p1_vec: 1,k*n array, giving p1 parameter for each token in batch
                    p2_vec: as above
                    p3_vec: as above
                    p4_vec: as above
        """

        if torch.is_tensor(index):
            index = index.to_list(index)

        if isinstance(index, slice):
            index = list(range(index.start or 0, index.stop or self.nitems, index.step or 1))

        if isinstance(index, numbers.Integral):
            index = [int(index)]

        if type(index) is not list:
            index = [index]

        item = self.data.iloc[index, :]
        # Get numpy or torch vectors (numpy for the strings)
        texts = item['text'].to_numpy()
        year_vec = torch.tensor(item['year'].to_numpy(dtype="int32"), requires_grad=False)
        p1_vec = item['p1'].to_numpy()
        p2_vec = item['p2'].to_numpy()
        p3_vec = item['p3'].to_numpy()
        p4_vec = item['p4'].to_numpy()
        seq_vec = item['seq_id'].to_numpy(dtype="int32")
        runindex_vec = item['run_index'].to_numpy(dtype="int32")

        token_input_vec = []  # tensor of padded inputs with special tokens
        token_id_vec = []  # List of token ids for each sequence, later transformed to 1-dim tensor over batch

        pos_vec = []
        sentiment_vec = []
        subject_vec = []
        for text in texts:

            # Text tokenization
            ##
            indexed_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            # Need fixed size, so we need to cut and pad
            indexed_tokens = indexed_tokens[:self.fixed_seq_length]
            indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)
            inputs = torch.tensor([indexed_tokens], requires_grad=False)
            inputs = inputs.squeeze(dim=0).squeeze()
            token_input_vec.append(inputs)
            # Also save list of tokens
            token_id_vec.append(inputs[1:-1])

            if self.pos:
                reform_text = self.tokenizer.convert_ids_to_tokens(list(inputs[1:-1].numpy()))
                pos_tags=pos_tag(reform_text)
                if len(reform_text) != len(pos_tags):
                    raise AssertionError("NLTK POS Tagger could not tag all tokens!")
                # Use simplified Tag Set instead of Penn
                pos_tags= [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tags]
                # Get rid of unknown stuff
                pos = [x[1] if x[0]!=x[1] else "UNKNOWN" for x in pos_tags]
                pos = np.array(pos)
            else:
                pos = np.zeros_like(inputs[1:-1].numpy())
            pos_vec.append(pos)
            # Sentiment analysis
            if self.sentiment and textblob_available:
                reform_text=self.tokenizer.convert_ids_to_tokens(list(inputs[1:-1].numpy()))
                joined_text = " ".join(reform_text)
                txtblb = TextBlob(joined_text)
                sentiment= torch.tensor(txtblb.sentiment.polarity)
                subject = torch.tensor(txtblb.sentiment.subjectivity)
            else:
                sentiment = torch.tensor(0)
                subject = torch.tensor(0)
            sentiment_vec.append(sentiment)
            subject_vec.append(subject)

        # Add padding sequence
        token_input_vec.append(torch.zeros([self.fixed_seq_length + 2], dtype=torch.int))
        token_input_vec = pad_sequence(token_input_vec, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # Detele padding sequence
        token_input_vec = token_input_vec[:-1, :]

        # Prepare index vector and sequence vector by using sequence IDs found in database
        lengths = ([x.shape[0] for x in token_id_vec])
        index_vec = torch.repeat_interleave(torch.as_tensor(index), torch.as_tensor(lengths))
        seq_vec = torch.repeat_interleave(torch.as_tensor(seq_vec), torch.as_tensor(lengths))
        runindex_vec = torch.repeat_interleave(torch.as_tensor(runindex_vec), torch.as_tensor(lengths))
        year_vec = torch.repeat_interleave(torch.as_tensor(year_vec), torch.as_tensor(lengths))
        sentiment_vec =  torch.repeat_interleave(torch.as_tensor(sentiment_vec), torch.as_tensor(lengths))
        subject_vec =  torch.repeat_interleave(torch.as_tensor(subject_vec), torch.as_tensor(lengths))
        p1_vec = p1_vec.repeat(lengths)
        p2_vec = p2_vec.repeat(lengths)
        p3_vec = p3_vec.repeat(lengths)
        p4_vec = p4_vec.repeat(lengths)

        # Cat token_id_vec list into a single tensor for the whole batch
        token_id_vec = torch.cat([x for x in token_id_vec])
        pos_vec= np.concatenate([x for x in pos_vec])

        return token_input_vec, token_id_vec, index_vec, seq_vec, runindex_vec, year_vec, p1_vec, p2_vec, p3_vec, p4_vec, pos_vec, sentiment_vec, subject_vec

    def __len__(self):
        return len(self.data)



class bert_dataset(Dataset):
    # TODO: Padd sentences instead of joining and splitting!
    def __init__(self, tokenizer, database, where_string, block_size=30, check_vocab=False, freq_cutoff=10, logging_level=logging.DEBUG):
        """
        Loads data from database according to where string.
        This dataset is only used to train BERT and cuts texts without respecting sentence logic in database.
        For processing, another dataset is used.

        If check_vocab= True, this dataset will not load any data but merely save the set of missing tokens
        in self.missing_tokens. This can be used to augment BERT's vocabulary before training.

        We generally use a custom tokenizer subclass that gets rid of performance problems associated with
        adding tokens. This problem seems to be fixed in later versions of pyTorch transformers.

        Parameters
        ----------
        tokenizer: PyTorch tokenizer
        database: str
            HDFS data of sentences
        where_string: str
            query on database
        block_size: int
            length of distinct sentences
        check_vocab: bool
            If true, will not tokenize data but only check vocabulary
        freq_cutoff: int
            Number of occurrences required for a token to be considered for vocabulary check
        logging_level: logging.level
        """
        self.database = database
        #logging.disable(logging_level)
        logging.info("Creating features from database file at {} with query {}".format(database,where_string))

        self.tables = tables.open_file(self.database, mode="r")
        self.data = self.tables.root.textdata.table
        nr_rows = self.data.nrows
        logging.warning("HDF Table has {} rows.".format(nr_rows))

        self.where_string=where_string

        coordinates= self.data.get_where_list(where_string)
        nr_rows_to_read = len(coordinates)
        logging.warning("Query plan reading {} rows.".format(nr_rows_to_read))

        self.examples = []


        if check_vocab:

            items = self.data.read_where(where_string)['text']
            # Because of pyTables, we have to encode.
            items = [x.decode("utf-8") for x in items]
            text = ' '.join(items)
            # Get unique tokens

            nltk_tokens=nltk.word_tokenize(text)
            nltk_tokens=[w.lower() for w in nltk_tokens if (w.isalpha() and len(w) > 3)]
            stop_words = set(stopwords.words('english'))
            nltk_tokens = [w for w in nltk_tokens if not w in stop_words]
            #nltk_tokens=list(np.setdiff1d(nltk_tokens,['']))
            # Get frequencies
            freq_table = {}
            for word in nltk_tokens:
                #word=ps.stem(word)
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
            if len(freq_table)>0:
                freq_table = pd.DataFrame(list(freq_table.values()), index=list(freq_table.keys()))
                freq_table = freq_table.sort_values(by=0, ascending=False)
                freq_table = freq_table[freq_table > freq_cutoff].dropna()
                nltk_tokens=list(freq_table.index)
                ids,tokenizer_vocab=get_full_vocabulary(tokenizer)
                self.missing_tokens=list(np.setdiff1d(nltk_tokens, tokenizer_vocab))
                if len(self.missing_tokens) > 1:
                    logging.warning("{} missing tokens in vocabulary".format(len(self.missing_tokens)))
            else:
                logging.warning("Text with query {} has no words".format(where_string))
                self.missing_tokens=[]
            self.examples=[]
        else:

            # Get text
            for item in tqdm(self.data.itersequence(coordinates), desc="Loading Training data ", total=nr_rows_to_read):
            #for item in tqdm(self.data.where(where_string), desc="Loading Training data ", total=nr_rows_to_read):
                item=item['text']
                item = item.decode("utf-8")
                text=tokenizer.tokenize(item)
                text=tokenizer.convert_tokens_to_ids(text)

                # this will change so save it
                ln_text=len(text)

                if (ln_text < block_size):
                    text = tokenizer.build_inputs_with_special_tokens(text)
                    text[len(text):block_size] = np.repeat(tokenizer.pad_token_id, block_size - ln_text)
                else:
                    text=tokenizer.build_inputs_with_special_tokens(text[0:block_size])

                self.examples.append(text)


        self.tables.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])





class bert_dataset_old2(Dataset):
    # TODO: Padd sentences instead of joining and splitting!
    def __init__(self, tokenizer, database, where_string, block_size=30, check_vocab=False, freq_cutoff=10,
                 logging_level=logging.DEBUG):
        """
        Loads data from database according to where string.

        This is the new version that is based on sentences. In fact, it uses internally the dataset class used for inference


        Parameters
        ----------
        tokenizer: PyTorch tokenizer
        database: str
            HDFS data of sentences
        where_string: str
            query on database
        block_size: int
            length of distinct sentences
        check_vocab: bool
            If true, will not tokenize data but only check vocabulary
        freq_cutoff: int
            Number of occurrences required for a token to be considered for vocabulary check
        logging_level: logging.level
        """
        self.database = database
        logging.disable(logging_level)
        logging.info("Creating features from database file at {} with query {}".format(database, where_string))

        self.tables = tables.open_file(self.database, mode="r")
        self.data = self.tables.root.textdata.table
        self.where_string = where_string
        # Get text
        items = self.data.read_where(where_string)['text']

        # Because of pyTables, we have to encode.
        items = [x.decode("utf-8") for x in items]

        text = ' '.join(items)

        self.examples = []

        if check_vocab:
            # Get unique tokens

            nltk_tokens = nltk.word_tokenize(text)
            nltk_tokens = [w.lower() for w in nltk_tokens if (w.isalpha() and len(w) > 3)]
            stop_words = set(stopwords.words('english'))
            nltk_tokens = [w for w in nltk_tokens if not w in stop_words]
            # nltk_tokens=list(np.setdiff1d(nltk_tokens,['']))
            # Get frequencies
            freq_table = {}
            for word in nltk_tokens:
                # word=ps.stem(word)
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
            if len(freq_table) > 0:
                freq_table = pd.DataFrame(list(freq_table.values()), index=list(freq_table.keys()))
                freq_table = freq_table.sort_values(by=0, ascending=False)
                freq_table = freq_table[freq_table > freq_cutoff].dropna()
                nltk_tokens = list(freq_table.index)
                ids, tokenizer_vocab = get_full_vocabulary(tokenizer)
                self.missing_tokens = list(np.setdiff1d(nltk_tokens, tokenizer_vocab))
                if len(self.missing_tokens) > 1:
                    logging.warning("{} missing tokens in vocabulary".format(len(self.missing_tokens)))
            else:
                logging.warning("Text with query {} has no words".format(where_string))
                self.missing_tokens = []
            self.examples = []
        else:
            logging.info("Setting up sentence-based dataset for BERT training examples")
            queryds=query_dataset(data_path=database, tokenizer=tokenizer, fixed_seq_length=block_size, maxn=None, query=where_string,
                 logging_level=logging_level, pos=False, sentiment=False)
            self.examples = []
            for data in tqdm(queryds, desc="Tokenized sentence for query {}".format(where_string)):
                self.examples.append(data[0][0])
        self.tables.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]




class bert_dataset_old(Dataset):
    # TODO: Padd sentences instead of joining and splitting!
    def __init__(self, tokenizer, database, where_string, block_size=30, check_vocab=False, freq_cutoff=10, logging_level=logging.DEBUG):
        """
        Loads data from database according to where string.
        This dataset is only used to train BERT and cuts texts without respecting sentence logic in database.
        For processing, another dataset is used.

        If check_vocab= True, this dataset will not load any data but merely save the set of missing tokens
        in self.missing_tokens. This can be used to augment BERT's vocabulary before training.

        We generally use a custom tokenizer subclass that gets rid of performance problems associated with
        adding tokens. This problem seems to be fixed in later versions of pyTorch transformers.

        Parameters
        ----------
        tokenizer: PyTorch tokenizer
        database: str
            HDFS data of sentences
        where_string: str
            query on database
        block_size: int
            length of distinct sentences
        check_vocab: bool
            If true, will not tokenize data but only check vocabulary
        freq_cutoff: int
            Number of occurrences required for a token to be considered for vocabulary check
        logging_level: logging.level
        """
        self.database = database
        logging.disable(logging_level)
        logging.info("Creating features from database file at {} with query {}".format(database,where_string))

        self.tables = tables.open_file(self.database, mode="r")
        self.data = self.tables.root.textdata.table
        self.where_string=where_string
        # Get text
        items = self.data.read_where(where_string)['text']

        # Because of pyTables, we have to encode.
        items = [x.decode("utf-8") for x in items]

        text = ' '.join(items)

        self.examples = []


        if check_vocab:
            # Get unique tokens

            nltk_tokens=nltk.word_tokenize(text)
            nltk_tokens=[w.lower() for w in nltk_tokens if (w.isalpha() and len(w) > 3)]
            stop_words = set(stopwords.words('english'))
            nltk_tokens = [w for w in nltk_tokens if not w in stop_words]
            #nltk_tokens=list(np.setdiff1d(nltk_tokens,['']))
            # Get frequencies
            freq_table = {}
            for word in nltk_tokens:
                #word=ps.stem(word)
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
            if len(freq_table)>0:
                freq_table = pd.DataFrame(list(freq_table.values()), index=list(freq_table.keys()))
                freq_table = freq_table.sort_values(by=0, ascending=False)
                freq_table = freq_table[freq_table > freq_cutoff].dropna()
                nltk_tokens=list(freq_table.index)
                ids,tokenizer_vocab=get_full_vocabulary(tokenizer)
                self.missing_tokens=list(np.setdiff1d(nltk_tokens, tokenizer_vocab))
                if len(self.missing_tokens) > 1:
                    logging.warning("{} missing tokens in vocabulary".format(len(self.missing_tokens)))
            else:
                logging.warning("Text with query {} has no words".format(where_string))
                self.missing_tokens=[]
            self.examples=[]
        else:
            # Now use the tokenizer to correctly tokenize
            tokens= tokenizer.tokenize(text)
            tokenized_text = tokenizer.convert_tokens_to_ids(tokens)

            if(len(tokenized_text) < block_size):
                asdf=tokenizer.build_inputs_with_special_tokens(tokenized_text)
                asdf[len(asdf):block_size] = np.repeat(tokenizer.pad_token_id, block_size - len(asdf))
                self.examples.append(asdf)

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

        self.tables.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def text_dataset_collate_batchsample(batch):
    """
    This function collates batches of text_dataset, if a batch is
    determined by the batch_sampler and just needs to be passed
    """
    return batch
