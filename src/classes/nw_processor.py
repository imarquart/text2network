import logging
import time
import numpy as np
import tables
import torch
import tqdm
from torch.utils.data import BatchSampler, SequentialSampler
from src.datasets.dataloaderX import DataLoaderX
from src.datasets.text_dataset import query_dataset, text_dataset_collate_batchsample
from src.utils.delwords import create_stopword_list
from src.utils.rowvec_tools import simple_norm
from src.utils.get_uniques import get_uniques
from src.utils.load_bert import get_bert_and_tokenizer, get_full_vocabulary
import gc
from src.utils.hash_file import hash_string, check_step, complete_step


class nw_processor():
    def __init__(self, trained_folder, neograph, MAX_SEQ_LENGTH, processing_options, text_db=None, maxn=None,
                 nr_workers=0, split_hierarchy=None, processing_cache=None, prune_missing_tokens=True, logging_level=logging.NOTSET):
        """
        Extracts pre-processed sentences, gets predictions by BERT and creates a network

        Network is created for both context distribution and for replacement distribution
        Each sentence is added via parallel ties in a multigraph

        :param tokenizer: BERT tokenizer (pyTorch)
        :param bert: BERT model
        :param text_db: HDF5 File of processes sentences, string of tokens, ending with punctuation
        :param MAX_SEQ_LENGTH:  maximal length of sequences
        :param batch_size: batch size to send to BERT
        :param nr_workers: Nr workers for dataloader. Probably should be set to 0 on windows
        :param method: "attention": Weigh by BERT attention; "context_element": Sum probabilities unweighted
        :param cutoff_percent: Amount of probability mass to use to create links. Smaller values, less ties.
        """
        # Assign parameters
        self.trained_folder = trained_folder
        self.tokenizer = None
        self.bert = None
        self.text_db = text_db
        self.neograph = neograph
        self.prune_missing_tokens=prune_missing_tokens
        self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        self.DICT_SIZE = 0
        self.batch_size = int(processing_options['batch_size'])
        self.maxn = maxn
        self.nr_workers = nr_workers
        self.cutoff_percent = int(processing_options['cutoff_percent'])
        self.max_degree = int(processing_options['max_degree'])
        self.context_cutoff_percent = int(processing_options['context_cutoff_percent'])
        self.context_max_degree = int(processing_options['context_max_degree'])
        self.split_hierarchy = split_hierarchy
        self.logging_level = logging_level
        self.processing_cache=processing_cache

        # Set uniques
        self.setup_uniques(self.split_hierarchy)
        # Set logging level
        logging.disable(logging_level)

    def setup_uniques(self, split_hierarchy=None):
        """
        Set up unique queries
        :param split_hierarchy: can provide other hierarchy here
        :return:
        """
        if split_hierarchy is not None:
            self.split_hierarchy = split_hierarchy
        assert self.split_hierarchy is not None

        self.uniques = get_uniques(self.split_hierarchy, self.text_db)


    def setup_bert(self, fname):

        del self.bert
        del self.tokenizer

        bert_folder = ''.join([self.trained_folder, '/', fname])
        tokenizer, bert = get_bert_and_tokenizer(bert_folder, True)
        self.DICT_SIZE = len(tokenizer)
        return tokenizer, bert

    def run_all_queries(self, clean_database=True, split_hierarchy=None, logging_level=None, prune_database=True):

        # SEt up logging
        if logging_level is not None:
            logging.disable(logging_level)
        else:
            logging.disable(self.logging_level)

        # Check if new split hierarchy needs to be processed
        if split_hierarchy is not None:
            self.setup_uniques(split_hierarchy)

        # Clean the database
        if clean_database == True:
            logging.info("Cleaning Neo4j Database of all prior connections")
            self.neograph.db.clean_database()



        for idx,query_filename in enumerate(self.uniques['query_filename']):
            # Get File name
            query = query_filename[0]
            fname = query_filename[1]
            if self.processing_cache is not None:
                processing_folder= ''.join([self.processing_cache, '/', fname])
            else:
                processing_folder = 'None'
            # Set up Hash
            hash = hash_string(processing_folder, hash_factory="md5")
            if (check_step(processing_folder, hash) and (self.processing_cache is not None)):
                logging.info("Found processed cache for %s. Skipping" % processing_folder)
            else:
                gc.collect()
                torch.cuda.empty_cache()
                logging.info("Processing query {}".format(query))
                start_time = time.time()
                self.process_query(query, fname)
                logging.info("Processing Time: %s seconds" % (time.time()-start_time))
                if self.processing_cache is not None:
                    complete_step(processing_folder, hash)

        # Prune the database
        if prune_database == True:
            logging.info("Cleaning Neo4j Database of all prior connections")
            self.neograph.db.prune_database()

    def process_query(self, query, fname, text_db=None, logging_level=None):
        """
        Extracts pre-processed sentences, gets predictions by BERT and creates network ties
        Ties are then added to the neo4j database.

        Network is created for both context distribution and for replacement distribution
        Each sentence is added via parallel ties in a multigraph

        :param query: Based on parameters in the database, process this particular query
        """
        tables.set_blosc_max_threads(15)

        # SEt up logging
        if logging_level is not None:
            logging.disable(logging_level)
        else:
            logging.disable(self.logging_level)

        # Check if text database has changed
        if text_db is not None:
            self.text_db = text_db
        elif self.text_db == None:
            logging.error("No text database provided!")
            raise ConnectionError("No text database provided!")

        # Setup Bert and tokenizer
        self.tokenizer, self.bert = self.setup_bert(fname)

        # Setup Neo4j network
        # Setup network
        ids, tokens=get_full_vocabulary(self.tokenizer)

        tokens = [x.translate(x.maketrans({"\"": '#e1#', "'": '#e2#', "\\": '#e3#'})) for x in tokens]
        self.neograph.db.setup_neo_db(tokens, ids)

        # %% Initialize text dataset
        dataset = query_dataset(self.text_db, self.tokenizer, self.MAX_SEQ_LENGTH, maxn=self.maxn, query=query,
                                logging_level=self.logging_level)
        logging.info("Number of sentences found: %i" % dataset.nitems)
        logging.info("Number of unique tokens in dataset: {}".format(self.tokenizer.vocab_size-len(dataset.id_mask)))
        # Error fix: Batch size must not be larger than dataset size
        original_batch_size = self.batch_size
        if dataset.nitems < self.batch_size:
            logging.warning("Number of sentences less than batch size. Reducing batch size!")
            self.batch_size = dataset.nitems

        batch_sampler = BatchSampler(SequentialSampler(range(0, dataset.nitems)), batch_size=self.batch_size,
                                     drop_last=False)
        dataloader = DataLoaderX(dataset=dataset, batch_size=None, sampler=batch_sampler, num_workers=self.nr_workers,
                                 collate_fn=text_dataset_collate_batchsample, pin_memory=False)

        # Push BERT to GPU
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): logging.info("Using CUDA.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert.to(device)
        self.bert.eval()

        # Create Stopwords
        delwords = create_stopword_list(self.tokenizer)

        # Counter for timing
        #model_timings = []
        #process_timings = []
        #load_timings = []
        #start_time = time.time()
        for batch, token_ids, index_vec, seq_id_vec, runindex_vec, year_vec, p1_vec, p2_vec, p3_vec, p4_vec in tqdm.tqdm(dataloader,
                                                                                                        desc="Iteration"):
            # batch, seq_ids, token_ids
            # Data spent on loading batch
            #load_time = time.time() - start_time
            #load_timings.append(load_time)
            # This seems to allow slightly higher batch sizes on my GPU
            # torch.cuda.empty_cache()
            # Run BERT and get predictions
            predictions, attn = self.get_bert_tensor(0, self.bert, batch, self.tokenizer.pad_token_id,
                                                     self.tokenizer.mask_token_id, device)

            # compute model timings time
            #prepare_time = time.time() - start_time - load_time
            #model_timings.append(prepare_time)

            # %% Sequence Table
            # Iterate over sequences
            for run_index in np.unique(runindex_vec):
                # Extract only current sequence
                sequence_mask = runindex_vec == run_index
                sequence_size = sum(sequence_mask)

                # Get parameters
                seq_year = year_vec[sequence_mask]
                seq_ids = seq_id_vec[sequence_mask]
                seq_p1 = p1_vec[sequence_mask]
                seq_p2 = p2_vec[sequence_mask]
                seq_p3 = p3_vec[sequence_mask]
                seq_p4 = p4_vec[sequence_mask]

                # Pad and add sequence of IDs
                # TODO: Sanity check I don't need this anymore
                # idx = torch.zeros([1, MAX_SEQ_LENGTH], requires_grad=False, dtype=torch.int32)
                # idx[0, :sequence_size] = token_ids[sequence_mask]
                # Pad and add distributions per token, we need to save to maximum sequence size
                dists = torch.zeros([self.MAX_SEQ_LENGTH, self.DICT_SIZE], requires_grad=False)

                dists[:sequence_size, :] = predictions[sequence_mask, :]

                # Context element selection matrix
                seq_ce = torch.ones([sequence_size, sequence_size])
                # Delete position of focal token
                seq_ce[torch.eye(sequence_size).bool()] = 0
                # Context distributions
                context_index = np.zeros([self.MAX_SEQ_LENGTH], dtype=np.bool)
                # Index words that are in the sentence
                context_index[:sequence_size] = True
                # Populate distribution with distributions
                # context_dist = dists[context_index, :]

                # TODO: This should probably not be a loop
                # but done smartly with matrices and so forth
                for pos, token in enumerate(token_ids[sequence_mask]):
                    # Should all be np
                    token = token.item()
                    # Extract parameters for token
                    year = seq_year[pos]
                    sequence_id = seq_ids[pos]
                    p1 = seq_p1[pos]
                    p2 = seq_p2[pos]
                    p3 = seq_p3[pos]
                    p4 = seq_p4[pos]

                    # Ignore stopwords for network creation
                    if token not in delwords:

                        # %% Replacement distribution
                        replacement = dists[pos, :]
                        # Flatten, since it is one row each
                        replacement = replacement.numpy().flatten()
                        # Sparsify
                        # TODO: try without setting own-link to zero!
                        replacement[token] = 0
                        replacement[replacement == np.min(replacement)] = 0
                        # Get rid of delnorm links
                        replacement[delwords] = 0
                        # Get rid of tokens not in text
                        if self.prune_missing_tokens == True:
                            replacement[dataset.id_mask] = 0
                        # We norm the distributions here
                        replacement = self.norm(replacement)

                        # %% Context Element
                        context = (
                            torch.sum((seq_ce[pos] * dists[context_index, :].transpose(-1, 0)).transpose(-1, 0),
                                      dim=0).unsqueeze(0))
                        # Flatten, since it is one row each
                        context = context.numpy().flatten()
                        # Sparsify
                        # TODO: try without setting own-link to zero!
                        context[token] = 0
                        context[context == np.min(context)] = 0
                        # Get rid of delnorm links
                        context[delwords] = 0
                        # Get rid of tokens not in text
                        if self.prune_missing_tokens==True:
                            context[dataset.id_mask] = 0
                        # We norm the distributions here
                        context = self.norm(context)

                        # Add values to network
                        # Replacement ties
                        cutoff_number, cutoff_probability = self.calculate_cutoffs(replacement, method="percent",
                                                                                   percent=self.cutoff_percent,
                                                                                   max_degree=self.max_degree)
                        ties = self.get_weighted_edgelist(token, replacement, year, cutoff_number, cutoff_probability,
                                                          sequence_id, pos, p1=p1, p2=p2, p3=p3, p4=p4, run_index=run_index,
                                                          max_degree=self.max_degree)

                        # Context ties
                        cutoff_number, cutoff_probability = self.calculate_cutoffs(context, method="percent",
                                                                                   percent=self.context_cutoff_percent,
                                                                                   max_degree=self.context_max_degree)
                        context_ties = self.get_weighted_edgelist(token, context, year, cutoff_number,
                                                                  cutoff_probability,
                                                                  sequence_id, pos, p1=p1, p2=p2, p3=p3, p4=p4,
                                                                  max_degree=self.context_max_degree,run_index=run_index,
                                                                  simplify_context=True)

                        if (ties is not None) and (context_ties is not None):
                            self.neograph.insert_edges_context(token, ties, context_ties)

                del dists

            del predictions, attn
            # compute processing time
            #process_timings.append(time.time() - start_time - prepare_time - load_time)
            self.neograph.db.write_queue()
            # New start time

            #start_time = time.time()

        # Write remaining
        self.neograph.db.write_queue()
        torch.cuda.empty_cache()

        # Reset batch size
        if self.batch_size != original_batch_size:
            logging.info("Resetting original batch size")
            self.batch_size = original_batch_size

        del dataloader, dataset, batch_sampler
        #logging.debug("Average Load Time: %s seconds" % (np.mean(load_timings)))
        #logging.debug("Average Model Time: %s seconds" % (np.mean(model_timings)))
        #logging.debug("Average Processing Time: %s seconds" % (np.mean(process_timings)))
        #logging.debug(
        #    "Ratio Load/Operations: %s seconds" % (np.mean(load_timings) / np.mean(process_timings + model_timings)))

    def get_bert_tensor(self, args, bert, tokens, pad_token_id, mask_token_id, device=torch.device("cpu")):
        """
        Extracts tensors of probability distributions for each word in sentence from BERT.
        This is done by running BERT separately for each token, masking the focal token.

        :param args: future: config
        :param bert: BERT model
        :param tokens: tensor of sequences
        :param pad_token_id: Token id's from tokenizer
        :param mask_token_id: Token id's from tokenizer
        :param device: CPU or CUDA device
        :param return_max: only returns ID of most likely token
        :return: predictions: Tensor of logits for each token (dimension: sum(k_i)*vocab-length); attn: Attention weights for each token
        """

        # We use lists of tensors first
        list_tokens = []
        list_segments = []
        list_labels = []
        list_eye = []

        max_seq_length = tokens.shape[1]
        # Create a batch for each token in sentence
        for idx, text in enumerate(tokens):
            # Check how many non-zero tokens are in this text (including special tokens)
            seq_length = sum((text != pad_token_id).int()).item()
            # We repeat the input text for each actual word in sentence
            # -2 because we do not run BERT for <CLS> and <SEP>
            inputs = text.repeat(seq_length - 2, 1)

            # We do the same for the segments, which are just zero
            segments_tensor = torch.zeros(seq_length - 2, max_seq_length, dtype=torch.int64)

            # Create the basis version of our labels
            labels_tensor = inputs.clone()

            # Create Masking matrix
            # First, how many tokens to mask?
            nr_mask_tokens = seq_length - 2
            # Create square eye matrix of max_sequence length
            # But cut both the first token, as well as passing tokens as rows
            eye_matrix = torch.eye(max_seq_length, dtype=torch.bool)[1:seq_length - 1, :]

            # We Mask diagonal tokens
            inputs[eye_matrix] = mask_token_id
            # Set all other labels to -1
            labels_tensor[~eye_matrix] = -1

            # Append lists
            list_tokens.append(inputs)
            list_segments.append(segments_tensor)
            list_labels.append(labels_tensor)
            list_eye.append(eye_matrix.int())

        tokens = torch.tensor([], requires_grad=False)
        segments = torch.tensor([], requires_grad=False)
        labels = torch.tensor([], requires_grad=False)
        eyes = torch.tensor([], requires_grad=False)

        # Send to GPU
        tokens = torch.cat(list_tokens).to(device)
        segments = torch.cat(list_segments).to(device)
        labels = torch.cat(list_labels).to(device)
        eyes = torch.cat(list_eye).to(device)

        # Save some memory insallah
        del list_tokens
        del list_labels
        del list_segments
        del list_eye

        # Get predictions
        bert.eval()
        with torch.no_grad():
            loss, predictions, attn = bert(tokens, masked_lm_labels=labels, token_type_ids=segments)

        del tokens
        del labels
        del segments

        # Only return predictions of masked words (gives one per word for each sentence)
        predictions = predictions[eyes.bool(), :]

        # if return_max==True:
        #    predictions=torch.argmax(predictions, dim=1)

        # Softmax prediction probabilities
        softmax = torch.nn.Softmax(dim=1)
        predictions = softmax(predictions)

        # Operate on attention
        attn = torch.stack(attn)
        # Max over layers and attention heads
        attn, _ = torch.max(attn, dim=0)
        attn, _ = torch.max(attn, dim=1)
        # We are left with a one matrix for each batch
        # Select the attention of the focal tokens only
        attn = attn[eyes.bool(), :]
        # Note attention is now of (nr_tokens,sequence_size+2)
        # because we added <SEP> and <CLS> tokens
        return predictions.cpu(), attn.cpu()

    # %% Utilities
    def calculate_cutoffs(self, x, method="percent", percent=100, max_degree=100, min_cut=0.001):
        """
        Different methods to calculate cutoff probability and number.

        :param x: Contextual/Replacement vector
        :param method: mean: Only accept entries above the mean; percent: Take the k biggest elements that explain X% of mass.
        :return: cutoff_degree and probability
        """
        if method == "mean":
            cutoff_probability = max(np.mean(x), min_cut)
            sortx = np.sort(x)[::-1]
            cutoff_degree = len(np.where(sortx >= cutoff_probability)[0])

        elif method == "percent":
            sortx = np.sort(x)[::-1]
            # Get cumulative sum
            cum_sum = np.cumsum(sortx)
            # Get cutoff value as fraction of largest cumulative element (in case vector does not sum to 1)
            # This is the threshold to cross to explain 'percent' of mass in the vector
            cutoff = cum_sum[-1] * percent / 100
            # Determine first position where cumulative sum crosses cutoff threshold
            # Python indexing - add 1
            cutoff_degree = np.where(cum_sum >= cutoff)[0][0] + 1
            # Calculate corresponding probability
            cutoff_probability = sortx[cutoff_degree - 1]
        else:
            cutoff_probability = min_cut
            cutoff_degree = max_degree

        return min(cutoff_degree, max_degree), cutoff_probability

    def get_weighted_edgelist(self, token, x, time, cutoff_number=100, cutoff_probability=0, seq_id=0, pos=0, p1="0",
                              p2="0", p3="0", p4="0", run_index=0,max_degree=100, simplify_context=False):
        """
        Sort probability distribution to get the most likely neighbor nodes.
        Return a networksx weighted edge list for a given focal token as node.

        :param token: Numerical, token which to add
        :param x: Probability distribution
        :param cutoff_number: Number of neighbor token to consider. Not used if 0.
        :param cutoff_probability: Lowest probability to consider. Not used if 0.
        :return: List of tuples compatible with networkx
        """
        # Get the most pertinent words
        if cutoff_number > 0:
            neighbors = np.argsort(-x)[:cutoff_number]
        else:
            neighbors = np.argsort(-x)[:max_degree]

        # Cutoff probability (zeros)
        # 20.08.2020 Added norm
        if len(neighbors > 0):
            if cutoff_probability > 0:
                neighbors = neighbors[x[neighbors] >= cutoff_probability]
            weights = self.norm(x[neighbors], min_zero=False)
            if simplify_context == False:
                return [(int(token), int(x[0]), int(time),
                         {'weight': float(x[1]), 'seq_id': int(seq_id), 'pos': int(pos), 'run_index': int(run_index), 'p1': str(p1), 'p2': str(p2),
                          'p3': str(p3), 'p4': str(p4)}) for x in list(zip(neighbors, weights))]
            else:
                return [(int(token), int(x[0]), int(time),
                         {'weight': float(x[1])}) for x in list(zip(neighbors, weights))]
        else:
            return None

    def norm(self, x, min_zero=True):
        return simple_norm(x, min_zero=min_zero)
