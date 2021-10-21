# TODO: Check Comments
import logging
import unicodedata
from os.path import dirname, normpath
import os
import re
import glob
import nltk
import tables
from tqdm import tqdm

from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger


class nw_preprocessor():

    def __init__(self, config=None, database=None, MAX_SEQ_LENGTH=None, char_mult=None, split_symbol=None,
                 number_params=None,
                 logging_level=None, logging_path=None, import_folder=None):
        """
        Pre-process class
        reads text files of the format
        YYYY-p1-p2-....txt, where p1-p4 are arbitrary parameters to be included in the database (<=40 characters each)
        :param database: HDF5 file to use with pyTables
        :param number_params: How many parameters are in the file name? Maximum 4.
        :param MAX_SEQ_LENGTH: Maximal length of sequence in # token
        :param split_symbol: Split string for file name
        :param char_mult: Average token length; determining the maximum size of needed string arrays. String_size=MAX_SEQ_LENGTH*char_mult
        :param logging_level: Logging level to use
        """

        # Fill parameters from configuration file
        if logging_level is not None:
            self.logging_level = logging_level
        else:
            if config is not None:
                self.logging_level = config['General'].getint('logging_level')
            else:
                msg = "Please provide valid logging level."
                logging.error(msg)
                raise AttributeError(msg)
        # Set logging level
        logging.info("Setting loggging level {}".format(self.logging_level))
        logging.disable(self.logging_level)

        if database is not None:
            self.database = database
        else:
            if config is not None:
                self.database = config['Paths']['database']
            else:
                msg = "Please provide valid database path."
                logging.error(msg)
                raise AttributeError(msg)

        if MAX_SEQ_LENGTH is not None:
            self.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
        else:
            if config is not None:
                self.MAX_SEQ_LENGTH = config['Preprocessing'].getint(
                    'max_seq_length')
            else:
                msg = "Please provide valid max sequence length."
                logging.error(msg)
                raise AttributeError(msg)

        if char_mult is not None:
            self.char_mult = char_mult
        else:
            if config is not None:
                self.char_mult = config['Preprocessing'].getint('char_mult')
            else:
                msg = "Please provide valid char mult."
                logging.error(msg)
                raise AttributeError(msg)

        if split_symbol is not None:
            self.split_symbol = split_symbol
        else:
            if config is not None:
                self.split_symbol = config['Preprocessing']['split_symbol']
            else:
                msg = "Please provide valid split symbol."
                logging.error(msg)
                raise AttributeError(msg)

        if number_params is not None:
            self.number_params = number_params
        else:
            if config is not None:
                self.number_params = config['Preprocessing'].getint(
                    'number_params')
            else:
                msg = "Please provide valid number_params."
                logging.error(msg)
                raise AttributeError(msg)

        if logging_path is not None:
            self.logging_path = logging_path
        else:
            if config is not None:
                self.logging_path = config['Paths']['log']
            else:
                msg = "Please provide valid logging_path."
                logging.error(msg)
                raise AttributeError(msg)

        if import_folder is not None:
            self.import_folder = import_folder
        else:
            if config is not None:
                self.import_folder = config['Paths']['import_folder']
            else:
                msg = "Please provide valid logging_path."
                logging.error(msg)
                raise AttributeError(msg)

        # Check and create folders
        self.import_folder = check_create_folder(
            self.import_folder, create_folder=False)
        self.database = check_create_folder(self.database)
        self.logging_path = check_create_folder(self.logging_path)

        # Get nltk stuff
        #nltk.download('punkt')

    def setup_logger(self):
        """
        Sets up logging formats and file etc.
        Returns
        -------
        None
        """
        # Logging path
        setup_logger(self.logging_path, self.logging_level)

    def resplit_sentences(self, sentences, max_tokens):
        """
        Functions takes a list of tokenized sentences and returns a new list, where
        each sentence has a maximum length
        :param sentences: List of strings
        :param max_length: maximum length in words / tokens
        :return: List of strings conforming to maximum length

        Parameters
        ----------
        max_tokens
        """
        split_sentences = []
        # Iterate over sentences and add each, split appropriately, to list
        for sent in sentences:
            split_sentences = self.check_split_sentence(
                sent, split_sentences, max_tokens)

        return split_sentences

    def check_split_sentence(self, sent, text_list, max_tokens):
        """
        Recursively splits tokenizer sentences to conform to max sentence length.
        Adds fake periods in between cut sentences.
        :param sent: Tokenized sentence
        :param text_list: List to append to
        :param max_length: Maximum sentence length
        :return: split tokenized sentence list

        Parameters
        ----------
        max_tokens
        """
        # Split sentence
        sent = sent.split()
        sent = [w.lower() for w in sent]
        # First, add current sentence as string
        current_sent = ' '.join(sent[0:self.MAX_SEQ_LENGTH])
        if current_sent[-1] not in [',', '.', ';', '.', '"', ':']:
            current_sent = ''.join([current_sent, '.'])
        # Add to list
        text_list.append(current_sent)
        # Now check if further split is necessary
        if len(sent) > max_tokens:
            rest_sent = ' '.join(sent[self.MAX_SEQ_LENGTH:])
            # Recurse
            text_list = self.check_split_sentence(
                rest_sent, text_list, max_tokens)

        # Once done, return completed list
        return text_list

    def preprocess_folders(self, folder=None, max_seq=0, overwrite=True, excludelist=[]):
        if folder is None:
            folder = self.import_folder
        folder = normpath(folder)
        folder = check_create_folder(folder,create_folder=False)
        folders = [''.join([folder, '/', name]) for name in os.listdir(folder)]
        if overwrite:
            try:
                data_file = tables.open_file(
                    self.database, mode="w", title="Sequence Data")
            except:
                logging.error("Could not open existing database file.")
                raise
                #raise IOError("Could not open existing database file.")
            finally:
                data_file.close()
        for dir in folders:
            year = int(os.path.split(dir)[-1])
            if year >= 0:
                self.preprocess_files(dir, max_seq, False, year, excludelist)

    def preprocess_files(self, folder=None, max_seq=0, overwrite=True, ext_year=None, excludelist=[]):
        """
        Pre-processes files from raw data into a HD5 Table
        :param folder: folder with text files
        :param max_seq: How many sequences to process maximally from data
        :return: none

        Parameters
        ----------
        overwrite
        ext_year
        excludelist
        """

        # Define particle for pytable
        class sequence(tables.IsDescription):
            run_index = tables.UInt32Col()
            seq_id = tables.UInt32Col()
            text = tables.StringCol(self.MAX_SEQ_LENGTH * self.char_mult)
            year = tables.UInt32Col()
            p1 = tables.StringCol(40)
            p2 = tables.StringCol(40)
            p3 = tables.StringCol(40)
            p4 = tables.StringCol(40)
            source = tables.StringCol(40)
            max_length = tables.UInt32Col()

        if folder is None:
            folder = self.import_folder

        # Initiate pytable database
        if overwrite is not True:
            try:
                data_file = tables.open_file(
                    self.database, mode="a", title="Sequence Data")
            except:
                try:
                    data_file = tables.open_file(
                        self.database, mode="w", title="Sequence Data")
                except:
                    logging.error("Could not open existing database file.")
                    raise IOError("Could not open existing database file.")
        else:
            try:
                data_file = tables.open_file(
                    self.database, mode="w", title="Sequence Data")
            except:
                logging.error("Could not open existing database file.")
                raise IOError("Could not open existing database file.")
        try:
            data_table = data_file.root.textdata.table
            start_index = data_file.root.textdata.table.nrows - 1
            run_index = start_index
        except:
            group = data_file.create_group("/", 'textdata', 'Text Data')
            data_table = data_file.create_table(
                group, 'table', sequence, "Sentence Table")
            start_index = -1
            run_index = start_index

        logging.info("Loading files from %s", folder)
        # Get list of files
        # files = [abspath(f) for f in listdir(folder) if isfile(join(folder, f))]

        #folder = normpath(folder)
        folder = check_create_folder(folder, False)
        files = glob.glob(''.join([folder, '/*.txt']))

        # Main loop over files
        for file_path in tqdm(files, desc="Iterating files in {} ".format(folder), leave=False):
            # Derive file name and year
            file_dirname = dirname(file_path)
            file_name = os.path.split(file_path)[-1]
            logging.debug("Loading file %s", file_name)

            file_name = re.split(".txt", file_name)[0]
            file_source = file_name

            if ext_year is None:
                logging.debug("Loading file %s", file_name)
                year = re.split(self.split_symbol,
                                file_name)[0]
                year = int(year)
                offset = 1
            else:
                offset = 0
                year = ext_year

            # Set up params list
            params = []
            exclude = False
            for i in range(offset, self.number_params + offset):
                par = re.split(self.split_symbol, file_name)[i]
                if par in excludelist:
                    exclude = True
                # If we have reached the last iteration, the rest of the string is the parameter
                if i>=(self.number_params+offset-1):
                    par = ''.join(re.split(self.split_symbol, file_name)[i:])
                params.append(par)

            if exclude:
                text = ""
            else:
                try:
                    with open(file_path) as f:
                        text = f.read()
                except:
                    try:
                        with open(file_path, encoding="utf-8", errors='ignore') as f:
                            text = f.read()
                    except:
                        logging.error("Could not load %s", file_path)
                        raise ImportError(
                            "Could not open file. Make sure, only .txt files in folder!")


            # Get rid of line-breaks
            text = text.replace('\n', ' ')
            text = text.replace(" \'", "' ")
            # Other replacements
            #text = text.replace("n\'t", " not")
            text = text.replace(" .", ". ")
            text = text.replace(" ,", ", ")
            #text = text.replace("-", " ")
            text = text.replace("...", ". ")
            killchars = ['#', '<p>', '$', '%', '(', ')', '*', '/', '<', '>', '@', '\\', '{', '}', '[', ']', '+', '^',
                         '~',
                         '"']
            for k in killchars:
                text = str.replace(text, k, ' ')

            text = text.strip()
            # Replace all whitespaces by a single " "
            text = " ".join(re.split("\\s+", text, flags=re.UNICODE))

            # Strip numeric from beginning and end.
            text = re.sub(r'^\d+|\d+$', '', text)
            # Strip numeric words of length 5+
            text = re.sub(r'\b\d[\d]{5, 100}\b', '', text)

            # We do not load empty lines
            if len(text) == 0:
                logging.info(
                    "Skipping {}, excluded or no text found".format(file_path))
                continue

            if (run_index - start_index >= max_seq) & (max_seq != 0):
                break

            text = nltk.sent_tokenize(text)
            tokenized_text = self.resplit_sentences(text, self.MAX_SEQ_LENGTH)

            for idx, sent in enumerate(tokenized_text):
                # Create table row
                particle = data_table.row
                # Do not add sentences which are too short
                if len(sent) < 2:
                    continue
                # Increment run index if we actually seek to add row
                run_index = run_index + 1
                particle['run_index'] = run_index
                particle['seq_id'] = idx
                particle['source'] = file_source
                particle['year'] = year
                particle['max_length'] = self.MAX_SEQ_LENGTH
                # Add parameters
                for i, p in enumerate(params):
                    pidx = ''.join(['p', str(i + 1)])
                    particle[pidx] = params[i]

                # If the sentence has too many tokens, we cut using nltk tokenizer
                sent = sent.split()
                # Cut max word amounts
                sent = sent[:self.MAX_SEQ_LENGTH]
                # Re-join
                sent = ' '.join(sent)

                # Our array has a maximum size limit
                # if the string is to long, we want to cut only complete words
                while len(sent) > (self.MAX_SEQ_LENGTH * self.char_mult - 1):
                    # Take out one word
                    split = sent.split()[:-1]
                    sent = ' '.join(split)

                # Add punctuation if not present (for BERT)
                if sent[-1] not in [',', '.', ';', '.', '"', ':']:
                    sent = ''.join([sent, '.'])

                # pyTables does not work with unicode
                # 31.08.2021: trying to get unicode to pytables to work without breaking quotation marks
                transl_table = {ord(x): ord(y) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")}
                sent = sent.translate(transl_table)
                sent = unicodedata.normalize('NFKD', sent).encode(
                    'ascii', 'replace').decode('ascii')
                sent = sent.replace("?", " ? ")

                try:
                    particle['text'] = sent
                except:
                    logging.error("Saving failed.")
                    logging.error("Sentence: %s", sent)

                if idx % 100000 == 0:
                    logging.debug(
                        "Sentence at idx {}, year {}, parameters {}:".format(idx, year, params))
                    logging.debug("Sentence {}".format(sent))

                particle.append()

            data_file.flush()

        data_file.close()
        #f.close()
