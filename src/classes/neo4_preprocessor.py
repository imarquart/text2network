# TODO: Check Comments
import logging
import unicodedata
from os import listdir
from os.path import isfile, join, abspath
import re
import glob
import nltk
import tables


class neo4j_preprocessor():

    def __init__(self, database, MAX_SEQ_LENGTH, char_mult, split_symbol="-", number_params=4, logging_level=logging.NOTSET):
        """
        Pre-process class
        reads text files of the format
        YYYY-p1-p2-....txt, where p1-p4 are arbitrary parameters to be included in the database (<=40 characters each)
        :param folder: folder with text files
        :param database: HDF5 file to use with pyTables
        :param number_params: How many parameters are in the file name? Maximum 4.
        :param MAX_SEQ_LENGTH: Maximal length of sequence in # token
        :param char_mult: Average token length; determining the maximum size of needed string arrays. String_size=MAX_SEQ_LENGTH*char_mult
        """
        # Set logging level
        logging.disable(logging_level)
        # Create network
        self.database=database
        self.MAX_SEQ_LENGTH=MAX_SEQ_LENGTH
        self.char_mult=char_mult
        self.split_symbol = split_symbol
        self.number_params=number_params
        

    def preprocess_files(folder, max_seq=0):
        """
        Pre-processes files from raw data into a HD5 Table
        :param folder: folder with text files
        :param max_seq: How many sequences to process maximally from data
        :return: none
        """

        # TODO: Load Docs as Matrix or parallelize; speed optimization
        # TODO: Check with other datasets
        # TODO: Write more general function for other data (low priority)

        # Use spacy to split sentences (model based)

        # nlp = spacy.load("en_core_web_sm")

        # Define particle for pytable
        class sequence(tables.IsDescription):
            run_index = tables.UInt32Col()
            seq_id = tables.UInt32Col()
            text = tables.StringCol(MAX_SEQ_LENGTH * char_mult)
            year = tables.UInt32Col()
            p1 = tables.StringCol(40)
            p2 = tables.StringCol(40)
            p3 = tables.StringCol(40)
            p4 = tables.StringCol(40)
            source = tables.StringCol(40)
            max_length = tables.UInt32Col()

        # Initiate pytable database
        try:
            data_file = tables.open_file(database, mode="a", title="Sequence Data")
        except:
            data_file = tables.open_file(database, mode="w", title="Sequence Data")

        try:
            data_table = data_file.root.textdata.table
            start_index = data_file.root.textdata.table.nrows - 1
            run_index = start_index
        except:
            group = data_file.create_group("/", 'textdata', 'Text Data')
            data_table = data_file.create_table(group, 'table', sequence, "Sentence Table")
            start_index = -1
            run_index = start_index

        logging.info("Loading files from %s" % folder)
        # Get list of files
        #files = [abspath(f) for f in listdir(folder) if isfile(join(folder, f))]

        files = glob.glob(''.join([folder, '/*.txt']))

        # Main loop over files
        for file_path in files:
            # Derive file name and year
            file_name = re.split("/", file_path)[-1]
            logging.info("Loading file %s" %file_name)

            file_name = re.split(".txt", file_name)[0]
            file_source = file_name

            # Set up params list
            params=[]
            for i in range(1,number_params):
                params.append(re.split("_", file_name)[-i])

            year = re.split("_", file_name)[-(number_params+1)]
            year = int(year[-4:])

            try:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
            except:
                logging.error("Could not load %s" % file_path)
                raise ImportError("Could not open file. Make sure, only .txt files in folder!")

            text = text.replace('\n', '')

            # Some pre processing
            text = text.replace('<p>', '')
            text = text.replace('@', '')
            text = text.replace(" \'", "'")
            text = text.replace("n\'t", "not")
            text = text.replace(" .", ".")
            text = text.replace(" ,", ",")
            text = text.replace("...", ".")
            killchars = ['#', '<p>', '$', '%', '(', ')', '*', '/', '<', '>', '@', '\\', '{', '}', '[', ']', '+', '^', '~',
                        '"']
            for k in killchars: text = str.replace(text, k, '')
            text = text.strip()
            text = " ".join(re.split("\s+", text, flags=re.UNICODE))

            # We do not load empty lines
            if len(text) == 0:
                continue

            if (run_index - start_index >= max_seq) & (max_seq != 0):
                break

            # Use Spacy to iterate over sentences
            # Note: We could probably do without NLTK here and use just split
            for idx, sent in enumerate(nltk.sent_tokenize(text)):
                # Create table row
                particle = data_table.row
                # Do not add sentences which are too short
                if len(sent) < 3:
                    continue
                # Increment run index if we actually seek to add row
                run_index = run_index + 1
                particle['run_index'] = run_index
                particle['seq_id'] = idx
                particle['source'] = file_source
                particle['year'] = year
                particle['max_length'] = MAX_SEQ_LENGTH
                # Add parameters
                for i,p in enumerate(params):
                    idx=''.join(['p',i+1])
                    particle[idx]=params[i]

                # If the sentence has too many tokens, we cut using nltk tokenizer
                sent = sent.split()
                # Cut max word amounts
                sent = sent[:MAX_SEQ_LENGTH]
                # Re-join
                sent = ' '.join(sent)

                # Our array has a maximum size limit
                # if the string is to long, we want to cut only complete words
                while len(sent) > (MAX_SEQ_LENGTH * char_mult - 1):
                    # Take out one word
                    split = sent.split()[:-1]
                    sent = ' '.join(split)

                # Add punctuation if not present (for BERT)
                if sent[-1] not in [',', '.', ';', '.', '"', ':']:
                    sent = ''.join([sent, '.'])

                # pyTables does not work with unicode
                sent=unicodedata.normalize('NFKD', sent).encode('ascii', 'ignore').decode('ascii')

                try:
                    particle['text'] = sent
                except:
                    logging.error("Saving failed.")
                    logging.info("Sentence: %s" % sent)

                particle.append()

            data_file.flush()


        data_file.close()
        f.close()
