# TODO: Check Comments


import re
#import spacy
import nltk
import tables


def pre_process_sentences_COCA(files,database,MAX_SEQ_LENGTH,char_mult,max_seq=0,file_type="old"):
    """
    Pre-processes files (using spacy) from raw data into a HD5 Table

    :param files: list of raw text files from COCA
    :param database: HDF5 file to use with pyTables
    :param MAX_SEQ_LENGTH: Maximal length of sequence in token
    :param char_mult: Multiplier for MAX_SEQ_LENGTH for array size - average token length
    :param max_seq: How many sequences to process maximally from data
    :param file_type: old / new COCA format
    :return: none
    """
    # TODO: Load Docs as Matrix or parallelize; speed optimization
    # TODO: Check with other datasets
    # TODO: Write more general function for other data (low priority)

    # Use spacy to split sentences (model based)

    #nlp = spacy.load("en_core_web_sm")

    # Define particle for pytable
    class sequence(tables.IsDescription):
        run_index = tables.UInt32Col()
        seq_id = tables.UInt32Col()
        text_id = tables.UInt32Col()
        text = tables.StringCol(MAX_SEQ_LENGTH*char_mult)
        year = tables.UInt32Col()
        source = tables.StringCol(40)
        max_length = tables.UInt32Col()


    # Initiate pytable database
    try:
        data_file = tables.open_file(database, mode="a", title="Sequence Data")
    except:
        data_file = tables.open_file(database, mode="w", title="Sequence Data")


    try:
        data_table = data_file.root.textdata.table
        start_index=data_file.root.textdata.table.nrows-1
        run_index=start_index
    except:
        group = data_file.create_group("/", 'textdata', 'Text Data')
        data_table = data_file.create_table(group, 'table', sequence, "Sentence Table")
        start_index = -1
        run_index=start_index



    # Main loop over files
    for file_path in files:

        # Derive file name and year
        if file_type=="old":
            file_name=re.split("/",file_path)[-1]
            file_name = re.split(".txt", file_name)[0]
            file_source = re.split("_", file_name)[1]
            year = int(re.split("_", file_name)[2])
        elif file_type=="new":
            file_name=re.split("/",file_path)[-1]
            file_name = re.split(".txt", file_name)[0]
            file_source = re.split("_", file_name)[1]
            year = int(re.split("_", file_name)[0])
        else:
            AssertionError("File type not correctly specified.")


        f=open(file_path)
        # We read line by line
        for raw in f.readlines():
            # If max is reached, we will stop adding sentences
            if (run_index-start_index >= max_seq) & (max_seq != 0):
                break

            # Get rid of line breaks
            raw=raw.replace('\n','')

            # We do not load empty lines
            if len(raw)==0:
                continue

            # Skip if text id can not be identified
            text_id=re.findall("##\d+\s",raw)
            if len(text_id)==0:
                continue

            # Get text id and text
            text_id = [int(float(i.replace("#", ""))) for i in text_id]
            text = re.split("##\d+\s", raw)[1]

            # Some pre processing
            text=text.replace('<p>','')
            text = text.replace('@', '')
            text = text.replace(" \'", "'")
            text = text.replace("n\'t", "not")
            text = text.replace(" .", ".")
            text = text.replace(" ,", ",")
            text = text.replace("...", ".")
            killchars=['#','<p>','$','%','(',')','*','/','<','>','@','\\','{','}','[',']','+','^','~','"']
            for k in killchars: text=str.replace(text,k,'')
            text = text.strip()
            text = " ".join(re.split("\s+", text, flags=re.UNICODE))

            # Use Spacy to iterate over sentences
            # Note: We could probably do without NLTK here and use just split
            for idx,sent in enumerate(nltk.sent_tokenize(text)):
                # Create table row
                particle=data_table.row
                # Do not add sentences which are too short
                if len(sent) < 3:
                    continue
                # Increment run index if we actually seek to add row
                run_index = run_index + 1
                particle['run_index']=run_index
                particle['text_id']=text_id[0]
                particle['seq_id'] = idx
                particle['source'] = file_source
                particle['year'] = year
                particle['max_length'] = MAX_SEQ_LENGTH


                # If the sentence has too many tokens, we cut using nltk tokenizer
                #sent=nltk.word_tokenize(sent)

                # Actually, let's use python split for now
                sent=sent.split()
                # Cut max word amounts
                sent=sent[:MAX_SEQ_LENGTH]
                # Re-join
                sent=' '.join(sent)

                # Detokenize again, since we will be using BERT to tokenize
                #sent=TreebankWordDetokenizer().detokenize(sent)

                # Our array has a maximum size limit
                # if the string is to long, we want to cut only complete words
                while len(sent)>(MAX_SEQ_LENGTH*char_mult-1):
                    # Take out one word
                    split=sent.split()[:-1]
                    sent=' '.join(split)

                # Add punctuation if not present (for BERT)
                if sent[-1] not in [',','.',';','.','"',':']:
                    sent = ''.join([sent, '.'])

                particle['text'] = sent
                particle.append()

        data_file.flush()
        data_file.close()
        f.close()
