import numpy as np
import tables
import nltk
import os, sys, re
import spacy

files=['/home/ingo/PhD/BERT-NLP/data/w_news_1990.txt']
database='/home/ingo/PhD/BERT-NLP/data/texts.h5'
MAX_SEQ_LENGTH=50



def process_sentences_COCA(files,database,MAX_SEQ_LENGTH,char_mult,file_type="old"):
    """
    Pre-processes files (using spacy) from raw data into a HD5 Table

    Parameters
        files : list of raw text files

        database : HDF5 file to use with pyTables

        MAX_SEQ_LENGTH: Maximal length of sequence in token

        char_mult: Multiplier for MAX_SEQ_LENGTH for array size

        file_type : "old" / "new" file format

    Returns

    """

    nlp = spacy.load("en_core_web_sm")

    class sequence(tables.IsDescription):
        seq_id = tables.UInt32Col()
        text_id = tables.UInt32Col()
        text = tables.StringCol(MAX_SEQ_LENGTH*char_mult)
        year = tables.UInt32Col()
        source = tables.StringCol(40)
        max_length = tables.UInt32Col()

    try:
        data_file = tables.open_file(database, mode="a", title="Sequence Data")
    except:
        data_file = tables.open_file(database, mode="w", title="Sequence Data")

    try:
        data_table = data_file.root.textdata
    except:
        group = data_file.create_group("/", 'textdata', 'Text Data')
        data_table = data_file.create_table(group, 'table', sequence, "Sentence Table")


    for file_path in files:
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
        for raw in f.readlines():
            raw=raw.replace('\n','')

            if len(raw)==0:
                continue

            text_id=re.findall("##\d+\s",raw)
            if len(text_id)==0:
                continue


            text_id = [int(float(i.replace("#", ""))) for i in text_id]
            text = re.split("##\d+\s", raw)[1]

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
            text=nlp(text)


            for idx,sent in enumerate(text.sents):
                particle=data_table.row
                particle['text_id']=text_id[0]
                particle['seq_id'] = idx
                particle['source'] = file_source
                particle['year'] = year
                particle['max_length'] = MAX_SEQ_LENGTH

                # Do not add sentences which are too short
                if len(sent) < 3:
                    continue

                # If the sentence has too many tokens, we cut using Spacy (on tokens)
                sent=sent[:MAX_SEQ_LENGTH].string
                # Delete white space
                sent=sent.strip()

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


process_sentences_COCA(files,database,MAX_SEQ_LENGTH,10,file_type="old")