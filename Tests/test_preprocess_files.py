
import os
import time
from NLP.Experiments.preprocess_files import pre_process_sentences_COCA


start_time = time.time()
cwd= os.getcwd()
files=[os.path.join(cwd,'data/w_news_1990.txt')]
database=os.path.join(cwd,'data/texts.h5')
MAX_SEQ_LENGTH=50
char_mult=10
max_seq=100

pre_process_sentences_COCA(files,database,MAX_SEQ_LENGTH,10,max_seq=200,file_type="old")

print("--- %s seconds ---" % (time.time() - start_time))