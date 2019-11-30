
import os
import time
from NLP.Experiments.preprocess_files import pre_process_sentences_COCA
#os.chdir('/home/ingo/PhD/BERT-NLP/BERTNLP')
start_time = time.time()
cwd= os.getcwd()
files=[os.path.join(cwd,'NLP/data/w_news_1990.txt')]
database=os.path.join(cwd,'NLP/data/texts.h5')
MAX_SEQ_LENGTH=20
char_mult=10
max_seq=10000

pre_process_sentences_COCA(files,database,MAX_SEQ_LENGTH,char_mult,max_seq=max_seq,file_type="old")

print("--- %s seconds ---" % (time.time() - start_time))