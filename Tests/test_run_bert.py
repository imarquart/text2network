import time, os
import torch
from NLP.src.run_bert import bert_args, run_bert



cwd = os.getcwd()
do_train=False
block_size=30
gpu_batch=100
epochs=2000
warmup_steps=0
mlm_probability=0.15

train_data_file = ''.join(['NLP/data/w_news_1990.txt'])
train_data_file = os.path.join(cwd, train_data_file)

model_dir = ''.join(['NLP/models'])
model_dir = os.path.join(cwd, model_dir)


train_data_file="D:/NLP/BERT-NLP/NLP/data/w_news_1990.txt"
model_dir="D:/NLP/BERT-NLP/NLP/models"
#model_dir="E:/NLP/bert"

output_dir="E:/NLP/models"


args = bert_args(train_data_file,output_dir,do_train,model_dir,mlm_probability,block_size,gpu_batch, epochs,warmup_steps)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    results=run_bert(args)
    print(results)