import time, os
from NLP.src.run_bert import bert_args, run_bert



cwd = os.getcwd()
do_train=True
block_size=-1
gpu_batch=4
epochs=1
warmup_steps=0
mlm_probability=0.15

train_data_file = ''.join(['NLP/data/w_news_1991.txt'])
train_data_file = os.path.join(cwd, train_data_file)

model_dir = ''.join(['NLP/models'])
model_dir = os.path.join(cwd, model_dir)

output_dir="E:/NLP/models"


args = bert_args(train_data_file,output_dir,do_train,model_dir,mlm_probability,block_size,gpu_batch, epochs,warmup_steps)

if __name__ == '__main__':
    results=run_bert(args)