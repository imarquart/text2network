import torch
import tables
from Experiments.text_dataset import text_dataset
from Experiments.text_dataset import text_dataset_collate
from torch.utils.data import Dataset, DataLoader

database='/home/ingo/PhD/BERT-NLP/data/texts.h5'
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer','bert-base-uncased')  # Download vocabulary from S3 and cache.



dataset=text_dataset('/home/ingo/PhD/BERT-NLP/data/texts.h5',tokenizer,6)

#import time
#start_time = time.time()
#batch_size=10000

#for i in range(0, dataset.nitems-batch_size, batch_size):
#        batch = dataset[i,i+batch_size]
#        print(i)
#        print(batch[1])
#       print(batch[2])


dataloader = DataLoader(dataset, batch_size=2000, shuffle=False, num_workers=16,pin_memory=False,collate_fn=text_dataset_collate)
for i, batch in enumerate(dataloader):
        print(i)
        print(batch[1])
        print(batch[2])


dataset.close()
#print("--- %s seconds ---" % (time.time() - start_time))