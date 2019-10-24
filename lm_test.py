import torch
import numpy as np

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
#tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

text_1 = "Donald Trump is the president of the United States"
#text_1 = "A good leader is strong"
# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, add_special_tokens=True)
segments_tensors=torch.zeros(len(indexed_tokens),dtype=torch.int64)
tokens_tensor = torch.tensor([indexed_tokens])
#%%
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
#segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
#segments_tensors = torch.tensor([segments_ids])

# Convert inputs to PyTorch tensors


model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

with torch.no_grad():
    encoded_layers, param2 = model(tokens_tensor, token_type_ids=segments_tensors)

#%%
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 5
mindexed_tokens=indexed_tokens.copy()
mindexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([mindexed_tokens])

masked_lm__model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')

with torch.no_grad():
    predictions = masked_lm__model(tokens_tensor, token_type_ids=segments_tensors)

# Get the predicted token
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
asdf=torch.argmax(predictions[0][0], dim=1)
print(tokenizer.convert_ids_to_tokens(asdf.numpy()))
print(tokenizer.convert_ids_to_tokens(tokens_tensor.numpy()[0]))
print(tokenizer.convert_ids_to_tokens(indexed_tokens))
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
real_token=tokenizer.convert_ids_to_tokens([indexed_tokens[masked_index]])[0]
assert predicted_token == real_token

#%%
limit_token=100
pr=predictions[0][0]
softmax=torch.nn.Softmax(dim=1)
pr=softmax(pr)
pr=pr.numpy()
#part=np.argpartition(-pr,10,axis=1)[:,:10]
#part=part[np.argsort(pr[part])]
# #pr1=pr[part]
part2=np.argsort(pr,axis=1)[:,-limit_token:][:,::-1]
#pr2=pr[part2]

tokens=np.apply_along_axis(tokenizer.convert_ids_to_tokens, 0, part2)


#%%
masked_index = 5
ff=predictions[0][0][masked_index]
spred=np.argpartition((-ff.numpy()),160)[:160]
spred2=np.argsort(ff.numpy())[-35:]
spred=spred[np.argsort(-ff[spred])]
spred2=spred2[np.argsort(-ff[spred2])]

print(tokenizer.convert_ids_to_tokens(spred))
print(tokenizer.convert_ids_to_tokens(spred2))
print(tokenizer.convert_ids_to_tokens([indexed_tokens[masked_index]])[0])

softmax=torch.nn.Softmax(dim=0)
ff2=softmax(ff).numpy()
spred=np.argpartition((-ff2),25)[:25]
spred=spred[np.argsort(-ff[spred])]
print(tokenizer.convert_ids_to_tokens(spred))
print(ff2[spred]*100)
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt


plt.plot(ff2)
plt.show()


#%%
batchnr=8
# Get the predicted token
predicted_index = torch.argmax(predictions[batchnr], dim=1)[batchnr+1].item()
asdf=torch.argmax(predictions[batchnr], dim=1)

predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
real_token=tokenizer.convert_ids_to_tokens([indexed_tokens[batchnr+1]])[0]

ff=pr[batchnr][batchnr+1]
softmax=torch.nn.Softmax(dim=0)
ff2=softmax(ff).numpy()
#spred=np.argpartition((-ff.numpy()),160)[:160]
spred2=np.argsort(ff2)[-35:]
#spred=spred[np.argsort(-ff[spred])]
spred2=spred2[np.argsort(-ff2[spred2])]

print(predicted_token)
print(real_token)
print(tokenizer.convert_ids_to_tokens(spred2))
print(tokenizer.convert_ids_to_tokens(asdf.numpy()))
print(tokenizer.convert_ids_to_tokens(inputs.numpy()[batchnr]))
print(tokenizer.convert_ids_to_tokens(indexed_tokens))
#%%

#%%
spred=np.argpartition((-asdf),25,axis=1)

#%%
spred=spred[np.argsort(asdf[spred])]


print(tokenizer.convert_ids_to_tokens(spred[:25]))