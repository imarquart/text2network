import torch
import numpy as np

# Load models
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
bert = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')


# Enter sentence without punctuation (uncomment)
text_1 = "Barack Obama is the president of the United States."
text_2 = "Angela Merkel is still chancellor of Germany"

texts=[text_1,text_2]
# The punctuated sentence is predicted correctly (uncomment)
#text_1 = "Barack Obama is the president of the United States."

# Tokenize and create zero sequence (one sentence)
indexed_tokens = tokenizer.encode(text_1, add_special_tokens=True)
segments_tensors=torch.zeros(len(indexed_tokens),dtype=torch.int64)

# We will create a batch with one element per input token (<CLS> and <SEP>)
tokens_tensor = torch.tensor([indexed_tokens], requires_grad=False)
tokens_tensor=tokens_tensor.repeat(len(indexed_tokens)-2,1)
segments_tensors=segments_tensors.repeat(len(indexed_tokens)-2,1)
#%%


# Create inputs and labels (to return losses)
inputs=tokens_tensor.clone()
labels = tokens_tensor.clone()
# Size variables
seq_length=tokens_tensor.size()[1]
nr_mask_tokens=tokens_tensor.size()[0]
# Each word will be masked. In element n of the batch, word n+1 is masked!
eye_matrix=torch.eye(seq_length,dtype=torch.bool)[1:-1,:]
inputs[eye_matrix] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
labels[~eye_matrix] = -1

#print(tokens_tensor)
#print(inputs)
#print(labels)

#%%

# Feed complete batch into the model
with torch.no_grad():
    loss,predictions = masked_lm__model(inputs,masked_lm_labels=labels , token_type_ids=segments_tensors)

# Apply softmax to graph the distributions
softmax=torch.nn.Softmax(dim=2)
predictions=softmax(predictions)
#%%

# Use the masking matrix again to extract distributions of masked words only
masked_predictions=predictions[eye_matrix]
# Find which words were predicted
pred_tokens=torch.argmax(masked_predictions, dim=1)
# Print the sentence using only the masked words
print("Sentence as predicted by successive masking: %s" % tokenizer.convert_ids_to_tokens(pred_tokens.numpy()))


#%%

# Further analysis - what is the distribution for the last token
token_nr=seq_length-2 # Last token excluding <CLS> and <SEP>
batch_nr=token_nr-1 # Batch Nr is one less since <CLS> is not fed as masked
# Get predictions for batch
batchpred=torch.argmax(predictions[batch_nr], dim=1)

# Compare predicted and actual token
predicted_token = tokenizer.convert_ids_to_tokens(batchpred[token_nr].item())[0]
real_token=tokenizer.convert_ids_to_tokens([indexed_tokens[token_nr]])[0]

print("Predicted last token is: %s" % predicted_token)
print("Real last token is: %s" % real_token)

# Get distribution of this token according to BERT
masked_dist=masked_predictions[batch_nr]
# Get the top words
spred=np.argsort(masked_dist)[-10:]
spred=spred[np.argsort(-masked_dist[spred])]
print("Ten most likely words are: %s" % tokenizer.convert_ids_to_tokens(spred.numpy()))

#%%

# Print distribution of last token
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
plt.plot(np.transpose(masked_dist))
plt.show()