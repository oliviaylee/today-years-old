import transformers 
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel
import torch
import datasets
import json
import tqdm
from tqdm import tqdm

#Code Author: Jason Chen
#Average Embedding for Baseline Approach credit to John Hewitt
#Source: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

#collab mounting
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

pre_expansion_embeddings = model.get_input_embeddings().weight
print(len(pre_expansion_embeddings))
tokenizer.add_tokens(words_to_add)
model.resize_token_embeddings(len(tokenizer))

embeddings = model.get_input_embeddings().weight #params['embeddings.word_embeddings.weight']
print(len(embeddings))

#code to sample embeddings adapted from hewitt's blogsa 
mu = torch.mean(pre_expansion_embeddings, dim=0)
n = pre_expansion_embeddings.size()[0]
sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5*sigma)

#sample from distribution
num_words = 961
with torch.no_grad():
    new_embeddings = torch.stack(tuple((dist.sample() for _ in range(num_words))), dim=0)
    embeddings[-num_words:,:] = new_embeddings

#novel approach: retrieving trained model weights
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name, return_dict = True)
tokenizer.add_tokens(['[PAD]'])
model.resize_token_embeddings(len(tokenizer))
state_dict = torch.load('drive/MyDrive/eval/gpt2_final_model_ep3_commononly', map_location=torch.device('cuda'))
model.load_state_dict(state_dict)

from torch.nn import functional as F
import torch
import random

total_1 = 0
total_5 = 0
total_10 = 0
total_25 = 0
counter = 0
rank_total = 0
vocab = tokenizer.vocab
data_len = len(dataset['train'])
for i in tqdm(range(data_len)):
    entry = dataset['train'][i]
    example = entry['example'].lower()
    word = entry['lowercase_word']
     
    element_index = None
    if word in vocab: 
      element_index = vocab[word]
    else:
      print('not in vocab')
      continue
    
    input_tokens = tokenizer.encode(example, return_tensors="pt")
    try:
      mask_index = torch.nonzero(input_tokens[0] == element_index, as_tuple=True)[0][0]
    except Exception as e:
      # print(word)
      # print(example)
      print(e)
      continue
    #mask_index = torch.where(input_tokens["input_ids"][0] == tokenizer.mask_token_id)[1].item()
    
    sample_size = 50
    vocab_size = tokenizer.vocab_size
    sentences = [input_tokens[0].clone() for _ in range(sample_size)]
    
    rand_inds = random.sample(range(tok_len), 49)
    for i in range(len(sentences) - 1):
        sentence = sentences[i]
        sentence[mask_index] = rand_inds[i]
    sentences[len(sentences) - 1][mask_index] = element_index
    rand_inds.append(element_index)
    stack = torch.stack(sentences)
    
    with torch.no_grad():
      try:
        outputs = model(stack, labels=stack)
      except Exception as e:
        print(e)
    logits = outputs.logits
    try:
        mask_words = logits[:, mask_index, :]
    except Exception as e:
        print(e)
    probs = mask_words.sum(dim=-1).squeeze()
    sorted_indices = torch.argsort(probs)
    element_rank = torch.nonzero(sorted_indices == 49, as_tuple=True)[0] + 1
    #element_rank = sorted_indices.index(99) + 1
    
    rank_total += element_rank
    if element_rank < 1:
        total_1 += 1
    if element_rank < 5:
        total_5 += 1
    if element_rank < 10:
        total_10 += 1
    if element_rank < 25:
      total_25 += 1
    counter += 1

vars = [total_1, total_5, total_10, total_25, rank_total, counter]
for var in vars:
  print(var)

#toy code for debugging outputs 
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

sentences = [
    "A grocery store sells fruit",
    "A grocery store sells grocery", 
    "A grocery store sells boot", 
    "A grocery store sells run"
]
tokened = []
for sentence in sentences:
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    tokened.append(tokens[0])
stacked = torch.stack(tokened)
print(stacked.size())

with torch.no_grad():
    outputs = model(stacked, labels=stacked)
logits = outputs.logits
print(logits.size())
try:
    mask_words = logits[:, 4, :]
except Exception as e:
    print(e)
probs = mask_words.sum(dim=-1).squeeze()
print(probs.size())
print(probs)
# sorted_indices = torch.argsort(probs, descending=True)
# print(sorted_indices)
# element_rank = torch.nonzero(sorted_indices == 99, as_tuple=True)[0] + 1