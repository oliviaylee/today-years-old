import transformers 
from transformers import pipeline, RobertaForMaskedLM, AutoTokenizer
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

path = 'drive/MyDrive/eval/words2.json'
with open(path, 'r') as file:
  to_add = json.load(file)
print(len(to_add))

model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict = True)
tok = AutoTokenizer.from_pretrained("roberta-base")

pre_expansion_embeddings = model.get_input_embeddings().weight
print(len(pre_expansion_embeddings))
tok.add_tokens(to_add)
model.resize_token_embeddings(len(tok))

embeddings = model.get_input_embeddings().weight 
print(len(embeddings))


#baseline approach: average vector sampling from distributions
mu = torch.mean(pre_expansion_embeddings, dim=0)
n = pre_expansion_embeddings.size()[0]
sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5*sigma)

#sampling from this distribution
num_new = 5382
with torch.no_grad():
    new_embeddings = torch.stack(tuple((dist.sample() for _ in range(num_new))), dim=0)
    embeddings[-num_new:,:] = new_embeddings

#novel approach: retrieving trained model weights
model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict = True)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer.encoder = tok_dic
tokenizer.decoder = {v: k for k, v in tok_dic.items()}
model.resize_token_embeddings(len(tokenizer))
state_dict = torch.load('roberta_final_model_ep', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

#Check whether embeddings look accurate
print(embeddings[-5,:])

#load the urban dataset
path = 'drive/MyDrive/eval/entries2.json'
dataset = load_dataset('json', data_files = path)

otal_100 = 0
total_1000 = 0
total_5000 = 0
counter = 0
rank_total = 0

data_len = len(dataset['train'])
for i in tqdm(range(data_len)):
    entry = dataset['train'][i]
    example = entry['example'].lower()
    word = entry['lowercase_word']
    text = example.replace(word, tokenizer.mask_token, 1)
    ## should not be a problem since preprocessing!
    if text == example: 
        print('no change in sentence error')
        continue
     
    element_index = None
    tokens = tokenizer(word)
    ids = tokens['input_ids']
    if (len(ids) != 3):
        print('token length error')
    else:
        element_index = ids[1]
    
    input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
    with torch.no_grad():
        try:
          output = model(**input)
        except Exception as e:
          print(text)
          continue
        logits = output.logits
        
    #softmax = F.softmax(logits, dim = -1)
    try:
        #mask_word = softmax[0, mask_index, :]
        mask_word = logits[0, mask_index, 50265:]
    except Exception as e:
        print(text)
        print(e)
        continue
    sorted_indices = torch.argsort(mask_word[0], descending=True)
    try:
      element_rank = (sorted_indices == (element_index - 50265)).nonzero(as_tuple=True)[0].item() + 1
    except Exception as e:
      print(text)
      print(e)
      continue

    rank_total += element_rank
    if element_rank <= 10:
        total_10 += 1
    if element_rank <= 100:
        total_100 += 1
    if element_rank <= 1000:
        total_1000 += 1
    if element_rank <= 5000:
        total_5000 += 1
    counter += 1

vars = [total_10, total_100, total_1000, total_5000, counter, rank_total]
for var in vars:
  print(var)

#toy code for debugging outputs
model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict = True)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

text = "The author is signing" + tokenizer.mask_token + " tonight."
input = tokenizer.encode_plus(text, return_tensors = "pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
with torch.no_grad():
    output = model(**input)
logits = output.logits
mask_word = logits[0, mask_index, :].squeeze()

words = ['fruit', 'fruits', 'food', 'coffee', 'books']
for word in words:
    tokens = tokenizer(word)
    ids = tokens['input_ids'][1]
    print('Word: ' + word)
    print('Rank: ' + str(mask_word[ids].item()))