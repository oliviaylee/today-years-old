import json
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForMultipleChoice
from dataset import JSonDataset

# https://github.com/huggingface/transformers/issues/1458

# GPT-2
gpt2_pt_model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)  # or any other checkpoint
word_embeddings = gpt2_pt_model.transformer.wte.weight  # Word Token Embeddings
# position_embeddings = gpt2_pt_model.transformer.wpe.weight  # Word Position Embeddings 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_pt_model.resize_token_embeddings(len(tokenizer))

common_data = JSonDataset('dict_wn.json', 'gpt2', tokenizer, word_embeddings)

counter = 0
for line in open('urban_words_reformat.json', "r"):
    entry = json.loads(line)
    word = entry['lowercase_word']
    if word in common_data.all_words:
        counter += 1

print(counter)
