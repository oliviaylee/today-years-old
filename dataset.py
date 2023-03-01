import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2LMHeadModel
import json

gpt2_pt_model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
word_embeddings = gpt2_pt_model.transformer.wte.weight  # Word Token Embeddings 
# position_embeddings = model.transformer.wpe.weight  # Word Position Embeddings 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

class JSonDataset(Dataset):
    def __init__(self, json_path):
        assert(isinstance(json_path, str) and json_path[:8] == 'datasets') 
        f = open(json_path)
        raw_data = json.load(f)
        f.close()

        self.samples = []
        for k in raw_data.keys():
            y = self.extract_embed_y(k)
            if isinstance(y, bool): continue # Remove words that are split into multiple tokens from training set
            X = raw_data[k] # TO-DO: CONCATENATE DEF LISTS?
            self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def extract_embed_y(self, word):
        """
        Returns ground truth pretrained embedding for a given word
        """
        text_index = tokenizer.encode(word,add_prefix_space=True)
        if len(text_index) > 1: # Remove words that are split into multiple tokens from training set
            # TO-DO: Alternative: return average of embeddings
            return False
        embed_y = word_embeddings[text_index,:]
        return embed_y


# def preprocess_json(data):
#     assert(data == 'common' or data == 'urban' or data == 'both')
#     common_f = open('datasets/dict_wn.json')
#     common_dict = json.load(common_f) #TO-DO: CONCATENATE DEF LISTS?
#     common_f.close()
#     common_data = JSonDataset('datasets/dict_wn.json')
#     urban_data = None
#     if data == 'urban':
#         # TO-DO: Unzip and preprocess Urban Dictionary
#         pass
#     return common_data, urban_data