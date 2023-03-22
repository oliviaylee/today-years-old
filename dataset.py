"""
Code Author: Olivia Lee
Loads and preprocesses JSon datasets to generate training data (inputs and labels)
Used in `gpt2_embed_predict.py` and `roberta_embed_predict.py`.
"""

import torch
from torch.utils.data import Dataset
import json

class JSonDataset(Dataset):
    def __init__(self, json_path, model, tokenizer, word_embeddings):
        assert(isinstance(json_path, str) and json_path[:8] == 'datasets')

        self.model = model
        self.tokenizer = tokenizer
        self.word_embeds = word_embeddings
        self.samples = []

        f = open(json_path)
        raw_data = json.load(f)
        f.close()
        input_defns, ground_truths = [], []
        for k in raw_data.keys():
            y = self.extract_embed_y(k)
            ground_truths.append(y)
            flat_defn = " ".join([tok for defn in raw_data[k] for tok in defn]) # Flatten def list into list of strings
            input_defns.append(flat_defn)
        assert(len(input_defns) == len(ground_truths))
        for i in range(len(ground_truths)):
            X, y = None, None
            if self.model == 'gpt2':
                X, y = tokenizer(input_defns[i], padding='max_length', return_tensors="pt"), torch.FloatTensor(ground_truths[i])
            elif self.model == 'roberta':
                X, y = tokenizer(input_defns[i], padding='max_length', truncation=True, max_length=512, return_tensors="pt"), torch.FloatTensor(ground_truths[i])
            self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def extract_embed_y(self, word):
        """
        Returns ground truth pretrained embedding for a given word
        """
        text_index, embed_y = None, None
        if self.model == 'gpt2':
            text_index = self.tokenizer.encode(word, add_prefix_space=True)
            embed_y = self.word_embeds[text_index,:].detach()
        elif self.model == 'roberta':
            text_index = self.tokenizer.encode(word, truncation=True, max_length=512)
            embed_y = self.word_embeds(torch.LongTensor(text_index)).detach()
        if len(text_index) > 1: # Return average of embeddings
            embed_y_avg = torch.stack(embed_y.unbind()).mean(dim=0)
            return embed_y_avg
        return embed_y.squeeze(dim=0)