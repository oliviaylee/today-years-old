import torch
from torch.utils.data import Dataset
import json

class JSonDataset(Dataset):
    def __init__(self, json_path, tokenizer, word_embeddings):
        assert(isinstance(json_path, str) and json_path[:8] == 'datasets') 
        f = open(json_path)
        raw_data = json.load(f)
        f.close()
        self.tokenizer = tokenizer
        self.word_embeds = word_embeddings
        self.samples = []
        flattened_defns, ground_truths = [], []
        for k in raw_data.keys():
            y = self.extract_embed_y(k)
            ground_truths.append(y)
            flat_defn = " ".join([tok for defn in raw_data[k] for tok in defn]) # Flatten def list into list of strings
            flattened_defns.append(flat_defn)
        assert(len(flattened_defns) == len(ground_truths))
        for i in range(len(ground_truths)):
            X, y = tokenizer(flattened_defns[i], padding=True, return_tensors="pt"), ground_truths[i] # GPT-2
            # X, y = tokenizer(flattened_defns[i], padding=True, truncation=True, max_length=512, return_tensors="pt"), ground_truths[i] # Bert
            self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def extract_embed_y(self, word):
        """
        Returns ground truth pretrained embedding for a given word
        """
        # GPT-2
        text_index = self.tokenizer.encode(word, add_prefix_space=True)
        embed_y = self.word_embeds[text_index,:].detach()

        # Bert
        # text_index = self.tokenizer.encode(word, truncation=True, max_length=512)
        # embed_y = self.word_embeds(torch.LongTensor(text_index)).detach()
        if len(text_index) > 1: # Return average of embeddings
            embed_y_avg = torch.stack(embed_y.unbind()).mean(dim=0)
            return embed_y_avg
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