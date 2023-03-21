import json
from tqdm import tqdm
from functools import lru_cache

words_to_add = []
seen_urban = set()
evals = []

#processing datasets based on upvotes/ phrases, common words
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
vocab = tokenizer.vocab
print(len(tokenizer))
data_len = len(dataset['train'])
print(data_len)
for i in tqdm(range(data_len)): 
    entry = dataset['train'][i]

    word, defn = entry['lowercase_word'], entry['definition'].lower()
    upv, downv = int(entry["thumbs_up"]), int(entry["thumbs_down"])
    example = entry['example']

    if (upv < 1000) or (downv > upv): 
        continue
    if word.lower() not in example.lower():
        continue
    if word in vocab: #or word in common:
        continue
    evals.append(entry)
    words_to_add.append(word)

with open('entries2.json', 'w') as outfile:
    json.dump(evals, outfile)


#processing common word dataset
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tok_dic = {}
with open('drive/MyDrive/eval/gpt2_tokenizer_vocab.json', 'r') as file:
    tok_dic = json.load(file)

to_add = []
for key, val in tok_dic.items():
    if val < 50257: 
        continue
    to_add.append(key)
tokenizer.add_tokens(to_add)
print(len(tokenizer))

#common word intersection between dictionaries
result = {}
for k, v in bert_vocab.items():
    if k not in gpt_vocab:
        result[k] = v

for k, v in gpt_vocab.items():
    if k not in bert_vocab:
        result[k] = v
print(len(result))