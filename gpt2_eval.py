from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model
import pandas as pd
import json

def get_train_set(train_json):    
    with open(train_json, "rb") as f:
        train_list = list(f)
        
    rows = []
    for train_json_str in train_list:
        train_line = json.loads(train_json_str)
        # to modify based on json structure
        row = [train_line["id"], train_line["img"], train_line["label"], train_line["text"]]
        rows.append(row)

    # to modify based on json structure
    df = pd.DataFrame(rows, columns=["id", "img", "label", "text"])
    return df


