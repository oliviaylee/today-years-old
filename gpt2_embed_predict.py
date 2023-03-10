"""
Method 2: Train a separate neural network to predict the word embedding given the 
definition, using a dictionary of common words as the input and the word embeddings 
already in the model as the output.
"""

import json
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import JSonDataset

# https://github.com/huggingface/transformers/issues/1458

# GPT-2
gpt2_pt_model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)  # or any other checkpoint
word_embeddings = gpt2_pt_model.transformer.wte.weight  # Word Token Embeddings
# position_embeddings = gpt2_pt_model.transformer.wpe.weight  # Word Position Embeddings 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_pt_model.resize_token_embeddings(len(tokenizer))
trained_model_path = None

def split_data(dataset):
    train_size, val_size = int(0.9 * len(dataset)), int(0.1 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dl, val_dl = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    return train_dl, val_dl

def train(timestamp, tb_writer, lr=0.00003, eps=3, batch_size=32):
    common_data = JSonDataset('datasets/dict_wn.json', 'gpt2', tokenizer, word_embeddings)
    train_dl, val_dl = split_data(common_data)
    model = gpt2_pt_model
    loss_fn = torch.nn.MSELoss() #torch.nn.CosineEmbeddingLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(eps):
        print('EPOCH {}:'.format(ep + 1))
        # One pass through data
        model.train(True)
        running_loss, avg_loss = 0.0, 0.0
        for i, data in enumerate(train_dl):
            input, label = data # input = tokenized+padded defn, label = ground truth pretrained embedding
            output = model(**input) # odict_keys(['logits', 'past_key_values', 'hidden_states'])
            # output['hidden states'] is a Tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
            last_hidden_state = (output['hidden_states'][-1].squeeze())[0].unsqueeze(dim=0)
            # Sometimes last hidden state is [1]. Sometimes label.size() is [1, 1, 768]. Not sure why
            if (last_hidden_state.size() == torch.Size([1])): 
                continue
            elif (label.size() == torch.Size([1, 1, 768])):
                label = label.squeeze(dim=0)
            assert(last_hidden_state.size() == label.size()) # torch.Size([1, 768])
            loss = loss_fn(last_hidden_state, label)
            loss.backward()
            if (i + 1) % batch_size == 0: # Sub-batching
                optimizer.step()
                optimizer.zero_grad()
            # Logging
            running_loss += loss.item()
            if i % 100 == 99:
                avg_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, avg_loss))
                tb_x = ep * len(train_dl) + i + 1
                tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
                running_loss = 0.
        
        # One set of eval
        model.train(False)
        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(val_dl):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        ep + 1)
        tb_writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, ep)
            trained_model_path = model_path
            torch.save(model.state_dict(), model_path)

def learn_urban():
    model = gpt2_pt_model
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    with torch.no_grad():
        for line in open('datasets/urban_words.json', "r"):
            entry = json.loads(line)
            word = entry['lowercase_word']
            if len(word.split(' ')) > 1: # Phrase
                pass # TO-DO: Handling phrases
            defn = entry['definition'].lower()
            # input is tokenized + padded defn
            input = tokenizer(defn, padding='max_length', return_tensors="pt")
            output = model(**input) # output is predicted word embedding
            tokenizer.add_tokens(word)
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight[:,-1] = output
    return model.transformer.wte.weight

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    # PHASE 1: Train model on dict of common words to learn r/s between defns and embeddings 
    train(timestamp, writer)

    # PHASE 2: Add add new word embeddings to GPT2 given the new definitions
    expanded_embedding_matrix = learn_urban()
    return expanded_embedding_matrix

if __name__ == '__main__':
    main()