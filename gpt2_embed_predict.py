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

def split_data(dataset):
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # val_size, test_size = int(0.1 * len(dataset)), len(dataset) - train_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dl, val_dl = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    return train_dl, val_dl

def train(device, timestamp, tb_writer, lr=0.00003, eps=1, batch_size=16):
    common_data = JSonDataset('datasets/dict_wn.json', 'gpt2', tokenizer, word_embeddings)
    train_dl, val_dl = split_data(common_data)
    model = gpt2_pt_model
    model.to(device)
    loss_fn = torch.nn.MSELoss() #torch.nn.CosineEmbeddingLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_vloss, trained_model_path = float('inf'), None
    for ep in range(eps):
        print('EPOCH {}:'.format(ep + 1))
        # One pass through data
        model.train(True)
        running_loss, avg_loss = 0.0, 0.0
        for i, data in enumerate(train_dl):
            input, label = data # input = tokenized+padded defn, label = ground truth pretrained embedding
            input['input_ids'] = input['input_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            label = label.to(device)
            output = model(input_ids=input['input_ids'], attention_mask=input['attention_mask']) # odict_keys(['logits', 'past_key_values', 'hidden_states'])
            # output['hidden states'] is a Tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
            last_hidden_state = (output['hidden_states'][-1].squeeze())[0].unsqueeze(dim=0)
            # Sometimes last hidden state is [1]. Sometimes label.size() is [1, 1, 768]. Not sure why
            if (last_hidden_state.size() == torch.Size([1])): 
                continue
            elif (label.size() == torch.Size([1, 1, 768])):
                label = label.squeeze(dim=0)
            if (last_hidden_state.size() != label.size()): # torch.Size([1, 768])
                continue # remove assert so as not to crash training
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
        running_vloss, val_count = 0.0, 0
        with torch.no_grad():
            for i, vdata in enumerate(val_dl):
                vinputs, vlabels = vdata
                vinputs['input_ids'] = vinputs['input_ids'].to(device)
                vinputs['attention_mask'] = vinputs['attention_mask'].to(device)
                vlabels = vlabels.to(device)
                voutputs = model(input_ids=vinputs['input_ids'], attention_mask=vinputs['attention_mask'])
                vlast_hidden_state = (voutputs['hidden_states'][-1].squeeze())[0].unsqueeze(dim=0)
                if (vlast_hidden_state.size() == torch.Size([1])): 
                    continue
                elif (vlabels.size() == torch.Size([1, 1, 768])):
                    vlabels = vlabels.squeeze(dim=0)
                if (vlast_hidden_state.size() != vlabels.size()): # torch.Size([1, 768])
                    continue # remove assert so as not to crash training
                vloss = loss_fn(vlast_hidden_state, vlabels)
                running_vloss += vloss
                val_count += 1
        avg_vloss = running_vloss / val_count
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
            model_path = 'gpt2_model_{}_{}'.format(timestamp, ep)
            torch.save(model.state_dict(), model_path)
            trained_model_path = model_path
    return trained_model_path

def learn_urban(device, trained_model_path, num_words=10000):
    model = gpt2_pt_model
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)
    model.eval()
    with torch.no_grad():
        counter = 0
        for line in open('datasets/urban_words_reformat.json', "r"):
            if counter == num_words:
                break
            entry = json.loads(line)
            word, defn, upv, downv = entry['lowercase_word'], entry['definition'].lower(), int(entry["thumbs_up"]), int(entry["thumbs_down"])
            if (len(word.split(' ')) > 1) or (downv > upv) or (upv < 10): continue # skip phrases, words with more downvotes than upvotes, or too few upvotes
            if len(tokenizer(word, return_tensors='pt')['input_ids']) == 1: continue # skip words that are common but in UD (naive test)
            # input is tokenized + padded defn
            input = tokenizer(defn, padding='max_length', return_tensors="pt")
            input['input_ids'] = input['input_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            outputs = model(input_ids=input['input_ids'], attention_mask=input['attention_mask']) # output is predicted word embedding
            last_hidden_state = (outputs['hidden_states'][-1].squeeze())[0].unsqueeze(dim=0)
            tokenizer.add_tokens(word)
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight[:,-1] = last_hidden_state
            counter += 1
        torch.save(model.state_dict(), 'gpt2_final_model')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    # PHASE 1: Train model on dict of common words to learn r/s between defns and embeddings 
    # common_data = JSonDataset('datasets/dict_wn.json', 'gpt2', tokenizer, word_embeddings)
    trained_model_path = train(device, timestamp, writer)

    # PHASE 2: Add add new word embeddings to GPT2 given the new definitions
    learn_urban(device, trained_model_path)

if __name__ == '__main__':
    main()