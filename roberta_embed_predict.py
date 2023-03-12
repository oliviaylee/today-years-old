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
from transformers import RobertaForMaskedLM, RobertaTokenizer
from dataset import JSonDataset

# https://github.com/huggingface/transformers/issues/1458

# RoBERTa
# roberta-large to avoid truncating?
roberta_pt_model = RobertaForMaskedLM.from_pretrained('roberta-base', output_hidden_states=True, is_decoder=True)  # or any other checkpoint
word_embeddings = roberta_pt_model.get_input_embeddings() # Word Token Embeddings # roberta_pt_model.embeddings.word_embeddings.weight
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
roberta_pt_model.resize_token_embeddings(len(tokenizer))
trained_model_path = None

def split_data(dataset):
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # val_size, test_size = int(0.1 * len(dataset)), len(dataset) - train_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dl, val_dl = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl

def train(device, timestamp, tb_writer, lr=0.00003, eps=3, batch_size=32):
    common_data = JSonDataset('datasets/dict_wn.json', 'roberta', tokenizer, word_embeddings)
    train_dl, val_dl = split_data(common_data)
    model = roberta_pt_model
    model.to(device)
    loss_fn = torch.nn.MSELoss() #torch.nn.CosineEmbeddingLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_vloss = float('inf')
    for ep in range(eps):
        print('EPOCH {}:'.format(ep + 1))
        # One pass through data
        model.train(True)
        running_loss, avg_loss = 0.0, 0.0
        for i, data in enumerate(train_dl):
            input, label = data # input = tokenized+padded defn, label = ground truth pretrained embedding
            input['input_ids'] = input['input_ids'].squeeze(dim=1).to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            label = label.to(device)
            outputs = model(input_ids=input['input_ids'], attention_mask=input['attention_mask']) # odict_keys(['logits', 'past_key_values', 'hidden_states'])
            # output['hidden states'] is a Tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
            last_hidden_state = (outputs['hidden_states'][-1].squeeze())[0].unsqueeze(dim=0)
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
        for i, vdata in enumerate(val_dl):
            vinputs, vlabels = vdata
            vinputs['input_ids'] = vinputs['input_ids'].squeeze(dim=1).to(device)
            vinputs['attention_mask'] = vinputs['attention_mask'].to(device)
            vlabels = vlabels.to(device)
            voutputs = model(input_ids=vinputs['input_ids'], attention_mask=vinputs['attention_mask'])
            vloss = loss_fn(voutputs, vlabels)
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
            model_path = 'model_{}_{}'.format(timestamp, ep)
            torch.save(model.state_dict(), model_path)

def learn_urban(device, num_words=5000):
    model = roberta_pt_model
    model.load_state_dict(torch.load(trained_model_path))
    model.to(device)
    model.eval()
    with torch.no_grad():
        counter = 0
        for line in open('datasets/urban_words.json', "r"):
            if counter == num_words:
                break
            counter += 1
            entry = json.loads(line)
            word = entry['lowercase_word']
            defn = entry['definition'].lower()
            # input is tokenized + padded defn
            input = tokenizer(defn, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            input['input_ids'] = input['input_ids'].squeeze(dim=1).to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            output = model(input_ids=input['input_ids'], attention_mask=input['attention_mask']) # output is predicted word embedding
            tokenizer.add_tokens(word)
            model.resize_token_embeddings(len(tokenizer))
            model.transformer.wte.weight[:,-1] = output
        torch.save(model.state_dict(), 'final_model')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    # PHASE 1: Train model on dict of common words to learn r/s between defns and embeddings 
    train(device, timestamp, writer)

    # PHASE 2: Add add new word embeddings to GPT2 given the new definitions
    learn_urban(device)

if __name__ == '__main__':
    main()