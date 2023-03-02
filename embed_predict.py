"""
Method 2: Train a separate neural network to predict the word embedding given the 
definition, using a dictionary of common words as the input and the word embeddings 
already in the model as the output.
"""

import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from transformers import GPT2Tokenizer, TFGPT2Model, GPT2LMHeadModel
from dataset import JSonDataset

# https://github.com/huggingface/transformers/issues/1458
gpt2_pt_model = GPT2LMHeadModel.from_pretrained('gpt2')  # or any other checkpoint
word_embeddings = gpt2_pt_model.transformer.wte.weight  # Word Token Embeddings 
# position_embeddings = model.transformer.wpe.weight  # Word Position Embeddings 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def split_data(dataset):
    train_size, val_size = int(0.8 * len(dataset)), int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_dl, val_dl, test_dl = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2), DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    return train_dl, val_dl, test_dl

def train(timestamp, tb_writer, eps=100, lr=0.00003): # TO TEST: How many eps?
    common_data = JSonDataset('datasets/dict_wn.json', tokenizer, word_embeddings)
    train_dl, val_dl, test_dl = split_data(common_data)
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
            optimizer.zero_grad()
            outputs = model(**input)
            loss = loss_fn(outputs.last_hidden_state, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 99:
                avg_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, avg_loss))
                tb_x = ep * len(train_dl) + i + 1
                tb_writer.add_scalar('Loss/train', avg_loss, tb_x)
                running_loss = 0.
        
        # One set of eval
        running_vloss = 0.0
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
            torch.save(model.state_dict(), model_path)

def learn_urban():
    # TO-DO: Unzip Urban Dict
    urban_data = JSonDataset('datasets/urban.json', tokenizer, word_embeddings)
    for urban_word in urban_data.keys():
        text_index = tokenizer.encode(urban_word, add_prefix_space=True)
        embed_y = word_embeddings[text_index,:]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    # PHASE 1: Train model on dict of common words to learn r/s between defns and embeddings 
    train(timestamp, writer)

    # PHASE 2: Add add new word embeddings to GPT2 given the new definitions 

if __name__ == '__main__':
    main()