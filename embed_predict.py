"""
Method 2: Train a separate neural network to predict the word embedding given the 
definition, using a dictionary of common words as the input and the word embeddings 
already in the model as the output.
"""

import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import EmbedPredictor
from dataset import JSonDataset

def split_data(dataset):
    train_size, val_size = int(0.8 * len(dataset)), int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_dl, val_dl, test_dl = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2), DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2), DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    return train_dl, val_dl, test_dl

def train(timestamp, tb_writer, eps=100, lr=0.001):
    common_data = JSonDataset('datasets/dict_wn.json')
    train_dl, val_dl, test_dl = split_data(common_data)
    model = EmbedPredictor()
    loss_fn = torch.nn.CrossEntropyLoss() # TO-DO: Ask which loss fn?
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(eps):
        print('EPOCH {}:'.format(ep + 1))

        # One pass through data
        model.train(True)
        running_loss, avg_loss = 0.0, 0.0
        for i, data in enumerate(train_dl):
            input, label = data
            optimizer.zero_grad()
            outputs = model(input)
            loss = loss_fn(outputs, label)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    train(timestamp, writer)

if __name__ == '__main__':
    main()