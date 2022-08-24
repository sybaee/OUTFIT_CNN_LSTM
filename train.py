import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN_LSTM

RANDOM_SEED = 1207

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def build_model(input_size, hidden_size, num_layers=1):    
    model = CNN_LSTM(input_size, hidden_size, num_layers)
    
    return model

def train_data_loader(model, train_loader, valid_loader, lr, num_epochs, threshold):
    fix_seed(RANDOM_SEED)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    model.cuda(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('-' * 80)
    print('Training Starts...')
    print('-' * 80)

    train_loss_list = []
    valid_loss_list = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0
        for i, train_data in enumerate(train_loader):
            if i % 50 == 0:
                print('Epoch: {:3d}/{:3d} | Train batch: {:6d}/{:6d}'.format(epoch+1, num_epochs, i+1, len(train_loader)))
            outfits, outfit_lens, labels, _ = train_data
            if cuda_available:
                outfits = outfits.cuda(device)
                outfit_lens = outfit_lens.tolist()
                labels = labels.cuda(device)

            optimizer.zero_grad()

            output = model(outfits, outfit_lens)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (output > threshold).int()
            train_accuracy += (labels == predicted).type(torch.FloatTensor).mean()

            train_loss_list.append(train_loss / len(train_loader))

        if epoch % 1 == 0:
            if valid_loader:
                with torch.no_grad():
                    valid_loss = 0.0
                    valid_accuracy = 0
                    for j, valid_data in enumerate(valid_loader):
                        if j % 50 == 0:
                            print('Epoch: {:3d}/{:3d} | Valid batch: {:6d}/{:6d}'.format(epoch+1, num_epochs, j+1, len(valid_loader)))
                        val_outfits, val_outfit_lens, val_labels, _ = valid_data

                        if cuda_available:
                            val_outfits = val_outfits.cuda(device)
                            val_outfit_lens = val_outfit_lens.tolist()
                            val_labels = val_labels.cuda(device)

                        valid_output = model(val_outfits, val_outfit_lens)

                        loss = criterion(valid_output, val_labels)
                        valid_loss += loss

                        predicted = (valid_output > threshold).int()
                        valid_accuracy += (val_labels == predicted).type(torch.FloatTensor).mean()

                print('-' * 80)
                print('Epoch: {}/{} | Train loss: {:.4f} | Train accuracy: {:.4f} | Valid loss: {:.4f} | Valid accuracy: {:.4f}'.format(
                    epoch+1, num_epochs, train_loss / len(train_loader), train_accuracy / len(train_loader), 
                    valid_loss / len(valid_loader), valid_accuracy / len(valid_loader)))
                print('-' * 80)      

                valid_loss_list.append(valid_loss / len(valid_loader))
                
            else:
                print('-' * 80)
                print('Epoch: {}/{} | Train loss: {:.4f} | Train accuracy: {:.4f}'.format(
                    epoch+1, num_epochs, train_loss / len(train_loader), train_accuracy / len(train_loader)))
                print('-' * 80)

        torch.save(model.state_dict(), 'efficientNet_b0_lstm_20220824')

    return model, train_loss_list, valid_loss_list