import json
import random

import torch
import torch.nn as nn

from data import get_data_loaders
from train import build_model, train_data_loader

if __name__ == "__main__":
    ## Data Parameters
    BATCH_SIZE = 32

    ## Dataset
    with open('../all_good_top_bottoms.json') as f:
        good_outfits = json.load(f)
    with open('../all_bad_top_bottoms.json') as f:
        bad_outfits = json.load(f)

    random.seed(1207)
    # random.shuffle(good_outfits)
    # good_outfits = good_outfits[:100000]
    # bad_outfits = bad_outfits[:]

    train_loader, valid_loader, test_loader = get_data_loaders(
        bad_outfits, good_outfits, BATCH_SIZE)

    ## Model Parameters    
    INPUT_SIZE = 512
    HIDDEN_SIZE = 1024
    # NUM_LAYERS = 1
    
    model = build_model(INPUT_SIZE, HIDDEN_SIZE)
    # model = nn.DataParallel(model) # multiple-GPU

    ## Train
    LR = 0.00005
    NUM_EPOCHS = 10
    THRESHOLD = 0.5
    
    model, train_loss_list, valid_loss_list = train_data_loader(
        model, train_loader, valid_loader, LR, NUM_EPOCHS, THRESHOLD)

    torch.save(model.state_dict(), 'efficientNet_b0_lstm_20220824')