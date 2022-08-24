import os

from PIL import Image

import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

IMAGE_PATH = '/home/aimlab/images/total/'
IMAGE_H, IMAGE_W = 224, 224
CHANNEL = 3
MAX_OUTFIT = 6

class OutfitDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        outfit_items = list(self.dataset[idx].values())
        outfit_len = len(outfit_items)
                
        outfit_features = []
        for item_code in outfit_items: 
            img = Image.open(os.path.join(IMAGE_PATH, item_code + '.jpg'))
            img = self.transform(img)
            
            outfit_features.append(img)
            
        for _ in range(MAX_OUTFIT - outfit_len): ## zero padding
            outfit_features.append(torch.zeros(CHANNEL, IMAGE_H, IMAGE_W))
        
        outfit_features = torch.cat(outfit_features)
        outfit_features = outfit_features.reshape(MAX_OUTFIT, CHANNEL, IMAGE_H, IMAGE_W)
        
        return (outfit_features, outfit_len, torch.tensor(self.label, dtype=torch.float), str(self.dataset[idx]))

def get_train_valid_test(dataset):
    train_len = int(len(dataset) * 8 / 10)
    valid_len = int(len(dataset) * 1 / 10)
    test_len = len(dataset) - train_len - valid_len
    
    return torch.utils.data.random_split(dataset, [train_len, valid_len, test_len])

def get_data_loaders(data_1, data_2, batch_size):
    dataset_1 = OutfitDataset(data_1, 0)
    dataset_2 = OutfitDataset(data_2, 1)

    train_dataset_1, valid_dataset_1, test_dataset_1 = get_train_valid_test(dataset_1)
    train_dataset_2, valid_dataset_2, test_dataset_2 = get_train_valid_test(dataset_2)

    train_dataset = train_dataset_1 + train_dataset_2
    valid_dataset = valid_dataset_1 + valid_dataset_2
    test_dataset = test_dataset_1 + test_dataset_2

    print('train: {}, valid: {}, test: {}'.format(
        len(train_dataset), len(valid_dataset), len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)

    return train_loader, valid_loader, test_loader