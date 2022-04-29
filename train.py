import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from preprocess import read_imgs
import random
import numpy as np
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import ViT

seed = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_data, train_label, test_data, test_label, criterion, optimizer, scheduler, epochs: int = 100):
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num = 0
        batch_size = 32
        for i in range(int(len(train_data)/batch_size) - 1):
            num += 1
            data = torch.tensor(train_data[i*32: i*32+32])
            label = torch.tensor(train_label[i*32: i*32+32])
            model.train()
            optimizer.zero_grad()
            #Load data into cuda
            data = data.to(device)
            label = label.to(device)
            #Pass data to model
            output = model(data)
            loss = criterion(output, label)
            #Optimizing
            loss.backward()
            optimizer.step()
            #Calculate Accuracy
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_data)
            epoch_loss += loss / len(train_data)
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for i in range(int(len(test_data)/batch_size) - 1):
                    model.eval()
                    data = torch.tensor(test_data[i * 32: i * 32 + 32])
                    label = torch.tensor(test_label[i * 32: i * 32 + 32])
                    #Load val_data into cuda
                    data = data.to(device)
                    label = label.to(device)
                    #Pass val_data to model
                    val_output = model(data)
                    val_loss = criterion(val_output, label)
                    #Calculate Validation Accuracy
                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(test_data)
                    epoch_val_loss += val_loss / len(test_data)
            print('Batch:', num)
            print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n")


def read_config(config_path: str = "config.txt"):
    with open(config_path) as f:
        lines = f.readlines()
    lines = [word.strip('\n') for word in lines]
    return {'batch_size': int(lines[0]),
            'epochs': int(lines[1]),
            'learning_rate': float(lines[2]),
            'gamma': float(lines[3]),
            'img_size': int(lines[4]),
            'patch_size': int(lines[5]),
            'num_class': int(lines[6]),
            'd_model': int(lines[7]),
            'n_head': int(lines[8]),
            'n_layers': int(lines[9]),
            'd_mlp': int(lines[10]),
            'channels': int(lines[11]),
            'dropout': float(lines[12]),
            'pool': lines[13]}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(seed)
    configs = read_config()
    img_size = configs['img_size']
    root = 'flowers'
    dirs = [os.path.join(root, i) for i in os.listdir(root)]
    imgs_list = read_imgs(dirs)

    data = imgs_list[0]
    label = imgs_list[1]
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.1, random_state=42)

    print('data_load!')

    vision_transformer = ViT(img_size = configs['img_size'],
                            patch_size = configs['patch_size'],
                            num_class = configs['num_class'],
                            d_model = configs['d_model'],
                            n_head = configs['n_head'],
                            n_layers = configs['n_layers'],
                            d_mlp = configs['d_mlp'],
                            channels = configs['channels'],
                            dropout = configs['dropout'],
                            pool = configs['pool']).to(device)
    #epochs
    epochs = configs['epochs']
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(vision_transformer.parameters(), lr=configs['learning_rate'])
    # scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    print('train start!')
    train(vision_transformer, data_train, label_train, data_test, label_test, criterion, optimizer, scheduler, epochs)
