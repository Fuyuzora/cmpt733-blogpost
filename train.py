
from datetime import datetime
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from CNNClasses import LeNet, AlexNet, VGGNet, ResNet

import pickle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
EPOCHS = 10
N_CLASSES = 10
run_lenet = False
run_alexnet =False
run_vggnet = False
run_resnet = True


def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for batch, labels in train_loader:
        optimizer.zero_grad()
        batch, labels = batch.to(device), labels.to(device)

        prediction = model(batch)
        loss = criterion(prediction, labels)
        running_loss += loss.item() * batch.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0
    for batch, labels in valid_loader:

        batch, labels = batch.to(device), labels.to(device)
        prediction = model(batch)
        loss = criterion(prediction, labels)
        running_loss += loss.item() * batch.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    train_losses = []
    valid_losses = []
    train_acces = []
    valid_acces = []

    for epoch in range(0, epochs):
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):

            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            train_acces.append(train_acc.item())
            valid_acces.append(valid_acc.item())

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    return model, optimizer, (train_acces, valid_acces)


def get_accuracy(model, data_loader, device):
    correct_prediction = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for batch, labels in data_loader:

            batch, labels = batch.to(device), labels.to(device)

            prediction = model(batch)
            _, predicted_labels = torch.max(prediction, 1)

            n += labels.size(0)
            correct_prediction += (predicted_labels == labels).sum()

    return correct_prediction.float() / n

# Accuracies
lenet_train = []
lenet_valid = []
alexnet_train = []
alexnet_valid = []
vggnet_train = []
vggnet_valid = []
resnet_train = []
resnet_valid = []

transforms32 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(
    root='mnist_data', train=True, transform=transforms32, download=True)
valid_dataset = datasets.FashionMNIST(
    root='mnist_data', train=False, transform=transforms32)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE, shuffle=False)

# LeNet ================================================================
if run_lenet:
    lenet = LeNet(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(lenet.parameters())
    criterion = nn.CrossEntropyLoss()

    lenet, optimizer, (lenet_train, lenet_valid) = training_loop(
        lenet, criterion, optimizer, train_loader, valid_loader, EPOCHS, DEVICE)

# Reshape images for more complex architecture ================================
transforms224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(
    root='mnist_data', train=True, transform=transforms224, download=True)
valid_dataset = datasets.FashionMNIST(
    root='mnist_data', train=False, transform=transforms224)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE, shuffle=False)

# AlexNet ================================================================\
if run_alexnet:
    alexnet = AlexNet(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(alexnet.parameters())
    criterion = nn.CrossEntropyLoss()

    alexnet, optimizer, (alexnet_train, alexnet_valid) = training_loop(
        alexnet, criterion, optimizer, train_loader, valid_loader, EPOCHS, DEVICE)

# VGGNet ================================================================
if run_vggnet:
    vggnet = VGGNet(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(vggnet.parameters())
    criterion = nn.CrossEntropyLoss()

    vggnet, optimizer, (vggnet_train, vggnet_valid) = training_loop(
        vggnet, criterion, optimizer, train_loader, valid_loader, EPOCHS, DEVICE)


# ResNet ================================================================
if run_resnet:
    resnet = ResNet(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(resnet.parameters())
    criterion = nn.CrossEntropyLoss()

    resnet, optimizer, (resnet_train, resnet_valid) = training_loop(
        resnet, criterion, optimizer, train_loader, valid_loader, EPOCHS, DEVICE)

acc_dict = {
    'lenet_train': lenet_train,
    'lenet_valid': lenet_valid,
    'alexnet_train': alexnet_train,
    'alexnet_valid': alexnet_valid,
    'vggnet_train': vggnet_train,
    'vggnet_valid': vggnet_valid,
    'resnet_train': resnet_train,
    'resnet_valid': resnet_valid,
}

if os.path.isfile('accuracy.pkl'):
    records = {}
    with open('accuracy.pkl', 'rb') as f:
        records = pickle.load(f)

    for key, value in acc_dict.items():
        if key not in records.keys() and len(value) > 0:
            records[key] = value

    with open('accuracy.pkl', 'wb') as f:
        pickle.dump(records, f)
else:
    with open('accuracy.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)

print('================= Accuracy Saved ====================')
print(acc_dict)
print('================= back up ====================')
print(lenet_train)
print(lenet_valid)
print(alexnet_train)
print(alexnet_valid)
print(vggnet_train)
print(vggnet_valid)
print(resnet_train)
print(resnet_valid)