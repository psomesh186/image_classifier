import torch
import numpy as np
import matplotlib.pyplot as plt
from model import CNNClassifier
from data_utils import get_dataloader
from tqdm import tqdm

def compute_accuracy(outputs, labels):
    outputs = torch.argmax(outputs, dim=1)
    acc = 0
    for idx in range(len(outputs)):
        if outputs[idx] == labels[idx]:
            acc += 1
    acc /= len(outputs)
    return acc


def train_classifier(device, root_dir, num_labels, patience=5, numEpochs=40):
    """Trains model on dataset according to given hyperparameters.
    
    Args:
        device (str): Device to train the model on.
        root_dir (str): Root directory of dataset.
        num_labels (int): Number of classes present in dataset.
        patience (int): Number of epochs to wait without improvement in loss. 
        (Default: 5)
        numEpochs (int): Epochs to perform training for. (Default: 40)
    """

    # Load data
    trainloader = get_dataloader(root_dir + '/train')
    validloader = get_dataloader(root_dir + '/valid')

    # Graph setup
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_title('Loss plot')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    
    ax[1].set_title('Accuracy plot')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.ion()
    plt.show()

    # Train setup
    device = torch.device(device)
    model = CNNClassifier(num_labels)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    best_loss = torch.inf
    patience_counter = 0

    # Training
    for epoch in range(numEpochs):
        # Train section
        model.train()
        train_loss = 0
        train_accuracy = []
        for i, (images, labels) in enumerate(
            tqdm(trainloader, desc=f"Epoch: {epoch}")
        ):
            # Shift data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Back propogation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            train_accuracy.append(compute_accuracy(outputs, labels))
        train_loss /= (i + 1)
        train_accuracy = sum(train_accuracy) / (i + 1)

        # Validation section
        model.eval()
        valid_loss = 0
        valid_accuracy = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(validloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()
                valid_accuracy.append(compute_accuracy(outputs, labels))
            valid_loss /= (i + 1)
            valid_accuracy = sum(valid_accuracy) / (i + 1)
        
        # Report metrics
        print(f"Train loss:{train_loss}, Validation loss:{valid_loss}, \
              Train acc:{train_accuracy}, Validation accuracy:{valid_accuracy}")
        if epoch != 0:
            ax[0].plot(
                [epoch - 1, epoch],
                [prev_trainloss, train_loss],
                marker='o',
                color='blue',
                label='Train loss'
                )
            ax[0].plot(
                [epoch - 1, epoch],
                [prev_validloss, valid_loss],
                marker='o',
                color='orange',
                label='Validation loss'
                )
            ax[1].plot(
                [epoch - 1, epoch],
                [prev_trainacc, train_accuracy],
                marker='o',
                color='blue',
                label='Train accuracy'
                )
            ax[1].plot(
                [epoch - 1, epoch],
                [prev_validacc, valid_accuracy],
                marker='o',
                color='orange',
                label='Validation accuracy'
                )
            plt.pause(0.0001)
        prev_trainloss = train_loss
        prev_validloss = valid_loss
        prev_trainacc = train_accuracy
        prev_validacc = valid_accuracy

        # Check improvement and save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": train_loss
                }, f"checkpoints/best.pt")
        else:
            patience_counter += 1
            if patience_counter > patience:
                break