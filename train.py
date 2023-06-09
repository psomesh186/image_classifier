import torch
import matplotlib.pyplot as plt

from model import CNNClassifier
from utils import get_dataloader, compute_accuracy
from tqdm import tqdm


def train_classifier(device, root_dir, batch_size, patience=5, numEpochs=40):
    """Trains model on dataset according to given hyperparameters.
    
    Args:
        device (str): Device to train the model on.
        root_dir (str): Root directory of dataset.
        batch_size (int): Mini-batch size used during training.
        patience (int): Number of epochs to wait without improvement in loss. 
        (Default: 5)
        numEpochs (int): Epochs to perform training for. (Default: 40)
    """

    # Load data
    trainloader, labels = get_dataloader(
        root_dir + '/train',
        batch_size=batch_size
        )
    validloader, _ = get_dataloader(
        root_dir + '/valid',
        batch_size=batch_size
        )
    num_labels = len(labels)

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
        train_accuracy = 0
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
            train_accuracy += compute_accuracy(outputs, labels)
        train_loss /= (i + 1)
        train_accuracy = train_accuracy / (i + 1) * 100

        # Validation section
        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(validloader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()
                valid_accuracy += compute_accuracy(outputs, labels)
            valid_loss /= (i + 1)
            valid_accuracy = valid_accuracy / (i + 1) * 100
        
        # Report metrics
        metrics = f"Train loss: {train_loss:.3f},"
        metrics += f"Validation loss: {valid_loss:.3f},"
        metrics += f"Train accuracy: {train_accuracy:.3f},"
        metrics += f"Validation accuracy: {valid_accuracy:.3f}"
        print(metrics)
        if epoch != 0:
            ax[0].plot(
                [epoch - 1, epoch],
                [prev_trainloss, train_loss],
                color='blue',
                label='Train loss' if epoch == 1 else ''
                )
            ax[0].plot(
                [epoch - 1, epoch],
                [prev_validloss, valid_loss],
                color='orange',
                label='Validation loss' if epoch == 1 else ''
                )
            ax[1].plot(
                [epoch - 1, epoch],
                [prev_trainacc, train_accuracy],
                color='blue',
                label='Train accuracy' if epoch == 1 else ''
                )
            ax[1].plot(
                [epoch - 1, epoch],
                [prev_validacc, valid_accuracy],
                color='orange',
                label='Validation accuracy' if epoch == 1 else ''
                )
            ax[0].legend()
            ax[1].legend()
            plt.pause(0.1)
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
        
    # Save plots to device
    plt.savefig("outputs/plot.png")