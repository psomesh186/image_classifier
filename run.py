import argparse
import os
import torch

from train import train_classifier
from test import test_classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Running image classifier model"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs to train model for")
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="Size of the mini-batches used in training"
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping of training"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for running the model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path of root data folder"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Toggle to skip training and perform testing on trained model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best.pt",
        help="Path to trained model"
    )
    args = parser.parse_args()
    
    # Create relevant folders
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    
    # Get GPU device if available
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    if not args.test:
        train_classifier(
            device=device,
            root_dir=args.data_path,
            batch_size=args.batch_size,
            patience=args.patience,
            numEpochs=args.epochs
        )
    test_classifier(
        device=device,
        batch_size=args.batch_size,
        test_data_path=args.data_path,
        model_path=args.model_path
    )