import torch
import matplotlib.pyplot as plt

from utils import get_dataloader, compute_accuracy, denormalize
from model import CNNClassifier
from tqdm import tqdm


def test_classifier(device, test_data_path, batch_size, model_path=None):
    """Tests trained model on test data.
    
    Args:
        device (str): Device to run the model on.
        test_data_path (str): Root directory of dataset.
        batch_size (int): Mini-batch size used for dataloader.
        model_path (str): Path to trained model. If none is given, sets to
        'checkpoints/best.pt'  
    """

    # Load test data
    testloader, class2idx = get_dataloader(
        root_dir=test_data_path + '/test',
        shuffle=True,
        batch_size=batch_size,
        num_workers=0
    )
    num_labels = len(class2idx)

    # Load trained model
    if not model_path:
        model_path = 'checkpoints/best.pt'
    checkpoint = torch.load(model_path)
    model = CNNClassifier(num_labels=num_labels)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(device)
    model.to(device)

    # Set up test
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(testloader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_accuracy += compute_accuracy(outputs, labels)
        test_accuracy = test_accuracy / (i + 1) * 100
        print(f"Test Accuracy: {test_accuracy:.3f}%")
        
        # Visualize results on first 9 images
        cols = 3
        rows = 3
        num_images = rows * cols
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(9, 9))
        plt.tight_layout()
        images, labels = next(iter(testloader))
        images = images[:num_images].to(device)
        labels = labels[:num_images]
        outputs = model(images)
        images = images.cpu()
        outputs = torch.argmax(outputs, dim=1)
        idx2class = {idx:label for label, idx in class2idx.items()}
        sample = 0
        for r in range(rows):
            for c in range(cols):
                image = denormalize(images[sample])
                ax[r][c].imshow(image.permute(1, 2, 0))
                ground_truth = idx2class[labels[sample].item()]
                text = f"Ground Truth: {ground_truth}"
                pred = idx2class[outputs[sample].item()]
                text += f"\nPrediction: {pred}"
                ax[r][c].set_xlabel(text)
                ax[r][c].set_xticks([])
                ax[r][c].set_yticks([])
                sample += 1
        plt.savefig("outputs/results.png")