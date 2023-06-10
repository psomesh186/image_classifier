import torch

from utils import get_dataloader, compute_accuracy
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