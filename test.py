import torch
from tqdm import tqdm

def run(model, test_loader, device):

    model.eval()

    for batch_index, (images, targets, paths, shapes) in enumerate(tqdm(test_loader)):

        images = images.to(device)
        targets = targets.to(device)

        predicts, _ = model(images)

    
    pass
