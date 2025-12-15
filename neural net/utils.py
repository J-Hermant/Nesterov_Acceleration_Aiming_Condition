import itertools
import numpy as np
import torch
from tqdm.auto import tqdm

def sort_batches(train_loader, batch_size, device):
    """
    Take batches of the train_loader and generate batches with only one class inside and 
    sort batches by their class.
    """
    batch_sort = [[] for i in range(10)]
    targets_sort = [[] for i in range(10)]
    num_im = 0
    for i, (batch, targets) in tqdm(enumerate(train_loader)):
        index_sort = np.argsort(targets)
        batch_sort_i = batch[index_sort]
        targets_sort_i = targets[index_sort]
        for j in range(len(targets_sort_i)):
            num_im += 1
            batch_sort[targets_sort_i[j]].append(batch_sort_i[j].numpy())
            targets_sort[targets_sort_i[j]].append(int(targets_sort_i[j]))

    batch_sort_2 = []
    targets_sort_2 = []
    for k in range(10):
        batch_sort_2 = batch_sort_2 + batch_sort[k]
        targets_sort_2 = targets_sort_2 + targets_sort[k]
    batch_shuffle = []
    targets_shuffle = []
    for i in range(len(targets_sort_2)//batch_size):
        batch_shuffle.append(batch_sort_2[i*batch_size:(i+1)*batch_size])
        targets_shuffle.append(targets_sort_2[i*batch_size:(i+1)*batch_size])
    batch_shuffle = np.array(batch_shuffle)
    batch_shuffle = torch.tensor(batch_shuffle).to(device)
    targets_shuffle = torch.tensor(targets_shuffle).to(device)
    return batch_shuffle, targets_shuffle

def calculate_training_loss(model, train_loader, criterion, device,network_type,batch_size_train):
    """
    Return the total loss value
    """
    model.eval()  # Model in evaluate mode
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # No gradient computation
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if network_type == "CNN":
                inputs = inputs.view((batch_size_train, 3, 32, 32))
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Loss accumulation
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    # Average loss on training dataset
    average_loss = total_loss / total_samples
    return average_loss