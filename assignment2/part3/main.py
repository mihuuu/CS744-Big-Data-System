import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model as mdl
import time
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# Training parameters
batch_size = 64  # Per node
SEED = 14
torch.manual_seed(SEED)

DEVICE = "cpu"

iteration_times = [0 for _ in range(39)]

# Initialize process group
def setup(master_ip, rank, world_size):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = '12399'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, model, train_loader, optimizer, criterion, epoch, world_size):
    
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        start = time.time()

        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(data)
        
        loss = criterion(outputs, target)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()

        end = time.time()

        if batch_idx>0 and batch_idx<40:
            iteration_times[batch_idx-1] = end-start
        
        if batch_idx%20==0 and rank == 0:
            print(f"Rank {rank} | Epoch {epoch} [{batch_idx}/{len(train_loader)}] - Loss: {running_loss / (batch_idx + 1):.4f}")

    return None

def test_model(rank, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    if rank == 0:
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def main(master_ip, rank, world_size):

    setup(master_ip, rank, world_size)
    
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
    ])
    
    training_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set)
    train_loader = DataLoader(training_set, sampler=train_sampler, shuffle=False, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = mdl.VGG11().to(DEVICE)
    model = nn.parallel.DistributedDataParallel(model)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    for epoch in range(1):
        epoch_start_time = time.time()
        train_model(rank, model, train_loader, optimizer, criterion, epoch, world_size)
        test_model(rank, model, test_loader, criterion)
        epoch_time = time.time() - epoch_start_time
        if rank == 0:
            print(f"Epoch {epoch+1} complete time: {epoch_time:.4f} sec")

    print(f"Average running time for iteration 1-39: {sum(iteration_times)/len(iteration_times)}")
    
    cleanup()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--master-ip', type=str, required=True)
    parser.add_argument('--num-nodes', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)

    args = parser.parse_args()
    
    main(args.master_ip, args.rank, args.num_nodes)

