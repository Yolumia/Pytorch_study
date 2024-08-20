import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet18

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
def main(rank, world_size):
    setup(rank, world_size)

    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    train_set = torchvision.datasets.CIFAR10(root='/mnt/sda/zjb/data/cifar10', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=16, sampler=train_sampler)

    test_set = torchvision.datasets.CIFAR10(root='/mnt/sda/zjb/data/cifar10', train=False, download=True, transform=transform)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(test_set, batch_size=16, sampler=test_sampler)

    # 模型定义
    model = resnet18(pretrained=False, num_classes=10).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练过程
    for epoch in range(10):
        model.train()
        for data in train_loader:
            inputs, labels = data[0].cuda(rank), data[1].cuda(rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")
        # 评估模型
        accuracy = evaluate(model, rank, test_loader)
        print(f"Rank {rank}, Test Accuracy: {accuracy}%")

    cleanup()

if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
