import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def gen_mean_std(dataset):
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=False, num_workers=2)
        train = iter(dataloader).next()[0]
        mean = np.mean(train.numpy(), axis=(0, 2, 3))
        std = np.std(train.numpy(), axis=(0, 2, 3))
        return mean, std

if __name__=='__main__':
    # cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    mean,std = gen_mean_std(cifar100)
    print(mean, std)

