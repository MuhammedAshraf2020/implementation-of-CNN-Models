import torchvision
import torchvision.transforms as transforms
import torch

def data_pt( batch_size = 150):
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
  trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                          shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())

  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

  return trainloader , testloader
