from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

trainset = CIFAR10(root="./data", train = True, download=True,transform = transforms.ToTensor())