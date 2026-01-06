import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t
import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
from tqdm import tqdm
from models_architecture import create_mlp, create_cnn, create_cnn_bn, create_logistic_regression
from argparse import ArgumentParser
import time as time

start = time.time()
parser = ArgumentParser()
parser.add_argument('--network_type', type=str, default = "CNN", choices=["CNN", "MLP", "Logistic"])
parser.add_argument('--batch_sample', type=str, default = "random_with_rpl", choices=["random_with_rpl", "determinist", "sort"])
parser.add_argument('--device', type=int, default = 0)
parser.add_argument('--n_epoch', type=int, default = 5)
parser.add_argument('--step', type=int, default = 100, help = "interval between each RACOGA computation")
parser.add_argument('--alg', type=str, default = "SNAG", choices = ["SNAG", "SGD", "GD", "NAG","ADAM","RMSprop"])
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--momentum', type=float, default = 0.9)
parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--alpha_rms',type=float, default = 0.99)
parser.add_argument('--data', type=str, default = "CIFAR10", choices = ["CIFAR10", "SPHERE", "MNIST", "FashionMNIST", "KMNIST", "EMNIST"])
parser.add_argument('--beta_adam',type=float, default = 0.999)
parser.add_argument('--batch_normalization', type=bool, default = False, choices = [True, False])
parser.add_argument('--n_data', type = int, default = None)
hparams = parser.parse_args()

device = torch.device('cuda:'+str(hparams.device) if torch.cuda.is_available() else 'cpu')

batch_sample = hparams.batch_sample
network_type = hparams.network_type
batch_normalization = hparams.batch_normalization
n_epoch = hparams.n_epoch
alg = hparams.alg
momentum = hparams.momentum
lr = hparams.lr
current_seed = hparams.seed
data_choice = hparams.data
beta = hparams.beta_adam
alpha = hparams.alpha_rms
# define network structure 

if network_type == "MLP":# MLP architecture
    net = create_mlp().to(device)
if network_type == "CNN" and batch_normalization == False:# Light CNN architecture
    if data_choice == "MNIST" or data_choice == "FashionMNIST" or data_choice == "KMNIST" or data_choice == "EMNIST":
        net = create_cnn(1).to(device)
    elif data_choice == "CIFAR10":
        net = create_cnn(3).to(device)
if network_type == "CNN" and batch_normalization:# Light CNN architecture with batch normalization
    net = create_cnn_bn().to(device)
if network_type == "Logistic":
    if data_choice == "CIFAR10":
        net = create_logistic_regression(input_dim=3072, n_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
if alg == "ADAM":
    optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas = (momentum,beta))
elif alg == "RMSprop":
    optimizer = torch.optim.RMSprop(net.parameters(), lr= lr, alpha = alpha)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=momentum) 

#Dataset load
to_tensor =  t.ToTensor()
if data_choice == "CIFAR10":
    normalize = t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
elif data_choice == "MNIST" or data_choice == "FashionMNIST" or data_choice == "KMNIST" or data_choice == "EMNIST":
    normalize = t.Normalize((0.5,), (0.5,))
flatten =  t.Lambda(lambda x:x.view(-1))
transform_list = t.Compose([to_tensor, normalize, flatten])
train_set = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=transform_list, download=True)

data_choice = hparams.data
if data_choice == "CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=transform_list, download=True)
elif data_choice == "SPHERE":
    # Sphere data
    checkpoint_train = torch.load('../dataset/sphere/train_dataset_sphere.pth')
    train_set = torch.utils.data.TensorDataset(checkpoint_train['data'], checkpoint_train['labels'])
elif data_choice == "MNIST":
    train_set = torchvision.datasets.MNIST(root='../dataset', train=True, transform=transform_list, download=True)
elif data_choice == "FashionMNIST":
    train_set = torchvision.datasets.FashionMNIST(root='../dataset', train=True, transform=transform_list, download=True)
elif data_choice == "KMNIST":
    train_set = torchvision.datasets.KMNIST(root='../dataset', train=True, transform=transform_list, download=True)
elif data_choice == "EMNIST":
    train_set = torchvision.datasets.KMNIST(root='../dataset', train=True, transform=transform_list, download=False)

# Load results
# Save the training loss
path_results = "results/"
if not os.path.exists(path_results):
    os.mkdir(path_results)
path_results = os.path.join(path_results, data_choice)
if not os.path.exists(path_results):
    os.mkdir(path_results)
path_results = os.path.join(path_results, alg)
if not os.path.exists(path_results):
    os.mkdir(path_results)
if alg == "SGD" or alg == "GD":
    path_results = os.path.join(path_results, 'lr_' +  str(lr)) 
    if not os.path.exists(path_results):
        os.mkdir(path_results)
elif alg == "RMSprop":
    name_dir = 'lr_' + str(lr) + '_alpha_' + str(alpha)
    path_results = os.path.join(path_results, name_dir) 
    if not os.path.exists(path_results):
        os.mkdir(path_results)
else:
    name_dir = 'lr_' + str(lr) + '_momentum_' + str(momentum)
    path_results = os.path.join(path_results, name_dir) 
    if not os.path.exists(path_results):
        os.mkdir(path_results)
if hparams.n_data != None:
    path_results = os.path.join(path_results, "n_data_"+str(hparams.n_data))
    if not os.path.exists(path_results):
        os.mkdir(path_results)

if alg == "ADAM":
    suffix = "_lr_" + str(lr) + "_momentum_" + str(momentum) + "_beta_" + str(beta) + "_seed_" + str(current_seed)
elif alg == "RMSprop":
    suffix = "_lr_" + str(lr) + "_alpha_" + str(alpha) + "_seed_" + str(current_seed)
else:
    suffix = "_lr_" + str(lr) + "_momentum_" + str(momentum) + "_seed_" + str(current_seed)
if batch_normalization:
    suffix += "_BN"
dict_path = path_results+"/"+network_type+'_n_epoch_'+str(n_epoch)+'_batch_'+batch_sample+ '_alg_' + alg  + suffix +'_dict_results.pth'
dict_results = torch.load(dict_path)
weights_trajectory = dict_results["weights_trajectory"]
loss_trajectory = dict_results["loss_trajectory"]

print("Number of iterations = {}".format(len(weights_trajectory)))

###
# Computation of RACOGA
###
post_process_loader = torch.utils.data.DataLoader(train_set, batch_size=128)
if hparams.n_data != None and data_choice == "CIFAR10":
    post_process_loader = list(post_process_loader)
    n_batch_data = hparams.n_data // 128
    post_process_loader = post_process_loader[:n_batch_data]

step = hparams.step # interval between each RACOGA computation
racoga_list = []
scalar_prod_list = []
iteration_list = []

if alg == "SGD" or alg == "SNAG" or alg == "ADAM" or alg == "RMSprop":
    for k in tqdm(range(len(weights_trajectory)//step)):
        iteration_list.append(step*k)
        x = weights_trajectory[step*k]
        for j, param in enumerate(net.parameters()):
            param.data = x[j].to(device)
        sum_gradient = torch.tensor([]).to(device)
        sum_gradient_norm = 0

        for i, (batch, targets) in enumerate(post_process_loader):
            batch = batch.to(device)
            if network_type == "CNN":
                batch_size = batch.size()[0]
                if data_choice == "CIFAR10":
                    batch = batch.view((batch_size, 3, 32, 32))
                elif data_choice == "MNIST" or data_choice == "FashionMNIST" or data_choice == "KMNIST" or data_choice == "EMNIST":
                    batch = batch.view((batch_size, 1, 28, 28))
            output = net(batch)
            targets = targets.to(device)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            #save gradients
            gradient_i = torch.tensor([]).to(device)
            for param in net.parameters():
                gradient_i = torch.cat((gradient_i,param.grad.flatten()))
            if sum_gradient.size() != torch.Size([0]):
                sum_gradient += gradient_i
                sum_gradient_norm += torch.sum(gradient_i**2)
            else:
                sum_gradient = gradient_i

        scalar_prod = (1/2) * (torch.sum(sum_gradient**2) - sum_gradient_norm)
        racoga = scalar_prod/sum_gradient_norm
        racoga_list.append(racoga.detach().cpu().numpy())
        scalar_prod_list.append(scalar_prod.detach().cpu().numpy())
else:
    for k in tqdm(range(len(weights_trajectory))):
        iteration_list.append(k)
        x = weights_trajectory[k]
        for j, param in enumerate(net.parameters()):
            param.data = x[j].to(device)
        sum_gradient = torch.tensor([]).to(device)
        sum_gradient_norm = 0

        for i, (batch, targets) in enumerate(post_process_loader):
            batch = batch.to(device)
            if network_type == "CNN":
                batch_size = batch.size()[0]
                if data_choice == "CIFAR10":
                    batch = batch.view((batch_size, 3, 32, 32))
                elif data_choice == "MNIST" or data_choice == "FashionMNIST" or data_choice == "KMNIST" or data_choice == "EMNIST":
                    batch = batch.view((batch_size, 1, 28, 28))
            output = net(batch)
            targets = targets.to(device)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            #save gradients
            gradient_i = torch.tensor([]).to(device)
            for param in net.parameters():
                gradient_i = torch.cat((gradient_i,param.grad.flatten()))
            if sum_gradient.size() != torch.Size([0]):
                sum_gradient += gradient_i
                sum_gradient_norm += torch.sum(gradient_i**2)
            else:
                sum_gradient = gradient_i

        scalar_prod = (1/2) * (torch.sum(sum_gradient**2) - sum_gradient_norm)
        racoga = scalar_prod/sum_gradient_norm
        racoga_list.append(racoga.detach().cpu().numpy())
        scalar_prod_list.append(scalar_prod.detach().cpu().numpy())
duration = time.time() - start
#Save the RACOGA evolution
dict = {
    "racoga_list" : racoga_list,
    "scalar_prod_list" : scalar_prod_list,
    "iteration_list" : iteration_list,
    "computation_time" : duration
}
if alg == "ADAM":
    suffix = "_lr_" + str(lr) + "_momentum_" + str(momentum) + "_beta_" + str(beta) + "_seed_" + str(current_seed)
elif alg == "RMSprop":
    suffix = "_lr_" + str(lr) + "_alpha_" + str(alpha) + "_seed_" + str(current_seed)
else:
    suffix = "_lr_" + str(lr) + "_momentum_" + str(momentum) + "_seed_" + str(current_seed)
if batch_normalization:
    suffix += "_BN"
save_name = path_results+"/"+network_type+'_n_epoch_'+str(n_epoch)+'_batch_'+batch_sample+'_alg_'+ alg+ suffix
np.save(save_name+'_racoga_results.npy', dict)

plt.plot(iteration_list, racoga_list)
plt.xlabel("number of iterations")
plt.ylabel("RACOGA")
plt.savefig(save_name+"_racoga_evolution.png")

# Save info
log_print = '\nracoga : '
if alg == "ADAM":
    log_print += 'datset = ' + data_choice + ', n_epoch = ' + str(n_epoch) +   ', alg = ' + alg + ', lr = ' + str(lr) + ', momentum = ' + str(momentum) + ', beta = ' + str(beta) + '. Computation time : ' + str(duration)
elif alg == "RMSprop":
    log_print += 'datset = ' + data_choice + ', n_epoch = ' + str(n_epoch) +   ', alg = ' + alg + ', lr = ' + str(lr) + ', alpha = ' + str(alpha) + '. Computation time : ' + str(duration)
else:
    log_print += 'datset = ' + data_choice + ', n_epoch = ' + str(n_epoch) +   ', alg = ' + alg + ', lr = ' + str(lr) + ', momentum = ' + str(momentum) +  '. Computation time : ' + str(duration)
log_print += 'datset = ' + data_choice + ', n_epoch = ' + str(n_epoch) +   ', alg = ' + alg + ', lr = ' + str(lr) + ', momentum = ' + str(momentum) +  '. Computation time : ' + str(duration)
if batch_normalization:
    log_print += ' . BN'
fichier = open("log_file.txt", "a")
fichier.write(log_print)
fichier.close()
