import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
import torch.optim as opt
import copy
import numpy as np
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser(description='pipline')

parser.add_argument('--device', '-d', type=int, default=0, help="Which gpu to use; default cpu")
parser.add_argument('-b', type=int, default=0, help="local batch size")
parser.add_argument('-e', type=int, default=20, help="local epoch")
parser.add_argument('--round', '-r', type=int, default=10, help="number of rounds to execute")
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-c', type=int, default=100, help='number of client machines')
parser.add_argument('-k', type=int, default=10, help='number of client machines running for each round')

args = parser.parse_args()

if args.device < 0:
    args.device = 'cpu'
else:
    args.device = torch.device(f"cuda:{args.device}")

print(args)

run = wandb.init(project="fl-flow", 
           config={
               arg: val for arg, val in vars(args).items()    
            }
        )


class Client():
    def __init__(self, client_number, dataset, model, B, E, idxes, lr = 0.01, device='cpu'):
        self.client_number = client_number
        self.local_batch_size = B if B != 0 else len(dataset)
        self.local_epoch = E
        self.lr = lr
        self.device = device
        self.model = model.to(self.device)
        self.sampleID = idxes
        
        self.dataset = [dataset[i] for i in self.sampleID]
        self.nk = len(idxes)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataloader = DataLoader(self.dataset, batch_size=self.local_batch_size, shuffle=True)
    
    def update(self, state_dict):
        self.model.load_state_dict(state_dict)

        optimizer = opt.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()

        pbar = tqdm(range(self.local_epoch), leave=False)
        acc_loss = 0
        for _ in pbar:
            for data,label in self.dataloader:
                optimizer.zero_grad()

                data, label = data.to(self.device), label.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, label)
                acc_loss += loss.item()
                
                loss.backward()
                optimizer.step()

        return self.model.state_dict(), (acc_loss / len(self.dataloader)/self.local_epoch)

class MyModel(nn.Module):
    def __init__(self, input_dim, ):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(5, 2),

            nn.Conv2d(16, 32, 3), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(5, 2),

            nn.Flatten(),
            nn.Linear(512, 32),
            nn.Linear(32, 10),
        )
    
    def forward(self, x):
        out = self.layers(x)
        return out

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=(ToTensor()))
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=(ToTensor()))

# sample for faster training
# train_dataset = Subset(train_dataset, np.random.choice(len(train_dataset), 5000, replace=False),)

### Dummy dataset
# X = torch.rand((10,2))
# y = torch.ones(10).long()
# train_dataset = TensorDataset(X, y)
# test_dataset = TensorDataset(X, y)

n = len(train_dataset)

nclients = args.c # number of clients
m = args.k # number of clients updating in a single round
clients = []

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# dataset for clients
# D = random_split(train_dataset, [n//nclients for _ in range(nclients)])
D = cifar_iid(train_dataset, nclients)

# initilize clients
for i in range(nclients):
    client = Client(i, train_dataset, B=args.b, model = MyModel(3), E = args.e, device= args.device, idxes = D[i])
    clients.append(client)

# server
model = MyModel(3).to(args.device)

pbar = tqdm(range(args.round))
for round in pbar:
    subset = np.random.choice(nclients, m, replace=False)
    old_params = copy.deepcopy(model.state_dict())
    cur_params = {}

    acc_loss = 0
    for idx in tqdm(subset, leave=False):
        client = clients[idx]
        new_params, loss = client.update(old_params)
        acc_loss += loss

        for key in new_params.keys():
            if key in cur_params:
                cur_params[key] += (( client.nk / (m * client.nk)) * new_params[key]).type_as(cur_params[key])
            else:
                cur_params[key] = (( client.nk / (m * client.nk)) * new_params[key]).float()
    

    if (round % 100) == 0:
        model.eval()
        with torch.no_grad():
            correct = 0
            for data, label in DataLoader(test_dataset, batch_size=32):

                data, label = data.to(args.device), label.to(args.device)
                output = model(data)

                pred = torch.argmax(output, dim=1)
                acc = torch.sum(pred == label)

                correct += acc

        run.log({"test acc ": correct / len(test_dataset)})
        print('Model accuracy: ' , correct / len(test_dataset))

    run.log({"Training loss": acc_loss / len(subset), "combination weight (1/m)": 1/m})
    model.load_state_dict(cur_params)


### Dummy test; check if global model is trained 
model.eval()
with torch.no_grad():
    correct = 0
    for data, label in DataLoader(test_dataset, batch_size=32):

        data, label = data.to(args.device), label.to(args.device)
        output = model(data)

        pred = torch.argmax(output, dim=1)
        acc = torch.sum(pred == label)

        correct += acc

run.log({"test acc ": correct / len(test_dataset)})
print('Model accuracy: ' , correct / len(test_dataset))







