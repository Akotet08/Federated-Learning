import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import torch.optim as opt
from tqdm import tqdm
import wandb
import sklearn.metrics as metrics
import os
from utils import wesad_get_groups, simulate_missing_modality, client_datasets
import copy

torch.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='pipline')

parser.add_argument('--device', '-d', type=int, default=0, help="Which gpu to use; default cpu")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-p', type=float, default=1, help='modality sampling')
parser.add_argument('-b', type=int, default=16, help="local batch size")
parser.add_argument('-drop', type=int, default=-1, help='id of modality to drop')
parser.add_argument('-e', type=int, default=100, help="local epoch")
parser.add_argument('--round', '-r', type=int, default=10, help="number of rounds to execute")
parser.add_argument('-k', type=int, default=10, help='number of client machines running for each round')

args = parser.parse_args()
args.device = 'cpu' if args.device < 0 else torch.device(f"cuda:{args.device}")
args.drop = None if args.drop < 0 else args.drop

print(args)

run = wandb.init(
    project='FL wesad (missing in test only)',
    config = {key: value for key, value in vars(args).items()}
)

def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class CNNLSTM(nn.Module):
    def __init__(self, n_hidden=64, n_layers=1, n_filters=64, 
                 n_classes=3, filter_size=5, drop_prob=0.5):
        super(CNNLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
             
        self.conv1 = nn.Conv1d(10, n_filters, filter_size, padding=2)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=2)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=2)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size, padding=2)
        
        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        
        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x, hidden, batch_size):
        # x = x.view(-1, 10, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.shape[-1], -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)
        
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x)
        
        out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
        return out, hidden
    
    def init_hidden(self, batch_size, gpu=True):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(args.device),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(args.device))
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

class Client():
    def __init__(self, client_number, datasets, model, B, E, lr = 0.01, device='cpu'):
        self.client_number = client_number
        self.local_batch_size = B if B != 0 else len(datasets[1])
        self.local_epoch = E
        self.lr = lr
        self.device = device
        self.model = model.to(self.device)

        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets
        self.nk = len(self.X_train)     

        self.criterion = torch.nn.CrossEntropyLoss()
        
        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.local_batch_size, shuffle=True)
        # self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.local_batch_size, shuffle=True)
    
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=True):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def update(self, state_dict):
        self.model.load_state_dict(state_dict)

        optimizer = opt.SGD(self.model.parameters(), lr=self.lr)  
        pbar = tqdm(range(self.local_epoch), leave=False)
        h = self.model.init_hidden(self.local_batch_size) 
        train_loss = 0
        val_losses = []
        f1_scores = []
        for _ in pbar:
            self.model.train()
            for data, label in self.iterate_minibatches(self.X_train, self.y_train, self.local_batch_size):
                optimizer.zero_grad()

                h = tuple([each.data for each in h])

                data, label = data.to(self.device), label.to(self.device)

                output, h = self.model(data, h, self.local_batch_size)

                loss = self.criterion(output, label)
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()

            val_h = self.model.init_hidden(self.local_batch_size)
            accuracy = 0
            self.model.eval()
            with torch.no_grad():
                for batch in self.iterate_minibatches(self.X_test, self.y_test, self.local_batch_size):
                    inputs, targets = batch     

                    val_h = tuple([each.data for each in val_h])
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                    output, val_h = self.model(inputs, val_h, self.local_batch_size)

                    val_loss = self.criterion(output, targets)
                    val_losses.append(val_loss.item()) 

                    _, top_class = output.topk(1, dim=1)
                    equals = top_class == targets.view(*top_class.shape).long()
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    f1_scores.append(metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(), average='macro'))

        assert len(val_losses) != 0
        assert len(f1_scores) != 0
        return self.model.state_dict(), (np.mean(val_losses)), np.mean(f1_scores)

def initialize_clients(p = 0.5, run_idx=1):
    CLIENT_LIST = []

    dataset_dic = client_datasets(p = p, run_idx=run_idx)

    for key in dataset_dic.keys():
        mdl = CNNLSTM()
        init_weights(mdl)
        client = Client(key, dataset_dic[key], mdl, args.b, args.e, args.lr, args.device)
        CLIENT_LIST.append(client)
    
    return CLIENT_LIST, len(CLIENT_LIST)


global_model = CNNLSTM()
init_weights(global_model)

global_model = global_model.to(args.device)
clients, nclients = initialize_clients(args.p, run_idx=1)

pbar = tqdm(range(args.round))
logs = {c.client_number: ([], []) for c in clients}

for round_num in pbar:
    subset = np.random.choice(nclients, args.k, replace=False)
    old_params = copy.deepcopy(global_model.state_dict())
    cur_params = {key: torch.zeros_like(value) for key, value in old_params.items()}

    tot_nk = sum([clients[idx].nk for idx in subset])

    acc_loss = 0
    acc_f1score = 0
    for idx in tqdm(subset, leave=False):
        client = clients[idx]

        new_params, val_loss, f1_score = client.update(old_params)
        acc_loss += val_loss   
        acc_f1score += f1_score

        logs[client.client_number][0].append(val_loss)
        logs[client.client_number][1].append(f1_score)

        for key in new_params.keys():
            cur_params[key] += (clients[idx].nk / tot_nk) * new_params[key].type_as(cur_params[key])

    global_model.load_state_dict(cur_params)
    print(f"Round: {round_num+1}/{args.round}... \tVal Loss: {acc_loss / len(subset):.4f}...\tF1-Score: {acc_f1score / len(subset):.4f}...")

data = [[client.client_number, np.mean(logs[client.client_number][0]), np.mean(logs[client.client_number][1])] for client in clients]
table = wandb.Table(data=data, columns=["client name", "val Loss", "F1-Score"])
run.log({'perfomance': table})
# print(table.get_dataframe())
# run.log(
#     {
#         "client performace": wandb.plot.bar(
#             table, "client name", "val Loss", "F1-Score", title="Performace"
#         )
#     }
# )