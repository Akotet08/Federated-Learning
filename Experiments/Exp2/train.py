import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import torch.optim as optim
from tqdm import tqdm
import wandb
import sklearn.metrics as metrics
import os
from utils import wesad_get_groups, simulate_missing_modality

torch.manual_seed(42)
np.random.seed(42)


def parse_arguments():
    parser = argparse.ArgumentParser(description='pipline')

    parser.add_argument('--device', '-d', type=int, default=0, help="Which gpu to use; default cpu")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epoch', type=int, default=500, help="epoch")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-p', type=float, default=1, help='modality sampling')
    parser.add_argument('-drop', type=int, default=-1, help='id of modality to drop')


    args = parser.parse_args()
    args.device = 'cpu' if args.device < 0 else torch.device(f"cuda:{args.device}")
    args.drop = None if args.drop < 0 else args.drop

    print(args)

    return args

def load_dataset(run_idx=1, ngroups = 2, p=0.5):    
    '''
        train and test_grouped, preprocessed data with 10 sec non overlapping window
    '''
    train_df = pd.read_csv('/home/akotet/FL/data/wesad_processed/wesad_train_grouped.csv')
    test_df = pd.read_csv('/home/akotet/FL/data/wesad_processed/wesad_test_grouped.csv')

    train_groups = wesad_get_groups(train_df, ngroup=ngroups)
    test_groups = wesad_get_groups(test_df, ngroup=ngroups)

    train_groups_info = simulate_missing_modality(train_groups, run_idx=run_idx, p=p)
    test_groups_info = simulate_missing_modality(test_groups, run_idx=run_idx, p=p)

    y_train = train_df['label'].to_numpy()  
    X_train = train_df.drop(columns=['label', 'user_id'], axis=1).to_numpy()
    y_test = test_df['label'].to_numpy()
    X_test = test_df.drop(columns=['label', 'user_id'], axis=1).to_numpy()

    X_train = X_train.astype(np.float32).reshape(-1, 10, 40)
    X_test = X_test.astype(np.float32).reshape(-1, 10, 40)

    # for g in train_groups_info.keys():
    #     idxes  = train_groups_info[g]['indexes']
    #     missing = train_groups_info[g]['missing_modalities']

    #     #simulate missing modalities from train
    #     for m in range(len(missing)):
    #         if missing[m] == 1 and g != 0: # g != 0 to create full dataset for first group 
    #             X_train[idxes, m, :] = 0

    for g in test_groups_info.keys():
        idxes  = test_groups_info[g]['indexes']
        missing = test_groups_info[g]['missing_modalities']

        #simulate missing modalities from train
        for m in range(len(missing)):
            if missing[m] == 1 and g!=0:
                X_test[idxes, m, :] = 0

    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))
    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

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

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
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

train_on_gpu = torch.cuda.is_available()

def train(net, epochs=10, batch_size=125, lr=0.01):
    
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    if(train_on_gpu):
        net.to(args.device)

    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)         
        train_losses = []    
        net.train()
        for batch in iterate_minibatches(X_train, y_train, batch_size):
            x, y = batch    

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if(train_on_gpu):
                inputs, targets = inputs.to(args.device), targets.to(args.device)

            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            
            # zero accumulated gradients
            opt.zero_grad()   
            
            # get the output from the model
            output, h = net(inputs, h, batch_size)
            
            loss = criterion(output, targets.long())
            train_losses.append(loss.item())
            loss.backward()
            opt.step()
            
        val_h = net.init_hidden(batch_size)
        val_losses = []
        accuracy=0
        f1score=0
        net.eval()
        with torch.no_grad():
            for batch in iterate_minibatches(X_test, y_test, batch_size):
                x, y = batch     

                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, targets = inputs.to(args.device), targets.to(args.device)
                    
                output, val_h= net(inputs, val_h, batch_size)

                val_loss = criterion(output, targets.long())
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == targets.view(*top_class.shape).long()
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(), average='weighted')
            
        net.train() # reset to train mode after iterationg through validation data
                
        # print("Epoch: {}/{}...".format(e+1, epochs),
        # "Train Loss: {:.4f}...".format(np.mean(train_losses)),
        # "Val Loss: {:.4f}...".format(np.mean(val_losses)),
        # "Val Acc: {:.4f}...".format(accuracy/(len(X_test)//batch_size)),
        # "F1-Score: {:.4f}...".format(f1score/(len(X_test)//batch_size)))

        run.log({'Train Loss': np.mean(train_losses), 
                 'Val loss' : np.mean(val_losses), 
                 'Val acc': accuracy/(len(X_test)//batch_size), 
                 'F1-Score': f1score/(len(X_test)//batch_size)})

    return ({'Train Loss': np.mean(train_losses), 
            'Val loss' : np.mean(val_losses), 
            'Val acc': accuracy/(len(X_test)//batch_size), 
            'F1-Score': f1score/(len(X_test)//batch_size)})


if __name__ == '__main__':
    args = parse_arguments()

    run = wandb.init(project=f'Experiment (centralized grouped) missing test only', 
        config= {key: value for key, value in vars(args).items()}
        )
    
    for i in range(1):
        X_train, y_train, X_test, y_test = load_dataset(run_idx=i*2000, p = args.p)

        model = CNNLSTM()
        model.apply(init_weights)   

        stat = train(model, epochs=args.epoch, batch_size=args.batch_size, lr=args.lr)
        print(stat)
        # # save model
        # path = 'models/'
        # torch.save(model.state_dict(), path + f'removed_{args.drop}_{i}')