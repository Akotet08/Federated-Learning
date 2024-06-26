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
from model import CNNLSTM

torch.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='pipline')

parser.add_argument('--device', '-d', type=int, default=0, help="Which gpu to use; default cpu")
parser.add_argument('--batch_size', type=int, default=16, help="batch size")
parser.add_argument('--epoch', type=int, default=100, help="epoch")
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-p', type=float, default=1, help='modality sampling')
parser.add_argument('-drop', type=int, default=-1, help='id of modality to drop')


args = parser.parse_args()
args.device = 'cpu' if args.device < 0 else torch.device(f"cuda:{args.device}")
args.drop = None if args.drop < 0 else args.drop

print(args)

def get_windowed_features(curr_df):
    WIN_LEN = 40 # 10 secs
    FEAT_IDX_START = 0
    FEAT_IDX_END = 9

    windowed_features = []
    windowed_labels = []
    window_start_idx= 0
    while window_start_idx < len(curr_df):
        window_end_idx = (window_start_idx) + WIN_LEN
        if window_end_idx >= len(curr_df):
            break
        # print(f'[{idx}/N], window_start_idx: {window_start_idx}, window_end_idx: {window_end_idx}')
        feature_window = curr_df.iloc[window_start_idx:window_end_idx, FEAT_IDX_START:FEAT_IDX_END + 1].values
        feature_window = feature_window.T
        
        # define lables for window
        label_window= curr_df.iloc[window_start_idx:window_end_idx, :]['label']
        if label_window.nunique() == 1:
            windowed_features.append(feature_window)
            windowed_labels.append(label_window.iloc[0])
        window_start_idx = window_end_idx

    return np.array(windowed_features), np.array(windowed_labels)

test_path_dir = '../../data/wesad_processed/wesad_test_scaled.csv'
test_df = pd.read_csv(test_path_dir)
def load_dataset(drop_index = None):    
    X_test, y_test = [], []

    clients = test_df['user_id'].unique()
    for cli in clients:
        df_test = test_df[test_df['user_id'] == cli]

        feat, label = get_windowed_features(df_test)
        X_test.append(feat)
        y_test.append(label)
    
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_test = y_test.astype(np.uint8)

    if drop_index != None:
        X_test[:, drop_index] = 0

    return X_test, y_test

# run = wandb.init(project='Experiment (single det. drop third) ', 
#                  config= {key: value for key, value in vars(args).items()})

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

# save model
path = 'models/'
batch_size = args.batch_size
train_on_gpu = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()

### for modeling Test time only removal
stats = []
for modalitiy in range(10):
    X_test, y_test = load_dataset(None) 
    
    val_losses = []
    accuracy=0
    f1score=0

    for i in range(5):
        net = CNNLSTM()
        net.load_state_dict(torch.load(path + f'removed_{None}_{i}'))
        net = net.to(args.device)
        val_h = net.init_hidden(batch_size)
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
                
    print("Val Loss: {:.4f} ...".format(np.mean(val_losses)),
    "Val Acc: {:.4f} ...".format(accuracy/(len(X_test)//batch_size)/5),
    "F1-Score: {:.4f} ...".format(f1score/(len(X_test)//batch_size)/5),
    'removed modality during test {}'.format(test_df.columns[modalitiy]))

    stats.append({'Val loss' : np.mean(val_losses), 
            'Val acc': (accuracy/(len(X_test)//batch_size)/5), 
            'F1-Score': (f1score/(len(X_test)//batch_size)/5), 
            'removed modality during test': test_df.columns[modalitiy]})

# data = [[stat['removed modality during test'], stat['F1-Score']] for  stat in stats]
# table = wandb.Table(data=data, columns=["removed modality", "F1-Score"])
# run.log(
#     {
#         "train_test": wandb.plot.bar(
#             table, "removed modality", "F1-Score", title="Only Train time removal"
#         )
#     }
# )