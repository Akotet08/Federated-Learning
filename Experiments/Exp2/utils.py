import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset

def wesad_get_groups(df, method='simple', ngroup=3):
    '''
    Input: train or test DataFrame
    Return: Dictionary of groups and the corresponding index
    '''
    if method != 'simple':
        raise NotImplementedError

    groups = {g: [] for g in range(ngroup)}
    users = df['user_id'].unique()
    labels = df['label'].unique()

    for user in users:
        df_user = df[df['user_id'] == user]

        for l in labels:
            df_user_label = df_user[df_user['label'] == l]
            df_user_label_index = list(df_user_label.index)
            total_samples = len(df_user_label_index)
            nsamples = total_samples // ngroup
            leftover = total_samples % ngroup
                        
            start_index = 0
            for g in range(ngroup):
                if g < leftover:
                    # Give extra sample to this group
                    end_index = start_index + nsamples + 1
                else:
                    end_index = start_index + nsamples

                selected_indices = df_user_label_index[start_index:end_index]
                groups[g].extend(selected_indices)
                start_index = end_index

    return groups

def simulate_missing_modality(groups, p=0.5, run_idx = 1):
    def bernoulli_missingness(seed, size= 1, p = 0.5):
        np.random.seed(seed)
        return np.random.binomial(
            size = size,
            n = 1,
            p = p
        )


    group_info = {g : {} for g in groups.keys()}

    for group in groups.keys():
        group_info[group]['indexes'] = groups[group]

        # missingness for each modality (10)
        miss =list(bernoulli_missingness((group + 1) * run_idx, size=10, p=p))

        group_info[group]['missing_modalities'] = miss
    
    return group_info

def client_datasets(p = 0.5, run_idx = 1, ngroups=2):
    train_df = pd.read_csv('/home/akotet/FL/data/wesad_processed/wesad_train_grouped.csv')
    test_df = pd.read_csv('/home/akotet/FL/data/wesad_processed/wesad_test_grouped.csv')

    clients = train_df['user_id'].unique()

    dic = {}

    for client in clients:
        client_train_df = train_df[train_df['user_id'] == client].reset_index(drop=True)
        client_test_df = test_df[test_df['user_id'] == client].reset_index(drop=True)

        train_groups = wesad_get_groups(client_train_df, ngroup=ngroups)
        test_groups = wesad_get_groups(client_test_df, ngroup=ngroups)

        train_groups_info = simulate_missing_modality(train_groups, run_idx=run_idx, p=p)
        test_groups_info = simulate_missing_modality(test_groups, run_idx=run_idx, p=p)

        y_train = client_train_df['label'].to_numpy()  
        X_train = client_train_df.drop(columns=['label', 'user_id'], axis=1).to_numpy()
        y_test = client_test_df['label'].to_numpy()
        X_test = client_test_df.drop(columns=['label', 'user_id'], axis=1).to_numpy()

        X_train = X_train.astype(np.float32).reshape(-1, 10, 40)
        X_test = X_test.astype(np.float32).reshape(-1, 10, 40)

        for g in train_groups_info.keys():
            idxes  = train_groups_info[g]['indexes']
            missing = train_groups_info[g]['missing_modalities']

            #simulate missing modalities from train
            for m in range(len(missing)):
                if missing[m] == 1 and g != 0: # g != 0 to create full dataset for first group 
                    X_train[idxes, m, :] = 0

        # for g in test_groups_info.keys():
        #     idxes  = test_groups_info[g]['indexes']
        #     missing = test_groups_info[g]['missing_modalities']

        #     #simulate missing modalities from train
        #     for m in range(len(missing)):
        #         if missing[m] == 1 and g!=0:
        #             X_test[idxes, m, :] = 0

        # print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))
        # The targets are casted to int8 for GPU compatibility.
        y_train = y_train.astype(np.uint8)
        y_test = y_test.astype(np.uint8)

        # create dataset
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train).long()

        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test).long()

        train_dataset = (X_train, y_train)
        test_dataset =  (X_test, y_test)

        dic[client] = [train_dataset, test_dataset]
    
    return dic


        