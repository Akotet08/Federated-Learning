import pandas as pd
import numpy as np


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