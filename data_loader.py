"""
!user/bin/env python
Author: Peng Hao
Email: peng.hao@student.uts.edu.au
Purpose: This file is used to load original data,
         and split it into train and test sets.
"""

import scipy.sparse as ssp
import pandas as pd
import numpy as np


def loadrating_Kholdout(file_path, k_hold_out):
    """
    Load rating data into train,
    for each user, hold one observed rating data as test.
    """
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_csv(file_path, names=names, delim_whitespace=True)
    df_table = pd.pivot_table(df, index='user_id', columns='item_id', values='rating')
    # Convert NaN to 0
    df_table = df_table.fillna(0)
    data = df_table.values

    train = ssp.lil_matrix(data)
    test = ssp.lil_matrix(data.shape)

    # Split train and test
    for u in range(train.shape[0]):
        i = np.random.choice(train.tocsr().indices, k_hold_out)
        train[u, i] = 0.
        test[u, i] = data[u, i]

    assert train.multiply(test).nnz == 0
    return train, test


