"""
!user/bin/env python
Author: Peng Hao
Email: peng.hao@student.uts.edu.au

This demo is used to test graph-based recommender system.
"""

import os
import numpy as np
import subprocess
import scipy.sparse as ssp
from dual_graph_rec import GraphRec
from data_loader import loadrating_Kholdout


DATA_DIR = './movielens/'
SIZE = '100k'


def run_movielens100k():
    # Download data from GroupLens website
    if not os.path.isdir('./movielens'):
        subprocess.call(['./download_data.sh'])
    for i in xrange(1, 6):
        # Load train data
        train = ssp.lil_matrix((943, 1682))
        with open(DATA_DIR + 'ml-{0}/u{1}.base'.format(SIZE, i), 'r') as f:
            for line in f:
                data = line.split('\t')
                user_id = int(data[0])
                item_id = int(data[1])
                rating = float(data[2])
                train[user_id - 1, item_id - 1] = rating
        # Load test data
        test = ssp.lil_matrix((943, 1682))
        with open(DATA_DIR + 'ml-{0}/u{1}.test'.format(SIZE, i), 'r') as f:
            for line in f:
                data = line.split('\t')
                user_id = int(data[0])
                item_id = int(data[1])
                rating = float(data[2])
                test[user_id - 1, item_id - 1] = rating
        # Check non-overlapping
        assert train.multiply(test).nnz == 0.

        # Call GraphRec to run
        graphrec = GraphRec()
        predictions = graphrec.fit(train)
        hit_rate = graphrec.evaluate(predictions, train, test)
        print("Fold-{0}: hit-rate={1:.4f}".format(i, hit_rate))


def run_filmtrust():
    file_path = './filmtrust/ratings.txt'
    hr = []
    # Run five times, each time with Leave-One-Out
    for i in range(1, 6):
        train, test = loadrating_Kholdout(file_path, k_hold_out=1)
        # Call GraphRec to run
        graphrec = GraphRec()
        predictions = graphrec.fit(train)
        hit_rate = graphrec.evaluate(predictions, train, test)
        print("Fold-{0}: hit-rate={1:.4f}".format(i, hit_rate))
        hr.append(hit_rate)
    ave_hr = np.mean(hr)
    print("Average Hit-rate is: {:.4f}".format(ave_hr))


if __name__ == '__main__':
    # run_movielens100k()
    run_filmtrust()
