"""
!user/bin/env python
Author: Peng Hao
Email: peng.hao@student.uts.edu.au
Purpose: This file is used to implement the algorithm proposed in:
         'Top-N Recommendation on Graphs',
         by Zhao Kang, et. al. at CIKM 2016.
"""

import random
import math
import heapq
import numpy as np
import scipy.sparse as ssp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import solve_sylvester
random.seed(12345)


class GraphRec(object):
    def __init__(self, alpha=1e-4, beta=1e-5):
        # Regularization for user graph
        self.alpha = alpha
        # Regularization for item graph
        self.beta = beta
        # Top-N
        self.top_n = 10

    def fit(self, train):
        """
        Train the model.

        Parameters
        -------------
        train: 2D sparse training array, shape of (num of users, num of items)
        """
        num_user, num_item = train.shape
        # Compute inter-user similarity
        user_sim = cosine_similarity(train, dense_output=False)
        # Build user graph Laplacian matrix
        L_user = ssp.spdiags(data=user_sim.sum(axis=1).A1, diags=0, m=num_user, n=num_user,
                            format='csr') - user_sim.tocsr()
        # Compute inter-item similarity
        item_sim = cosine_similarity(train.T, dense_output=False)
        # Build item graph Laplacian matrix
        L_item = ssp.spdiags(data=item_sim.sum(axis=1).A1, diags=0, m=num_item, n=num_item,
                           format='csr') - item_sim.tocsr()
        # Solve Sylvester equation (AX + XB = Q)
        predictions = solve_sylvester(np.array((self.beta * L_user + np.eye(num_user))),
                                      self.alpha * L_item.toarray(),
                                      train.toarray())
        return predictions

    def evaluate(self, predictions, train, test):
        hr = ndcg = []
        for u in xrange(test.shape[0]):
            map_item_score = {}
            test_items = test.tocsr()[u].indices
            if len(test_items):
                for gtItem in test_items:
                    # Get the score of the test item first
                    maxScore = predictions[u, gtItem]
                    # Early stopping if there are K items larger than maxScore.
                    countLarger = 0
                    # For each user, all unrated items, including test item, are considered to be
                    # ranked
                    for i in set(np.arange(predictions.shape[1])) - set(train.tocsr()[u].indices):
                        early_stop = False
                        score = predictions[u, i]
                        map_item_score[i] = score

                        if score > maxScore:
                            countLarger += 1
                        if countLarger > self.top_n:
                            hr.append(0)
                            ndcg.append(0)
                            early_stop = True
                            break
                    # Generate topK rank list
                    if not early_stop:
                        ranklist = heapq.nlargest(self.top_n, map_item_score, key=map_item_score.get)
                        one_hr = self.getHitRatio(ranklist, gtItem)
                        one_ndcg = self.getNDCG(ranklist, gtItem)
                        hr.append(one_hr)
                        ndcg.append(one_ndcg)
        return np.mean(hr)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in xrange(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0


