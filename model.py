"""model"""
import math
import re
import sys
import numpy as np

import param
from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

class Model:
    """パラメータ初期化"""
    def __init__(self, wordVecs, vocab_len, vec_len):
        self.wordVecs = wordVecs
        self.vocab_len = vocab_len  # V
        self.vec_len = vec_len  # L
        self.avg_err = 1
        self.prev_avg_err = 0

        """AとDの作成"""
        keys = list(wordVecs.keys())
        # A : (V, K), A[key] : (K,)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len*factor)
        self.atom = {}
        for key in keys:
            self.atom[key] = (0.6*(1/np.sqrt(factor*vec_len))) * np.random.randn(1, factor*vec_len)
        # D : (L, K)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len+vec_len*factor))
        self.dict = (0.6*(1/np.sqrt(self.vec_len + factor*vec_len))) * np.random.randn(vec_len, factor * vec_len)

    """Dとa_iの内積を計算する"""
    def PredictVector(self, key):
        # word2vec[key] : (1, L)
        # self.dict : (L, K)
        # self.atom[key] : (1, K)
        # 内積 : (1, L)
        return np.dot(self.atom[key], self.dict.T)

    """Sparse_Overfitting (adaptiveな処理)"""
    def Sparse_Overfitting(self):
        """param"""
        Optimizer = param.Param(self.atom, self.dict, self.vocab_len, self.vec_len)
        # 指定のnumIter回以下の処理を繰り返す
        for time in range(1, numIter):
            num_words = 0
            total_error = 0
            atom_l1_norm = 0
            print("time : {}".format(time))
            # adaptiveな手続き, A[key]を対象
            for key in self.wordVecs.keys():
                """error算出"""
                # 更新単語数
                num_words += 1
                # predict i-th word, DとAの内積を計算, (1, L)
                pred_vec = self.PredictVector(key)
                # true_vec - pred_vecの復元誤差, (1, L)
                diff_vec = self.wordVecs[key] - pred_vec

                """AとDの更新"""
                Optimizer.UpdateParams(time, key, diff_vec)
                self.atom = Optimizer.atom
                self.dict = Optimizer.dict
                """error更新"""
                atom_l1_norm += np.linalg.norm(self.atom[key], ord=1)
                error = np.linalg.norm(diff_vec, ord=2)
                total_error += error

            self.prev_avg_err = self.avg_err
            self.avg_err = total_error / num_words
        
            print("Error per example : {}".format(self.avg_err))
            print("Dict L2 norm : {}".format(np.linalg.norm(self.dict, ord=2)/num_words))
            print("Avg Atom L1 norm : {}\n".format(atom_l1_norm/num_words))

