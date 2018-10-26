"""model"""
import math
import re
import sys
import numpy as np

import param

class Model:
    """パラメータ初期化"""

    def __init__(self, wordVecs, Dict, Atom, vocab_len, vec_len, factor):
        self.wordVecs = wordVecs
        self.dict = Dict  # (L, K)
        self.atom = Atom  # (V, K)
        self.vocab_len = vocab_len  # V
        self.vec_len = vec_len  # L
        self.factor = factor
        self.avg_err = 1
        self.prev_avg_err = 0

    """Dとa_iの内積を計算する"""
    def PredictVector(self, key):
        # word2vec[key] : (1, L)
        # self.dict : (L, K)
        # self.atom[key] : (1, K)
        # 内積 : (1, L)
        return np.dot(self.atom[key], self.dict.T)

    """train (adaptiveな処理)"""
    def train(self, l1_reg, l2_reg, numIter):
        # 指定のnumIter回以下の処理を繰り返す
        for time in range(1, numIter):
            num_words = 0
            total_error = 0
            print("time : {}".format(time))
            # 学習回数の制限
            # if time < 20 or (self.avg_err > 0.05 and time < 50 and abs(self.avg_err - self.prev_avg_err) > 0.005):
            # adaptiveな手続き, A[key]を対象
            for key in self.wordVecs.keys():
                """error算出"""
                # 更新単語数
                num_words += 1
                # predict i-th word, DとAの内積を計算, (1, L)
                pred_vec = self.PredictVector(key)
                # true_vec - pred_vecの復元誤差, (1, L)
                diff_vec = self.wordVecs[key] - pred_vec
                # np.sqrt(np.sum(np.abs(diff_vec**2)))
                error = np.linalg.norm(diff_vec, ord=2)

                """error更新"""
                # keyまでのtotal_error
                total_error += error
                # keyまでのerrorの平均値
                self.avg_err = total_error / num_words
                if num_words % 50 == 0:
                    print("prev_avg_err - avg_err : {}".format(self.prev_avg_err - self.avg_err))
                self.prev_avg_err = self.avg_err

                """AとDの更新"""
                Optimizer = param.Param(self.atom, self.dict, self.vocab_len,
                                        self.vec_len, self.factor, key, diff_vec, error, l1_reg, l2_reg)
                # 最適化
                Optimizer.UpdateParams()
                self.atom = Optimizer.atom
                self.dict = Optimizer.dict
            # else:  # 処理終了
            #     break
