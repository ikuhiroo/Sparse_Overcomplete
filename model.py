"""model"""
import math
import re
import sys
import numpy as np

import param

class Model:
    """パラメータ初期化"""

    def __init__(self, wordVecs, Dict, Atom):
        self.wordVecs = wordVecs
        self.dict = Dict
        self.atom = Atom
        self.avg_err = 1
        self.prev_avg_err = 0

    """Dとa_iの内積を計算する"""
    def PredictVector(self, key):
        return np.dot(self.dict, self.atom[key])

    """train (adaptiveな処理)"""
    def train(self, l1_reg, l2_reg, numIter):
        # 指定のnumIter回以下の処理を繰り返す
        for time in range(1, numIter):
            num_words = 0  # time回目に更新した単語数
            total_error = 0  # time回目のtotal_error
            # atom_l1_norm = 0  # AのL1ノルム
            # dict_l2_norm = 0  # DのL2ノルム
            print("time : {}".format(time))
            # 学習回数の制限
            if time < 20 or (self.avg_err > 0.05 and time < 50 and abs(self.avg_err - self.prev_avg_err) > 0.005):
                # adaptiveな手続き, A[key]を対象
                for key in self.wordVecs.keys():
                    """error算出"""
                    # 更新単語数
                    num_words += 1
                    # predict i-th word, DとAの内積を計算, (300,)
                    pred_vec = self.PredictVector(key)
                    # compute error, true_vec - pred_vecの復元誤差
                    diff_vec = self.wordVecs[key] - pred_vec
                    # diff_vecのL2ノルム
                    error = np.linalg.norm(diff_vec, ord=2)

                    """error更新"""
                    # A[key]のL1ノルム
                    # atom_l1_norm += np.linalg.norm(self.atom[key], ord=1)
                    # DのL2ノルム
                    # dict_l2_norm += np.linalg.norm(self.dict, ord=2)
                    # time回目のtotal_error
                    total_error += error
                    # time回目のerrorの平均値
                    self.avg_err = total_error / num_words
                    print("num_words : {}, prev_avg_err - avg_err : {}".format(
                        num_words, self.prev_avg_err - self.avg_err))
                    self.prev_avg_err = self.avg_err

                    """AとDの更新"""
                    Optimizer = param.Param(time, self.atom, self.dict, key, num_words, diff_vec, error, l1_reg, l2_reg)
                    # 最適化
                    Optimizer.UpdateParams()
                    self.atom = Optimizer.atom
                    self.dict = Optimizer.dict
            else:  # 処理終了
                break
