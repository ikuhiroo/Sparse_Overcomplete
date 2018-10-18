"""model"""
import math
import re
import sys
import numpy as np

import param

class Model:
    """パラメータ初期化"""
    def __init__(self, wordVecs, Dict, Atom, factor, vec_len, vocab_len):
        self.vec_len = vec_len
        self.vocab_len = vocab_len
        self.factor = factor
        self.wordVecs = wordVecs
        self.atom = Atom
        self.dict = Dict

    """Dとa_iの内積を計算する"""
    def PredictVector(self, key):
        return np.dot(self.dict, self.atom[key])

    """train (adaptiveな処理)"""
    def train(self, numIter, l1_reg, l2_reg):
        # lossの初期化
        ave_error = 1
        prev_avg_err = 0

        # 指定のnumIter回以下の処理を繰り返す
        for time in range(numIter):
            num_words = 0  # 更新した単語数
            total_error = 0  # total_error
            atom_l1_norm = 0  # AのL1ノルム
            # 学習回数の制限
            # if time < 20 or (ave_error > 0.05 and time < 50 and abs(ave_error - prev_avg_err) > 0.005):
                # adaptiveな手続き
            for key in self.wordVecs.keys():
                # 更新単語数
                num_words += 1
                # predict i-th word, DとAの内積を計算
                pred_vec = self.PredictVector(key)  # (300,)
                # compute error
                diff_vec = self.wordVecs[key] - pred_vec  # (300,)
                error = np.linalg.norm(diff_vec, ord=2)
                total_error += error
                # AのL1ノルムの更新
                atom_l1_norm += np.linalg.norm(self.atom[key], ord=1)
                # 損失の平均値
                avg_error = total_error / num_words
                # print("word_id : {}, Error per example : {}".format(key, error))
                print("error : {}, Ave_error : {}".format(error, avg_error))
                # print("atom_l1_norm : {}".format(atom_l1_norm))
                # print("Avg atom_l1_norm : {}".format(atom_l1_norm / num_words))
                # print("dict_l2_norm : {}".format(np.linalg.norm(self.dict, ord=2)))

                # AとDの更新
                Optimizer = param.Param(
                    time, self.atom, self.dict, key, num_words, diff_vec, error, l1_reg, l2_reg)
                Optimizer.UpdateParams()
            # else:  # 処理終了
            #     break
