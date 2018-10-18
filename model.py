"""model"""
import math
import re
import sys
import numpy as np

import param

class Model:
    """パラメータ初期化"""
    def __init__(self, factor, vector_len, vocab_len):
        # word_vecs : (vocab_len, vec_len)
        # vec_len, len(word_vecs[0])
        self.vec_len = vector_len
        # vocab_len, len(word_vecs)
        self.vocab_len = vocab_len
        # factor
        self.factor = factor

        # vectorの初期化
        # A : (factor * vec_len, vec_len)
        self.atom = np.random.rand(self.factor * self.vec_len, self.vocab_len)
        # D : (vocab_len, factor * vec_len)
        self.dict = np.random.rand(self.vocab_len, self.factor * self.vec_len)

    """Dとa_iの内積を計算する"""
    def PredictVector(self, word_index):
        return np.dot(self.dict, self.atom[word_index])

    """train (adaptiveな処理)"""
    def train(self, wordVecs, newVecs, numIter, l1_reg, l2_reg, factor, vocab):
        # lossの初期化
        ave_error = 1
        prev_avg_err = 0
        pred_vec = wordVecs

        # 指定のnumIter回以下の処理を繰り返す
        for time in range(numIter):
            num_words = 0  # 更新した単語数
            total_error = 0  # total_error
            atom_l1_norm = 0  # AのL1ノルム
            # 学習回数の制限
            if time < 20 or (ave_error > 0.05 and time < 50 and abs(ave_error - prev_avg_err) > 0.005):
                # adaptiveな手続き
                for word_id in range(len(wordVecs)):
                    # predict i-th word, DとAの内積を計算
                    pred_vec = self.PredictVector(word_id)
                    # compute error
                    diff_vec = wordVecs[word_id] - pred_vec
                    error = np.linalg.norm(diff_vec, ord=1)
                    total_error += error
                    # 更新単語数
                    num_words += 1
                    # AのL1ノルムの更新
                    atom_l1_norm += np.linalg.norm(self.atom[word_id], ord=1)
                    # 損失の平均値
                    avg_error = total_error / num_words
                    prev_avg_err = avg_error
                    print("word_id : {}, Error per example : {}".format(word_id, error))
                    print("Ave error : {}".format(avg_error))
                    print("atom_l1_norm : {}".format(atom_l1_norm))
                    print("Avg atom_l1_norm : {}".format(atom_l1_norm / num_words))
                    print("dict_l2_norm : {}".format(np.linalg.norm(vocab, ord=2)))
                    # AとDの更新
                    Optimizer = param.Param(
                        pred_vec, self.dict, word_id, num_words, diff_vec, l1_reg, l2_reg)
                    Optimizer.UpdateParams()
            else:  # 処理終了
                break
