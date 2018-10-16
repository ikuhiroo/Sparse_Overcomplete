"""model"""
import math
import re
import sys
import numpy as np

class Model:
    """パラメータ初期化"""
    def __init__(self, factor, vector_len, vocab_len):
        # vec_len
        self.vec_len = vector_len
        # vocab_len
        self.vocab_len = vocab_len
        # A : (factor * vec_len, vec_len)
        # A の各要素は (factor * vec_len, 1)
        self.atom = []
        # D : (vector_len, factor * vec_len)
        self.dict = []

    """Dとa_iの内積を計算する"""
    def PredictVector(self, word_vec, word_index):
        return np.dot(self.dict, self.atom[word_index])

    def AdagradUpdate(self, rate, grad):
        _del_grad += cwiseAbs2(grad)
        _grad_sum += grad
        var -= rate * cwiseQuotient(grad, cwiseSqrt(_del_grad))

    """勾配を求め，Adagradでパラメータを更新する"""
    def UpdateParams(self, word_id, RATE, diff_vec, l1_reg, l2_reg):
        # D に関する勾配値
        dict_grad = (-2) * np.dot(diff_vec, self.atom[word_id].T) + 2 * l2_reg * self.dict
        # AdagradUpdate
        self.AdagradUpdate(RATE, dict_grad)
        # A の勾配
        atom_elem_grad = (-2) * self.dict.T * diff_vec
        # AdagradUpdateWithL1Reg
        atom[word_index].AdagradUpdateWithL1Reg(rate, atom_elem_grad, l1_reg)

    # それぞれの要素の二乗和を求める
    def squaredNorm(self):
        pass

    # L2ノルムを計算
    def lpNorm(self, dict, p):
        pass

    """train
    ・word_vecs : init vectors A
    ・outfilename : newvec
    ・numIter : 20
    ・l1_reg : 正則化係数
    ・l2_reg : 正則化係数
    ・factor : newvecの要素数の要因
    ・vocab : 辞書 D

    adaptiveな処理
    """
    def train(self, wordVecs, outFileName, numIter, l1_reg, l2_reg, factor, vocab):
        # lossの初期化
        ave_error = 1
        prev_avg_err = 0
        pred_vec = wordVecs
        # 指定のnumIter回以下の処理を繰り返す
        for time in range(numIter):
            num_words = 0  # 更新した単語数
            total_error = 0  # loss
            atom_l1_norm = 0  # A の L1ノルム
            RATE = 0.05
            # 学習回数の制限
            if time < 20 or (ave_error > 0.05 and time < 50 and abs(ave_error - prev_avg_err) > 0.005):
                for word_id in range(len(wordVecs)):
                    # predict i-th word and compute error
                    pred_vec = self.PredictVector(wordVecs[word_id], word_id)
                    diff_vec = wordVecs[word_id] - pred_vec
                    # (2)の第1項目の値
                    error = diff_vec.squaredNorm()
                    # (2)の第1項目のsum
                    total_error += error
                    # 対象単語の更新
                    num_words += 1
                    # atom_l1_normの更新
                    atom_l1_norm += self.lpNorm(self.atom[word_id], 1)
                    self.atom, self.dict = self.UpdateParams(word_id, RATE, diff_vec, l1_reg, l2_reg)
                    # 損失の平均値
                    prev_avg_err = avg_error
                    avg_error = total_error / num_words
                    print("Error per example : {}".format(avg_error))
                    print("Dict L2 norm : {}".format(self.lpNorm(vocab, 2)))
                    print("Avg Atom L1 norm : {}".format(atom_l1_norm / num_words))
            else:  # 処理終了
                break
