"""Param update
・AdaGrad
学習が進むにつれて学習係数を小さくする手法．
adaptiveな手続きのため相性が良い．
"""
import math
import numpy as np
import re
import sys

# global変数の設定
EPSILON = 0.000001
RHO = 0.95

class Param:
    def __init__(self, A_vec, D_vec, word_id, num_words, diff_vec, l1_reg, l2_reg):
        self.atom = A_vec  # newvec
        self.dict = D_vec  # dict
        self.word_id = word_id  # word2vecのkey id
        self.num_words = num_words  # num_words
        self.diff_vec = diff_vec  # lossの第1項
        self.l1_reg = l1_reg  # AのL1ノルムの係数
        self.l2_reg = l2_reg  # DのL2ノルムの係数
        self.rate = 0.05  # 学習係数
        self.Totalgrads = 0 

    # ステップ関数
    def step_func(self, val):
        return 1 if val > 0 else 0

    # AdagradUpdate : Dを固定してAを更新
    def AdagradUpdate(self, grads):
        # hの初期化
        self.h = {}  # h[key] : (factor * vec_len, 1)
        for key, val in self.atom.items():
            self.h[key] = np.zeros_like(val)
        # hとAの更新
        for key in self.atom.keys():
            self.h[key] += grads[key] * grads[key]
            # 更新幅 : 0で割る可能性があるため1e-7を加える
            update = 1 / (np.sqrt(self.h[key]) + 1e-7)
            self.atom[key] -= (self.rate * grads[key]) * update

    # AdagradUpdateWithL1Reg : Aを固定してDを更新
    def AdagradUpdateWithL1Reg(self, grads):
        for key in self.atom.keys():
            # average gradient
            Avegrads = grads / self.num_words
            self.Totalgrads += np.linalg.norm(grads, ord=2)
            # λ = l1_reg
            diff = abs(Avegrads[key]) - self.l1_reg
            # l2_reg
            self.l2_reg = -(self.step_func(Avegrads[key]) *
                            self.rate * diff * self.num_words) / (np.sqrt(self.Totalgrads))
            # Aの更新
            if diff <= 0:
                self.atom[key] = 0
            else:
                self.atom[key] = self.l2_reg

    # AdagradUpdateWithL1RegNonNeg
    def AdagradUpdateWithL1RegNonNeg(self, grads):
        for key in self.atom.keys():
            # average gradient
            Avegrads = grads / self.num_words
            self.Totalgrads += np.linalg.norm(grads, ord=2)
            # λ = l1_reg
            diff = abs(Avegrads[key]) - self.l1_reg
            # l2_reg
            self.l2_reg = -(self.step_func(Avegrads[key]) *
                       self.rate * diff * self.num_words) / (np.sqrt(self.Totalgrads))
            # Aの更新
            if diff <= 0:
                self.atom[key] = 0
            elif self.l2_reg < 0:
                self.atom[key] = 0
            else:
                self.atom[key] = self.l2_reg

    def UpdateParams(self):
        # Dに関する勾配値 (Aは固定)
        dict_grad = (-2) * np.dot(self.diff_vec, self.atom[self.word_id].T) + 2 * self.l2_reg * self.dict
        # AdagradUpdate
        self.AdagradUpdate(dict_grad)
        # Aに関する勾配値 (Dは固定)
        atom_elem_grad = (-2) * self.dict.T * self.diff_vec
        # AdagradUpdateWithL1Reg
        self.AdagradUpdateWithL1Reg(atom_elem_grad)
        # self.AdagradUpdateWithL1RegNonNeg(atom_elem_grad)
