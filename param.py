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
    def __init__(self, time, Atom, Dict, key, num_words, diff_vec, error, l1_reg, l2_reg):
        self.time = time
        self.atom = Atom  # newvec
        self.dict = Dict  # dict
        self.key = key  # word2vecのkey id
        self.num_words = num_words  # num_words
        self.diff_vec = diff_vec  # lossの第1項
        self.error = error  # diff_vecのL2ノルム
        self.l1_reg = l1_reg  # AのL1ノルムの係数
        self.l2_reg = l2_reg  # DのL2ノルムの係数
        self.rate = 0.05  # 学習係数
        self.h = 0
        self.sum_dict_grads = np.zeros((300, 3000))
        self.Squared_Avegrads = np.zeros((300, 3000))

    # AdagradUpdate : Aを固定してDを更新
    def AdagradUpdate(self, dict_grads):
        # hの初期化
        # hとDの更新
        # 更新幅 : 0で割る可能性があるため1e-7を加える
        self.h += dict_grads * dict_grads
        update = 1 / (np.sqrt(self.h) + 1e-7)
        self.dict -= (self.rate * dict_grads) * update

    # AdagradUpdateWithL1Reg : Dを固定してAを更新
    def AdagradUpdateWithL1Reg(self, atom_elem_grads):
        G_grads = np.linalg.norm(self.Squared_Avegrads, ord=2)
        # λ = l1_reg
        diff = np.linalg.norm(self.Average_grads, ord=1) - self.l1_reg
        # l2_reg
        # self.l2_reg = - (int(np.sign(self.Average_grads[j])) * self.rate *
        #      diff * self.time) / (np.sqrt(G_grads))
        self.l2_reg = - (1 * self.rate * diff * self.time) / (np.sqrt(G_grads))
        for j in range(len(self.atom[self.key])):
            # Aの更新
            if diff <= 0:
                self.atom[self.key][j] = 0
            else:
                self.atom[self.key][j] = self.l2_reg

    # # AdagradUpdateWithL1RegNonNeg
    def AdagradUpdateWithL1RegNonNeg(self, grads):
        G_grads = np.linalg.norm(self.Squared_Avegrads, ord=2)
        # λ = l1_reg
        diff = np.linalg.norm(self.Average_grads, ord=1) - self.l1_reg
        # l2_reg
        self.l2_reg = -(np.sign(self.atom[self.key][j]) * self.rate * diff * self.time) / (np.sqrt(G_grads))
        for j in range(len(self.atom[self.key])):
            # Aの更新
            if diff <= 0:
                self.atom[self.key][j] = 0
            elif self.l2_reg < 0:
                self.atom[self.key][j] = 0
            else:
                self.atom[self.key][j] = self.l2_reg

    def UpdateParams(self):
        # print(self.diff_vec.shape)  # (300,)
        # print(self.atom[self.key].shape)  # (3000,)
        # Dに関する勾配 (Aは固定, (300, 3000)
        dict_grads = (-2)*self.error*self.atom[self.key] + 2*self.l2_reg*self.dict
        # AdagradUpdate
        self.AdagradUpdate(dict_grads)
        # A_iに関する勾配 (Dは固定), (300, 3000)
        atom_elem_grads = (-2)*self.error*self.dict + self.l1_reg
        """
        SAG (Stochastic Average Gradient)
        full gradientの計算が重いので一部の計算をサボって近似しよう、という出発点はSGDと変わらない。
        しかし、SAGの場合はfull gradientを少しずつ更新していく、という形を取る。
        つまり、ランダムに1つのデータを取ってきて、現在のパラメーターに対するgradientを計算し、
        それを使ってfull gradientを更新する。そのfull gradientを使ってパラメーターを更新する。
        """
        self.Average_grads = atom_elem_grads
        # 二乗和
        self.Squared_Avegrads += self.Average_grads * self.Average_grads
        # AdagradUpdateWithL1Reg
        self.AdagradUpdateWithL1Reg(atom_elem_grads)
        # self.AdagradUpdateWithL1RegNonNeg(atom_elem_grad)
