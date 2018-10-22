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
        self.time = time  # 更新回数
        self.atom = Atom  # newvec
        self.dict = Dict  # dict
        self.key = key  # word2vecの対象key
        self.num_words = num_words  # time回目に更新した単語数
        self.diff_vec = diff_vec  # lossの第1項
        self.error = error  # diff_vecのL2ノルム
        self.l1_reg = l1_reg  # AのL1ノルムの係数
        self.l2_reg = l2_reg  # DのL2ノルムの係数
        self.rate = 0.05  # 学習係数
        self.h = 0
        self.sum_atom_elem_grads = np.zeros_like(self.atom[self.key])
        self.ave_atom_elem_grads = np.zeros_like(self.atom[self.key])
        self.square_atom_elem_grads = np.zeros_like(self.atom[self.key])

    # AdagradUpdate : Aを固定してDを更新
    def AdagradUpdate(self, dict_grads):
        # hとDの更新
        self.h += np.linalg.norm(dict_grads, ord=2)
        update = 1 / (np.sqrt(self.h) + 1e-7)
        self.dict -= (self.rate * update * dict_grads)

    # AdagradUpdateWithL1Reg : Dを固定してAを更新
    def AdagradUpdateWithL1Reg(self):
        # A[key]のj番目の要素に対する処理
        for j in range(len(self.atom[self.key])):
            # λ = l1_reg
            diff = abs(self.ave_atom_elem_grads[j]) - self.l1_reg
            # gamma
            gamma = -(np.sign(self.ave_atom_elem_grads[j]) * self.rate * diff * self.time) / (
                np.sqrt(self.square_atom_elem_grads[j]))
            # Aの更新
            if diff <= 0:
                self.atom[self.key][j] = 0
            else:
                self.atom[self.key][j] = gamma

    # AdagradUpdateWithL1RegNonNeg
    def AdagradUpdateWithL1RegNonNeg(self):
        for j in range(len(self.atom[self.key])):
            # λ = l1_reg
            diff = abs(self.ave_atom_elem_grads[j]) - self.l1_reg
            # l2_reg
            gamma = -((np.sign(self.ave_atom_elem_grads[j]) * self.rate * diff * self.time) / (
                np.sqrt(self.square_atom_elem_grads[j])))
            # Aの更新
            if diff <= 0:
                self.atom[self.key][j] = 0
            elif self.l2_reg < 0:
                self.atom[self.key][j] = 0
            else:
                self.atom[self.key][j] = gamma

    def UpdateParams(self):
        """Dの更新 (A[key]は固定)"""
        # gradient, (300, 3000), L2ノルムgrad
        dict_grads = (-2) * np.dot(np.array([self.diff_vec]).T, np.array([self.atom[self.key]])) + 2 * self.l2_reg * self.dict
        # AdagradUpdate
        self.AdagradUpdate(dict_grads)
        """A[key]の更新 (Dは固定)"""
        # atom_elem_grads, (1, 3000)
        atom_elem_grads = (-2) * np.dot(np.array([self.diff_vec]), self.dict) + self.l1_reg
        
        # timeごとのatom_elem_gradsの要素の和
        self.sum_atom_elem_grads += atom_elem_grads[0]
        # timeごとのatom_elem_gradsの要素の和の平均
        self.ave_atom_elem_grads = self.sum_atom_elem_grads / self.time
        # timeごとのatom_elem_gradsの要素の二乗和
        self.square_atom_elem_grads += atom_elem_grads[0] * atom_elem_grads[0]
        # AdagradUpdateWithL1Reg
        # self.AdagradUpdateWithL1Reg()
        # Binarizing Transformation
        self.AdagradUpdateWithL1RegNonNeg()
