"""Param update
・AdaGrad
学習が進むにつれて学習係数を小さくする手法．
adaptiveな手続きのため相性が良い．
"""
import math
import numpy as np
import re
import sys

class Param:
    def __init__(self, Atom, Dict, vocab_len,
                 vec_len, factor, key, diff_vec, error, l1_reg, l2_reg):
        self.atom = Atom  # (L, V)
        self.dict = Dict  # (L, K)
        self.vocab_len = vocab_len  # V
        self.vec_len = vec_len  # L
        self.factor = factor  # K = L * factor
        self.key = key  # 対象key
        self.diff_vec = diff_vec  # (1, L)
        self.error = error  # diff_vecのL2ノルム
        self.l1_reg = l1_reg  # AのL1ノルムの係数
        self.l2_reg = l2_reg  # DのL2ノルムの係数
        self.rate = 0.05  # 学習係数
        
        self._del_grad_D = np.zeros((vec_len, vec_len*factor))  # (L, K)
        self._grad_sum_D = np.zeros((vec_len, vec_len*factor))  # (L, K)
        self._del_grad_A = np.zeros(vec_len*factor)  # (K,)
        self._grad_sum_A = np.zeros(vec_len*factor)  # (K,)
        self._update_num = 0 # 更新単語数

    # AdagradUpdate : Aを固定してDを更新
    def AdagradUpdate(self, grads):
        # hとDの更新
        # grads : (L, K)
        # _del_grad_D : (L, K)
        self._del_grad_D += np.power(grads, 2)
        # _grad_sum : (L, K)
        self._grad_sum_D += grads
        # cwiseQuotient : (L, K)
        cwiseQuotient = grads / (np.sqrt(self._del_grad_D) + 1e-7)
        # dict : (L, K)
        self.dict -= self.rate * cwiseQuotient

    # AdagradUpdateWithL1Reg : Dを固定してAを更新
    def AdagradUpdateWithL1Reg(self, grads):
        # 更新回数
        self._update_num += 1
        #  grads: (1, K)
        # _del_grad_A : (K,)
        self._del_grad_A += np.power(grads[0], 2)
        # _grad_sum_A : (K,)
        self._grad_sum_A += grads[0]
        # A[key]の更新
        for j in range(self.vec_len*self.factor): # K
            diff = abs(self._grad_sum_A[j]) - self.l1_reg*self._update_num
            gamma = -(np.sign(self._grad_sum_A[j])*self.rate*diff)/(np.sqrt(self._del_grad_A[j]))
            # A[key][j]の更新
            if diff <= 0:
                self.atom[self.key][0][j] = 0
            else:
                self.atom[self.key][0][j] = gamma

    # AdagradUpdateWithL1RegNonNeg
    def AdagradUpdateWithL1RegNonNeg(self, grads):
        # 更新回数
        self._update_num += 1
        #  grads: (1, K)
        # _del_grad_A : (K,)
        self._del_grad_A += np.power(grads[0], 2)
        # _grad_sum_A : (K,)
        self._grad_sum_A += grads[0]
        for j in range(self.vec_len*self.factor):  # K
            diff = abs(self._grad_sum_A[j]) - self.l1_reg*self._update_num
            gamma = -((np.sign(self._grad_sum_A[j])*self.rate*diff)/(np.sqrt(self._del_grad_A[j])))
            # Aの更新
            if diff <= 0:
                self.atom[self.key][0][j] = 0
            else:
                if gamma < 0:
                    self.atom[self.key][0][j] = 0
                else:
                    self.atom[self.key][0][j] = gamma

    def UpdateParams(self):
        """Dの更新 (A[key]は固定)"""
        # gradientの計算
        # diff_vec : (1, L)
        # atom[self.key] : (1, K)
        # dict : (L, K)
        # dict_grads : (L, K)
        dict_elem_grads = (-2)*np.dot(self.diff_vec.T, self.atom[self.key]) + 2*self.l2_reg*self.dict
        # AdagradUpdate
        self.AdagradUpdate(dict_elem_grads)
        """A[key]の更新 (Dは固定)"""
        # gradientの計算
        # diff_vec : (1, L)
        # dict : (L, K)
        # atom_elem_grads, (1, K)
        atom_elem_grads = (-2)*np.dot(self.diff_vec, self.dict)
        # AdagradUpdateWithL1Reg
        # self.AdagradUpdateWithL1Reg(atom_elem_grads)
        # Binarizing Transformation
        self.AdagradUpdateWithL1RegNonNeg(atom_elem_grads)
