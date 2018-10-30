import math
import numpy as np
import re
import sys

from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

class Param:
    def __init__(self, Atom, Dict, vocab_len, vec_len):
        self.atom = Atom  # (L, V)
        self.dict = Dict  # (L, K)
        self.vocab_len = vocab_len  # V
        self.vec_len = vec_len   # L

        self._del_grad_D = np.zeros((vec_len, vec_len*factor))  # (L, K)
        self._grad_sum_D = np.zeros((vec_len, vec_len*factor))  # (L, K)
        self._del_grad_A = np.zeros(vec_len*factor)  # (K,)
        self._grad_sum_A = np.zeros(vec_len*factor)  # (K,)

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
        self.dict -= rate * cwiseQuotient

    # AdagradUpdateWithL1Reg : Dを固定してAを更新
    def AdagradUpdateWithL1Reg(self, time, key, grads):
        #  grads: (1, K)
        # _del_grad_A : (K,)
        self._del_grad_A += np.power(grads[0], 2)
        # _grad_sum_A : (K,)
        self._grad_sum_A += grads[0]
        # A[key]の更新
        for j in range(factor*self.vec_len):# K
            diff = abs(self._grad_sum_A[j]) - l1_reg*time
            gamma = -(np.sign(self._grad_sum_A[j]) * rate * diff) / (np.sqrt(self._del_grad_A[j]))
            # A[key][j]の更新
            if diff <= 0:
                self.atom[key][0][j] = 0
            else:
                self.atom[key][0][j] = gamma

    # AdagradUpdateWithL1RegNonNeg
    def AdagradUpdateWithL1RegNonNeg(self, time, key, grads):
        #  grads: (1, K)
        # _del_grad_A : (K,)
        self._del_grad_A += np.power(grads[0], 2)
        # _grad_sum_A : (K,)
        self._grad_sum_A += grads[0]
        for j in range(factor * self.vec_len):  # K
            diff = abs(self._grad_sum_A[j]) - l1_reg * time
            gamma = -((np.sign(self._grad_sum_A[j]) * rate * diff) / (np.sqrt(self._del_grad_A[j])))
            # Aの更新
            if diff <= 0:
                self.atom[key][0][j] = 0
            else:
                if gamma < 0:
                    self.atom[key][0][j] = 0
                else:
                    self.atom[key][0][j] = gamma
    
    """パラメータの更新"""
    def UpdateParams(self, time, key, diff_vec):
        """Dの更新 (A[key]は固定)"""
        # diff_vec : (1, L)
        # atom[self.key] : (1, K)
        # dict : (L, K)
        # dict_grads : (L, K)
        dict_elem_grads = (-2)*np.dot(diff_vec.T, self.atom[key]) + 2*l2_reg*self.dict
        # AdagradUpdate
        self.AdagradUpdate(dict_elem_grads)
        """A[key]の更新 (Dは固定)"""
        # gradientの計算
        # diff_vec : (1, L)
        # dict : (L, K)
        # atom_elem_grads, (1, K)
        # atom_elem_grads = (-2)*np.dot(diff_vec, self.dict)
        # AdagradUpdateWithL1Reg
        # self.AdagradUpdateWithL1Reg(time, key, atom_elem_grads)
        # Binarizing Transformation
        # self.AdagradUpdateWithL1RegNonNeg(time, key, atom_elem_grads)
