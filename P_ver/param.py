import math
import numpy as np
import re
import sys

from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

from memory_profiler import profile
import gc

class Param:
    """
    以下はクラス変数（全ての単語で共通）である．
    ・_del_grad_D
    ・_grad_sum_D
    以下はインスタンス変数（単語ごとに異なる）である．
    ・_del_grad_A
    ・_grad_sum_A
    """
    def __init__(self, Atom, Dict, vocab_len, vec_len):
        self.atom = Atom  # (L, V)
        self.dict = Dict  # (L, K)
        # self.vocab_len = vocab_len  # V
        # self.vec_len = vec_len   # L

        # 全単語で共通
        self._del_grad_D = np.zeros((vec_len, vec_len*factor), dtype=np.float32)  # (L, K)
        self._grad_sum_D = np.zeros((vec_len, vec_len*factor), dtype=np.float32)  # (L, K)

        # 単語ごと作成
        """メモリを食っている原因
        １つの単語が(300, 3000)のオブジェクトをもつ
        しかも，単語数が754069個
        """
        self._del_grad_A = {}
        self._grad_sum_A = {}
        for key in self.atom.keys():
            self._del_grad_A[key] = np.zeros(vec_len*factor, dtype=np.float32)  # (K,)
            self._grad_sum_A[key] = np.zeros(vec_len*factor, dtype=np.float32)  # (K,)

    # AdagradUpdate : Aを固定してDを更新
    # @profile
    def AdagradUpdate(self, grads):
        # hとDの更新
        # grads : (L, K)
        # _del_grad_D : (L, K)
        self._del_grad_D += np.power(grads, 2, dtype=np.float32)
        # _grad_sum : (L, K)
        self._grad_sum_D += grads
        # cwiseQuotient : (L, K)
        cwiseQuotient = grads / (np.sqrt(self._del_grad_D, dtype=np.float32) + 1e-7)
        # dict : (L, K)
        self.dict -= rate * cwiseQuotient

    # AdagradUpdateWithL1Reg : Dを固定してAを更新
    # @profile
    def AdagradUpdateWithL1Reg(self, time, key, grads, vec_len):
        #  grads: (1, K)
        # _del_grad_A : (K,)
        """単語単位で_del_grad_Aや_grad_sum_Aを持たせる"""
        self._del_grad_A[key] += np.power(grads[0], 2, dtype=np.float32)
        # _grad_sum_A : (K,)
        self._grad_sum_A[key] += grads[0]
        # A[key]の更新
        for j in range(factor*vec_len):  # K
            diff = abs(self._grad_sum_A[key][j]) - l1_reg*time
            gamma = - (np.sign(self._grad_sum_A[key][j], dtype=np.float32) *
                       rate * diff) / (np.sqrt(self._del_grad_A[key][j], dtype=np.float32))
            # A[key][j]の更新
            if diff <= 0:
                self.atom[key][0][j] = 0
            else:
                self.atom[key][0][j] = gamma

    # AdagradUpdateWithL1RegNonNeg
    def AdagradUpdateWithL1RegNonNeg(self, time, key, grads, vec_len):
        #  grads: (1, K)
        # _del_grad_A : (K,)
        self._del_grad_A[key] += np.power(grads[0], 2, dtype=np.float32)
        # _grad_sum_A : (K,)
        self._grad_sum_A[key] += grads[0]
        for j in range(factor * vec_len):  # K
            diff = abs(self._grad_sum_A[key][j]) - l1_reg*time
            gamma = - ((np.sign(self._grad_sum_A[key][j], dtype=np.float32) * rate *
                        diff) / (np.sqrt(self._del_grad_A[key][j], dtype=np.float32)))
            # Aの更新
            if diff <= 0:
                self.atom[key][0][j] = 0
            else:
                if gamma < 0:
                    self.atom[key][0][j] = 0
                else:
                    self.atom[key][0][j] = gamma
    
    """パラメータの更新"""
    # @profile
    def UpdateParams(self, time, key, diff_vec, vec_len):
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
        """iter2の2つ目の単語のdiff_vecまでは同じ"""
        atom_elem_grads = (-2)*np.dot(diff_vec, self.dict).astype(np.float32)
        # AdagradUpdateWithL1Reg
        self.AdagradUpdateWithL1Reg(time, key, atom_elem_grads, vec_len)
        # Binarizing Transformation
        # self.AdagradUpdateWithL1RegNonNeg(time, key, atom_elem_grads)
