import math
import numpy as np
from numpy import inf
from numpy import nan
import re
import sys

from hyperparameter import numIter, l1_reg, l2_reg, factor, rate
from memory_profiler import profile
import gc

from tqdm import tqdm


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
        self.Dict = Dict  # (L, K)

        # 全単語で共通
        self._del_grad_D = np.zeros(
            (vec_len, vec_len * factor), dtype=np.float16
        )  # (L, K)
        self._grad_sum_D = np.zeros(
            (vec_len, vec_len * factor), dtype=np.float16
        )  # (L, K)

        # 単語ごと作成
        self._del_grad_A = {}
        self._grad_sum_A = {}
        for key in self.atom.keys():
            self._del_grad_A[key] = np.zeros(vec_len * factor, dtype=np.float16)  # (K,)
            self._grad_sum_A[key] = np.zeros(vec_len * factor, dtype=np.float16)  # (K,)

    # AdagradUpdate : Aを固定してDを更新
    def AdagradUpdate(self, grads):
        # hとDの更新
        # grads : (L, K)
        # _del_grad_D : (L, K)
        """RuntimeWarning: overflow encountered in add
        gradsのclippingし，一定の幅で絞る
        小さい値はイプシロンで埋める
        """
        self._del_grad_D += np.power(grads, 2).astype(np.float16)
        # print(self._del_grad_D)
        # _grad_sum : (L, K)
        self._grad_sum_D += grads.astype(np.float16)
        # print(self._grad_sum_D)
        # cwiseQuotient : (L, K)
        # cwiseQuotient = grads / (np.sqrt(self._del_grad_D.astype(np.float64) + 1e-7))
        # dict : (L, K)
        self.Dict -= rate * (
            grads / (np.sqrt(self._del_grad_D.astype(np.float64) + 1e-7))
        ).astype(np.float16)
        # print(self.Dict)

    # AdagradUpdateWithL1Reg : Dを固定してAを更新
    def AdagradUpdateWithL1Reg(self, time, key, grads, vec_len):
        #  grads: (1, K)
        # _del_grad_A : (K,)
        """単語単位で_del_grad_Aや_grad_sum_Aを持たせる"""
        """RuntimeWarning: overflow encountered in power"""
        self._del_grad_A[key] += np.power(grads[0], 2).astype(np.float16)
        # print(self._del_grad_A[key])
        # _grad_sum_A : (K,)
        self._grad_sum_A[key] += grads[0].astype(np.float16)
        # print(self._grad_sum_A[key])
        # A[key]の更新
        for j in range(factor * vec_len):  # K
            """RuntimeWarning: invalid value encountered in double_scalars"""
            # A[key][j]の更新
            if abs(self._grad_sum_A[key][j].astype(np.float64)) - l1_reg * time <= 0:
                self.atom[key][0][j] = 0
            else:
                self.atom[key][0][j] = (
                    -(
                        np.sign(self._grad_sum_A[key][j])
                        * rate
                        * abs(self._grad_sum_A[key][j].astype(np.float64))
                        - l1_reg * time
                    )
                    / (np.sqrt(self._del_grad_A[key][j].astype(np.float64)))
                ).astype(np.float16)

    # AdagradUpdateWithL1RegNonNeg
    def AdagradUpdateWithL1RegNonNeg(self, time, key, grads, vec_len):
        #  grads: (1, K)
        # _del_grad_A : (K,)
        self._del_grad_A[key] += np.power(grads[0], 2).astype(np.float16)
        # _grad_sum_A : (K,)
        self._grad_sum_A[key] += grads[0].astype(np.float16)
        for j in range(factor * vec_len):  # K
            diff = abs(self._grad_sum_A[key][j]) - l1_reg * time
            gamma = -(
                (np.sign(self._grad_sum_A[key][j], dtype=np.float16) * rate * diff)
                / (np.sqrt(self._del_grad_A[key][j], dtype=np.float16))
            )
            gamma = gamma.astype(np.float16)
            # Aの更新
            if diff <= 0:
                self.atom[key][0][j] = 0
            else:
                if gamma < 0:
                    self.atom[key][0][j] = 0
                else:
                    self.atom[key][0][j] = gamma

    """パラメータの更新"""

    def UpdateParams(self, time, key, diff_vec, vec_len):
        """Dの更新 (A[key]は固定)
        # diff_vec : (1, L)
        # atom[self.key] : (1, K)
        # dict : (L, K)
        # dict_grads : (L, K)
        """
        # AdagradUpdate
        self.AdagradUpdate(
            np.clip(
                (-2)
                * np.dot(
                    diff_vec.astype(np.float64).T, self.atom[key].astype(np.float64)
                )
                + 2 * l2_reg * self.Dict.astype(np.float64),
                -1,
                1,
            )
        )

        """A[key]の更新 (Dは固定)
        iter2の2つ目の単語のdiff_vecまでは同じ
        # gradientの計算
        # diff_vec : (1, L)
        # dict : (L, K)
        # atom_elem_grads, (1, K)
        """
        # AdagradUpdateWithL1Reg
        self.AdagradUpdateWithL1Reg(
            time,
            key,
            np.clip(
                (-2)
                * np.dot(diff_vec.astype(np.float64), self.Dict.astype(np.float64)),
                -1,
                1,
            ),  # atom_elem_grads
            vec_len,
        )
        # Binarizing Transformation
        # self.AdagradUpdateWithL1RegNonNeg(time, key, atom_elem_grads)
