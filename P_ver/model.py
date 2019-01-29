"""model"""
import math
import re
import sys
import numpy as np
np.random.seed(100)
import util

import param
from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

from memory_profiler import profile
import gc

import argparse
import sys
import numpy as np

import util
import model
import param
from hyperparameter import numIter, l1_reg, l2_reg, factor, rate

from memory_profiler import profile
import gc

class Model:
    """パラメータ初期化"""
    # @profile
    def __init__(self, wordVecs, vocab_len, vec_len):
        # self.wordVecs = wordVecs
        # self.vocab_len = vocab_len  # V
        # self.vec_len = vec_len  # L

        """AとDの初期化
        # A : (V, K), A[key] : (K,)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len*factor)
        # メモリ計算式：単語数 * SOVのvecの次元数 * 0.000002[MB]
        
        # D : (L, K)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len+vec_len*factor))
        # メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000002[MB]
        """
        keys = list(wordVecs.keys()) 
        self.atom = {}
        # 3.6MB (32bit, factor=10)増える
        for key in keys:
            self.atom[key] = 0.6*(1/np.sqrt(factor*vec_len, dtype=np.float16)) * \
                np.random.randn(1, factor*vec_len).astype(np.float16)
        
        # 3.6MB (32bit, factor=10)増える
        self.dict = (0.6*(1/np.sqrt(vec_len + factor*vec_len))*np.random.randn(vec_len, factor*vec_len)).astype(np.float16)
        # メモリ再利用
        del keys, wordVecs, vocab_len, vec_len
        gc.collect()

    """Dとa_iの内積を計算する
    # word2vec[key] : (1, L)
    # self.dict : (L, K)
    # self.atom[key] : (1, K)
    # 内積 : (1, L)
    """
    def PredictVector(self, key):
        return np.dot(self.atom[key], self.dict.T).astype(np.float32)

    """Sparse_Overfitting (adaptiveな処理)
    ・Dataインスタンス(data)生成
    ・paramインスタンス (Optimizer) 生成
        self.atom = Atom  # (L, V)
        -> メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000002[MB]
        self.dict = Dict  # (L, K)
        -> メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000002[MB]
    
        self._del_grad_D  # (L, K)
        -> メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000002[MB]
        self._grad_sum_D  # (L, K)
        -> メモリ計算式：初期vecの次元数 * SOVのvecの次元数 * 0.000002[MB]
        self._del_grad_A[key] # (K,)
        -> メモリ計算式：単語数 * SOVのvecの次元数 * 0.000002[MB]
        self._grad_sum_A[key] # (K,)
        -> メモリ計算式：単語数 * SOVのvecの次元数 * 0.000002[MB]
    """
    @profile
    def Sparse_Overfitting(self, wordVecs, vocab_len, vec_len):
        # mainとmodelは分ける必要があるのか
        data = util.Data()
        # 
        Optimizer = param.Param(self.atom, self.dict, vocab_len, vec_len)
        for time in range(1, numIter):
            num_words = 0 # 更新単語数
            total_error = np.array(0, dtype=np.float16)  # 総ロス
            atom_l1_norm = np.array(0, dtype=np.float16) # Aに関するノルム値
            # adaptiveな手続き, A[key]を対象
            for key in wordVecs.keys():
                """error算出"""
                # predict i-th word, DとAの内積を計算, (1, L)
                pred_vec = self.PredictVector(key)
                # true_vec - pred_vecの復元誤差, (1, L)
                diff_vec = wordVecs[key] - pred_vec
                # メモリの再利用
                del pred_vec
                gc.collect()

                """AとDの更新
                全単語共通でtimeごとに値を保持している
                -> 単語ごとに値を保持するべき

                atom[key].UpdateParams(time, diff_vec)
                """
                Optimizer.UpdateParams(time, key, diff_vec, vec_len)
                self.atom = Optimizer.atom
                self.dict = Optimizer.dict
                
                num_words += 1  # 更新単語数
                error = np.sum(np.square(diff_vec), dtype=np.float16)
                total_error += error
                # メモリの再利用
                del error
                gc.collect()
                atom_l1_norm += np.sum(self.atom[key][0])
            print("Error per example : {}".format(total_error / num_words))
            print("Dict L2 norm : {}".format(np.linalg.norm(self.dict, ord=2)))
            print("Avg Atom L1 norm : {}\n".format(atom_l1_norm/num_words))

            # 保存先
            output = '../sample_Sparse/newvec_{}.model'.format(time)
            """save（sparse）A and D"""
            # 0.4 MiB増える
            data.WriteVectorsToFile(self.atom, str(output))
            # dict
            data.WriteDictToFile(self.dict, str(output+'_dict'))
            # """save（sparse + binary）A and D"""
            # # Sparse + Binary
            # data.WriteVectorsToFile_non(self.atom, str(output + '_non'))
            # # dict
            # data.WriteDictToFile(self.dict, str(output+'_dict'))



