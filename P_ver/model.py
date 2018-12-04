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

class Model:
    """パラメータ初期化"""
    # @profile
    def __init__(self, wordVecs, vocab_len, vec_len):
        # self.wordVecs = wordVecs
        # self.vocab_len = vocab_len  # V
        # self.vec_len = vec_len  # L

        """AとDの作成
        # A : (V, K), A[key] : (K,)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len*factor)
        
        # D : (L, K)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len+vec_len*factor))
        """
        keys = list(wordVecs.keys()) 
        self.atom = {}
        # 3.6MiB増える -> 3.6MBで一致
        for key in keys:
            self.atom[key] = 0.6*(1/np.sqrt(factor*vec_len, dtype=np.float32)) * \
                np.random.randn(1, factor*vec_len).astype(np.float32)
        
        """14.4 MiB増える -> 3000 * 300 * 4B = 3600,000B = 3.6MB"""
        self.dict = (0.6*(1/np.sqrt(vec_len + factor*vec_len))*np.random.randn(vec_len, factor*vec_len)).astype(np.float32)
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

    """Sparse_Overfitting (adaptiveな処理)"""
    @profile
    def Sparse_Overfitting(self, wordVecs, vocab_len, vec_len):
        data = util.Data()
        # 7.0 MiB増える -> 3.6MB + 3.6MB = 7.2MBで一致
        Optimizer = param.Param(self.atom, self.dict, vocab_len, vec_len)
        for time in range(1, numIter):
            num_words = 0 # 更新単語数
            total_error = np.array(0, dtype=np.float32)  # 総ロス
            atom_l1_norm = np.array(0, dtype=np.float32) # Aに関するノルム値
            print("Iteration : {}".format(time))
            # adaptiveな手続き, A[key]を対象
            for key in wordVecs.keys():
                """error算出"""
                # predict i-th word, DとAの内積を計算, (1, L)
                pred_vec = self.PredictVector(key)
                # true_vec - pred_vecの復元誤差, (1, L)
                diff_vec = wordVecs[key] - pred_vec
                del pred_vec
                gc.collect()

                """AとDの更新
                全単語共通でtimeごとに値を保持している
                -> 単語ごとに値を保持するべき

                atom[key].UpdateParams(time, diff_vec)
                """
                # 17.3 MiB増える -> 
                Optimizer.UpdateParams(time, key, diff_vec, vec_len)
                self.atom = Optimizer.atom
                self.dict = Optimizer.dict
                
                num_words += 1  # 更新単語数
                # error = (diff_vec**2).sum()  # 4.435537832891
                # error = np.sum(diff_vec**2, dtype=np.float32)  # 4.4052725
                error = np.sum(np.square(diff_vec), dtype=np.float32)  # 4.4492593
                total_error += error
                del error
                gc.collect()
                # atom_l1_norm += (self.atom[key][0]).sum() # -1.6117806434631348
                # -1.6117806434631348
                atom_l1_norm += np.sum(self.atom[key][0])
                # # Aは更新後
            print("Error per example : {}".format(total_error / num_words))
            # Dは更新前
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



