"""modelA"""
import argparse
import sys
import numpy as np

import util
import model
import param

def main():
    """コマンドライン引数の設定
    $ python main.py -i ./sample/sample_vecs.txt -o ./newvec.txt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
    parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    parser.add_argument("-n", "--numiter", type=int, default=2, help="Num iterations")
    parser.add_argument("-l1", "--l1_reg", type=float, default=0.5, help="L1 Reg (Atom)")
    parser.add_argument("-l2", "--l2_reg", type=float, default=1e-5, help="L2 Reg (Dict)")
    args = parser.parse_args()

    """ベクトルの初期化"""
    factor = 10
    data = util.Data()
    # wordVecsのread
    wordVecs = data.ReadVecsFromFile(args.input)
    # word2vecのkey数
    vocab_len = len(wordVecs)
    print("wordVecs vocab_len : {}".format(vocab_len))
    # word2vecのvalueの次元数out
    for key in wordVecs.keys():
        vec_len = len(wordVecs[key])
        print("wordVecs per example : {}".format(vec_len))
        break
    # AとDの初期化
    Dict, Atom = data.CreateVecs(wordVecs, factor, vec_len)
    for key in Atom.keys():
        print("A : {}".format(Atom[key].shape))
        print("D : {}".format(Dict.shape))
        print("DA : {}".format(np.dot(Dict, Atom[key]).shape))
        break

    """ハイパーパラメータ設定"""
    numIter = int(args.numiter) + 1 # trainのiter数
    l1_reg = float(args.l1_reg)  # AのL1ノルムの正則化項
    l2_reg = float(args.l2_reg)  # DのL1ノルムの正則化項
    print("num_iter : {}".format(numIter))
    print("L1 Reg (Atom) : {}".format(l1_reg))
    print("L2 Reg (Dict) : {}".format(l2_reg))


    """train"""    
    trainer = model.Model(wordVecs, Dict, Atom)
    trainer.train(l1_reg, l2_reg, numIter)

    """save"""
    #  Non Binarizing Transformation
    # data.WriteVectorsToFile(trainer.atom, str(args.output))
    #  Binarizing Transformation
    data.WriteVectorsToFile_non(trainer.atom, str(args.output))

if __name__ == '__main__':
    main()
