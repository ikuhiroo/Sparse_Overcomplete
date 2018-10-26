import gzip
import math
import re
import sys
import numpy as np
from copy import deepcopy
import codecs

class Data:
    def __init__(self):
        pass

    """vectorsのread + normalize"""
    def ReadVecsFromFile(self, filename):
        wordVectors = {}
        # ファイル読み込み
        if filename.endswith('.gz'):
            fileObject = gzip.open(filename, 'r')
        else:
            fileObject = codecs.open(filename, "r", "utf-8", 'ignore')

        for line in fileObject:
            # line = line.strip().lower()
            line = line.strip()
            word = line.split()[0]
            wordVectors[word] = np.zeros(len(line.split())-1, dtype=float)  # (L,)
            for index, vecVal in enumerate(line.split()[1:]):
                wordVectors[word][index] = float(vecVal)
            """normalize weight vector"""
            wordVectors[word] /= [math.sqrt((wordVectors[word]**2).sum() + 1e-6)]
            wordVectors[word] = np.array([wordVectors[word]]) # (1, L)

        sys.stderr.write("Vectors read from: "+filename+" \n")
        return wordVectors

    """AとDの作成"""
    def CreateVecs(self, wordVecs, factor, vec_len):
        keys = list(wordVecs.keys())
        # A : (V, K), A[key] : (K,)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len*factor)
        Atom = {}
        for key in keys:
            Atom[key] = (0.6*(1/np.sqrt(vec_len*factor))) * np.random.rand(1, factor*vec_len)
        # D : (L, K)
        # 初期値の係数 : 0.6*(1/np.sqrt(vec_len+vec_len*factor))
        Dict = (0.6*(1/np.sqrt(vec_len+vec_len*factor))) * np.random.rand(vec_len, factor * vec_len)
        return Dict, Atom

    """vector A の書き込み"""
    def WriteVectorsToFile(self, newvec, outFileName):
        """Write word vectors to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for word in newvec.keys():
            outFile.write(word+' ')
            for val in newvec[word][0]:
                outFile.write('%.4f' % (val)+' ')
            outFile.write('\n')
        outFile.close()

    """vector A non の書き込み"""
    def WriteVectorsToFile_non(self, newvec, outFileName):
        """binary + Write word vectors to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for word in newvec.keys():
            outFile.write(word+' ')
            for val in newvec[word][0]:
                if val > 0:
                    val = 1
                else:
                    val = 0
                outFile.write(str(val)+' ')
            outFile.write('\n')
        outFile.close()

    """dict D の書き込み"""
    def WriteDictToFile(self):
        pass
