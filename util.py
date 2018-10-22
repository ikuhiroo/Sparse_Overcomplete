import gzip
import math
import numpy
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
            wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
            for index, vecVal in enumerate(line.split()[1:]):
                wordVectors[word][index] = float(vecVal)
            """normalize weight vector"""
            wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)

        sys.stderr.write("Vectors read from: "+filename+" \n")
        return wordVectors

    """AとDの作成"""
    def CreateVecs(self, wordVecs, factor, vec_len):
        keys = list(wordVecs.keys())
        # A : (factor * vec_len, vec_len)
        Atom = {}
        for key in keys:
            Atom[key] = np.random.rand(factor * vec_len)  # (3000, 1)
        # D : (vec_len, factor * vec_len)
        Dict = np.random.rand(vec_len, factor * vec_len)  # (300, 3000)
        return Dict, Atom

    """vector A の書き込み"""
    def WriteVectorsToFile(self, newvec, outFileName):
        """Write word vectors to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for word, val in newvec.items():
            outFile.write(word+' ')
            for val in newvec[word]:
                outFile.write('%.4f' % (val)+' ')
            outFile.write('\n')
        outFile.close()

    """vector A の書き込み"""
    def WriteVectorsToFile_non(self, newvec, outFileName):
        """binary + Write word vectors to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for word, val in newvec.items():
            outFile.write(word+' ')
            for val in newvec[word]:
                if val > 0:
                    val = 1
                else:
                    val = 0
                outFile.write('%.4f' % (val)+' ')
            outFile.write('\n')
        outFile.close()

    """dict D の書き込み"""
    def WriteDictToFile(self):
        pass
