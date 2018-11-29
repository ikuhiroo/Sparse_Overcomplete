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

        print('wordVecsのread中')
        cnt = 0
        for line in fileObject:
            if cnt == 0:
                pass
            else:
                # line = line.strip().lower()
                line = line.strip()
                word = line.split()[0]
                wordVectors[word] = np.zeros(len(line.split())-1, dtype=float)  # (L,)
                for index, vecVal in enumerate(line.split()[1:]):
                    wordVectors[word][index] = float(vecVal)
                """normalize"""
                # wordVectors[word] /= [math.sqrt((wordVectors[word]**2).sum() + 1e-6)]
                wordVectors[word] = np.array([wordVectors[word]]) # (1, L)
            cnt += 1
        print('wordVecsのread完了')
        sys.stderr.write("Vectors read from: "+filename+" \n")
        return wordVectors

    """vector A の書き込み"""
    def WriteVectorsToFile(self, newvec, outFileName):
        """Write word vectors to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for word in newvec.keys():
            outFile.write(word+' ')
            for val in newvec[word][0]:
                outFile.write('%.3f' % (val)+' ')
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
    def WriteDictToFile(self, dic, outFileName):
        print(dic.shape)
        """Write dict to file"""
        sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
        outFile = open(outFileName, 'w')
        for i in range(len(dic)):
            for j in range(len(dic[i])):
                outFile.write(str(dic[i][j])+' ')
        outFile.close()
