'''
Created on 28 Feb 2018

@author: it41
'''
from hmm import HMM

CONLL2000_UNIVERSAL = 1
CONLL2000 = 2
CONLL2002 = 3
BROWN_UNIVERSAL = 4
BRWON = 5

if __name__ == '__main__':
    hmm = HMM()
    hmm.viterbi()