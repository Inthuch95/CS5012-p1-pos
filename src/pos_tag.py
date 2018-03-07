'''
Created on 28 Feb 2018
'''
from hmm import HMM
from nltk.corpus import brown, conll2000, conll2002

CONLL2000_UNIVERSAL = 1
CONLL2000 = 2
CONLL2002 = 3
BROWN_UNIVERSAL = 4
BRWON = 5

def get_corpus():
#     corpus = conll2000
#     corpus = brown
    corpus = conll2002
    tagset = "default"
    return corpus, tagset

if __name__ == '__main__':
    corpus, tagset = get_corpus()
    hmm = HMM(corpus, tagset)
    hmm.viterbi()