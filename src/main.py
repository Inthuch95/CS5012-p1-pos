'''
Created on 28 Feb 2018
'''
from hmm import HMM
from tagger import pos_tagger
from nltk.corpus import brown, conll2000, alpino, treebank
import sys

CONLL2000_UNIVERSAL = 1
CONLL2000 = 2
TREEBANK = 3
BROWN_UNIVERSAL = 4
ALPINO = 5

def get_corpus(selected_corpus):
    tagset = ""
    if selected_corpus == CONLL2000_UNIVERSAL:
        corpus = conll2000
        tagset = "universal"
    elif selected_corpus == CONLL2000:
        corpus = conll2000
    elif selected_corpus == TREEBANK:
        corpus = treebank
    elif selected_corpus == BROWN_UNIVERSAL:
        corpus = brown
        tagset = "universal"
    elif selected_corpus == ALPINO:
        corpus = alpino
    else:
        print("corpus unavailable")
        quit() 
    return corpus, tagset

if __name__ == '__main__':
    selected_corpus = int(sys.argv[1])
    smoothing = sys.argv[2]
    corpus, tagset = get_corpus(selected_corpus=1)
    hmm = HMM(corpus, tagset, smoothing)
    tagger = pos_tagger(hmm)
    tagger.viterbi()