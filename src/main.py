'''
Created on 28 Feb 2018
'''
from hmm import HMM
from tagger import pos_tagger
from nltk.corpus import brown, conll2000, treebank
import sys

CONLL2000 = 1
CONLL2000_UNIVERSAL = 2
TREEBANK = 3
TREEBANK_UNIVERSAL = 4
BROWN_UNIVERSAL = 5

def get_corpus(selected_corpus):
    # get the right corpus and tagset
    tagset = ""
    corpus_name = ""
    if selected_corpus == CONLL2000_UNIVERSAL:
        corpus = conll2000
        tagset = "universal"
        corpus_name = "Conll2000"
    elif selected_corpus == CONLL2000:
        corpus = conll2000
        corpus_name = "Conll2000"
    elif selected_corpus == TREEBANK:
        corpus = treebank
        corpus_name = "Treebank"
    elif selected_corpus == BROWN_UNIVERSAL:
        corpus = brown
        tagset = "universal"
        corpus_name = "Brown"
    elif selected_corpus == TREEBANK_UNIVERSAL:
        corpus = treebank
        tagset = "universal"
        corpus_name = "Treebank"
    else:
        print("corpus unavailable")
        quit() 
    return corpus, tagset, corpus_name

if __name__ == '__main__':
    # load selected corpus
    selected_corpus = int(sys.argv[1])
    smoothing = sys.argv[2]
    corpus, tagset, corpus_name = get_corpus(selected_corpus)
    print("Loaded corpus: ", corpus_name)
    
    # train HMM
    hmm = HMM(corpus, tagset, smoothing)
    
    # pos tagging with viterbi algorithm 
    tagger = pos_tagger(hmm)
    tagger.viterbi()
    
    # print confusion matrix
    print("")
    if tagset == "universal":
        tagger.print_confusion_matrix()
    else:
        tagger.print_summary()
    print("")
    
    # print overall accuracy
    print("overall accuracy, correct: %d/%d percentage: %0.2f \n" % \
      (tagger.overall_accuracy["correct"], tagger.overall_accuracy["words"], tagger.overall_accuracy["percentage"]))