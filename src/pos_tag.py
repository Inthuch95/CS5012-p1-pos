'''
Created on 28 Feb 2018

@author: it41
'''
import nltk
from nltk import ngrams
from nltk.corpus import brown
from nltk.probability import FreqDist

if __name__ == '__main__':
    brown_tagged_sents = brown.tagged_sents()
    brown_sents = brown.sents()
    
    # 90/10 train/test split
    size = int(len(brown_tagged_sents) * 0.9)
    size = 50
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_sents[size:size+5]
    
    words = []
    tags = []
    # create list of words and tags
    for sent in train_sents:
        words = words + [w for (w,_) in sent]
        tags = tags + [t for (_,t) in sent]
    # set of unique tags
    tags_freq = FreqDist(tags)
    # dictionary for tag transition probability
    pairs_freq_dict = dict((tag,0) for tag in tags_freq)
    for key in pairs_freq_dict.keys():
        pairs_freq_dict[key] = dict((tag,0) for tag in tags_freq) 
    # create tag pairs count
    bigrams = list(ngrams(tags, 2))
    for i in bigrams:
        key_row = i[0]
        key_col = i[1]
        pairs_freq_dict[key_row][key_col] += 1
    print(pairs_freq_dict)