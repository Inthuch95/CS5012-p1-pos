'''
Created on Feb 28, 2018

@author: User
'''
import nltk
from nltk import ngrams
from nltk.corpus import brown
from nltk.probability import FreqDist
import csv

def train_test_split():
    brown_tagged_sents = brown.tagged_sents(tagset="universal")
    brown_sents = brown.sents()
    
    # 90/10 train/test split
    size = int(len(brown_tagged_sents) * 0.9)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_sents[size:]
    return train_sents, test_sents

def get_words_and_tags():
    # create list of words and tags
    words = []
    tags = []
    start = ["<s>"]
    end = ["</s>"]
    for sent in train_sents:
        words += start + [w for (w,_) in sent] + end
        tags += start + [t for (_,t) in sent] + end
    return words, tags

def create_transition_tables():
    # dictionary for tag transition count table
    transition_count = dict((tag,0) for tag in tags_dist)
    for key in transition_count.keys():
        transition_count[key] = dict((tag,0) for tag in tags_dist) 
    bigrams = list(ngrams(tags, 2))
    for i in bigrams:
        key_row = i[0]
        key_col = i[1]
        transition_count[key_row][key_col] += 1
    # dictionary for tag transition probability table
    transition_prob = dict((tag,0) for tag in tags_dist)
    for key in transition_prob.keys():
        transition_prob[key] = dict((tag,0) for tag in tags_dist) 
    for key_row in transition_prob.keys():
        for key_col in transition_prob[key_row].keys():
            transition_prob[key_row][key_col] = 1.0 * transition_count[key_col][key_row] / tags_dist[key_col]
    return transition_count, transition_prob

def create_emission_tables():
    emission_count = dict((tag,0) for tag in tags_dist)
    for key in emission_count.keys():
        emission_count[key] = dict((word,0) for word in words_dist)
    return emission_count

def write_tables_to_files():
    with open("../emission_output.csv", "w") as f:
        w = csv.writer( f )
        tag_names = list(emission_count.values())[0].keys()
        w.writerow([""] + [key for key in list(emission_count.values())[0].keys()])
        for key in emission_count.keys():
            w.writerow([key] + [emission_count[key][tag_name] for tag_name in tag_names])    
    with open("../transition_output.csv", "w") as f:
        w = csv.writer( f )
        tag_names = list(transition_prob.values())[0].keys()
        w.writerow([""] + [key for key in list(transition_prob.values())[0].keys()])
        for key in transition_prob.keys():
            w.writerow([key] + [transition_prob[key][tag_name] for tag_name in tag_names])
 
if __name__ == '__main__':
    # split train and test sets
    train_sents, test_sents = train_test_split()
    words, tags = get_words_and_tags()
    
    # tags and words distribution
    tags_dist = FreqDist(tags)            
    words_dist = FreqDist(words)
    
    # create tables
    transition_count, transition_prob = create_transition_tables()
    emission_count = create_emission_tables()
        
    # write the tables to csv
    write_tables_to_files()