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
        row = i[0]
        col = i[1]
        transition_count[row][col] += 1
    # dictionary for tag transition probability table
    transition_prob = dict((tag,0) for tag in tags_dist)
    for key in transition_prob.keys():
        transition_prob[key] = dict((tag,0) for tag in tags_dist) 
    for row in transition_prob.keys():
        for col in transition_prob[row].keys():
            transition_prob[row][col] = (1.0 * transition_count[col][row]) / tags_dist[col]
    return transition_count, transition_prob

def create_word_tag_pairs():
    words_tags = []
    for word, tag in zip(words, tags):
        words_tags += list(ngrams([tag,word], 2))
    return words_tags

def create_emission_tables():
    # create word/tag pairs
    words_tags = create_word_tag_pairs()
    # create tables
    emission_count = dict((tag,0) for tag in tags_dist)
    for key in emission_count.keys():
        emission_count[key] = dict((word,0) for word in words_dist)
    for pair in words_tags:
        row = pair[0]
        col = pair[1]
        emission_count[row][col] += 1
    # dictionary for word/tag emission probability table
    emission_prob = dict((tag,0) for tag in tags_dist)
    for key in emission_prob.keys():
        emission_prob[key] = dict((word,0) for word in words_dist) 
    for row in emission_prob.keys():
        for col in emission_prob[row].keys():
            emission_prob[row][col] = (1.0 * emission_count[row][col]) / tags_dist[row]
    return emission_prob

def write_tables_to_files():
    with open("../emission_output.csv", "w") as f:
        w = csv.writer( f )
        tag_names = list(emission_prob.values())[0].keys()
        w.writerow([""] + [key for key in list(emission_prob.values())[0].keys()])
        for key in emission_prob.keys():
            w.writerow([key] + [emission_prob[key][tag_name] for tag_name in tag_names])    
    with open("../transition_output.csv", "w") as f:
        w = csv.writer( f )
        tag_names = list(transition_prob.values())[0].keys()
        w.writerow([""] + [key for key in list(transition_prob.values())[0].keys()])
        for key in transition_prob.keys():
            w.writerow([key] + [transition_prob[key][tag_name] for tag_name in tag_names])
 
if __name__ == '__main__':
    # split train and test sets
    print("splitting train/test sets")
    train_sents, test_sents = train_test_split()
    print("creating list of words and tags")
    words, tags = get_words_and_tags()
    
    # tags and words distribution
    print("calculating words and tags distribution")
    tags_dist = FreqDist(tags)            
    words_dist = FreqDist(words)
    
    # create tables
    print("creating transition table")
    transition_count, transition_prob = create_transition_tables()
    print("creating observation likelihood tables")
    emission_prob = create_emission_tables()
        
    # write the tables to csv
    print("wrting to file")
    write_tables_to_files()