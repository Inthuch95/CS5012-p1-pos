'''
Created on Mar 2, 2018
'''
from nltk.util import ngrams
from nltk import FreqDist
from sklearn.linear_model import LinearRegression
import numpy as np
from math import log10
from sklearn.metrics.regression import mean_squared_error
from collections import defaultdict

class HMM():
    def __init__(self, corpus, tagset="", smoothing="-g"):
        self.corpus = corpus
        self.tagged_sents, self.sents = self.get_sentences(tagset)
        # 90/10 train/test split
#         self.train_size = int(len(self.tagged_sents) * 0.9)
#         print("training with " + str(self.train_size) + " sentences")
#         self.test_size = int(len(self.tagged_sents) * 0.1)
        self.train_size = 5000
        self.test_size = 500
        print("creating list of words and tags")
        self.train_sents, self.test_sents = self.train_test_split()
        self.words, self.tags = self.get_words_and_tags()
#         print(self.brown_tags_words[:10])
        # tags and words distribution
        print("replacing unknown words with 'UNK'")
        self.change_unknown_words()
        self.tags_dist = FreqDist(self.tags)            
        self.words_dist = FreqDist(self.words)
        # create tables
        print("creating transition table")
        if smoothing == "-l":
            print("applying laplace smoothing")
            self.transition_prob = self.create_transition_table()
        else:
            print("applying good-turing smoothing")
            self.transition_prob = self.good_turing_transition_table()
        print("creating observation likelihood table")
        self.emission_prob = self.create_emission_table()
        print("hmm training completed")
    
    def viterbi_test(self):
        allPossibleTags = self.transition_prob.keys()
        print(allPossibleTags)
        for sent in self.test_sents:
            probability = defaultdict(dict)
            backpointer = defaultdict(dict)
            
            words = sent.split()
            
            T = len(words)
            ##### Start of Viterbi #####
            
            ## Initialization at t=1 ##
            word = words[0]
            ##### If the word is seen #####
            if word in self.emission_prob["<s>"].keys():
                for eachTag in self.emission_prob.keys():
                    probability[eachTag][0] = transitionProb['S0'][eachTag] * emissionProb[word][eachTag]
                    backpointer[eachTag][0] = 'S0'
            ##### If the word is unseen #####
        
    def viterbi(self):
        print("testing pos tagger")
        test_tagged_sents = self.tagged_sents[self.train_size:self.train_size+self.test_size]
        print("testing with " + str(len(test_tagged_sents)) + " sentences")
        num_sent = 0
        correct_tags = 0
        num_words = 0
        accuracy_tag = {}
        
        test_tags = []
        for sent in test_tagged_sents:
            test_tags += [t for (_,t) in sent] 
        test_tags_dist = FreqDist(test_tags) 
        for tag in test_tags_dist.keys():
            accuracy_tag[tag] = {"right":0, "all":0}
        
        for test_sent in self.test_sents:
            if num_sent % 10 == 0:
                print("sentence processed: " + str(num_sent))
            actual_tags =  ["<s>"] + [t for (_,t) in test_tagged_sents[num_sent]] + ["</s>"]
            for i in range(len(test_sent)):
                if test_sent[i] not in self.words:
                    test_sent[i] = "UNK"
            viterbi_mat = {}
            states = self.tags_dist.keys()
            
            '''Initialise viterbi matrix'''
            for tag in self.tags_dist.keys():
                if tag not in viterbi_mat.keys():
                    viterbi_mat[tag] = {}
                for word in test_sent:
                    viterbi_mat[tag][word] = 0.0
            '''prediction list'''
            predictions = [0 for _ in range(len(test_sent)+2)]
            '''initialise step'''
            for state in states:
                if state not in ["<s>","</s>"]:
                    viterbi_mat[state][test_sent[0]] = self.transition_prob[state]["<s>"] * \
                        self.emission_prob[state][test_sent[0]]
                    predictions[0] = "<s>"
            for t in range(1,len(test_sent)):
                word = test_sent[t]
                word_p = test_sent[t-1]
                backpointer = {}
                backpointer["tag"] = []
                backpointer["value"] = []
                for state in states:
                    if state not in ["<s>","</s>"]:
                        transition_p = [self.transition_prob[state][prev_state] * \
                                        viterbi_mat[prev_state][word_p] for prev_state in viterbi_mat.keys()]
                        max_transition_p = max(transition_p)
                        emission_p = self.emission_prob[state][word]
                        for prev_state in states:
                            if viterbi_mat[prev_state][word_p] * self.transition_prob[state][prev_state] == max_transition_p:
                                viterbi_mat[state][word] = max_transition_p * emission_p
                                if viterbi_mat[state][word] != 0:
                                    backpointer["tag"] += [prev_state]
                                    backpointer["value"] += [max_transition_p]
                                break
                actual_prev_pos = backpointer["tag"][backpointer["value"].index(max(backpointer["value"]))]
               
                predictions[t] = actual_prev_pos
            transition_p = [viterbi_mat[prev_state][test_sent[-1]] * \
                            self.transition_prob["<s>"][prev_state] for prev_state in viterbi_mat.keys()]
            max_transition_p = max(transition_p)
            for prev_state in states:
                if viterbi_mat[prev_state][test_sent[-1]] * self.transition_prob["</s>"][prev_state] == max_transition_p:
                    viterbi_mat["</s>"][test_sent[-1]] = max_transition_p
                    if viterbi_mat["</s>"][test_sent[-1]] != 0:
                        predictions[len(test_sent)] = prev_state
                    break
            predictions[-1] = "</s>"
            num_sent += 1
            for i in range(len(predictions)):
                if predictions[i] not in ["<s>","</s>"]:
                    if predictions[i] == actual_tags[i]:
                        accuracy_tag[actual_tags[i]]["right"] += 1
                        correct_tags += 1
                    num_words += 1
                    accuracy_tag[actual_tags[i]]["all"] += 1
        for key in accuracy_tag.keys():
            print(key, accuracy_tag[key])
        print("overall accuracy, correct: %d  from: %d percentage: %f \n" % \
              (correct_tags, num_words, float(correct_tags*100.0/num_words)))
    
    def get_sentences(self, selected_tagset):
        tagged_sents = self.corpus.tagged_sents(tagset=selected_tagset)
        sents = self.corpus.sents()
        return tagged_sents, sents 
            
    def change_unknown_words(self):           
        words_dist = FreqDist(self.words)
        for index, w in enumerate(self.words):
            if words_dist[w] == 1:
                self.words[index] = "UNK"
        
    def train_test_split(self):
        train_sents = self.tagged_sents[:self.train_size]
        test_sents = self.sents[self.train_size:self.train_size+self.test_size]
        return train_sents, test_sents
    
    def get_words_and_tags(self):
        # create list of words and tags
        words = []
        tags = []
        start = ["<s>"]
        end = ["</s>"]
        for sent in self.train_sents:
            words += start + [w for (w,_) in sent] + end
            tags += start + [t for (_,t) in sent] + end
        return words, tags
    
    def create_transition_table(self):
        # dictionary for tag transition count table
        transition_count = dict((tag,0) for tag in self.tags_dist)
        for key in transition_count.keys():
            transition_count[key] = dict((tag,0) for tag in self.tags_dist) 
        bigrams = list(ngrams(self.tags, 2))
        for i in bigrams:
            row = i[0]
            col = i[1]
            transition_count[row][col] += 1
        # dictionary for tag transition probability table
        transition_prob = dict((tag,0) for tag in self.tags_dist)
        for key in transition_prob.keys():
            transition_prob[key] = dict((tag,0) for tag in self.tags_dist) 
        for row in transition_prob.keys():
            for col in transition_prob[row].keys():
                transition_prob[row][col] = (1.0 * transition_count[col][row] + 1) / (self.tags_dist[col] + len(self.tags))
        return transition_prob
    
    def get_nc(self, c, linreg):
        x = [log10(c)]
        x = np.c_[np.ones_like(x), x]
        y_hat = linreg.predict(x)
        return pow(10, y_hat[0])
    
    def good_turing_transition_table(self):
        bigrams = list(ngrams(self.tags, 2))
        # dictionary for tag transition count table
        transition_count = dict((tag,0) for tag in self.tags_dist)
        for key in transition_count.keys():
            transition_count[key] = dict((tag,0) for tag in self.tags_dist) 
        for i in bigrams:
            row = i[0]
            col = i[1]
            transition_count[row][col] += 1
        bigrams_dist = FreqDist(bigrams)
        Nc = {}
        N = len(list(bigrams))
        for key in bigrams_dist.keys():
            if bigrams_dist[key] not in Nc.keys():
                Nc[bigrams_dist[key]] = 1
            else:
                Nc[bigrams_dist[key]] += 1
        x = [np.real(log10(count)) for count in Nc.keys()]
        x = np.c_[np.ones_like(x), x]
        y = [log10(Nc[key]) for key in Nc.keys()]
        linreg = LinearRegression()
        linreg.fit(x, y)
        y_hat = linreg.predict(x)
        print("MSE = ", mean_squared_error(y, y_hat))
        
        transition_prob = dict((tag,0) for tag in self.tags_dist)
        for key in transition_prob.keys():
            transition_prob[key] = dict((tag,0) for tag in self.tags_dist) 
        for row in transition_prob.keys():
            for col in transition_prob[row].keys():
                if transition_count[row][col] == 0:
                    transition_prob[row][col] = (self.get_nc(1, linreg) * 1.0) / N
                else:
                    c_star = (transition_count[row][col] + 1) * \
                    (self.get_nc(transition_count[row][col] + 1, linreg)/self.get_nc(transition_count[row][col], linreg))
                    transition_prob[row][col] = c_star / N 
        return transition_prob
    
    def create_word_tag_pairs(self):
        words_tags = []
        for word, tag in zip(self.words, self.tags):
            words_tags += list(ngrams([tag,word], 2))
        return words_tags
    
    def create_emission_table(self):
        # create word/tag pairs
        words_tags = self.create_word_tag_pairs()
        # create tables
        emission_count = dict((tag,0) for tag in self.tags_dist)
        for key in emission_count.keys():
            emission_count[key] = dict((word,0) for word in self.words_dist)
        for pair in words_tags:
            row = pair[0]
            col = pair[1]
            emission_count[row][col] += 1
        # dictionary for word/tag emission probability table
        emission_prob = dict((tag,0) for tag in self.tags_dist)
        for key in emission_prob.keys():
            emission_prob[key] = dict((word,0) for word in self.words_dist) 
        for row in emission_prob.keys():
            for col in emission_prob[row].keys():
                emission_prob[row][col] = (1.0 * emission_count[row][col]) / self.tags_dist[row]
        return emission_prob

if __name__ == '__main__':
    pass