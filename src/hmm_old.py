'''
Created on Feb 28, 2018

@author: User
'''
import nltk
from nltk import ngrams
from nltk.corpus import brown
from nltk.probability import FreqDist
import csv

class hmm():
    def __init__(self):
        # split train and test sets
        print("splitting train/test sets")
        self.train_sents, self.test_sents = self.train_test_split()
        print("creating list of words and tags")
        self.words, self.tags = self.get_words_and_tags()
        
        # tags and words distribution
        print("calculating words and tags distribution")
        self.tags_dist = FreqDist(self.tags)            
        self.words_dist = FreqDist(self.words)
        
        # create tables
        print("creating transition table")
        self.transition_prob = self.create_transition_table()
        print("creating observation likelihood table")
        self.emission_prob = self.create_emission_table()
        self.observation = self.get_test_data()
        print("hmm completed")
        
    def viterbi(self):
        obs_len = len(self.observation)
        v = []
        backpointer = []
        first_viterbi = {}
        first_backpointer = {}
        
        for tag in self.tags_dist:
            # don't record anything for the START tag
            if tag == "<s>": 
                continue
            first_viterbi[tag] = 1.0 * self.transition_prob["<s>"][tag] * self.emission_prob[tag][self.observation[0]]
            first_backpointer[tag] = "<s>"
        print(self.transition_prob["<s>"][tag], self.emission_prob[tag][self.observation[0]])
        print(first_backpointer)
        v.append(first_viterbi)
        backpointer.append(first_backpointer)
        currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[tag])
        print( "Word", "'" + self.observation[0] + "'", "current best two-tag sequence:", first_backpointer[currbest], currbest)
        for wordindex in range(1, len(self.observation)):
            this_viterbi = {}
            this_backpointer = {}
            prev_viterbi = v[-1]
            
            for tag in self.tags_dist:
                # don't record anything for the START tag
                if tag == "START": 
                    continue
        
                # if this tag is X and the current word is w, then 
                # find the previous tag Y such that
                # the best tag sequence that ends in X
                # actually ends in Y X
                # that is, the Y that maximizes
                # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
                # The following command has the same notation
                # that you saw in the sorted() command.
                best_previous = max(prev_viterbi.keys(),
                                    key = lambda prevtag: \
                    prev_viterbi[prevtag] * self.transition_prob[prevtag][tag] * self.emission_prob[tag][self.observation[wordindex]])
        
                # Instead, we can also use the following longer code:
                # best_previous = None
                # best_prob = 0.0
                # for prevtag in distinct_tags:
                #    prob = prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
                #    if prob > best_prob:
                #        best_previous= prevtag
                #        best_prob = prob
                #
                this_viterbi[tag] = prev_viterbi[best_previous] * \
                    self.transition_prob[best_previous][tag] * self.emission_prob[tag][self.observation[wordindex]]
                this_backpointer[ tag ] = best_previous
        
            currbest = max(this_viterbi.keys(), key=lambda tag: this_viterbi[tag])
            print( "Word", "'" + self.observation[ wordindex] + "'", "current best two-tag sequence:", this_backpointer[currbest], currbest)
            # print( "Word", "'" + sentence[ wordindex] + "'", "current best tag:", currbest)
        
        
            # done with all tags in this iteration
            # so store the current viterbi step
            v.append(this_viterbi)
            backpointer.append(this_backpointer)
        
        
        # done with all words in the sentence.
        # now find the probability of each tag
        # to have "END" as the next tag,
        # and use that to find the overall best sequence
        prev_viterbi = v[-1]
        best_previous = max(prev_viterbi.keys(),
                            key = lambda prevtag: prev_viterbi[prevtag] * self.transition_prob[prevtag]["</s>"])
        
        prob_tagsequence = prev_viterbi[ best_previous ] * self.transition_prob[best_previous]["</s>"]
        
        # best tagsequence: we store this in reverse for now, will invert later
        best_tagsequence = ["</s>", best_previous]
        # invert the list of backpointers
        backpointer.reverse()
        
        # go backwards through the list of backpointers
        # (or in this case forward, because we have inverter the backpointer list)
        # in each case:
        # the following best tag is the one listed under
        # the backpointer for the current best tag
        current_best_tag = best_previous
        for bp in backpointer:
            best_tagsequence.append(bp[current_best_tag])
            current_best_tag = bp[current_best_tag]
        
        best_tagsequence.reverse()
        print("The sentence was:", end = " ")
        for w in self.observation: 
            print(w, end = " ")
        print("\n")
        print("The best tag sequence is:", end = " ")
        for t in best_tagsequence: 
            print (t, end = " ")
        print("\n")
        print( "The probability of the best tag sequence is:", prob_tagsequence)
    
    def train_test_split(self):
        brown_tagged_sents = brown.tagged_sents(tagset="universal")
        brown_sents = brown.sents()
        
        # 90/10 train/test split
        size = int(len(brown_tagged_sents) * 0.9)
        train_sents = brown_tagged_sents[:size]
        test_sents = brown_sents[size:size+1]
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
    
    def get_test_data(self):
        # create list of words and tags
        test_data = []
        start = ["<s>"]
        end = ["</s>"]
        for sent in self.test_sents:
            test_data += start + [w for w in sent] + end
        return test_data
    
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
                transition_prob[row][col] = (1.0 * transition_count[col][row]) / self.tags_dist[col]
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
    
    def write_tables_to_files(self, transition_prob, emission_prob):
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
    hmm = hmm()
    hmm.viterbi()