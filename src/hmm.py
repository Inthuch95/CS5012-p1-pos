'''
Created on Mar 2, 2018

@author: User
'''
import nltk
from nltk.corpus import brown

class HMM():
    def __init__(self):
        self.train_sents, self.test_sents = self.train_test_split()
        self.brown_tags_words = self.get_tags_words()
        print(self.brown_tags_words[:10])
        # conditional frequency distribution
        freq_tagwords = nltk.ConditionalFreqDist(self.brown_tags_words)
        # Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
        # P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
        self.brown_tags = [tag for (tag,_) in self.brown_tags_words]
        # make conditional frequency distribution:
        # count(t{i-1} ti)
        freq_tags= nltk.ConditionalFreqDist(nltk.bigrams(self.brown_tags))
        # make conditional probability distribution, using
        # maximum likelihood estimate:
        # P(ti | t{i-1})
        self.transition_tags = nltk.ConditionalProbDist(freq_tags, nltk.MLEProbDist)
        # conditional probability distribution
        self.emission_tagwords = nltk.ConditionalProbDist(freq_tagwords, nltk.MLEProbDist)
        self.sentence = self.get_test_sentence()
        
    def viterbi(self):
        distinct_tags = set(self.brown_tags)
        viterbi = []
        backpointer = []
        
        first_viterbi = {}
        first_backpointer = {}
        for tag in distinct_tags:
            # don't record anything for the START tag
            if tag == "<s>": 
                continue
            first_viterbi[tag] = self.transition_tags["<s>"].prob(tag) * \
                self.emission_tagwords[tag].prob(self.sentence[0])
            first_backpointer[tag] = "<s>"
        
#         print(first_viterbi)
#         print(first_backpointer)
        
        viterbi.append(first_viterbi)
        backpointer.append(first_backpointer)
        
        currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])
#         print( "Word", "'" + self.sentence[0] + "'", "current best tag sequence:", currbest)
        
        for wordindex in range(1, len(self.sentence)):
            this_viterbi = { }
            this_backpointer = { }
            prev_viterbi = viterbi[-1]
            
            for tag in distinct_tags:
                if tag == "<s>": 
                    continue
                best_previous = max(prev_viterbi.keys(), key = lambda prevtag: prev_viterbi[ prevtag ] * \
                                    self.transition_tags[prevtag].prob(tag) * \
                                    self.emission_tagwords[tag].prob(self.sentence[wordindex]))
                
                this_viterbi[tag] = prev_viterbi[best_previous] * \
                    self.transition_tags[ best_previous ].prob(tag) * \
                    self.emission_tagwords[ tag].prob(self.sentence[wordindex])
                this_backpointer[tag] = best_previous
        
            currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[tag])
#             print( "Word", "'" + self.sentence[wordindex] + "'", "current best tag:", currbest)
        
            # done with all tags in this iteration
            # so store the current viterbi step
            viterbi.append(this_viterbi)
            backpointer.append(this_backpointer)
        # done with all words in the sentence.
        # now find the probability of each tag
        # to have "END" as the next tag,
        # and use that to find the overall best sequence
        prev_viterbi = viterbi[-1]
        best_previous = max(prev_viterbi.keys(), key = lambda prevtag: prev_viterbi[prevtag] * \
                            self.transition_tags[prevtag].prob("</s>"))
        
        prob_tagsequence = prev_viterbi[best_previous] * self.transition_tags[best_previous].prob("</s>")
        
        # best tag sequence: we store this in reverse for now, will invert later
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
        print( "The sentence was:", end = " ")
        for w in self.sentence: 
            print( w, end = " ")
        print("\n")
        print( "The best tag sequence is:", end = " ")
        for t in best_tagsequence: 
            print (t, end = " ")
        print("\n")
        print( "The probability of the best tag sequence is:", prob_tagsequence)
        
    def train_test_split(self):
        brown_tagged_sents = brown.tagged_sents()
        brown_sents = brown.sents()
        # 90/10 train/test split
        size = int(len(brown_tagged_sents) * 0.9)
        train_sents = brown_tagged_sents[:size]
        test_sents = brown_sents[size:size+2]
        return train_sents, test_sents
    
    def get_tags_words(self):
        brown_tags_words = []
        for sent in brown.tagged_sents(tagset="universal"):
            brown_tags_words.append(("<s>", "<s>"))
            # then all the tag/word pairs for the word/tag pairs in the sentence.
            # shorten tags to 2 characters each
            brown_tags_words.extend([(tag, word) for (word, tag) in sent])
            # then END/END
            brown_tags_words.append( ("</s>", "</s>") )
        return brown_tags_words
    
    def get_test_sentence(self):
        # create list of words and tags
        sentence = []
        for sent in self.test_sents:
            sentence += [w for w in sent]
        return sentence

if __name__ == '__main__':
    hmm = HMM()
    for sent in hmm.test_sents:
        hmm.sentence = sent
        hmm.viterbi()
        print("\n")