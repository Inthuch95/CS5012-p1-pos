'''
Created on Mar 8, 2018

@author: User
'''
from nltk import FreqDist
import csv 

class pos_tagger():
    def __init__(self, hmm):
        self.hmm = hmm
        
    def viterbi(self):
        # get tagged sentences
        test_tagged_sents = self.hmm.tagged_sents[self.hmm.train_size:self.hmm.train_size+self.hmm.test_size]
        print("tagging {} sentences with viterbi algorithm".format(str(len(test_tagged_sents))))
        num_sent = 0
        correct_tags = 0
        num_words = 0
        accuracy_tag = {}
        
        # create tags distribution
        test_tags = []
        for sent in test_tagged_sents:
            test_tags += [t for (_,t) in sent] 
        test_tags_dist = FreqDist(test_tags)
        # create tagging accuracy dictionary 
        for tag in test_tags_dist.keys():
            accuracy_tag[tag] = {"correct":0, "all":0, "accuracy":0}
        
        # intialise confusion matrix
        confusion_matrix = dict((tag,0) for tag in test_tags_dist.keys())
        for key in confusion_matrix.keys():
            confusion_matrix[key] = dict((tag,0) for tag in test_tags_dist.keys()) 
        
        # test HMM with untagged sentences
        for test_sent in self.hmm.test_sents:
            # print progress every 50 sentences
            if num_sent % 50 == 0 and num_sent != 0:
                print("sentence processed: " + str(num_sent))
            actual_tags =  ["<s>"] + [t for (_,t) in test_tagged_sents[num_sent]] + ["</s>"]
            
            # handle unknown words
            for i in range(len(test_sent)):
                if test_sent[i] not in self.hmm.words:
                    test_sent[i] = "UNK"
            viterbi_mat = {}
            states = self.hmm.tags_dist.keys()
            
            '''initialise viterbi matrix'''
            for tag in self.hmm.tags_dist.keys():
                if tag not in viterbi_mat.keys():
                    viterbi_mat[tag] = {}
                for word in test_sent:
                    viterbi_mat[tag][word] = 0.0
            '''prediction list'''
            predictions = [0 for _ in range(len(test_sent)+2)]
            '''initialise step'''
            for state in states:
                if state not in ["<s>","</s>"]:
                    viterbi_mat[state][test_sent[0]] = self.hmm.transition_prob[state]["<s>"] * \
                        self.hmm.emission_prob[state][test_sent[0]]
                    predictions[0] = "<s>"
                    
            '''recursive step'''
            for t in range(1,len(test_sent)):
                word = test_sent[t]
                word_p = test_sent[t-1]
                backpointer = {}
                backpointer["tag"] = []
                backpointer["value"] = []
                for state in states:
                    if state not in ["<s>","</s>"]:
                        transition_p = [self.hmm.transition_prob[state][prev_state] * \
                                        viterbi_mat[prev_state][word_p] for prev_state in viterbi_mat.keys()]
                        max_transition_p = max(transition_p)
                        emission_p = self.hmm.emission_prob[state][word]
                        for prev_state in states:
                            if viterbi_mat[prev_state][word_p] * self.hmm.transition_prob[state][prev_state] == max_transition_p:
                                viterbi_mat[state][word] = max_transition_p * emission_p
                                if viterbi_mat[state][word] != 0:
                                    backpointer["tag"] += [prev_state]
                                    backpointer["value"] += [max_transition_p]
                                break
                actual_prev_pos = backpointer["tag"][backpointer["value"].index(max(backpointer["value"]))]
               
                predictions[t] = actual_prev_pos
            transition_p = [viterbi_mat[prev_state][test_sent[-1]] * \
                            self.hmm.transition_prob["<s>"][prev_state] for prev_state in viterbi_mat.keys()]
            max_transition_p = max(transition_p)
            
            '''termination step'''
            for prev_state in states:
                if viterbi_mat[prev_state][test_sent[-1]] * self.hmm.transition_prob["</s>"][prev_state] == max_transition_p:
                    viterbi_mat["</s>"][test_sent[-1]] = max_transition_p
                    if viterbi_mat["</s>"][test_sent[-1]] != 0:
                        predictions[len(test_sent)] = prev_state
                    break
            predictions[-1] = "</s>"
            '''update results'''
            num_sent += 1
            for i in range(len(predictions)):
                if predictions[i] not in ["<s>","</s>"]:
                    if predictions[i] == actual_tags[i]:
                        accuracy_tag[actual_tags[i]]["correct"] += 1
                        correct_tags += 1
                    if predictions[i] in confusion_matrix[actual_tags[i]].keys():
                        confusion_matrix[actual_tags[i]][predictions[i]] += 1 
                    num_words += 1
                    accuracy_tag[actual_tags[i]]["all"] += 1
                    accuracy_tag[actual_tags[i]]["accuracy"] = float(accuracy_tag[actual_tags[i]]["correct"] * \
                                                                     100.0/accuracy_tag[actual_tags[i]]["all"])
        accuracy = {"correct":correct_tags,
                    "words":num_words,
                    "percentage":float(correct_tags*100.0/num_words)}
        self.save_results(predictions, actual_tags, accuracy, accuracy_tag, confusion_matrix)
    
    def save_results(self, predictions, actual_tags, accuracy, accuracy_tag, confusion_matrix):
        self.predictions = predictions
        self.actual_tags = actual_tags
        self.overall_accuracy = accuracy
        self.accuracy_tag = accuracy_tag
        self.confusion_matrix = confusion_matrix
        
    def print_confusion_matrix(self):
        # print confusion matrix in table format
        row_format ="{:>10}" * (len(self.confusion_matrix) + 3)
        print (row_format.format("", *self.confusion_matrix.keys(), "Total", "Accuracy"))
        tagging_res = []
        for row in self.confusion_matrix.keys():
            for col in self.confusion_matrix.keys():
                tagging_res.append(self.confusion_matrix[row][col])
            tag_accuracy = "{0:.2f}".format(self.accuracy_tag[row]["accuracy"])
            print(row_format.format(row, *tagging_res, self.accuracy_tag[row]["all"], tag_accuracy))
            tagging_res.clear()
    
    def print_summary(self):
        for tag in self.accuracy_tag.keys():
            result = "Tag: %4s \t Correct: %4d/%4d \t Accuracy: %0.2f" % (tag, self.accuracy_tag[tag]["correct"], 
                                                       self.accuracy_tag[tag]["all"], self.accuracy_tag[tag]["accuracy"])
            print(result)  