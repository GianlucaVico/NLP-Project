import re
import numpy as np
import time
from sklearn.naive_bayes import ComplementNB, MultinomialNB

class BayesModel:
    def __init__(self, split, ignore_chars, stop_words, categories, size = 3, alpha = 1):
        '''
        split: character used to split words
        ignore_chars: characters removed
        stop_words: words removed
        categories: set of possible labels
        size: n-gram size, default is 3
        alpha: used for alpha-smoothing, default is 1
        '''
        self._split = split        
        self._ignore_chars = ignore_chars
        self._re_ignore = "[" +ignore_chars + "]"
        self._stop = stop_words
        
        self._cat = categories
        self._size = size
        
        self._nbm = MultinomialNB(alpha = alpha)
        self._voc = {}
        
    def fit(self, corpus, class_vector):
        '''
        Corpus: list of sentences used to train the model
        
        For example: corpus = ["This is a sentence", "This is the second sentence]
        '''
        tot = len(corpus)
        self._voc = {"<s>": 1} #token: index, 0 for oov
        
        for x,y in zip(enumerate(corpus), class_vector):
            n = x[0]
            x = x[1]
            tokens = self._tokenize(x)
            #update n-gram counters
            tokens = ["<s>"]*(self._size-1)+tokens + ["<s>"]*(self._size-1)
            
            for i in tokens:
                if self._voc.get(i, 0) == 0: #new word
                    self._voc[i] = len(self._voc) + 1
                    
            ngrams = []
            
            for i in range(self._size, len(tokens)):
                ngrams.append([self._voc[j] for j in tokens[i-self._size:i]])            
            
            self._nbm.partial_fit(ngrams, [y]*len(ngrams), self._cat)
            
            train_bar(n+1, tot, "Training:")
        self._voc_size = len(self._voc)
        
    def predict_proba(self, corpus):
        '''
        Corpus: list of sentences
        
        For example: corpus = ["This is a sentence", "This is the second sentence]
        
        Return the probability of each category, one row for each sentence
        '''
        y = []
        tot = len(corpus)
        for n, i in enumerate(corpus):
            tokens = self._tokenize(i)
            tokens = ["<s>"]*(self._size-1)+tokens + ["<s>"]*(self._size-1)
            ngrams = []            
            for j in range(self._size, len(tokens)):                
                ngrams.append([self._voc.get(k, 0) for k in tokens[j-self._size:j]])            
            tot_prob = np.sum(self._nbm.predict_log_proba(ngrams), axis = 0)
            y.append(np.exp(tot_prob / len(ngrams)))
            #train_bar(n+1, tot, "Predicting:")
        return y
    
    def predict(self, corpus):
        """
        Corpus: list of sentences
        
        For example: corpus = ["This is a sentence", "This is the second sentence]
        
        Return the most probable category for each sentence
        """
        y = self.predict_proba(corpus)
        #y = np.shape(y)[1] - np.argmax(np.flip(y), axis = 1) - 1
        y = np.argmax(y, axis = 1)
        out = np.zeros(y.shape)
        for i,c in enumerate(y):
            out[i] = self._cat[c]
        return out
        
    def _tokenize(self, sentence):
        tok = sentence.split(self._split)
        tok = [re.sub(self._re_ignore, "", i) for i in tok]
        tok = [i.strip().lower() for i in tok if i != "" and i not in self._stop]
        return tok
        
def train_bar (i, tot, text, l = 50):    
    """Print progress bar"""
    p = ("{:2.2f}").format(100 * (i / tot)) # percentage
    filled = int(l * i // tot) #Number of characters of the filled part of thebar
    bar = "|" * filled + '-' * (l - filled) #bar as str
    print('\r{}: [{}] {}% completed'.format(text, bar, p), end = "", flush=True)    #Print bar, no new line and \r    
    if i >= tot: 
        print()
