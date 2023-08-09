import numpy as np
import operator
from functools import reduce

def geometric_mean(values):
    return (reduce(operator.mul, values)) ** (1.0 / len(values))

class Dist(object):
    def __init__(self):
        super(Dist, self).__init__()

    def _get_ngram(self, line, n):
        words = line.strip().split(" ")
        length = len(words)
        ngrams = []
        k = n - 1
        for i in range(0, length-k):
            ngrams.append(" ".join(words[i:i+k+1]))

        return ngrams

    def get_ngram_ratio(self, sents, n):
        dic = {}
        total_ngram_num = 1e-12
        for sent in sents:
            ngrams = self._get_ngram(sent.strip(), n)
            total_ngram_num += len(ngrams)

            for token in ngrams:
                dic[token] = 1

        #print (len(dic), total_ngram_num)
        ngram_ratio = len(dic) / float(total_ngram_num)
        return ngram_ratio


    def calculate(self, sents, max_order=4):
        
        dist_vec = [self.get_ngram_ratio(sents, n) for n in range(1, max_order+1)]

        dist = geometric_mean(dist_vec)

        return (dist, list(dist_vec))