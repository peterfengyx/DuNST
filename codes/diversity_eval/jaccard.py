import numpy as np
import operator
from functools import reduce


def geometric_mean(values):
    return (reduce(operator.mul, values)) ** (1.0 / len(values))


class Jaccard(object):
    """docstring for Diversity"""
    def __init__(self, max_order=4):
        super(Jaccard, self).__init__()
        self._n = max_order # consider 1-guram to n-gram


    def _get_ngram(self, line, n):
        words = line.strip().split(" ")
        length = len(words)
        ngrams = []
        k = n - 1
        for i in range(0, length-k):
            ngrams.append(" ".join(words[i:i+k+1]))

        return ngrams

    '''
    jaccard similarity
    '''
    def _jaccardsim(self, set1, set2):
        jaccard = len(set1 & set2) / float(len(set1 | set2))

        return jaccard


    def get_ngram_jaccard(self, sents, n):
        # build inverted index
        data = []
        inverted_dic = {}
        lens = []
        for i, sent in enumerate(sents):
            ngrams = set(self._get_ngram(sent.strip(), n))
            data.append(ngrams)
            lens.append(len(ngrams))
            for ng in ngrams:
                if ng in inverted_dic:
                    inverted_dic[ng].append(i)
                else:
                    inverted_dic[ng] = [i]

        #----------------------------------------
        ans = []
        N = len(data)
        step = int(len(data) / 100)
        for i in range(N):
            data[i] = set(data[i])
        for i in range(0, N-1):
            '''
            if i % step == 0 and i != 0:
                print ("%.2f" % (float(i)/N))
            '''
            tokens1 = data[i]

            # get indices
            indices_set = set()
            for ng in tokens1:
                indices_set.update(inverted_dic[ng])

            cache = 0.0
            for j in indices_set:
                if j > i:
                    tokens2 = data[j]
                    intersection_num = len(tokens1 & tokens2)
                    cache += (intersection_num / (lens[i] + lens[j] - intersection_num))
            cache = cache / (N-i)

            ans.append(cache)

        ans = np.mean(ans)
        #ans = 1.0 - np.max(ans)
        return ans



    def calculate(self, data):

        js_vec = []
        for n in range(1, self._n+1):
            #print ("{}-gram diversity".format(n))
            js = self.get_ngram_jaccard(data, n)

            js_vec.append(js)


        # ----------------------------
        js = geometric_mean(js_vec)

        return js, list(js_vec)