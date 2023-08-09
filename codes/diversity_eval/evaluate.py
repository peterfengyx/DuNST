from enum import EnumMeta
import json
import pickle
import bert_score
import numpy as np
import os
import csv
import argparse
import traceback

from datasets import load_metric
from bert_score import BERTScorer
from nltk.tokenize import sent_tokenize

from fast_bleu import SelfBLEU, BLEU
from .dist import Dist
from .jaccard import Jaccard
from tqdm import tqdm


class Evaluator(object):

    def __init__(self):
        self.dist_metric = Dist()
        self.jaccard_metric = Jaccard()
        
        self.sbleu_weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.), '4gram': (1/4., 1/4., 1/4., 1/4.)}
    

    def get_dist(self, ori_sents):
        
        results = self.dist_metric.calculate(ori_sents)
        print(results[1:])
        return results[0] * 100

    def get_jaccard(self, ori_sents):
        js, js_vec = self.jaccard_metric.calculate(ori_sents)
        print(js_vec)
        return js * 100
    
    def get_selfbleu(self, ori_sents):
        sents = [sent.strip().split(" ") for sent in ori_sents]
        tool = SelfBLEU(sents, self.sbleu_weights)
        scores = tool.get_score()
        res = {}
        for key, ele in scores.items():
            res[key] = sum(ele) / len(ele)
        
        print(res)
        #input(">")
        return res['4gram'] * 100
    
    def evaluate_file(self, ori_sents):
        dist = self.get_dist(ori_sents)
        jaccard = self.get_jaccard(ori_sents)
        selfbleu = self.get_selfbleu(ori_sents)
        #mauve = self.get_mauve(ori_sents)
        print("dist: %.2f" % (np.round(dist, 2)))
        print("jaccard: %.3f" % (np.round(jaccard, 3)))
        print("self bleu: %.2f" % (np.round(selfbleu, 2)))
        print("\n\n")
        #return bleu, b2, b4, length_ratio, rouge2, rouge3, rougel, rougew, bert_score, cnd, dist, jaccard, selfbleu
        return dist, jaccard, selfbleu
    
    
    def evaluate_file_inner(self, sents_vec):
        
        dist_vec, jaccard_vec, selfbleu_vec = [], [], []
        for i, sents in enumerate(sents_vec):
            if i % 100 == 0:
                print(i)
            dist_vec.append(self.get_dist(sents))
            jaccard_vec.append(self.get_jaccard(sents))
            selfbleu_vec.append(self.get_selfbleu(sents))
            
        print("inner dist: %.2f" % (np.round(np.mean(dist_vec), 2)))
        print("inner jaccard: %.3f" % (np.round(np.mean(jaccard_vec), 3)))
        print("inner self bleu: %.2f" % (np.round(np.mean(selfbleu_vec), 2)))
        print("\n\n")