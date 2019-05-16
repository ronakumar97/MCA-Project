# -*- coding: UTF-8 -*-
import emoji
import csv
from nltk.tokenize import TweetTokenizer
import pickle
import gensim.models.keyedvectors as word2vec
import re
from nltk.corpus import stopwords
import string
import numpy as np
import gensim.models as gs
import gensim
import tqdm

from sklearn.naive_bayes import GaussianNB

data = []
labels = []
w2v = word2vec.KeyedVectors.load_word2vec_format('Dump/wiki-news-300d-1M.vec',limit=200000)
e2v = gensim.models.KeyedVectors.load_word2vec_format('Dump/emoji2vec.bin', binary=True)

def pickleLoad(filename):
	with open(filename, "rb") as f:
		filetype = pickle.load(f)
	return filetype

def pickleUnload(filename,filetype):
	with open(filename, "wb") as f:
		pickle.dump(filetype, f)

class Tweet2Vec:
	def __init__(self, dim, w2v, e2v=None):
		self.wordVecModel = w2v
		if e2v is not None:
			self.emojiVecModel = e2v
			print ("yay")
		else:
			self.emojiVecModel = dict()
		self.dimension = dim

	@classmethod
	def from_word2vec_paths(cls, dim, w2v_path='/data/word2vec/GoogleNews-vectors-negative300.bin',
							e2v_path=None):
		if not os.path.exists(w2v_path):
			print(str.format('{} not found. Either provide a different path, or download binary from '
							 'https://code.google.com/archive/p/word2vec/ and unzip', w2v_path))
		w2v = gs.Word2Vec.load_word2vec_format(w2v_path, binary=True)
		if e2v_path is not None:
			e2v = gs.Word2Vec.load_word2vec_format(e2v_path, binary=True)
		else:
			e2v = dict()
		return cls(dim, w2v, e2v)

	def __getitem__(self, item):
		phr_sum = np.zeros(self.dimension, np.float32)
		for token in item:
			if token in self.wordVecModel:
				phr_sum += self.wordVecModel[token]
				print ('word')
			elif token in self.emojiVecModel:
				phr_sum += self.emojiVecModel[token]
				print ('emoji')
		return phr_sum.flatten()

	def __setitem__(self, key, value):
		self.wordVecModel[key] = value



p2v = Tweet2Vec(300,w2v,e2v)
rows = pickleLoad("Dump/data_extraction.pkl")

for row in rows:
	data.append(p2v[row[0]])
	labels.append(row[1])

pickleUnload("Data/data.pkl",data)
pickleUnload("Data/labels.pkl",labels)
