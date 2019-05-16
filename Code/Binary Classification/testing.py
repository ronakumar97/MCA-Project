# -*- coding: UTF-8 -*-
import json
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
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
from sklearn.svm import SVC
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mahotas
import os
import h5py
from scipy.spatial import distance
import pickle
from operator import itemgetter
from sklearn.decomposition import PCA
import imutils
import csv
from collections import Counter
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import emoji
import csv
from nltk.tokenize import TweetTokenizer
import pickle
import re
from nltk.corpus import stopwords
import string

tknzr = TweetTokenizer()
fields = []
rows = []
i = 0
stop_words = set(stopwords.words('english'))

w2v = word2vec.KeyedVectors.load_word2vec_format('Preprocessing/Dump/wiki-news-300d-1M.vec',limit=100000)
e2v = gensim.models.KeyedVectors.load_word2vec_format('Preprocessing/Dump/emoji2vec.bin', binary=True)
with open('Preprocessing/Data/emojis.json') as f:
	emoji_data = json.load(f)

def pickleLoad(filename):
	with open(filename, "rb") as f:
		filetype = pickle.load(f)
	return filetype

def pickleUnload(filename,filetype):
	with open(filename, "wb") as f:
		pickle.dump(filetype, f)

def text_with_emoji(weirdInput):
	smiley = (weirdInput
	  .decode("raw_unicode_escape")
	  .encode('utf-16', 'surrogatepass')
	  .decode('utf-16')
	)
	return smiley

def preprocessing(tweet,emoji_data):
	emoji_meaning = []
	Tweet = text_with_emoji(tweet.encode("latin_1"))
	Tweet = re.sub('@[^\s]+','',Tweet)
	Tweet = re.sub(r"http\S+", "", Tweet)
	filtered_sentence = tknzr.tokenize(Tweet)
	filtered_sentence = [w.lower() for w in filtered_sentence if not w in stop_words and w is not '']
	filtered_sentence = [w.translate(str.maketrans('', '', string.punctuation)) for w in filtered_sentence]
	for a in filtered_sentence:
		for b in emoji_data:
			if(a == emoji_data[b]['char']):
				emoji_meaning+=(emoji_data[b]['keywords'])
	new_list = filtered_sentence + emoji_meaning
	return filtered_sentence + emoji_meaning

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
			elif token in self.emojiVecModel:
				phr_sum += self.emojiVecModel[token]
		return phr_sum.flatten()

	def __setitem__(self, key, value):
		self.wordVecModel[key] = value

p2v = Tweet2Vec(300,w2v,e2v)

X = pickleLoad("Preprocessing/Data/data.pkl")
y = pickleLoad("Preprocessing/Data/labels.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#Naive Bayes
# print ("Naive Bayes")
# clf = GaussianNB()
# rfclf = clf.fit(X_train, y_train)
# rfpredictions = rfclf.predict(X_test)
# accuracy = rfclf.score(X_test, y_test)
# print (accuracy)
# print ("\n\n")
#svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
#svm_model_multi = SVC(gamma='scale', decision_function_shape='ovo').fit(X_train, y_train)
#pickleUnload("svm_model.pkl",svm_model_multi)
svm_model_multi = pickleLoad("Preprocessing/Dump/eval_model.pkl")

while(1):
	k = input("\nEnter a sentence: ")
	svm_predictions = svm_model_multi.predict([p2v[preprocessing(k,emoji_data)]])
	print (svm_predictions)
# accuracy = svm_model_multi.score(X_test, y_test)
# print ("Accuracy".format(accuracy))
# print ("\n\n")

#Random Forest
# print ("Random Forest Classifier")
# clf =  RandomForestClassifier(n_estimators=100,
#                            random_state=0)
# rfclf = clf.fit(X_train, y_train)
# rfpredictions = rfclf.predict(X_test)
# accuracy = rfclf.score(X_test, y_test)
# print ("Accuracy: ".format(accuracy))
# print ("\n\n")

#LDA
# print ("KNN")
# clf = KNeighborsClassifier(2)
# lda_model = clf.fit(X_train, y_train)
# lda_predictions = lda_model.predict(X_test)
# accuracy = lda_model.score(X_test, y_test)
# print (accuracy)
# print ("\n\n")

#Linear Regression
# print ("Linear Regression")
# clf = LinearRegression()
# nb_model = clf.fit(X_train, y_train)
# accuracy = nb_model.score(X_test, y_test)
# print (accuracy)
# print ("\n\n")

#MLPClassifier
# print ("MLP Classifier")
# clf = MLPClassifier()
# model = clf.fit(X_train, y_train)
# accuracy = model.score(X_test,y_test)
# print (accuracy)
# print ("\n\n")

#Adaboost
# print ("Adaboost")
# bdt_real = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=2),
#     n_estimators=10,
#     learning_rate=1)
# bdt_real.fit(X_train, y_train)
# accuracy_score(real_test_predict, y_test)
# print ("\n\n")
