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
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import scikitplot as skplt1
from sklearn.manifold import TSNE
from sklearn import metrics

filename = "dataset.csv"
tknzr = TweetTokenizer()
fields = []
rows = []
i = 0
stop_words = set(stopwords.words('english'))
#
# w2v = word2vec.KeyedVectors.load_word2vec_format('Preprocessing/Dump/wiki-news-300d-1M.vec',limit=1000)
# e2v = gensim.models.KeyedVectors.load_word2vec_format('Preprocessing/Dump/emoji2vec.bin', binary=True)
# with open('Preprocessing/Data/emojis.json') as f:
# 	emoji_data = json.load(f)

def ROC_Curve(FPR,TPR):
    #ROC Curve Plot
    lw = 2
    plt.plot(FPR,TPR,color='darkorange',lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()

def plot_cmat_seaborn(y_test, y_pred,name):
	import seaborn as sn
	from sklearn.metrics import confusion_matrix
	conf_mat = confusion_matrix(y_test, y_pred)
	conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
	sn.set(font_scale=1.4)
	sn.heatmap(conf_mat, fmt='g', annot=True,annot_kws={"size": 16})# font size
	title = "Confusion Matrix: "+name
	plt.title(title)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

def plot_roc(y_test, y_pred,name):
	import seaborn as sn
	y_test = np.array(y_test)
	y_pred = np.array(y_pred)
	y_test =  y_test.reshape(len(y_test),1)
	y_pred =  y_pred.reshape(len(y_pred),1)
	skplt1.metrics.plot_roc_curve(y_test, y_pred)
	title = "ROC Curve : "+name
	plt.title(title)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()


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
	# for a in filtered_sentence:
	# 	for b in emoji_data:
	# 		if(a == emoji_data[b]['char']):
	# 			emoji_meaning+=(emoji_data[b]['keywords'])
	# new_list = filtered_sentence + emoji_meaning
	# print (new_list)
	return filtered_sentence

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

# p2v = Tweet2Vec(300,w2v,e2v)
X = pickleLoad("Preprocessing/Data/data.pkl")
y = pickleLoad("Preprocessing/Data/labels.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#Naive Bayes
print ("************ Naive Bayes ************")
clf = GaussianNB()
rfclf = clf.fit(X_train, y_train)
rfpredictions = rfclf.predict(X_test)
accuracy = rfclf.score(X_test, y_test)
print ("---Accuracy---")
print(accuracy)
print ("---F1-Score---")
print(f1_score(y_test, rfpredictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test, rfpredictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test, rfpredictions, average="macro"))
# plot_cmat_seaborn(y_test,rfpredictions,"Naive Bayes")
print ("\n\n")
fpr, tpr, _ = metrics.roc_curve(y_test,  rfpredictions)
auc = metrics.roc_auc_score(y_test, rfpredictions)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

skplt1.metrics.plot_roc_curve(y_test, rfpredictions)
plt.show()

#SVM
print ("************ SVM ************")
svm_model_multi = SVC(gamma='scale', decision_function_shape='ovo').fit(X_train, y_train)
svm_predictions = svm_model_multi.predict(X_test)
accuracy = svm_model_multi.score(X_test, y_test)
print ("---Accuracy---")
print(accuracy)
print ("---F1-Score---")
print(f1_score(y_test, svm_predictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test, svm_predictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test, svm_predictions, average="macro"))
# plot_cmat_seaborn(y_test,svm_predictions,"SVM")
print ("\n\n")

#Decision Tree Classifier
print ("************ Decision Tree Classifier ************")
clf =  DecisionTreeClassifier(max_depth=5)
rfclf = clf.fit(X_train, y_train)
rfpredictions = rfclf.predict(X_test)
accuracy = rfclf.score(X_test, y_test)
print ("---Accuracy---")
print(accuracy)
print ("---F1-Score---")
print(f1_score(y_test, rfpredictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test, rfpredictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test, rfpredictions, average="macro"))
plot_cmat_seaborn(y_test,rfpredictions,"Decision Tree Classifier")
print ("\n\n")

#Random Forest
print ("************ Random Forest Classifier ************")
clf =  RandomForestClassifier(n_estimators=100,
                           random_state=0)
rfclf = clf.fit(X_train, y_train)
rfpredictions = rfclf.predict(X_test)
accuracy = rfclf.score(X_test, y_test)
print ("---Accuracy---")
print(accuracy)
print ("---F1-Score---")
print(f1_score(y_test, rfpredictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test, rfpredictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test, rfpredictions, average="macro"))
plot_cmat_seaborn(y_test,rfpredictions,"Random Forest Classifier")
print ("\n\n")

#KNN
print ("************ KNN ************")
clf = KNeighborsClassifier(2)
lda_model = clf.fit(X_train, y_train)
lda_predictions = lda_model.predict(X_test)
accuracy = lda_model.score(X_test, y_test)
print ("---Accuracy---")
print(accuracy)
print ("---F1-Score---")
print(f1_score(y_test, lda_predictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test, lda_predictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test, lda_predictions, average="macro"))
plot_cmat_seaborn(y_test,lda_predictions,"KNN")
print ("\n\n")

#MLPClassifier
print ("************ MLP Classifier ************")
clf = MLPClassifier()
model = clf.fit(X_train, y_train)
mlp_predictions = model.predict(X_test)
accuracy = model.score(X_test,y_test)
print ("---Accuracy---")
print(accuracy)
print ("---F1-Score---")
print(f1_score(y_test, mlp_predictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test, mlp_predictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test, mlp_predictions, average="macro"))
# plot_cmat_seaborn(y_test,mlp_predictions,"Multi Layer Perceptron")
print ("\n\n")

#Adaboost
print ("************ Adaboost ************")
bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=100)
model = bdt_real.fit(X_train, y_train)
ada_predictions = model.predict(X_test)
accuracy = model.score(X_test,y_test)
print ("---Accuracy---")
print(accuracy)
print ("---F1-Score---")
print(f1_score(y_test, ada_predictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test, ada_predictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test, ada_predictions, average="macro"))
plot_cmat_seaborn(y_test,ada_predictions,"Adaboost")
print ("\n\n")


# #Voting Classifier
# print ("************ Voting Classifier ************")
# eclf1 = VotingClassifier(estimators=[('mlp', model), ('svm', svm_model_multi)],voting='soft')
# eclf1 = eclf1.fit(X_train, y_train)
# ada_predictions = eclf1.predict(X_test)
# accuracy = eclf1.score(X_test,y_test)
# print ("---Accuracy---")
# print(accuracy)
# print ("---F1-Score---")
# print(f1_score(y_test, ada_predictions, average="macro"))
# print ("---Precision---")
# print(precision_score(y_test, ada_predictions, average="macro"))
# print ("---Recall---")
# print(recall_score(y_test, ada_predictions, average="macro"))
# plot_cmat_seaborn(y_test,ada_predictions,"Voting Classifier")
# print ("\n\n")
