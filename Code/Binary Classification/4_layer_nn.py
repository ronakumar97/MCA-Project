import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import keras
import tensorflow as tf
from keras.layers import LSTM,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import pickle
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


top_words = 6000
epoch_num = 5
batch_size = 64


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


def pickleLoad(filename):
	with open(filename, "rb") as f:
		filetype = pickle.load(f)
	return filetype

def pickleUnload(filename,filetype):
	with open(filename, "wb") as f:
		pickle.dump(filetype, f)

def plot_cmat(yte, ypred):
    skplt.plot_confusion_matrix(yte, ypred)
    plt.show()

def plot_cmat_seaborn(y_test, y_pred):
    import seaborn as sn
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    sn.set(font_scale=1.4)
    sn.heatmap(conf_mat, fmt='g', annot=True,annot_kws={"size": 16})# font size
    plt.title("Confusion Matrix: Neural Network")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

embed_dim = 128
lstm_out = 196
max_fatures = 2000
batch_size = 32

X = pickleLoad("Preprocessing/Data/data.pkl")
y = pickleLoad("Preprocessing/Data/labels.pkl")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train1 = y_train
y_test1 = y_test
y_train = np.array(to_categorical(y_train))
y_test = np.array(to_categorical(y_test))

model = Sequential()
model.add(Dense(300, input_dim=300, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=300)
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.predict_classes(X_test)
scores1 = model.evaluate(X_test, y_test)
predictions = [int(a) for a in scores]
print ("---F1-Score---")
print(f1_score(y_test1, predictions, average="macro"))
print ("---Precision---")
print(precision_score(y_test1, predictions, average="macro"))
print ("---Recall---")
print(recall_score(y_test1, predictions, average="macro"))
plot_cmat_seaborn(y_test1,predictions)
