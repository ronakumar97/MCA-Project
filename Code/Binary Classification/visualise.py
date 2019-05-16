import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gensim
from sklearn.decomposition import PCA
from matplotlib import font_manager as fm, rcParams
import os
import seaborn as sns
import numpy as np
import pickle
from sklearn.manifold import TSNE
import json
from collections import Counter

# fpath = os.path.join(rcParams["datapath"], "emojione-mac.ttc")
# prop = fm.FontProperties(fname=fpath)
with open('Preprocessing/Data/emojis.json') as f:
	emoji_data = json.load(f)

def preprocess(tweet,emoji_data):
	emoji_meaning = []
	for a in tweet:
		for b in emoji_data:
			if(a == emoji_data[b]['char']):
				emoji_meaning.append(a)
	return emoji_meaning

def pickleLoad(filename):
	with open(filename, "rb") as f:
		filetype = pickle.load(f)
	return filetype

def pickleUnload(filename,filetype):
	with open(filename, "wb") as f:
		pickle.dump(filetype, f)

def scatterplot(x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):
	_, ax = plt.subplots()
	ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)
	if yscale_log == True:
		ax.set_yscale('log')
	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)

#Dataset Visualisation
# X = pickleLoad("Preprocessing/Data/data.pkl")
# y = pickleLoad("Preprocessing/Data/labels.pkl")
# result = TSNE(n_components=2).fit_transform(X[:5000])
# colors = ['red','green','blue','purple']
# fig = plt.figure(figsize=(8,8))
# plt.scatter(result[:, 0], result[:, 1], c=y[:5000])
# cb = plt.colorbar()
# loc = np.arange(0,max(y[:5000]),max(y[:5000])/float(len(colors)))
# cb.set_ticks(loc)
# cb.set_ticklabels(colors)
# plt.show()

pos = []
neg = []
# X = pickleLoad("Preprocessing/Dump/data_extraction.pkl")
# for row in X:
# 	if row[1] == 1:
# 		pos+=preprocess(row[0],emoji_data)
# 	else:
# 		neg+=preprocess(row[0],emoji_data)
#
# pickleUnload("Preprocessing/Dump/pos_emoji.pkl",pos)
# pickleUnload("Preprocessing/Dump/neg_emoji.pkl",neg)
#

pos = pickleLoad("Preprocessing/Dump/pos_emoji.pkl")
neg = pickleLoad("Preprocessing/Dump/neg_emoji.pkl")
print ("Top 20 Emojis & its Occurence from Positive Labeled Tweets")
print (Counter(pos).most_common(20)) # 4, 6 times
print ("\n\n")
print ("Top 20 Emojis & its Occurence from Negative Labeled Tweets")
print (Counter(neg).most_common(20)) # 4, 6 times
