# -*- coding: UTF-8 -*-
# b = "\U0001F600"
# print (b)
# #😀
import emoji
import csv
from nltk.tokenize import TweetTokenizer
import pickle
import re
from nltk.corpus import stopwords
import string

filename = "dataset1.csv"
tknzr = TweetTokenizer()
fields = []
rows = []
i = 0
stop_words = set(stopwords.words('english'))

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

def preprocessing(tweet):
	Tweet = text_with_emoji(tweet.encode("latin_1"))
	Tweet = re.sub('@[^\s]+','',Tweet)
	Tweet = re.sub(r"http\S+", "", Tweet)
	filtered_sentence = tknzr.tokenize(Tweet)
	filtered_sentence = [w.lower() for w in filtered_sentence if not w in stop_words and w is not '']
	filtered_sentence = [w.translate(str.maketrans('', '', string.punctuation)) for w in filtered_sentence]
	return filtered_sentence

with open(filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		rows.append([preprocessing(row[0]),int(row[1])])

pickleUnload("Dump/data_extraction.pkl",rows)
