from phrase2vec import Phrase2Vec
import gensim as gsm
import pandas as pd
e2v = gsm.models.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
print("till here 1")
w2v = gsm.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print("till here 2")
p2v = Phrase2Vec(300, w2v, e2v)
print("till here 3")
header = ['tweet', 'label']
data = pd.read_csv("myCleanData3.csv", sep=',', names=header)
with open("tweet_vecs.txt", 'w+') as tweetfile, open("tweet_label.txt", 'w+') as labelfile:
    for line in data.itertuples():
        vec = p2v[str(line[1])]
        tweetfile.write(','.join(map(str, vec)))
        tweetfile.write("\n")
        labelfile.write(str(line[2]))
        labelfile.write("\n")
