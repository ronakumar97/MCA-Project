import preprocessor as p
import pandas as pd


p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED)
header = ['id', 'label', 'tweet']
data = pd.read_csv("myData3.csv", sep=',', names=header)

""""#Cleaning the data and encoding the labels"""
with open("myCleanData.csv", 'w+') as file:
    for line in data.itertuples():
        y = " ".join(line[3].split())
        s = p.clean(y)
        s += ", "
        if(line[2].strip() == 'Positive'):
            s += '3'
        elif(line[2].strip() == 'Neutral'):
            s += '2'
        elif(line[2].strip() == 'Negative'):
            s += '1'
        file.write(s)
        file.write("\n")

"""Storing the tweets with utf-8 encoding"""
with open("myCleanData.csv", 'r') as infile, open("myCleanData2.csv", 'w+', encoding='utf-8') as outfile:
    for line in infile:
        t = line.split(",")
        y = " ".join(str(t[0]).encode('utf-8').decode('unicode-escape').split())
        outfile.write(y+","+t[1].strip())
        outfile.write("\n")

"""A little more cleaning, "myCleanData3.csv" is final tweet data """
with open("myCleanData2.csv", 'r', encoding='utf-8') as infile, open("myCleanData3.csv", 'w+', encoding='utf-8') as outfile:
    for line in infile:
        t = line.strip(":")
        s = t.strip()
        outfile.write(s)
        outfile.write("\n")

