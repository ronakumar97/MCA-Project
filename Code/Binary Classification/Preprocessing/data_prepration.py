from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

score_value = 0.65
analyzer = SentimentIntensityAnalyzer()
scores = []
sentences = []

with open('dataset.csv') as f:
	rows = csv.reader(f)
	for row in rows:
		vs = analyzer.polarity_scores(row[1])
		if(vs['compound'] >= 0.05):
			scores.append([row[1],1])
		elif (vs['compound'] <= -0.05):
			scores.append([row[1],0])

with open('dataset1.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerows(scores)
