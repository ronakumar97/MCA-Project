import numpy as np

""" Onehot encoding of the labels """
Y = np.loadtxt("tweet_label.txt", delimiter=',')
onehot = np.zeros((Y.shape[0],3))
for i, label in enumerate(Y): #<3,2,1>
    if label == 3:
        onehot[i,0] = 1
    elif label == 2:
        onehot[i,1] = 1
    elif label == 1:
        onehot[i,2] = 1
with open("tweet_label_onehot.txt", 'w+') as file:
    for i, row in enumerate(onehot):
        file.write(','.join(map(str, row)))
        file.write("\n")