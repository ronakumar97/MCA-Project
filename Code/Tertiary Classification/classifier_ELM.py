from hpelm import ELM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.loadtxt("tweet_vecs.txt", delimiter=',')
Y = np.loadtxt("tweet_label.txt", delimiter=',')
Y_onehot = np.loadtxt("tweet_label_onehot.txt", delimiter=',')

X_train, X_test, Y_train, Y_test, Y_onehot_train, Y_onehot_test = train_test_split(X, Y, Y_onehot, test_size=0.20, shuffle=False)

print("Starting training...")
elm = ELM(X_train.shape[1], Y_onehot_train.shape[1])
elm.add_neurons(200, "sigm")
elm.add_neurons(100, "tanh")
elm.add_neurons(100, "sigm")
elm.add_neurons(100, "sigm")
elm.add_neurons(100, "tanh")
elm.train(X_train, Y_onehot_train, "CV", "OP", "c", k=5)
print("Finished training...")

Y_predicted_elm = elm.predict(X_test)
Y_predicted = np.zeros((Y_predicted_elm.shape[0]))
for i, row in enumerate(Y_predicted_elm):
    idx_of_max = np.argmax(row)
    Y_predicted[i] = idx_of_max+1

with open("Y_predicted.txt", 'w+') as predfile, open("Y_true.txt", 'w+') as trufile:
    for i in Y_predicted:
        predfile.write(str(i))
        predfile.write("\n")
    for i in Y_test:
        trufile.write(str(i))
        trufile.write("\n")

score = accuracy_score(Y_test, Y_predicted)

with open("score.txt", 'w+') as file:
    file.write(str(score))

print("ELM_Accuracy", score)