import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sn
import pandas as pd
import random

def get_shuffled_data(a,b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a,b

def get_split_data(a,split = 0.7):
    k = len(a)
    split_k = int(k*split)
    return a[:split_k],a[split_k:]

def get_results(predictions,Y_test,C1 = 1,C2 = 0):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    for i in range(len(Y_test)):
        if(Y_test[i]==C1 and predictions[i]==C1):
            TP+=1.0
        elif(Y_test[i]==C2 and predictions[i]==C1):
            FP+=1.0
        elif(Y_test[i]==C1 and predictions[i]==C2):
            FN+=1.0
        elif(Y_test[i]==C2 and predictions[i]==C2):
            TN+=1.0
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    return {'TP':TP,'FP':FP,'FN':FN,'TN':TN,'FPR':FPR,'TPR':TPR}

def get_confusion_matrix(TP,FP,FN,TN):
    l1 = [[TP,FP],[FN,TN]]
    return np.array(l1)

def get_mean(prediction):
    return np.mean(prediction)

def get_stdev(prediction,axis = 0):
    return np.std(prediction,axis = axis)

def pickleLoad(filename):
    with open(filename, "rb") as f:
        filetype = pickle.load(f)
    return filetype

def pickleUnload(filename,filetype):
    with open(filename, "wb") as f:
        pickle.dump(filetype, f)

def get_error_rate(pred, Y):
    return sum(np.array(pred) != np.array(Y)) / float(len(np.array(Y)))

def get_accuracy(pred, Y):
    return sum(np.array(pred) == np.array(Y)) / float(len(np.array(Y)))

def get_minmax(prediction):
    prediction = np.array(prediction)
    return (prediction-min(prediction))/(max(prediction)-min(prediction))

def get_zscore(prediction):
    prediction = np.array(prediction)
    return np.sign((prediction - np.mean(prediction))/(np.std(prediction,axis = 0)))

def get_tanh(prediction):
    prediction = np.array(prediction)
    t1 = np.exp([float(x) for x in prediction])
    t2 = np.exp([-1.0*x for x in prediction])
    return np.sign((t1-t2)/(t1+t2))

def confusion_matrix_show(array):
    array = np.array(array)
    df_cm = pd.DataFrame(array, range(array.shape[0]),
                      range(array.shape[1]))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    plt.show()

def get_score_fusion(zscore_prediction,minmax_prediction,tanh_prediction):
    final = (np.array(zscore_prediction) +
             np.array(minmax_prediction) +
             np.array(tanh_prediction))/3.0
    return np.sign(final)

def get_ROC_Curve(FPR,TPR):
    lw = 2
    plt.plot(FPR,TPR,color='darkorange',lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()

def get_k_fold_generator(X, y, k_fold = 5):
    subset_size = int(len(X) / k_fold)
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_test = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_test = y[k * subset_size:][:subset_size]
        yield X_train, y_train, X_test, y_test

#for X_train_k, y_train_k, X_test_k, y_test_k in k_fold_generator(X_train, y_train, num_folds):
#x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
#y = [1,1,1,1,1,1,0]
#predictions = [1,0,1,1,0,1,0]
