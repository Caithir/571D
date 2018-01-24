
import numpy as np

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math



#data is a set of tuples where first element is a vector data point, and the second is the lable.
def perceptron(data, epochs):
    w =  np.zeros(data[0][0].size)
    accuracy = []
    N = len(data)
    for e in range(epochs):
        correctPrediction = 0
        falsePrediction = 0
        for dataPoint in data:
            point = dataPoint[0]
            label = dataPoint[1]
            if label * np.dot(w, point) <= 0:
                w += label*point
                update = True
                falsePrediction +=1
            else:
                correctPrediction += 1
        accuracy.append((correctPrediction/(falsePrediction+correctPrediction), e))

    return w, accuracy

#might have to loop through and do the update for each individual feature
def BalancedWinnow(data, epochs, eda):
    #p is number of features
    p = data[0][0].size
    wp = np.full(data[0][0].size, 1/(2*p))
    wn = np.full(data[0][0].size, 1/(2*p))
    update = False
    accuracy = []
    N = len(data)
    for e in range(epochs):
        correctPrediction = 0
        for dataPoint in data:
            point = dataPoint[0]
            label = dataPoint[1]
            if label * (np.dot(wp, point) - np.dot(wn, point)) <= 0:

                for j in range(wn.size):
                    wn[j] = wn[j] * math.exp(eda*label*point[j])
                for j in range(wp.size):
                    wp[j] = wp[j] * math.exp((-1)*eda * label* point[j])

                # wp = wp*np.exp(eda*label*point)
                # wn = wn * np.exp(-eda * label * point)
                s = 0
                for j in range(wn.size):
                    s += wn[j]+wp[j]
                    # if j == 159:
                    #     print(wn[j])
                    #     print(wp[j])
                    #     print("wtf")

                wp = wp/s
                wn = wn/s
                update = True
            else:
                correctPrediction += 1

        if not update:
            print("broke early, no update")
            break
        update = False
        accuracy.append((correctPrediction / N, e))
    return (wp, wn), accuracy


def perceptronROC(data, epochs):
    w = np.zeros(data[0][0].size)
    ROCS = []
    dotValues = []
    for e in range(epochs):
        for dataPoint in data:
            point = dataPoint[0]
            label = dataPoint[1]
            if label * np.dot(w, point) <= 0:
                w += label*point
        if e == epochs//3 or e == epochs-1:
            #ROC curve computation
            ROC = []
            for b in range(900, -450, -1):
                tN, tP, fN, fP = (0, 0, 0, 0)
                for dataPoint in data:
                    point = dataPoint[0]
                    label = dataPoint[1]
                    dotValues.append(np.dot(w, point)*label)

                    if label * (np.dot(w, point) - b) <= 0:
                        # inccorect label
                        if label == 1:
                            fN += 1
                        else:
                            fP += 1
                    else:
                        if label == 1:
                            tP += 1
                        else:
                            tN += 1
                #x is FPR and y is TPR
                ROC.append((fP/(fP+tN), tP/(tP+fN)))
            ROCS.append(ROC)
    # print(max(dotValues))
    # print(min(dotValues))
    return ROCS

def confusionMatrix(data, w):
    tN, tP, fN, fP = (0, 0, 0, 0)
    for dataPoint in data:
        point = dataPoint[0]
        label = dataPoint[1]
        if label * np.dot(w, point) <= 0:
            #inccorect label
            if label == 1:
                fN+=1
            else:
                fP+=1
        else:
            if label ==1:
                tP +=1
            else:
                tN+=1
    return (tP, tN, fP, fN)
#simple left corner reimman sum
def AUC(points):
    auc = 0
    for index in range(len(points)-1):
        x = points[index][0]
        y = points[index][1]
        x2 = points[index+1][0]

        auc += (x2-x)*y
    return auc



def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    epochs = 100

    train_data_pos = [(x_train[i].flatten()/256, 1) for i in range(len(x_train)) if y_train[i] == 9]
    train_data_neg = [(x_train[i].flatten()/256, -1) for i in range(len(x_train)) if y_train[i] == 4]

    test_data_pos = [(x_test[i].flatten()/256, 1) for i in range(len(x_test)) if y_test[i] == 9]
    test_data_neg = [(x_test[i].flatten()/256, -1) for i in range(len(x_test)) if y_test[i] == 4]

    train_data = train_data_neg + train_data_pos
    test_data = test_data_neg + test_data_pos



    w_train, accuracy_train = perceptron(train_data, epochs)
    w_test, accuracy_test = perceptron(test_data, epochs)

    # wwinn, accuracy_train = BalancedWinnow(train_data, epochs, 2)

    tP, tN, fP, fN = confusionMatrix(test_data, w_test)
    acc = (tP+tN)/(tP+tN+fN+fP)
    print(f"tp: {tP}")
    print(f"fp: {fP}")
    print(f"tn: {tN}")
    print(f"fn: {fN}")
    print(f"accuracy: {acc}")

    ROC_data = perceptronROC(train_data, epochs)

    print(f"w' AUC: {AUC(ROC_data[0])}")
    print(f"w* AUC: {AUC((ROC_data[1]))}")

    df_train = pd.DataFrame({"x": range(epochs), "y": [x[0] for x in accuracy_train]})
    df_test = pd.DataFrame({"x": range(epochs), "y": [x[0] for x in accuracy_test]})

    winn_train = pd.DataFrame({"x": range(epochs), "y": [x[0] for x in accuracy_train]})
    winn_test = pd.DataFrame({"x": range(epochs), "y": [x[0] for x in accuracy_train]})

    df_ROC_first = pd.DataFrame({"x": [x[0] for x in ROC_data[0]], "y": [y[1] for y in ROC_data[0]]})
    df_ROC_end = pd.DataFrame({"x": [x[0] for x in ROC_data[1]], "y": [y[1] for y in ROC_data[1]]})

    # plt.plot('x', 'y', data=df_train, linestyle='-', marker='o')
    # plt.plot('x', 'y', data=df_test, linestyle='-', marker='o')

    # plt.plot('x', 'y', data=winn_train, linestyle='-', marker='o')
    # plt.plot('x', 'y', data=winn_test, linestyle='-', marker='o')

    plt.plot('x', 'y', data=df_ROC_first, linestyle='-', marker='o')
    plt.plot('x', 'y', data=df_ROC_end, linestyle='-', marker='o')

    plt.show()


if __name__ == '__main__':
    main()