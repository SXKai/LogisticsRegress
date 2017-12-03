# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 21:41:01 2017

@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
def loadData():
    labelVec = []
    dataMat = []
    with open('testSet.txt') as f:
        for line in f.readlines():
            dataMat.append([1.0,line.strip().split()[0],line.strip().split()[1]])
            labelVec.append(line.strip().split()[2])
    return dataMat,labelVec

def Sigmoid(inX):
    return 1/(1+np.exp(-inX))
    
def trainLR(dataMat,labelVec):
    dataMatrix = np.mat(dataMat).astype(np.float64)
    lableMatrix = np.mat(labelVec).T.astype(np.float64)
    m,n = dataMatrix.shape
    w = np.ones((n,1))
    alpha = 0.001
    for i in range(500):
        predict = Sigmoid(dataMatrix*w)
        error = predict-lableMatrix
        w = w - alpha*dataMatrix.T*error
    return w
    
    
def plotBestFit(wei,data,label):
    if type(wei).__name__ == 'ndarray':
        weights = wei
    else:
        weights = wei.getA()
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    xxx = np.arange(-3,3,0.1)
    yyy = - weights[0]/weights[2] - weights[1]/weights[2]*xxx
    ax.plot(xxx,yyy)
    cord1 = []
    cord0 = []
    for i in range(len(label)):
        if label[i] == 1:
            cord1.append(data[i][1:3])
        else:
            cord0.append(data[i][1:3])
    cord1 = np.array(cord1)
    cord0 = np.array(cord0)
    ax.scatter(cord1[:,0],cord1[:,1],c='red')
    ax.scatter(cord0[:,0],cord0[:,1],c='green')
    plt.show()

def stocGradAscent(dataMat,labelVec,trainLoop):
    m,n = np.shape(dataMat)
    w = np.ones((n,1))
    for j in range(trainLoop):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(i+j+1) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            predict = Sigmoid(np.dot(dataMat[dataIndex[randIndex]],w))
            error = predict - labelVec[dataIndex[randIndex]]
            w = w - alpha*error*dataMat[dataIndex[randIndex]].reshape(n,1)
            np.delete(dataIndex,randIndex,0)
    return w
    
if __name__ == "__main__":
    data,label = loadData()
    data = np.array(data).astype(np.float64)
    label = [int(item) for item in label]
#    weight = trainLR(data,label)
    weight = stocGradAscent(data,label,300)    
    plotBestFit(weight,data,label)