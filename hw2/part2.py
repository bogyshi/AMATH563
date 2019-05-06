import numpy as np
import struct
import cvxpy as cp
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.lines as lines
import matplotlib.cbook as cbook
import cvxopt as cx
from cvxopt.modeling import op, dot, variable
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import lsmr
from sklearn import linear_model
from decimal import Decimal
from scipy.interpolate import interp1d
import h5py

smallModel = True
grounded = False

def fitModel(derivs,A,penalty=0.1,iters=0):
    b=derivs
    if(iters>0):
        clf = linear_model.Lasso(alpha=penalty,max_iter=iters)
    else:
        clf = linear_model.Lasso(alpha=penalty)
    clf.fit(A,b)
    return (clf.coef_)

def transforms(inputs,single=False):
    numRows = inputs.shape[0]
    if(smallModel):
        numCols=4
        lib = np.ones([numRows,numCols])
        lib[:,1] = inputs
        lib[:,2] = np.gradient(inputs)
        lib[:,3] = np.gradient(np.gradient(inputs))
    else:
        numCols=7
        lib = np.ones([numRows,numCols])
        lib[:,1] = inputs
        lib[:,2] = np.gradient(inputs)
        lib[:,3] = np.gradient(np.gradient(inputs))
        lib[:,4] = np.sin(inputs)
        lib[:,5] = np.sin(np.gradient(inputs))
        lib[:,6] = np.sin(np.gradient(np.gradient(inputs)))
    return lib

def rePredict(initPoints,coefs,numsteps,actualPoints):
    myPoints=np.zeros([initPoints.shape[0],1])
    myPoints[:,0] = initPoints.flatten()
    myGuesses = np.zeros([initPoints.shape[0],numsteps])
    myADerivs = np.zeros([initPoints.shape[0],numsteps])

    counter = 0
    while(counter<numsteps):
        myGuesses[:,counter]=myPoints.flatten()
        if(grounded):
            myDerivs = transforms(actualPoints[counter])
        else:
            myDerivs = transforms(myPoints.flatten())
        myADerivs[:,counter] = myDerivs.dot(coefs).T.flatten()
        myPoints = myPoints.flatten() + myDerivs.dot(coefs).T.flatten()
        counter+=1
    return myGuesses,myADerivs

def calcMSE(trueData,myData):
    counter = 0
    totalErr = 0
    for y in trueData:
        totalErr += np.sum((myData[:,counter] -y)**2)/351
        counter+=1
    return (totalErr/counter)

def showSlice(newData):
    plt.clf()
    plt.imshow(np.reshape(newData,(351,1)),aspect='auto')
    plt.show()

def createBar(data,title):
    '''
    plots the magnitude of weights.
    '''
    plt.clf()
    xs = np.arange(len(data))
    newTitle = title
    plt.title(newTitle)
    plt.bar(x=xs,height=data)
    plt.ylabel('weight value')
    plt.xlabel('coefficient index')
    plt.axhline(0, color='red', lw=1)
    plt.show()

def main(penalty):
    file = h5py.File('BZ.mat','r')
    l = list(file.keys())
    data = file['BZ_tensor']
    counter=0
    altData = data[:,0,:]
    x = (np.gradient(altData))

    inputValues = altData.flatten()
    output = x[0].flatten()
    inputMatrix = transforms(inputValues)
    file.close()
    coefs = fitModel(output,inputMatrix,penalty)
    print('penalty: ' + str(penalty))
    print('coefs')
    print(coefs)
    guesses,mypderivs = rePredict(altData[0],coefs,len(altData),altData)
    print('time:'+str(0))
    showSlice(altData[0])
    showSlice(guesses[:,0])
    print('time: 600')
    showSlice(altData[600])
    showSlice(guesses[:,600])
    print('time: 1200')
    showSlice(altData[1199])
    showSlice(guesses[:,1199])
    print('diff: ' + str(guesses[:,0]-guesses[:,600]))

    print('MSE of this model:')
    print(calcMSE(altData,guesses))
    print('MSE of this model derivs:')
    print(calcMSE(np.array(x)[0],mypderivs))
    print('baseline')
    print(calcMSE(altData[0:-1,:],altData[1:,:].T))
    print('baseline')
    print(calcMSE(np.array(x)[0][0:-1,:],np.array(x)[0][1:,:].T))
    print('MSE of this model (first 5 steps):')
    print(calcMSE(altData[0:5],guesses[:,0:5]))
    print('MSE of this model derivs (first 5 steps):')
    print(calcMSE(np.array(x)[0][0:5],mypderivs[:,0:5]))
    print('baseline')
    print(calcMSE(altData[0:5,:],altData[1:6,:].T))
    print('baseline')
    print(calcMSE(np.array(x)[0][0:5,:],np.array(x)[0][1:6,:].T))
    print('MSE of this model (first step):')
    print(calcMSE(altData[0:1],guesses[:,0:1]))
    print('MSE of this model derivs (first step):')
    print(calcMSE(np.array(x)[0][0:1],mypderivs[:,0:1]))
    print('baseline')
    print(calcMSE(altData[0:1,:],altData[1:2,:].T))
    print('baseline')
    print(calcMSE(np.array(x)[0][0:1,:],np.array(x)[0][1:2,:].T))

    createBar(coefs,'Model 1 Coefficient Values')
main(0.00001)
