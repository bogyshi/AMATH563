import numpy as np
import struct
import cvxpy as cp
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
filePath = '/home/bdvr/Documents/GitHub/AMATH563/hw1/'

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
#the above stolen from here https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
def loss_fn(X, Y, beta):
    return cp.norm(cp.matmul(X, beta) - Y, 2)**2

def loss_fn_l1(X, Y, beta):
    return cp.norm1(cp.matmul(X, beta) - Y)

def regularizer(beta):
    return cp.pnorm(beta, p=2)**2

def objective_fn(X, Y, beta, lambd,oneOrTwo):
    if(oneOrTwo==2):
        return loss_fn(X, Y, beta) + lambd * regularizer(beta)
    else:
        return loss_fn_l1(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value
#the above four functions have been stolen from here https://www.cvxpy.org/examples/machine_learning/ridge_regression.html
def cvxExample(m,n):
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x = cp.Variable(n)
    #objective = cp.Minimize(cp.sum_squares(A*x - b))
    objective = cp.Minimize(objective_fn(A,b,x,1.1,1))

    #constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective)

    print(A)
    print("Optimal value", prob.solve())
    print("Optimal var")
    print('MSE: '+str(mse(A,b,x.value)))
    return(x.value)

def plotGridStyle(vals,indiv):
    plt.title('28x28 grid values colorized')
    #plt.imshow(vals)
    if(indiv):
        counter = 0
        length = len(vals[0])
        print(length)
        while(counter<length):
            data = np.reshape(vals[:,counter],[-1,28])
            print(data.shape)
            plt.clf()
            plt.imshow(data,aspect='auto', cmap='gray')
            plt.show()
            counter+=1

    else:
        plt.imshow(vals, aspect='auto', cmap='gray')
        plt.show()

def reshapeSols(sol):
    counter = 0
    getcol = np.array(sol[0,:])
    depth = len(getcol)
    while(counter<depth):
        data = np.array(sol[:,counter])
        print(np.sqrt(len(data)))
        temp = np.reshape(data,(28,28))
        plotGridStyle(temp,False)
        counter+=1

def getThisNumber(train,labels,number):
    pass

def simpleSolutionAXB(train,labels,oneOrTwo,penalty):
    np.random.seed(1)
    rows = train.shape[0]
    cols = train.shape[1]
    shape = (784,10)
    print(shape)
    x=cp.Variable(shape)
    objective = cp.Minimize(objective_fn(train,labels,x,penalty,oneOrTwo))
    prob = cp.Problem(objective)

    #print(A)
    print("Optimal value", prob.solve(verbose=True))
    #print('MSE: '+str(mse(train,labels,x.value)))
    return(x.value)

def cvxoptAttempt(train,labels):
    A = cx.matrix(train)
    B = cx.matrix(labels)
    x = variable()
    holdsol = op(objective_fn(A,B,x,0,2))
    sol = holdsol.solve()
    #print(sol['x'])
def reshapeLabels(numbers):
    numEntries = len(numbers)
    result = np.zeros([10,numEntries])
    i=0
    while i < numEntries:
        result[numbers[i],i]=1
        i+=1
    return result

def reshapeLeastSquareRes(data):
    res = np.ones([784,10])
    counter = 0
    for i in data:
        if(np.count_nonzero(i)==0):
            print(counter)
        res[counter][:] = i
        counter+=1
    return res

def saveData(data,filename):
    np.save(filename,data)

def calcError(A,b,x):
    return np.linalg.norm(b-A.dot(x))

def linAlgSol(A,b):
    x = np.linalg.lstsq(A,b)[0] #https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
    return x

def pInvSol(A,b):
    pinv = (np.linalg.pinv(A)) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html
    x= pinv.dot(b)
    return x

def regLstSqr(A,b,penalty):
    clf=Ridge(alpha=penalty)
    clf.fit(A,b)
    return clf.coef_

def lasso(A,b,penalty):
    clf = linear_model.Lasso(alpha=penalty)
    clf.fit(A,b)
    return (clf.coef_)

def createBar(sol):
    xs = np.arange(784)
    for i in np.arange(10):
        plt.clf()
        plt.bar(x=xs,height=sol[i*784:((i+1)*784)])
        plt.show()
    #return lsmr(A=A,b=b,damp=penalty)[0]https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

def tinytest(smallDataIn,smallDataOut,penalty):
    xlin = linAlgSol(smallDataIn,smallDataOut)
    xinv = pInvSol(smallDataIn,smallDataOut)
    lstsqErr = calcError(smallDataIn,smallDataOut,xlin)
    pinvErr = calcError(smallDataIn,smallDataOut,xinv)
    xRLS = regLstSqr(smallDataIn,smallDataOut,10000)
    plz = np.sum(xRLS.T-xlin)
    #print('cmon' + str(plz))
    xRLS = np.reshape(xRLS.T,(784,10))
    lassoRes = lasso(smallDataIn,smallDataOut,0.001)
    lassoRes = np.reshape(lassoRes.T,(784,10))
    print('lasso')
    printStats(lassoRes)
    reshapeSols(lassoRes)
    print('reg lst sqr')
    printStats(xRLS)
    reshapeSols(xRLS)
    print('pinv')
    printStats(xinv)
    reshapeSols(xinv)
    print('lstSqr')
    printStats(xlin)
    reshapeSols(xlin)

def printStats(xs):
    print('Mean: ' + str(np.mean(xs)))
    print('StdDev: ' + str(np.std(xs)))
    print('Max: ' + str(np.max(xs)))
    print('Min: ' + str(np.min(xs)))

def debugData(xs,ys):
    print('input')
    print(xs.shape)
    print('Mean: ' + str(np.mean(xs)))
    print('StdDev: ' + str(np.std(xs)))
    print('Max: ' + str(np.max(xs)))
    print('Min: ' + str(np.min(xs)))
    print('output')
    print(ys.shape)
    print('Mean: ' + str(np.mean(ys)))
    print('StdDev: ' + str(np.std(ys)))
    print('Max: ' + str(np.max(ys)))
    print('Min: ' + str(np.min(ys)))


def shapeData(rawIn,rawOut,smallSize=100):
    counter = 0
    location=0
    numRows = len(rawIn)
    modTrain = np.zeros([numRows,28*28])
    newY = reshapeLabels(rawOut)
    modInSmall =  np.zeros([smallSize,28*28])
    modOutSmall =np.zeros([10,smallSize])
    for x in rawIn:
        modTrain[counter] = rawIn[counter].flatten()/255
        if(numRows-smallSize<counter):
            modInSmall[location] = rawIn[counter].flatten()/255
            modOutSmall[:,location] = newY[:,counter]
            location+=1
        counter+=1
    newYT = newY.T
    modOutSmall = modOutSmall.T
    return modTrain,newYT,modInSmall,modOutSmall
#np.fromfile('/home/bdvr/Documents/GitHub/AMATH563/hw1/data/t10k-images-idx3-ubyte',)
recompData = False

trainInputRaw = read_idx(filePath+'data/train-images-idx3-ubyte')
trainOutputRaw = read_idx(filePath+'data/train-labels-idx1-ubyte')

trainIn,trainOut,smallTrainIn,smallTrainOut = shapeData(trainInputRaw,trainOutputRaw,200)
print(trainIn.shape)
print(trainOut.shape)
pinvFileName = filePath+'data/pInvSol'
lstSqrFileName = filePath+'data/lstSqrSol'

if(recompData is True):
    xlin = linAlgSol(trainIn,trainOut)
    xinv = pInvSol(trainIn,trainOut)
    saveData(xlin,lstSqrFileName)
    saveData(xinv,pinvFileName)
else:
    xlin = np.load(lstSqrFileName+'.npy')
    xinv = np.load(pinvFileName+'.npy')
#reshapeSols(xinv)
debugData(trainIn,trainOut)
tinytest(trainIn,trainOut,10)
#toshape = simpleSolutionAXB(modTrainR,modTestRT,2,0.5)# TODO test this for both over and under determined systems
#res = np.linalg.lstsq(modTrainR,modTestRT)
lstsqErr = calcError(trainIn,trainOut,xlin)
pinvErr = calcError(trainIn,trainOut,xinv)
#print(lstsqErr)
#print(pinvErr)
#xRLS = regLstSqr(trainIn,trainOut,10.0)
#xRLS = np.reshape(xRLS,(784,10))
#lassoRes = lasso(smallTrainIn,smallTrainOut,0)
#lassoRes = np.reshape(lassoRes,(784,10))
#print(lassoRes)
#print(xRLS)
#reshapeSols(xRLS)

#reshapeSols(lassoRes)

#createBar(lassoRes.flatten())

#reshapeSols(lassoRes)

#plotGridStyle(reshapeLeastSquareRes(res2[0]),True)
#print(res[0][783])
#might be useful https://glowingpython.blogspot.com/2012/03/solving-overdetermined-systems-with-qr.html
#cvxoptAttempt(modTrain,newYT)
#plotGridStyle(toshape)
#print(toshape)#returns none, why? need to examine this
#result = np.reshape(toshape,(-1,28))
#test = np.reshape(test,(-1,5))
#plotGridStyle(result)
#plotGridStyle(test)
#images are 28x28, so each entry in the images are 28 'columns' per row, with each column holding another 28 values.
#labels in hrm 2 seem to make sense

'''
useful links:
https://www.cvxpy.org/examples/machine_learning/ridge_regression.html

'''
