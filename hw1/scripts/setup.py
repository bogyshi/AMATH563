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
filePath = '/home/bdvr/Documents/GitHub/AMATH563/hw1/'
exploreIncorrect = False
debug = False

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

def plotGridStyle(data,type,number,filename = None,save=False):
    plt.clf()
    plt.title(type+' ' + str(number) + ' weights')
    plt.imshow(data,aspect='auto', cmap='gray')
    plt.colorbar()
    if(save):
        plt.savefig(filename)
    else:
        plt.show()

def reshapeSols(sol,penalty,type,filename = None,save=False):
    counter = 0
    getcol = np.array(sol[0,:])
    depth = len(getcol)
    while(counter<depth):
        if(debug):
            print(np.sqrt(len(data)))#debugging purposes
        data = np.array(sol[:,counter])
        penalty = str(penalty).replace('.','_')
        fileExt = filename + 'weights_p'+str(penalty)+'_' + str(counter)
        temp = np.reshape(data,(28,28))
        plotGridStyle(temp,type,counter,fileExt,save)
        counter+=1

def getThisNumber(train,labels,number):
    counter = 0
    size = len(labels)
    toFilter = np.zeros((size,1))
    while(counter < size):
        if(labels[counter,number] == 1 ):#or labels[counter,number+1] == 1
            toFilter[counter,0]=1
        counter+=1
    return toFilter #https://stackoverflow.com/questions/44142173/how-can-a-numpy-array-of-booleans-be-used-to-remove-filter-rows-of-another-numpy

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

def calcError(A,b,x,ord=None,axis=None):
    if(axis is None):
        return np.linalg.norm(b-A.dot(x),ord=ord) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
    else:
        return np.linalg.norm(b-A.dot(x),ord=ord,axis=axis)

def linAlgSol(A,b):
    x = np.linalg.lstsq(A,b)[0] #https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
    return x

def pInvSol(A,b):
    pinv = (np.linalg.pinv(A)) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html
    x= pinv.dot(b)
    return x

def regLstSqr(A,b,penalty):
    clf=Ridge(alpha=penalty,solver='svd')
    clf.fit(A,b)
    return clf.coef_

def lasso(A,b,penalty):
    clf = linear_model.Lasso(alpha=penalty)
    clf.fit(A,b)
    return (clf.coef_)

def createBar(sol,size,title,save,fileLoc,penalty,type,ext=''):
    strpenalty = str(penalty).replace('.','_')

    if(type == 'lasso'):
        imageLoc = fileLoc+'images/lasso/'+ext+'/'
    else:
        imageLoc = fileLoc+'images/RLS/'+ext+'/'

    if(ext):
        fileExt = imageLoc+ 'bar_p'+strpenalty+'_' + ext

    xs = np.arange(784)
    for i in np.arange(size):
        plt.clf()
        if(not ext):
            newTitle = title + ' on ' + str(i)
            fileExt = imageLoc+ 'bar_p'+strpenalty+'_' + str(i)
        else:
            newTitle = title
        plt.title(newTitle)
        plt.bar(x=xs,height=sol[i*784:((i+1)*784)])
        plt.ylabel('weight value')
        plt.xlabel('pixel index')
        plt.axhline(0, color='red', lw=1)
        if(not save):
            plt.show()
        else:
            plt.savefig(fileExt)

def averageClassificationError(input,output,x,isMult = False,num=0):
    res = input.dot(x)
    trueSize=0
    counter = 0
    correct=0
    size = len(input)
    while(counter<size):
        guess = round(res[counter][0])
        if(guess==(output[counter])):
            correct+=1
        counter+=1
    return correct/size

def batchClassificationError(input,output,x):
    res = input.dot(x)
    size = len(output)
    correct = 0
    counter = 0
    classifications = np.zeros(10)
    percentCorrect = np.zeros(10)
    actuals = np.sum(output,axis=0)
    while(counter<size):
        guess = np.argmax(res[counter])
        if(output[counter,guess]==1):
            percentCorrect[guess]+=(1/actuals[guess])
        counter+=1
    return percentCorrect


def createDirectories(path):
    try:
        os.mkdir(path)
    except OSError:
        if(debug):
            print ("Creation of the directory %s failed" % path)
    else:
        if(debug):
            print ("Successfully created the directory %s " % path)
    #return lsmr(A=A,b=b,damp=penalty)[0]https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

def trainData(smallDataIn,smallDataOut,penaltyRLS=10000,penaltyLasso=0.001,recompData=False,fileLoc=None,savefile=False,ext=""):
    if(ext):
        RLSFileName = fileLoc+'data/'+ext+'/RLSData_'+str(penaltyRLS)
        LassoFileName = fileLoc+'data/'+ext+'/LassoData_'+str(penaltyLasso)
        lassoImageLoc = fileLoc+'images/lasso/'+ext+'/'
        RLSImageLoc = fileLoc+'images/RLS/'+ext+'/'
    else:
        RLSFileName = fileLoc+'data/RLSData_'+str(penaltyRLS)
        LassoFileName = fileLoc+'data/LassoData_'+str(penaltyLasso)
        lassoImageLoc = fileLoc+'images/lasso/'
        RLSImageLoc = fileLoc+'images/RLS/'
    size=len(smallDataIn)
    if(recompData is True):
        xRLS = regLstSqr(smallDataIn,smallDataOut,penaltyRLS)
        lassoRes = lasso(smallDataIn,smallDataOut,penaltyLasso)
        saveData(xRLS,RLSFileName)
        saveData(lassoRes,LassoFileName)
    else:
        xRLS = np.load(RLSFileName+'.npy')
        lassoRes = np.load(LassoFileName+'.npy')

    if(ext):
        xRLS = np.reshape(xRLS.T,(784,1))
        lassoRes = np.reshape(lassoRes.T,(784,1))
    else:
        xRLS = np.reshape(xRLS.T,(784,10))
        lassoRes = np.reshape(lassoRes.T,(784,10))

    RLSErr = calcError(smallDataIn,smallDataOut,xRLS,2)
    LErr = calcError(smallDataIn,smallDataOut,lassoRes,2)
    #plz = np.sum(xRLS.T-xlin) #was for debugging
    #print('cmon' + str(plz))


    reshapeSols(lassoRes,penaltyLasso,'lasso',lassoImageLoc,savefile)
    reshapeSols(xRLS,penaltyRLS,'RLS',RLSImageLoc,savefile)
    if(debug):
        print('lasso')
        printStats(lassoRes,LErr,size)
        print('reg lst sqr')
        printStats(xRLS,RLSErr,size)
    if(exploreIncorrect is True):
        xlin = linAlgSol(trainIn,trainOut)
        xinv = pInvSol(trainIn,trainOut)
        print('pinv')
        reshapeSols(xinv,0,'pinv','',False)
        print('lstSqr')
        reshapeSols(xlin,0,'lstSqr','',False)
    return [lassoRes,xRLS]

def printStats(xs,err,size):
    print('Size:' + str(size))
    print('Mean: ' + str(np.mean(xs)))
    print('StdDev: ' + str(np.std(xs)))
    print('Max: ' + str(np.max(xs)))
    print('Min: ' + str(np.min(xs)))
    print('Total Squared Error: ' + str(err))
    print('Mean Squared Error: ' + str(err/size))

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

def writeLine(filename,line):
    with open(filename,'a') as fd:
        fd.write(line) #https://stackoverflow.com/questions/2363731/append-new-row-to-old-csv-file-python

def formRow(type,num,penalty,trainCE,trainMSE,testCE,testMSE,batch):
    finalString = str(num) +','+str(trainMSE)+','+str(trainCE)+','+str(testMSE)+','+str(testCE)+','+type+','+str(penalty)+','+str(batch)+'\n'
    return finalString

def gatherTopXData(weights):
    counter = 0
    absWeights = np.absolute(weights.flatten())
    size = len(weights)
    potentialSol = np.argsort(absWeights)[::-1]
    total = np.sum(absWeights)
    pixels=[]
    threshhold = 0.9
    totalCoverage=0.0
    newX = np.zeros([size,1])
    while(counter<size and totalCoverage<threshhold):
        curIndex = potentialSol[counter]
        curVal= weights[curIndex]
        counter+=1
        newX[curIndex]=curVal
        totalCoverage+=np.abs(curVal/total)
    return newX

def createNewSparseFullModel(weights,numbers):
    fullModel = np.zeros([784,10])
    for n in numbers:
        sparseWeights = np.reshape(gatherTopXData(weights[:,n]),(784,1))
        fullModel[:,n] = sparseWeights.flatten()
    return fullModel



#np.fromfile('/home/bdvr/Documents/GitHub/AMATH563/hw1/data/t10k-images-idx3-ubyte',)
types = ['lasso','RLS']
numbers = [0,1,2,3,4,5,6,7,8,9]

trainInputRaw = read_idx(filePath+'data/train-images-idx3-ubyte')
trainOutputRaw = read_idx(filePath+'data/train-labels-idx1-ubyte')

testInputRaw = read_idx(filePath+'data/t10k-images-idx3-ubyte')
testOutputRaw = read_idx(filePath+'data/t10k-labels-idx1-ubyte')


trainIn,trainOut,smallTrainIn,smallTrainOut = shapeData(trainInputRaw,trainOutputRaw,100)
testIn,testOut,smallTestIn,smallTestOut = shapeData(testInputRaw,testOutputRaw,200)

debugSum = 0
runIndiv = True
counter=0
write0 = False
recompData = False
writeToCSV = True
writeToSparse = True
indivPenalties = {'lasso':0.001,'RLS':10000}
fullPenalties = {'lasso':0.01,'RLS':1000.0}
trainSize = len(trainIn)
testSize = len(testIn)
csvFile = filePath+'results/fullresults0.csv'
sparseCsvFile = filePath+'results/sparseResults0.csv'

if(runIndiv):
    for x in numbers:
        counter=0
        createDirectories(filePath+'/data/'+str(x))
        createDirectories(filePath+'/images/lasso/'+ str(x))
        createDirectories(filePath+'/images/RLS/'+str(x))
        numts = getThisNumber(trainIn,trainOut,x)
        singleTest = getThisNumber(testIn,testOut,x)
        #plotGridStyle(np.reshape(numtr[5000,:],(-1,28)),'whatev',0,False)
        models = trainData(trainIn,numts,indivPenalties['RLS'],indivPenalties['lasso'],recompData,filePath,True,str(x)) #todo: bug here, somehow my results are always zero. idk if this is because my labels are weird or what
        for y in models:
            sparseWeights = gatherTopXData(y)
            title = 'pixel weights for ' + str(x) + ' using ' + types[counter]
            createBar(y.flatten(),y.shape[1],title,True,filePath,indivPenalties[types[counter]],types[counter],str(x))

            t = (types[counter])
            num = (x)
            TRCE = (1-averageClassificationError(trainIn,trainOut[:,num],y))
            TRMSE = (calcError(trainIn,np.reshape(trainOut[:,num],(-1,1)),y,2)/trainSize)
            TSCE = (1-averageClassificationError(testIn,testOut[:,num],y))
            TSMSE = (calcError(testIn,np.reshape(testOut[:,num],(-1,1)),y,2)/testSize)

            sparseTRCE = (1-averageClassificationError(trainIn,trainOut[:,num],sparseWeights))
            sparseTRMSE = (calcError(trainIn,np.reshape(trainOut[:,num],(-1,1)),sparseWeights,2)/trainSize)
            sparseTSCE = (1-averageClassificationError(testIn,testOut[:,num],sparseWeights))
            sparseTSMSE = (calcError(testIn,np.reshape(testOut[:,num],(-1,1)),sparseWeights,2)/testSize)
            if(writeToSparse):
                insert = formRow(t,num,indivPenalties[types[counter]],sparseTRCE,sparseTRMSE,sparseTSCE,sparseTSMSE,False)
                if(write0 is True):
                    if(types[counter]=='RLS'):
                        writeLine(sparseCsvFile,insert)
                else:
                    writeLine(sparseCsvFile,insert)

            else:
                print(t)
                print(num)
                print('sparse training classification error: ' + str(sparseTRCE))
                print('sparse training MSE: ' + str(sparseTRMSE))
                print('sparse testing classification error: ' + str(sparseTSCE))
                print('sparse testing MSE: ' + str(sparseTSMSE))
            if(writeToCSV):
                insert = formRow(t,num,indivPenalties[types[counter]],TRCE,TRMSE,TSCE,TSMSE,False)
                if(write0 is True):
                    if(types[counter]=='RLS'):
                        writeLine(csvFile,insert)
                else:
                    writeLine(csvFile,insert)
            else:
                print(t)
                print(num)
                print('training classification error: ' + str(TRCE))
                print('training MSE: ' + str(TRMSE))
                print('testing classification error: ' + str(TSCE))
                print('testing MSE: ' + str(TSMSE))

            counter+=1
    if(debug):
        print(debugSum)

if(debug):
    debugData(trainIn,trainOut)
models = trainData(trainIn,trainOut,fullPenalties['RLS'],fullPenalties['lasso'],recompData,filePath,True)
counter = 0

for x in models:
    title = 'pixel weights using ' + types[counter]
    createBar(x.flatten(),x.shape[1],title,True,filePath,fullPenalties[types[counter]],types[counter])
    sparseFull = createNewSparseFullModel(x,numbers)
    allSparseResTrain = batchClassificationError(trainIn,trainOut,sparseFull)
    allFullResTrain = batchClassificationError(trainIn,trainOut,x)
    allSparseResTest = batchClassificationError(testIn,testOut,sparseFull)
    allFullResTest = batchClassificationError(testIn,testOut,x)
    allSparseMSETrain = calcError(trainIn,trainOut,sparseFull,axis=0)
    allSparseMSETest = calcError(testIn,testOut,sparseFull,axis=0)
    allFullMSETrain = calcError(trainIn,trainOut,x,axis=0)
    allFullMSETest = calcError(testIn,testOut,x,axis=0)
    for y in numbers:
        t = (types[counter])
        num = y

        TRCE = (1-allFullResTrain[num])
        TRMSE = allFullMSETrain[num]/trainSize
        TSCE = (1-allFullResTest[num])
        TSMSE = allFullMSETest[num]/testSize

        sparseTRCE =  (1-allSparseResTrain[num])
        sparseTRMSE = allSparseMSETrain[num]/trainSize
        sparseTSCE =  (1-allSparseResTest[num])
        sparseTSMSE =  allSparseMSETest[num]/testSize
        if(writeToSparse):
            insert = formRow(t,num,fullPenalties[types[counter]],sparseTRCE,sparseTRMSE,sparseTSCE,sparseTSMSE,True)
            if(write0 is True):
                if(types[counter]=='RLS'):
                    writeLine(sparseCsvFile,insert)
            else:
                writeLine(sparseCsvFile,insert)
        else:
            print(t)
            print(num)
            print('sparse training classification error: ' + str(sparseTRCE))
            print('sparse training MSE: ' + str(sparseTRMSE))
            print('sparse testing classification error: ' + str(sparseTSCE))
            print('sparse testing MSE: ' + str(sparseTSMSE))
        if(writeToCSV):
            insert = formRow(t,num,fullPenalties[types[counter]],TRCE,TRMSE,TSCE,TSMSE,True)
            if(write0 is True):
                if(types[counter]=='RLS'):
                    writeLine(csvFile,insert)
            else:
                writeLine(csvFile,insert)

        else:
            print(t)
            print(num)
            print('training classification error: ' + str(TRCE))
            print('training MSE: ' + str(TRMSE))
            print('testing classification error: ' + str(TSCE))
            print('testing MSE: ' + str(TSMSE))


    counter+=1

#toshape = simpleSolutionAXB(modTrainR,modTestRT,2,0.5)# TODO test this for both over and under determined systems
#res = np.linalg.lstsq(modTrainR,modTestRT)
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
