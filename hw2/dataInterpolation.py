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
from scipy import io

'''
useful links
https://www.youtube.com/watch?v=gSCa78TIldg the meat of this assignment
https://www.pnas.org/content/113/15/3932
https://mathinsight.org/ordinary_differential_equation_introduction for refreshing ODE
https://imedea.uib-csic.es/master/cambioglobal/Modulo_V_cod101615/Theory/TSA_theory_part1.pdf for probablities and time series
'''
YI = 0
SHPI = 1
CLPI = 2

filePath = '/home/bdvr/Documents/Github/AMATH563/hw2/'
interps = [(1,1),(2,3),(3,7)] # represent every year, every 6 months, and every 3 months (year, half year, quarter year)
def createLibrary(data):
    xs= data[:,SHPI]
    ys = data[:,CLPI]

def splineInterpolation(xs,level=0,plotSplines = False,title='peltCounts',clearFigure=True,saveLoc = 'scrap',test=False):
    size = len(xs)
    y = xs
    #f = interp1d(x, y)
    allfactors = [2,4,8,24,730]
    factor=allfactors[level]
    #print(size)
    if(test is False):
        x = np.linspace(1845, 1903, num=size, endpoint=True)
        xnew = np.linspace(1845, 1903, num=factor*size-(factor-1), endpoint=True)
    else:
        x = np.linspace(-5, 5, num=size, endpoint=True)
        xnew = np.linspace(-5, 5, num=factor*size-(factor-1), endpoint=True)
    f2 = interp1d(x, y, kind='cubic')

    if(clearFigure):
        plt.clf()
    plt.plot(x, y, 'o', xnew, f2(xnew), '--')
    plt.title('Interpolated Pelt Collection (1845-1903)')
    plt.legend(['original', 'interpolated (cubic)'], loc='best')
    if(plotSplines is True):
        plt.show()
    else:
        plt.savefig(saveLoc)
    return xnew,f2(xnew)


def linearInterpolation(thisPoint,nextPoint):
    curYear = thisPoint[YI]
    nextYear = nextPoint[YI]-1
    startSHP = thisPoint[SHPI]
    startCLP = thisPoint[CLPI]
    yearDiff = nextPoint[YI] - thisPoint[YI]
    shpDelta = ((nextPoint[SHPI] - thisPoint[SHPI])/yearDiff)
    clpDelta = ((nextPoint[CLPI] - thisPoint[CLPI])/yearDiff)
    newPoints=[]
    counter=1
    while(curYear<nextYear):
        curYear+=1
        newSHP = startSHP + shpDelta*counter
        newCLP = startCLP + clpDelta*counter
        newPoints.append([curYear,newSHP,newCLP])
        counter+=1
    return np.array(newPoints)

testData = np.array([[2000,1,2],[2003,7,11],[2006,11,2],[2009,5,11]])
actualData = np.genfromtxt('peltData.csv',delimiter=',')
def interpolateData(data):
    numEntries = data.shape[0]
    counter = 0
    Times=[]
    SHP=[]
    CLP=[]
    while(counter < numEntries-1):
        thisDataPoint = data[counter]
        nextDataPoint= data[counter+1]
        allDelta = linearInterpolation(thisDataPoint,nextDataPoint)
        counter+=1

def simpleLibrary(y1,y2,single=False):
    types = ['const','y1','y2',]
    if(single):
        numRows=1
    else:
        numRows = len(y1)

    numCols = 29
    lib = np.ones([numRows,numCols])
    lib[:,1]= np.sin(y1)
    lib[:,2]= np.sin(y2)
    lib[:,3] = np.sin(y1)*np.cos(y2)
    lib[:,4] = np.cos(y1)
    lib[:,5] = np.cos(y2)
    lib[:,6] = np.sin(y2**2)*np.cos(y1)
    lib[:,7] = np.sin(y1**2)*np.cos(y2)
    lib[:,8] = np.sin(y2)*np.cos(y1**2)
    lib[:,9] = np.sin(y1)*np.cos(y2**2)
    lib[:,10] = y1
    lib[:,11] = y2
    lib[:,12] = y1*y2
    lib[:,13] = np.sin(y1*y2)
    lib[:,14] = np.sin(y1+y2)
    lib[:,15] = np.cos(y1*y2)
    lib[:,16] = np.cos(y1+y2)
    lib[:,17] = y1*np.sin(y1)
    lib[:,18] = y1*np.cos(y1)
    lib[:,19] = y2*np.sin(y1)
    lib[:,20] = y2*np.cos(y1)
    lib[:,21] = y2*np.sin(y2)
    lib[:,22] = y2*np.cos(y2)
    lib[:,23] = y1*np.sin(y2)
    lib[:,24] = y1*np.cos(y2)
    lib[:,25] = (y1+y2)*np.sin(y1)
    lib[:,26] = (y1+y2)*np.cos(y1)
    lib[:,27] = (y1-y2)*np.sin(y1)
    lib[:,28] = (y1-y2)*np.cos(y1)

    return lib

def simpleLibrary2(y1,y2,single=False):
    types = ['const','y1','y2',]
    if(single):
        numRows=1
    else:
        numRows = len(y1)

    numCols = 22
    lib = np.ones([numRows,numCols])
    lib[:,1]= np.sin(y1)
    lib[:,2]= np.sin(y2)
    lib[:,3] = np.sin(y1)*np.cos(y2)
    lib[:,4] = np.cos(y1)
    lib[:,5] = np.cos(y2)
    lib[:,6] = np.sin(y2**2)*np.cos(y1)
    lib[:,7] = np.sin(y1**2)*np.cos(y2)
    lib[:,8] = np.sin(y2)*np.cos(y1**2)
    lib[:,9] = np.sin(y1)*np.cos(y2**2)
    lib[:,10] = y1
    lib[:,11] = y2
    lib[:,12] = y1*y2
    lib[:,13] = (y1**2)*y2
    lib[:,14] = y2*y2
    lib[:,15] = y1*y1
    lib[:,16] = y1**3
    lib[:,17] = y2**3
    lib[:,18] = y1**4
    lib[:,19] = y2**4
    lib[:,20] = y1**5
    lib[:,21] = y2**5

    return lib

def getLibrary(y1,y2,single=False):
    types = ['const','y1','y2',]
    if(single):
        numRows=1
    else:
        numRows = len(y1)
    numCols = 15
    lib = np.ones([numRows,numCols])
    lib[:,0]#nothing as its our constant column
    lib[:,1] = y1
    lib[:,2] = y2
    lib[:,3]= y1*y1
    lib[:,4]= y2*y2
    lib[:,5] = y1*y2
    lib[:,6] = y1*y1*y2
    lib[:,7] = y2*y2*y1
    lib[:,8] = np.sin(y1)
    lib[:,9] = np.cos(y1)
    lib[:,10] = np.sin(y2)
    lib[:,11] = np.cos(y2)
    lib[:,12] = np.cos(y2)*np.sin(y1)
    lib[:,13] = np.sin(y2)*np.cos(y1)
    lib[:,14] = y1*y1*y1

    return lib
'''
try init sparse of 0.2, and then reregres son 0.5
'''
def getLibraryTest(y1,single=False):
    types = ['const','y1','y2',]
    if(single):
        numRows=1
    else:
        numRows = len(y1)
    lib = np.ones([numRows,4])
    lib[:,0] = np.sqrt(y1)#nothing as its our constant column
    lib[:,1] = y1
    lib[:,2] = y1**2
    lib[:,3]= y1**3
    return lib

def getFormulaTest(derivs,data,penalty,iters):
    A = getLibraryTest(data)
    b=derivs
    if(iters>0):
        clf = linear_model.Lasso(alpha=penalty,max_iter=iters)
    else:
        clf = linear_model.Lasso(alpha=penalty)
    clf.fit(A,np.reshape(derivs,(-1,1)))
    return (clf.coef_),A

def smallTest():
    #true func is x^2 with deriv 2x.
    xs=np.arange(-5,6)
    ys = xs**2
    interpxs,interpolatedDatay = splineInterpolation(ys,3,True,test=True)
    derivs = calcDiff(interpolatedDatay,np.diff(interpxs))
    penalty = 0.001
    print(np.diff(interpxs)[0])
    coefs,SA = getFormulaTest(derivs,interpolatedDatay[1:-1],penalty,100000)
    print(coefs)

def getFormula2(derivs,incomys,otherys,penalty=0.1,iters=0):
    A = getLibrary(incomys,otherys)
    #print(A)
    #b=np.reshape(derivs,(derivs.shape[0],1))
    b=derivs
    reg = linear_model.LinearRegression()
    reg.fit(A,b)
    return (reg.coef_),A

def getFormula(derivs,incomys,otherys,penalty=0.1,iters=0):
    A = getLibrary(incomys,otherys)
    #print(A)
    b=derivs
    if(iters>0):
        clf = linear_model.Lasso(alpha=penalty,max_iter=iters)
    else:
        clf = linear_model.Lasso(alpha=penalty)
    clf.fit(A,b)
    return (clf.coef_),A

def calcVals(SS,Lynx,time,coefSS,coefLynx,derivSS,derivLynx):
    counter = 0
    duration=len(time)
    repeatSize = len(coefSS)
    deltaSS=0
    deltaLynx=0
    mySS=SS[1]
    myLynx = Lynx[1]
    currentSS = SS[1]
    currentLynx = Lynx[1]
    myTimes = np.zeros([duration,5])
    myDerivs =  np.zeros([duration,5])
    stepSize = time[1]-time[0]
    while(counter<duration):

        toCoefSS = getLibrary(mySS,myLynx,True)
        toCoefLynx = getLibrary(myLynx,mySS,True)

        deltaSS = toCoefSS.dot(coefSS)
        actualSS = derivSS[counter]
        deltaLynx = toCoefLynx.dot(coefLynx)
        actualLynx = derivLynx[counter]

        mySS = mySS+stepSize*deltaSS
        myLynx = myLynx+stepSize*deltaLynx

        currentSS = SS[counter+1]
        currentLynx = Lynx[counter+1]
        myTimes[counter] = [time[counter],mySS,myLynx,currentSS,currentLynx]
        myDerivs[counter] = [time[counter],deltaSS,deltaLynx,actualSS,actualLynx]

        counter+=1
    return myTimes,myDerivs

def marchForward(SS,Lynx,time,coefSS,coefLynx,initSS,initLynx,filter=None):
    counter = 0
    duration=len(time)
    repeatSize = len(coefSS)
    deltaSS=0
    deltaLynx=0
    mySS=initSS
    myLynx = initLynx
    currentSS = SS[0]
    currentLynx = Lynx[0]
    myGuesses = np.zeros([duration,5])
    while(counter<duration):


        toCoefSS = getLibrary(mySS,myLynx,True)
        toCoefLynx = getLibrary(mySS,myLynx,True)
        if(filter is not None):
            tfilter = np.array(filter,dtype=bool)
            toCoefSS = (toCoefSS.T[tfilter]).T
            toCoefLynx = (toCoefLynx.T[tfilter]).T

        mySS = toCoefSS.dot(coefSS)
        myLynx = toCoefLynx.dot(coefLynx)

        currentSS = SS[counter]
        currentLynx = Lynx[counter]
        myGuesses[counter] = [time[counter],mySS,myLynx,currentSS,currentLynx]
        counter+=1
    return myGuesses

def smallCalcs(startPointSS,startPointLynx,time,coefSS,coefLynx,diffSS,diffLynx):
    counter = 0
    duration=len(time)
    repeatSize = len(coefSS)

    myTimes = np.zeros([duration,3])
    currentSS = startPointSS
    currentLynx = startPointLynx

    while(counter<10):
        print('Step: '+str(counter))
        myTimes[counter] = [time[counter],currentSS,currentLynx]
        toCoefSS = getLibrary(currentSS,currentLynx,True)
        #print(currentSS)
        #print(toCoefSS)
        #print(coefSS)
        #print(toCoefSS*coefSS)
        stepSize = time[counter+1]-time[counter]
        toCoefLynx = getLibrary(currentLynx,currentSS,True)
        deltaSS = np.sum(toCoefSS*coefSS)
        print('Truth SS')
        print(diffSS[counter]*stepSize)
        print('mycalc SS')
        print(deltaSS)
        deltaLynx = toCoefLynx.dot(coefLynx)
        print('Truth Lynx')
        print(diffLynx[counter]*stepSize)
        print('Mycalc lynx')
        print(deltaLynx)
        currentSS+= deltaSS
        currentLynx += deltaLynx
        counter+=1

def calcDiff(yvals, xdiffs):
    numPoints = len(yvals)-1
    allDifs = np.zeros(numPoints-1)
    counter = 1
    constDist = xdiffs[0]
    while(counter < numPoints):
        avgSlope = (yvals[counter+1]-yvals[counter-1])/(xdiffs[counter-1]+xdiffs[counter])
        allDifs[counter-1] = avgSlope
        counter+=1
    return allDifs

def plotMyAttempt(ogv,myv,time,title = 'Interpolated Pelt Collection (1845-1903)',ylab='Number of Pelts',legendData=['original', 'my guess']):
    plt.clf()
    plt.plot(time, ogv, 'o', time, myv, '--')
    plt.title(title)
    plt.ylabel(ylab)
    plt.xlabel('time')
    plt.legend(legendData, loc='best')
    plt.show()

def timeDelayImbedData(data1,data2,numOffset):
    size = len(data1)*2
    hankellMatrix = np.zeros([numOffset,size])
    counter =0
    arrayIndex = 0
    while(counter<numOffset):
        j=0
        arrayIndex=counter
        while(j<size-2*counter):
            hankellMatrix[counter,j] = data1[arrayIndex]
            hankellMatrix[counter,j+1] = data2[arrayIndex]
            arrayIndex+=1
            j+=2
        counter+=1
    return hankellMatrix[:,0:(size-(numOffset-1)*2)]

def binData(data,numBins=20):
    size = len(data)
    hist, bin_edges = np.histogram(data, numBins)
    probs = hist/size
    return probs,bin_edges

def getProb(val,binEdges,probs):
    counter = 0
    while(counter<len(binEdges)):
        if(val<=binEdges[counter+1]):
            return probs[counter]
        counter+=1
    return 0

def createPointPairs(data1,data2,mean1,mean2):
    size = len(data1)
    counter = 0
    bins=[1,1,1,1]
    while(counter<size):
        p1 = data1[counter]
        p2 = data2[counter]
        if(p1>mean1 and p2>mean2):
            bins[0]+=1
        elif(p1<=mean1 and p2>mean2):
            bins[1]+=1
        elif(p1>mean1 and p2<=mean2):
            bins[2]+=1
        elif(p1<=mean1 and p2<=mean2):
            bins[3]+=1
        counter += 1
    bins = np.array(bins)
    return (bins/size)

def lassoRegress(derivs,incomys,otherys,filter,penalty=0.1,iters=0):
    holdA = getLibrary(incomys,otherys)
    tfilter = np.array(filter,dtype=bool)
    finalA = holdA.T[tfilter]
    finalA = finalA.T
    #print(A)
    b=derivs
    if(iters>0):
        clf = linear_model.Lasso(alpha=penalty,max_iter=iters)
    else:
        clf = linear_model.Lasso(alpha=penalty)
    clf = linear_model.LinearRegression()
    clf.fit(finalA,b)
    return (clf.coef_),finalA

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

def calcKLDivergence(trueValsSS,myValsSS,trueValsLynx,myValsLynx):
    meanss = np.mean(trueValsSS)
    meanlynx = np.mean(trueValsLynx)
    probActual = createPointPairs(trueValsSS,trueValsLynx,meanss,meanlynx)
    probMine = createPointPairs(myValsSS,myValsLynx,meanss,meanlynx)
    counter = 0
    size = len(myValsSS)
    score = 0
    while(counter<size):
        mp = probMine[counter]
        ap = probActual[counter]
        temp = -ap*np.log(mp/ap)
        score+= temp
        counter+=1
    return score
'''
AIC/BIC
log(L(x|mu _hat)) ~~ -n/2 log(2pi) -n/2log(sigma^2) - 1/(2sigma^2)RSS
where sigma^2 = RSS/n (n is sample size, RSS = residual sum of squares = sum((y_i-f(x_i)))^2)
This assumes our error is gaussian (why? explain this in the paper)
Log likelihood for AIC and BIC. Likelihood of seeing the data given that model. Obviously we approximate this isnce we ont have a real distribution.

BOTH AIC AND BIC AND KL DIVERGENCE are both for one model in two dimensions, SO, you calculate KL Divergence over the bins of our data space and for the estimation of RSS and
conseuqeuntly sigma squared is by summing hte squared distance between actual and my calculated time points.

KL Divergence.
We are looking at the ratio of the distributions
equation is f(x,Beta) log((f(X,Beta)/g(X,Beta)))

where f is the amount of the animal

exploding to infinity is normal for this assignment when left on its own, so i got it right.

Now another approach is to do this

lookup ode45 equaivalent in python

difference equation is x_{n+1} = f(x_n,x_n-1...)

solved by look at x_n+1 rather than x', the derivative.

This difference equation and the x' are similar/ exact in certain situations


my x squared example is right .

for part 2, make it a small slice u(x,t), like a row of pixels.

SVD Can think of XV=UE (e is sum sigma) where sigma represents rank of matrix

want to throw both animals into this 2d matrix for the hankell matrix , like lynx ss lynx ss lynx ss

Part 2
come up with a pde very similar to how we did part one but for part two

plot trajectories against each other. (lynx versus snowshoe..)

for KL Divergence dont bother for part 2.
'''
def main(data,penalty,level=2):
    snowshoex , snowshoey = splineInterpolation(data[:,1],level,False,clearFigure=True,saveLoc = 'hareinterp'+str(level))
    lynxx , lynxy = splineInterpolation(data[:,2],level,False,clearFigure=True,saveLoc='lynxinterp'+str(level))
    print(timeDelayImbedData(snowshoey,lynxy,10))
    snowShoeDerivs = calcDiff(snowshoey,np.diff(snowshoex))
    lynxDerivs = calcDiff(lynxy,np.diff(lynxx))
    SSMarch = snowshoey[1:]
    LynxMarch = lynxy[1:]
    coefsSnowShoe,SA = getFormula(snowShoeDerivs,snowshoey[1:-1],lynxy[1:-1],penalty,100000)
    coefsLynx,LA = getFormula(lynxDerivs,lynxy[1:-1],snowshoey[1:-1],penalty,100000)

    altss,idk = getFormula2(SSMarch,snowshoey[0:-1],lynxy[0:-1],penalty,100000)
    altlynx,dik2 = getFormula2(LynxMarch,snowshoey[0:-1],lynxy[0:-1],penalty,100000)
    filter = [1,1,1,0,0,1,0,0,1,1,1,1,1,1,0]
    altss = altss*filter
    altlynx = altlynx*filter
    altss2,a2 = lassoRegress(SSMarch,snowshoey[0:-1],lynxy[0:-1],filter)
    altlynx2,a3 = lassoRegress(LynxMarch,snowshoey[0:-1],lynxy[0:-1],filter)
    #altss*[0,1,0,0]
    print('snowshoe hare')
    print(coefsSnowShoe)
    print('lynx')
    print(coefsLynx)
    print(snowshoex.shape)
    #vals,derivs= calcVals(snowshoey,lynxy,snowshoex[1:-1],coefsSnowShoe,coefsLynx,snowShoeDerivs,lynxDerivs)
    #vals = marchForward(SSMarch,LynxMarch,snowshoex[1:],altss2,altlynx2,snowshoey[0],lynxy[0],filter)
    vals = marchForward(SSMarch,LynxMarch,snowshoex[1:],altss,altlynx,snowshoey[0],lynxy[0],filter)

    print('total dist')
    #totaldist = np.sum(np.abs(derivs[:,1]-derivs[:,3]))
    #print(totaldist)
    #print(np.sum(np.abs(derivs[:,3])))

    MSSV = vals[:,1]
    RSSV = vals[:,3]

    MLV = vals[:,2]
    RLV = vals[:,4]

    #MSSD = derivs[:,1]
    #RSSD = derivs[:,3]

    #MLD = derivs[:,2]
    #RLD = derivs[:,4]
    plotMyAttempt(vals[:,3],vals[:,1],snowshoex[0:-1])
    plotMyAttempt(vals[:,4],vals[:,2],snowshoex[0:-1])

    #plotMyAttempt(RSSV,MSSV,snowshoex[1:-1])
    #plotMyAttempt(RLV,MLV,lynxx[1:-1])

    #print(calcKLDivergence(RSSV,MSSV))
    #print(calcKLDivergence(RLV,MLV))

def solveLasso(A,b,penalty,iters=0):
    if(iters>0):
        clf = linear_model.Lasso(alpha=penalty,max_iter=iters)
    else:
        clf = linear_model.Lasso(alpha=penalty)
    clf.fit(A,b)
    return (clf.coef_),A

def marchForward2(coefSS,coefLynx,initSS,initLynx,steps,filter=None):
    counter = 0
    myGuesses = np.zeros([steps,2])
    mySS=initSS
    myLynx = initLynx
    while(counter<steps):
        toCoefSS = simpleLibrary(mySS,myLynx,True)
        toCoefLynx = simpleLibrary(mySS,myLynx,True)
        if(filter is not None):
            tfilter = np.array(filter,dtype=bool)
            toCoefSS = (toCoefSS.T[tfilter]).T
            toCoefLynx = (toCoefLynx.T[tfilter]).T

        mySS = toCoefSS.dot(coefSS)
        myLynx = toCoefLynx.dot(coefLynx)
        myGuesses[counter] = [mySS,myLynx]
        counter+=1
    return myGuesses

def calcKLDivergence(trueBins,falseBins):
    score = np.sum(trueBins * np.log(trueBins/falseBins))
    return score

def calcAIC(trueData,falseData,numParam):
    size = len(trueData)
    counter=0
    RSS=0
    while(counter<size):
        diff1 = (trueData[counter,0]-falseData[counter,0])**2
        diff2 = (trueData[counter,1]-falseData[counter,1])**2
        RSS+=(diff1+diff2)
        counter+=1
    #sigma2 = RSS/counter
    #likelihood = (-1*counter/2)*np.log(2*np.pi) + (-1*counter/2)*np.log(sigma2) + (-1/(2*sigma2))*RSS

    #AIC = 2*numParam-likelihood
    AIC = counter*(np.log(RSS/counter)) +2*numParam
    return AIC

def calcBIC(trueData,falseData,numParam):
    size = len(trueData)
    counter=0
    RSS=0
    while(counter<size):
        diff1 = (trueData[counter,0]-falseData[counter,0])**2
        diff2 = (trueData[counter,1]-falseData[counter,1])**2
        RSS+=(diff1+diff2)
        counter+=1
    #sigma2 = RSS/counter
    #likelihood = (-1*counter/2)*np.log(2*np.pi) + (-1*counter/2)*np.log(sigma2) + (-1/(2*sigma2))*RSS

    #AIC = 2*numParam-likelihood
    AIC = counter*(np.log(RSS/counter)) + numParam*np.log(counter)
    return AIC

def main2(data,penalty,level=3):
    snowshoex , snowshoey = splineInterpolation(data[:,1],level,False,clearFigure=True,saveLoc = 'hareinterp'+str(level))
    lynxx , lynxy = splineInterpolation(data[:,2],level,False,clearFigure=True,saveLoc='lynxinterp'+str(level))
    hm =(timeDelayImbedData(snowshoey,lynxy,50))
    plotMyAttempt(snowshoey,lynxy,snowshoex,title = 'Interpolated Pelt Collection (1845-1903)',ylab='Number of Pelts',legendData=['snowshoe hares', 'lynx'])

    lib = simpleLibrary(snowshoey[:-1],lynxy[:-1])
    ssb = snowshoey[1:]
    lynxb = lynxy[1:]

    sscoef,idc1 = solveLasso(lib,ssb,penalty,10000)
    lynxcoef,idc2 = solveLasso(lib,lynxb,penalty,10000)

    sscoef0,idc3 = solveLasso(lib,ssb,1,10000)
    lynxcoef0,idc4 = solveLasso(lib,lynxb,1,10000)

    model0guesses = marchForward2(sscoef0,lynxcoef0,snowshoey[0],lynxy[0],len(ssb),filter=None)
    guesses = marchForward2(sscoef,lynxcoef,snowshoey[0],lynxy[0],len(ssb),filter=None)
    print('model 1 coefficeints SS:')
    print(sscoef0)

    plotMyAttempt(model0guesses[:,0],ssb,snowshoex[1:],'Interpolated SS Pelt Collection (1845-1903)')
    print('model 1 coefficeints Lynx:')
    print(lynxcoef0)

    plotMyAttempt(model0guesses[:,1],lynxb,snowshoex[1:],'Interpolated Lynx Pelt Collection (1845-1903)')

    print('model 2 coefficeints SS:')
    print(sscoef)

    plotMyAttempt(guesses[:,0],ssb,snowshoex[1:],'Interpolated SS Pelt Collection (1845-1903)')
    print('model 2 coefficeints Lynx:')
    print(lynxcoef)

    plotMyAttempt(guesses[:,1],lynxb,snowshoex[1:],'Interpolated Lynx Pelt Collection (1845-1903)')

    altss,AS = getFormula(ssb,snowshoey[0:-1],lynxy[0:-1],0.1,100000)
    altlynx,AL = getFormula(lynxb,snowshoey[0:-1],lynxy[0:-1],0.1,100000)
    print('model 3 coefficients SS: ')
    print(altss)
    print('model 3 coefficients Lynx: ')
    print(altlynx)
    filter = [1,1,1,0,0,1,0,0,1,1,1,1,1,1,0]

    '''
    For model 2, I threshold at ~0.0002

    '''
    altss = altss*filter
    altlynx = altlynx*filter
    vals = marchForward(ssb,lynxb,snowshoex[1:],altss,altlynx,snowshoey[0],lynxy[0])
    plotMyAttempt(vals[:,1],ssb,snowshoex[1:],'Interpolated SS Pelt Collection (1845-1903)')
    plotMyAttempt(vals[:,2],lynxb,snowshoex[1:],'Interpolated Lynx Pelt Collection (1845-1903)')
    ssmean = np.mean(vals[:,3])
    lynxmean = np.mean(vals[:,4])
    truDistrib = createPointPairs(vals[:,3],vals[:,4],ssmean,lynxmean)
    myDistrib0 = createPointPairs(model0guesses[:,0],model0guesses[:,1],ssmean,lynxmean)
    myDistrib1 = createPointPairs(guesses[:,0],guesses[:,1],ssmean,lynxmean)
    myDistrib2 = createPointPairs(vals[:,1],vals[:,2],ssmean,lynxmean)

    truModelScore=calcKLDivergence(truDistrib,truDistrib)
    myModel0Score = calcKLDivergence(truDistrib,myDistrib0)
    myModel1Score = calcKLDivergence(truDistrib,myDistrib1)
    myModel2Score = calcKLDivergence(truDistrib,myDistrib2)

    model0coefss = np.count_nonzero(sscoef0)
    model0coeflynx = np.count_nonzero(lynxcoef0)

    model1coefss = np.count_nonzero(sscoef)
    model1coeflynx = np.count_nonzero(lynxcoef)

    model2coefss = np.count_nonzero(altss)
    model2coeflynx = np.count_nonzero(altlynx)

    myModel0AIC = calcAIC(vals[:,3:5],model0guesses,model0coefss)
    myModel1AIC = calcAIC(vals[:,3:5],guesses,model1coefss)
    myModel2AIC = calcAIC(vals[:,3:5],vals[:,1:3],model2coefss)

    myModel0BIC = calcBIC(vals[:,3:5],model0guesses,model0coefss)
    myModel1BIC = calcBIC(vals[:,3:5],guesses,model1coefss)
    myModel2BIC = calcBIC(vals[:,3:5],vals[:,1:3],model2coefss)

    print(truDistrib)
    print(truModelScore)

    print('Bin values in order ')
    print(myDistrib0)
    print(myDistrib1)
    print(myDistrib2)
    print('KL Divergence values in order ')
    print(myModel0Score)
    print(myModel1Score)
    print(myModel2Score)
    print('AIC scores in order')
    print(myModel0AIC)
    print(myModel1AIC)
    print(myModel2AIC)
    print('BIC scores in order')
    print(myModel0BIC)
    print(myModel1BIC)
    print(myModel2BIC)

    createBar(sscoef0, "model 1 SS coefficients")
    createBar(lynxcoef0,"model 1 lynx coefficients")
    createBar(sscoef, "model 2 SS coefficients")
    createBar(lynxcoef,"model 2 lynx coefficients")
    createBar(altss,"model 3 SS coefficients")
    createBar(altlynx,"model 3 lynx coefficients")

    print('SS num coef in order')
    print(model0coefss)
    print(model1coefss)
    print(model2coefss)
    print('lynx num coef in order')
    print(model0coeflynx)
    print(model1coeflynx)
    print(model2coeflynx)


    '''
    a matrix A multiplied to a vector does a certain amount of stretching and rotation, two parts.
    https://www.youtube.com/watch?v=EokL7E6o1AE
    think of x many vectors defines a x dimensional hypersphere
    our A will alter these x that define a sphere S into a ellipse of n dimensions with a major and minor axis (by rotating and stretch)
    we can break up these new x into a sigma*mu where mu is a directional vector and sigma is the variance
    so in the AV=UE, V is a rotation, U is a rotation (also called unitary transformation)
    A=UEV^T is called the reduced singular value decomp. Says that to get correct rotations and stretching of our A, we say is this combination of stretching and rotations
    '''
    u,s,vh=np.linalg.svd(hm) # this is our results from our hankell matrix
    createBar(s/np.sum(s),"time delay imbedded singular values")

#smallTest()

main2(actualData,0.0001)
'''
part 2
use np.gradient to create the library.
so gradient over space is the transformation library for our data at a time steps

'''
#main(actualData,1)
#main(actualData,10)
#main(actualData,100)
#main(actualData,1000) #this returns negative KL scores!


#for time delay imbedding https://stackoverflow.com/questions/48967169/time-delay-embedding-of-time-series-in-python
