import numpy as np
import struct
import cvxpy as cp
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.lines as lines
import matplotlib.cbook as cbook
import pandas as pd
from cvxopt.modeling import op, dot, variable
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import lsmr
from sklearn import linear_model
from decimal import Decimal
latex = True
filePath = '/home/bdvr/Documents/GitHub/AMATH563/hw1/'

fullRes = pd.read_csv(filePath+'results/fullresults.csv')
sparseRes = pd.read_csv(filePath+'results/sparseResults.csv')

justLasso = fullRes.loc[fullRes['model']=='lasso']
lassoIndiv = justLasso.loc[justLasso['indiv'] == False]
lassoFull = justLasso.loc[justLasso['indiv'] == True]
grpby = fullRes.groupby(['indiv','model','penalty'])
grpbysparse = sparseRes.groupby(['indiv','model','penalty'])

testMeans = (grpby['Test_CE'].mean())
trainMeans = (grpby['Train_CE'].mean())

sparseTestMeans = (grpbysparse['Test_CE'].mean())
sparseTrainMeans = (grpbysparse['Train_CE'].mean())

newdffull= (trainMeans.reset_index())
tempdf = testMeans.reset_index()
newdffull['Test_CE'] = tempdf['Test_CE']


newdfsparse= (sparseTrainMeans.reset_index())
tempdf = sparseTestMeans.reset_index()
newdfsparse['Test_CE'] = tempdf['Test_CE']
if(latex):
    print(newdffull.to_latex(index=True))
    print(newdfsparse.to_latex(index=True))
    print(fullRes.to_latex())
    print(sparseRes.to_latex())
else:
    print(newdffull)
    print(newdfsparse)

#print(sparseTestMeans.reset_index()['Test_CE'] - testMeans.reset_index()['Test_CE'])
#print(sparseTrainMeans.reset_index()['Train_CE'] - trainMeans.reset_index()['Train_CE'])
