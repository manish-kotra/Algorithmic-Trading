from datetime import *
import datetime
from numpy import *
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import scipy.io as sio
import pandas as pd
import numpy as np

def normcdf(X):
    (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    L = abs(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2*pi)*exp(-L*L/2.) * (a1*K + a2*K*K + a3*pow(K,3) + a4*pow(K,4) + a5*pow(K,5))
    if X < 0:
        w = 1.0-w
    return w

def vratio(a, lag = 2, cor = 'hom'):
    """ the implementation found in the blog Leinenbock  
    http://www.leinenbock.com/variance-ratio-test/
    """
    #t = (std((a[lag:]) - (a[1:-lag+1])))**2;
    #b = (std((a[2:]) - (a[1:-1]) ))**2;
 
    n = len(a)
    mu  = sum(a[1:n]-a[:n-1])/n;
    m=(n-lag+1)*(1-lag/n);
    #print( mu, m, lag)
    b=sum(square(a[1:n]-a[:n-1]-mu))/(n-1)
    t=sum(square(a[lag:n]-a[:n-lag]-lag*mu))/m
    vratio = t/(lag*b);
 
    la = float(lag)
     
    if cor == 'hom':
        varvrt=2*(2*la-1)*(la-1)/(3*la*n)
 
    elif cor == 'het':
        varvrt=0;
        sum2=sum(square(a[1:n]-a[:n-1]-mu));
        for j in range(lag-1):
            sum1a=square(a[j+1:n]-a[j:n-1]-mu);
            sum1b=square(a[1:n-j]-a[0:n-j-1]-mu)
            sum1=dot(sum1a,sum1b);
            delta=sum1/(sum2**2);
            varvrt=varvrt+((2*(la-j)/la)**2)*delta
 
    zscore = (vratio - 1) / sqrt(float(varvrt))
    pval = normcdf(zscore);
 
    return  vratio, zscore, pval

def backshift(day,x):
    assert day > 0,'Invalid day'
    shift = np.zeros((np.shape(x)))
    shift[day:] = x[:-day]
    shift[shift==0] = np.nan
    return shift

def calculateReturns(prices, lag):
    prevPrices = backshift(lag, prices)
    rlag = (prices - prevPrices) / prevPrices
    return rlag

def fwdshift(day,x):
    assert day > 0,'Invalid day'
    shift = np.zeros((np.shape(x)))
    shift[:-day] = x[day:]
    shift[shift==0] = np.nan
    return shift

def calculateMaxDD(cumret):
    highwatermark = np.zeros(len(cumret))
    drawdown      = np.zeros(len(cumret))
    drawdownduration = np.zeros(len(cumret))
    for t in range(1, len(cumret)):
        highwatermark[t] = np.max([highwatermark[t-1], cumret[t]])
        drawdown[t] = (1+cumret[t]) / (1 + highwatermark[t]) - 1
        if (drawdown[t]==0):
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1
    return np.min(drawdown), np.max(drawdownduration)
