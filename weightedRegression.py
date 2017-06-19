import numpy as np
import pandas as pd
import macpy.utils.database as db
import collections
import datetime

import statsmodels.api as sm
import itertools
import constrainedmodel as cm
import argparse

import matplotlib.pyplot as plt
import pylab as pl

RegressionData = pd.read_csv('RegressionData.csv')
RegressionData = RegressionData.set_index(RegressionData.CurveShortName)

def compute_FactorReturns(d0, d1, RegressionData):
    df0 = RegressionData[RegressionData.TradeDate == d0]
    df1 =  RegressionData[RegressionData.TradeDate == d1]
    r = df1['Average'].fillna (0.0)
    estu = df1['estu']
    estu = estu[estu == 1.0].index
    estu = df0.index.intersection (estu)
    B = df0.drop (['TradeDate', 'CurveShortName', 'Average', 'estu', 'SumAmtOutstanding'], axis=1).fillna (0.0)
    mask1 = r.index.isin (estu)
    mask2 = B.index.isin (estu)
    r = r[mask1]
    B = B[mask2]

    industries = ['B_AUTOS', 'B_CDRTL', 'B_CNSDA', 'B_CNSSV', 'B_MEDIA']
    qualities = ['B_Q1', 'B_Q2', 'B_Q3', 'B_Q4']
    weights = np.sqrt(np.exp(B.B_SIZE))
    weights /= weights.sum()
    B[B == 0.0] = np.nan
    B = B.dropna (axis=1, how='all').fillna (0.0)
    C = pd.DataFrame (0.0, index=['Industry', 'Quality'], columns=B.columns)
    C.loc['Industry', C.columns.intersection (industries)] = 1.0
    C.loc['Quality', C.columns.intersection (qualities)] = 1.0
    model = cm.ConstrainedLinearModel (r, B, C, weights=weights)
    fr_W = model.params
    t_W = model.tvalues
    rsquared_W = model.rsquared

    return fr_W, t_W, rsquared_W


date1 = RegressionData.TradeDate.unique()
fr = []
tstats = []
rsquared = []
date = date1[0:1000]
for d0,d1 in itertools.izip(date[:-1], date[1:]):
    print 'date', d1
    (fr_W, t_W, rsquared_W) = compute_FactorReturns(d0, d1, RegressionData)
    fr.append (fr_W)
    tstats.append (t_W)
    rsquared.append (rsquared_W)
fr = pd.DataFrame (fr)
tstats = pd.DataFrame(tstats)
rsquared = pd.Series (rsquared)

d = date[1:]
fr = fr.set_index(d)
tstats = tstats.set_index(d)
rsquared.index = d

rollingr = rsquared.rolling(window=30,center=False).mean()
ts = rollingr.dropna()
ts.plot(title = 'Rolling Average Rsquared')
t = tstats.abs()
mean_t = t.mean()
mean_t.plot(kind = 'bar', title = 'Average t stat for Each Factor')


for iname in B.columns:
    name = iname
    # factor return plot
    df_chart_fr = fr[name] + 1
    ts1 = df_chart_fr.cumprod ()
    # tvalue plot
    ts2 = tstats[name].abs ()
    # plot
    fig, axes = plt.subplots (nrows=1, ncols=2)
    # factor return
    ts1.plot (ax=axes[0], title=name + ' Factor Return History')
    # abs(t) histogram
    ts2.hist (ax=axes[1])
    axes[1].set_title ('Absolute t-value Histogram')


name = 'B_MEDIA'
# factor return plot
df_chart_fr = fr[name] + 1
ts1 = df_chart_fr.cumprod()
# tvalue plot
ts2 = tstats[name].abs()

#plot
fig, axes = plt.subplots(nrows=1, ncols=2)
# factor return
ts1.plot(ax=axes[0], title = name + ' Factor Return History')
#abs(t) histogram
ts2.hist(ax = axes[1])
axes[1].set_title('Absolute t-value Histogram')


