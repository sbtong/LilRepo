import numpy as np
import pandas as pd
import macpy.utils.database as db
import collections

import statsmodels.api as sm
import itertools
import constrainedmodel as cm
import csv

import matplotlib.pyplot as plt


# CurveAverage
sql="""
use marketdata 
SELECT c.[AxiomaDataId]
   ,u.[CurveShortName]
      ,c.[TradeDate]
      ,c.[SumAmtOutstanding]
      ,[Average]
      ,[Alpha]
   , RE.CountryOfRisk
 , region.Region
 , region.RegionGroup
 , IndGrp.IndustryGroup
 , IndMap.Market
 , IndMap.Sector
 , IndMap.SectorGroup
from xcIssuerProperty c
join xcIssuerCurveUniverse u on u.AxiomaDataId = c.AxiomaDataId
left join [MarketData].[dbo].[RiskEntity]   RE  on RE.RiskEntityId = u.RiskEntityId
                and RE.ToDate = '31-Dec-9999'
left join [MarketData].[dbo].[AL_IssuerIndustryGroup] IndGrp on u.RiskEntityId = IndGrp.IssuerId
left join [MarketData].[dbo].[AL_IndustryMap]  IndMap on IndGrp.IndustryGroup = IndMap.IndustryGroup
left join [MarketData].[dbo].[AL_CountryToRegionMap]  region on region.iso_scntry_cd = RE.CountryOfRisk
where IndGrp.IndustryGroup in ('AUTOS', 'CDRTL', 'CNSDA', 'MEDIA')
and u.Currency = 'USD'
order by c.TradeDate, c.AxiomaDataId
"""

df_namesReturn = db.MSSQL.extract_dataframe(sql, environment='DEV')
df = df_namesReturn
df = df.rename(columns = {'Alpha':'Quote'})
df = df[df.TradeDate >= '2016-03-30']
df = df[df.TradeDate <= '2016-05-25']

# drop dates without adequate observations
IndGrp = df.IndustryGroup.unique()
df_drop = []
for iInd in IndGrp:
    df_indgrp = df[df['IndustryGroup'] == iInd]
    dropset = [item for item, count in collections.Counter (df_indgrp.TradeDate).items () if count <= 4]
    mask = df_indgrp.TradeDate.isin (dropset)
    df_indgrp = df_indgrp[~mask]
    df_drop.append(df_indgrp)
df = pd.concat(df_drop, axis = 0)
# df1 = df[df['IndustryGroup'] == 'AUTOS']
# df2 = df[df['IndustryGroup'] == 'CDRTL']
# dropset = [item for item, count in collections.Counter(df.TradeDate).items() if count <= 4]
# mask = df.TradeDate.isin(dropset)
# df = df[~mask]

# Market Factor
df['B_MKT'] = df['Market'].mask(df.Market == 'CR', 1.)
# Industry Factors
# IndGrps = df.IndustryGroup.unique()
IndGrps = ['AUTOS', 'CDRTL', 'CNSDA', 'CNSSV', 'MEDIA']
for ig in IndGrps:
    column = 'B_%s' % ig
    df[column] = df['IndustryGroup'].mask(df.IndustryGroup == ig, 1.)
# size factor
df.SumAmtOutstanding = df.SumAmtOutstanding / 1000.
df['B_SIZE'] = df['SumAmtOutstanding']

# Quality Factor
# Rank and quantile for each day (by each industry group)
IndGrp = df.IndustryGroup.unique()
appended_data = []
for iInd in IndGrp:
    df_indgrp = df[df['IndustryGroup'] == iInd]
    # a: date set
    a = df_indgrp.TradeDate.unique ()
    # loop for each tradedate
    for iDate in a:
        df_sub = df_indgrp[df_indgrp.TradeDate == iDate]
        df_sub['RANK'] = df_sub['Quote'].rank ()
        df_sub['B_Q'] = pd.qcut(df_sub['RANK'], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        appended_data.append(df_sub)
appended_data = pd.concat (appended_data, axis=0)

quality = ['Q1', 'Q2', 'Q3', 'Q4']
for iq in quality:
    colname = 'B_%s' % iq
    df[colname] = appended_data['B_Q'].mask(appended_data.B_Q == iq, 1.)

# Estimation Universe
estSelect = pd.DataFrame(df.CurveShortName.value_counts()).reset_index()
estSelect.columns = ['CurveShortName', 'NumObs']
estSelect = estSelect[estSelect['NumObs'] >= 40] #select curves with 1000 more obs
estuSeries = estSelect.CurveShortName
x = df['CurveShortName'].isin(estuSeries)
df['estu'] = x.astype(int)

# convert str to NaN
colnames = ['B_Q1', 'B_Q2', 'B_Q3', 'B_Q4', 'B_AUTOS', 'B_CDRTL', 'B_CNSDA', 'B_CNSSV', 'B_MEDIA']
for icol in colnames:
    df[icol] = pd.to_numeric(df[icol], 'coerce')

# exposure matrix for consumer discretionary sector
RegressionData = df[['TradeDate', 'CurveShortName', 'Average', 'estu', 'B_MKT', 'B_Q1', 'B_Q2', 'B_Q3', 'B_Q4', 'B_AUTOS', 'B_CDRTL', 'B_CNSDA', 'B_CNSSV', 'B_MEDIA', 'B_SIZE']]
RegressionData = RegressionData.set_index(RegressionData.CurveShortName)

# write each date data to csv
date = df.TradeDate.unique()
date = date[0:10]
for jDate in date:
    data = RegressionData[RegressionData.TradeDate == jDate]
    data.to_csv('data' + jDate + '.csv', index = True)

def computeFactorReturns(df0, df1):
    df0 = pd.DataFrame.from_csv ('data' + d0 + '.csv')
    df1 = pd.DataFrame.from_csv ('data' + d1 + '.csv')
    r = df1['Average'].fillna (0.0)
    estu = df1['estu']
    estu = estu[estu == 1.0].index
    estu = df0.index.intersection (estu)
    B = df0.drop (['TradeDate', 'CurveShortName.1', 'Average', 'estu'], axis=1).fillna (0.0)
    mask1 = r.index.isin (estu)
    mask2 = B.index.isin (estu)
    r = r[mask1]
    B = B[mask2]
    result = sm.OLS (r, B).fit ()
    fr_OLS = result.params
    t_OLS = result.tvalues
    rsquared_OLS = result.rsquared
    assets = r.index.intersection (B.index)
    resid = r - B.loc[assets].dot (fr_OLS)

    industries = ['B_AUTOS', 'B_CDRTL', 'B_CNSDA', 'B_CNSSV', 'B_MEDIA']
    B[B == 0.0] = np.nan
    B = B.dropna (axis=1, how='all').fillna (0.0)
    C = pd.DataFrame (0.0, index=['Industry'], columns=B.columns)
    C.loc['Industry', C.columns.intersection (industries)] = 1.0
    model = cm.ConstrainedLinearModel (r, B, C, weights=None)
    fr_COLS = model.params
    t_COLS = model.tvalues
    rsquared_COLS = model.rsquared

    df_fr = pd.concat ([fr_OLS, fr_COLS], axis=1)
    df_fr.columns = ['fr_OLS', 'fr_COLS']
    df_stats = pd.concat ([t_OLS, t_COLS], axis=1)
    df_stats.columns = ['t_OLS', 't_COLS']
    rsquared = [rsquared_OLS, rsquared_COLS]
    frame = pd.concat ([df_fr, df_stats], axis=1)

    return (fr_COLS, t_COLS, rsquared_COLS)


fr = []
tstats = []
rsquared = []
for d0,d1 in itertools.izip(date[:-1], date[1:]):
    print d0, d1
    df0 = pd.DataFrame.from_csv ('data' + str (d0) + '.csv')
    df1 = pd.DataFrame.from_csv ('data' + str (d1) + '.csv')
    (fr_COLS, t_COLS, rsquared_COLS) = computeFactorReturns (df0, df1)
    fr.append(fr_COLS)
    tstats.append(t_COLS)
    rsquared.append(rsquared_COLS)
fr = pd.concat(fr, axis = 0)
tstats = pd.concat(tstats, axis = 0)
factorset = fr_COLS.index
factorName = 'B_MKT'

    # frame.to_csv ('fr + tvalue' + str (d1) + '.csv')
    # resid.to_csv ('sr' + str (d1) + '.csv')
    # print rsquared
d = np.repeat(date[1:], len(factorset))
fr = pd.DataFrame(fr)
tstats = pd.DataFrame(tstats)
rsquared = pd.Series(rsquared)
fr = fr.reset_index().set_index(d)
tstats = tstats.reset_index().set_index(d)
rsquared.index = date[1:]
fr.columns = ['Factor', 'Return']
tstats.columns = ['Factor', 'tvalue']

# factor return plot
df_chart_fr = fr[fr['Factor'] == factorName] #factor can equal to every factor name
df_chart_fr.loc[:, '1+r'] = df_chart_fr.Return.values + 1
ts1 = df_chart_fr['1+r'].cumprod()

# tvalue plot
df_chart_tvalue = tstats[tstats['Factor'] == factorName]
df_chart_tvalue.loc[:, 'abstvalue'] = df_chart_tvalue.tvalue.abs()
ts2 = df_chart_tvalue['abstvalue']

#plot
fig, axes = plt.subplots(nrows=2, ncols=2)
# factor return
ts1.plot(ax=axes[0,0])
# tvalue
ts2.plot(ax=axes[0,1])
#rsquare
rsquared.plot(ax = axes[1,0])