import numpy as np
import pandas as pd
import collections
import macpy.utils.database as db

import statsmodels.api as sm
import constrainedmodel as cm
import itertools


sql="""
use marketdata 
SELECT c.[AxiomaDataId]
   ,u.[CurveShortName]
      ,c.[TradeDate]
      ,c.[TenorEnum]
      ,[Quote]
      ,[category]
   ,t.InYears
   ,p.[SumAmtOutstanding]
   , RE.CountryOfRisk
 , region.Region
 , region.RegionGroup
 , IndGrp.IndustryGroup
 , IndMap.Market
 , IndMap.Sector
 , IndMap.SectorGroup
from xcIssuerCurve c
join tenorenum t on t.TenorEnum = c.TenorEnum
join xcIssuerProperty p on p.AxiomaDataId = c.AxiomaDataId
      and p.TradeDate = c.TradeDate
join xcIssuerCurveUniverse u on u.AxiomaDataId = c.AxiomaDataId
left join [MarketData].[dbo].[RiskEntity]   RE  on RE.RiskEntityId = u.RiskEntityId
                and RE.ToDate = '31-Dec-9999'
left join [MarketData].[dbo].[AL_IssuerIndustryGroup] IndGrp on u.RiskEntityId = IndGrp.IssuerId
left join [MarketData].[dbo].[AL_IndustryMap]  IndMap on IndGrp.IndustryGroup = IndMap.IndustryGroup
left join [MarketData].[dbo].[AL_CountryToRegionMap]  region on region.iso_scntry_cd = RE.CountryOfRisk
where c.category in ('ll','lc','w')
--and IndGrp.IndustryGroup = 'AUTOS'
order by c.TradeDate, c.AxiomaDataId, c.category, t.InYears
"""
df_namesReturn = db.MSSQL.extract_dataframe(sql, environment='DEV')
df = df_namesReturn


df = pd.read_csv('AUTODATA.csv')

# df_namesReturn = db.MSSQL.extract_dataframe(sql, environment='DEV')
# df = df_namesReturn
# df=df.rename(columns = {'Average':'Quote'})


# drop dates without adequate observations
df1 = df[df['IndustryGroup'] == 'AUTOS']
df2 = df[df['IndustryGroup'] == 'CDRTL']
dropset = [item for item, count in collections.Counter(df.TradeDate).items() if count == 1]
mask = df.TradeDate.isin(dropset)
df = df[~mask]

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

# df['B_Q1'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q1', 1.)
# df['B_Q2'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q2', 1.)
# df['B_Q3'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q3', 1.)
# df['B_Q4'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q4', 1.)

# return column
name = df['CurveShortName'].unique()
append_rets = []
for iName in name:
    df_name = df[df['CurveShortName'] == iName]
    df_name['rets'] = df_name['Quote'].diff()
    append_rets.append(df_name)
append_rets = pd.concat(append_rets, axis = 0)
df['rets'] = append_rets.rets.sort_index()

# convert str to NaN
colnames = ['B_Q1', 'B_Q2', 'B_Q3', 'B_Q4', 'B_AUTOS', 'B_CDRTL', 'B_CNSDA', 'B_CNSSV', 'B_MEDIA']
for icol in colnames:
    df[icol] = pd.to_numeric(df[icol], 'coerce')

# Estimation Universe
estSelect = pd.DataFrame(df.CurveShortName.value_counts()).reset_index()
estSelect.columns = ['CurveShortName', 'NumObs']
estSelect = estSelect[estSelect['NumObs'] >= 1000] #select curves with 1000 more obs
estuSeries = estSelect.CurveShortName
x = df['CurveShortName'].isin(estuSeries)
df['estu'] = x.astype(int)

#run regression

# define regression function
def computeFactorReturns(df0, df1):
    r = df1['Average'].fillna(0.0)
    estu = df1['estu']
    estu = estu[estu == 1.0].index
    estu = df0.index.intersection(estu)
    B = df0.drop(['rets', 'mcap', 'estu'], axis=1).fillna(0.0)
    result = sm.WLS(r[estu], B.loc[estu], weights=np.sqrt(df0['mcap'].loc[estu])).fit()
    fr = result.params
    assets = r.index.intersection(B.index)
    resid = r[assets] - B.loc[assets].dot(fr)
    return (fr, resid)


# read and write

d0 = '2016-03-30'
d1 = '2016-03-31'
df0 = pd.DataFrame.from_csv ('data' + d0 + '.csv')
df1 = pd.DataFrame.from_csv ('data' + d1 + '.csv')
r = df1['Average'].fillna (0.0)
estu = df1['estu']
estu = estu[estu == 1.0].index
estu = df0.index.intersection (estu)
B = df0.drop (['TradeDate', 'CurveShortName.1', 'Average', 'estu'], axis=1).fillna (0.0)
mask1 = r.index.isin(estu)
mask2 = B.index.isin(estu)
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
model = cm.ConstrainedLinearModel (r, B, C, weights = None)
fr_COLS = model.params
t_COLS = model.tvalues
rsquared_COLS = model.rsquared

df_fr = pd.concat([fr_OLS, fr_COLS], axis = 1)
df_fr.columns = ['fr_OLS', 'fr_COLS']
df_stats = pd.concat([t_OLS, t_COLS], axis = 1 )
df_stats.columns = ['t_OLS', 't_COLS']
rsquared = [rsquared_OLS, rsquared_COLS]
frame = pd.concat([df_fr, df_stats], axis = 1)

# return (frame, rsquared, resid)

for d0,d1 in itertools.izip(date[:-1], date[1:]):
    print d0, d1
    df0 = pd.DataFrame.from_csv ('data' + str (d0) + '.csv')
    df0 = pd.DataFrame.from_csv ('data' + str (d1) + '.csv')
    (frame, rsquared, resid) = computeFactorReturns (df0, df1)
    frame.to_csv ('fr + tvalue' + str (d1) + '.csv')
    rsquared.to_csv('rsquared' + str(d1) + '.csv')
    resid.to_csv ('sr' + str (d1) + '.csv')