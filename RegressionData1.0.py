import numpy as np
import pandas as pd
import macpy.utils.database as db

# CurveAverage
sql="""
use marketdata 
SELECT c.[AxiomaDataId]
   ,u.[CurveShortName]
      ,c.[TradeDate]
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
where IndGrp.IndustryGroup = 'CDRTL'
and u.Currency = 'USD'
order by c.TradeDate, c.AxiomaDataId
"""

df_namesReturn = db.MSSQL.extract_dataframe(sql, environment='DEV')
df = df_namesReturn
df=df.rename(columns = {'Average':'Quote'})

# df_namesAvg = db.MSSQL.extract_dataframe(sql, environment='DEV')

# df = df_namesAvg[df_namesAvg.TradeDate > '2012-12-31']
# df = df.set_index('TradeDate')

# Market Factor
df['B_MKT'] = df['Market'].mask(df.Market == 'CR', 1.)
# Industry Factors
IndGrps = df.IndustryGroup.unique()
for ig in IndGrps:
    column = 'B_%s' % ig
    df[column] = df['IndustryGroup'].mask(df.IndustryGroup == ig, 1.)
# size factor
df['B_SIZE'] = df['SumAmtOutstanding']
df.SumAmtOutstanding = df.SumAmtOutstanding / 1000.



# return column
name = df['CurveShortName'].unique()
append_rets = []
for iName in name:
    df_name = df[df['CurveShortName'] == iName]
    df_name['rets'] = df_name['Quote'].diff()
    append_rets.append(df_name)
append_rets = pd.concat(append_rets, axis = 0)
df['rets'] = append_rets.rets.sort_index()


# normalize size exposure
# IndGrp = df.IndustryGroup.unique()
# # calculate size
# size = []
#!!normalize within each industry
# for iInd in IndGrp:
#     df_indgrp = df[df['IndustryGroup'] == iInd]
#     mean = np.mean(df_indgrp['SumAmtOutstanding'])
#     sigma = np.std(df_indgrp['SumAmtOutstanding'])
#     df_indgrp['SumAmtOutstanding'] = (df_indgrp['SumAmtOutstanding'] - mean) / sigma
#     size.append(df_indgrp)
# size = pd.concat(size, axis = 0)

#!OR nomalize in the whole universe
# mean = np.mean(df['SumAmtOutstanding'])
# sigma = np.std(df['SumAmtOutstanding'])
# df['SumAmtOutstanding'] = (df['SumAmtOutstanding'] - mean) / sigma



# Rank and quantile for each day (by each industry group)
IndGrp = df.IndustryGroup.unique()
appended_data = []
for iInd in IndGrp:
    df_indgrp = df[df['IndustryGroup'] == iInd]
    # a: date set
    a = df_indgrp.index.unique ()
    # loop for each tradedate
    for iDate in a:
        df_sub = df_indgrp[df_indgrp.index == iDate]
        df_sub['RANK'] = df_sub['Quote'].rank ()
        df_sub['B_Q'] = pd.qcut(df_sub['RANK'], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        appended_data.append(df_sub)
appended_data = pd.concat (appended_data, axis=0)

# Factor Exposure Matrix

# Quality Factor
df['B_Q1'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q1', 1.)
df['B_Q2'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q2', 1.)
df['B_Q3'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q3', 1.)
df['B_Q4'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q4', 1.)


# df['B_AUTOS'] = df['IndustryGroup'].mask(df.IndustryGroup == 'AUTOS', 1.)
# df['B_CDRTL'] = df['IndustryGroup'].mask(df.IndustryGroup == 'CDRTL', 1.)
# df['B_CNSDA'] = df['IndustryGroup'].mask(df.IndustryGroup == 'CNSDA', 1.)
# df['B_CNSSV'] = df['IndustryGroup'].mask(df.IndustryGroup == 'CNSSV', 1.)
# df['B_MEDIA'] = df['IndustryGroup'].mask(df.IndustryGroup == 'MEDIA', 1.)


# convert str to NaN
df['B_Q1'] = pd.to_numeric(df['B_Q1'],'coerce')
df['B_Q2'] = pd.to_numeric(df['B_Q2'],'coerce')
df['B_Q3'] = pd.to_numeric(df['B_Q3'],'coerce')
df['B_Q4'] = pd.to_numeric(df['B_Q4'],'coerce')
df['B_AUTOS'] = pd.to_numeric(df['B_AUTOS'],'coerce')
df['B_CDRTL'] = pd.to_numeric(df['B_CDRTL'],'coerce')
df['B_CNSDA'] = pd.to_numeric(df['B_CNSDA'],'coerce')
df['B_CNSSV'] = pd.to_numeric(df['B_CNSSV'],'coerce')
df['B_MEDIA'] = pd.to_numeric(df['B_MEDIA'],'coerce')
# df = df.fillna(0.0)

# Estimation Universe
estSelect = pd.DataFrame(df.CurveShortName.value_counts()).reset_index()
estSelect.columns = ['CurveShortName', 'NumObs']
estSelect = estSelect[estSelect['NumObs'] >= 1000] #select curves with 1000 more obs
estuSeries = estSelect.CurveShortName
x = df['CurveShortName'].isin(estuSeries)
df['estu'] = x.astype(int)



# exposure matrix for consumer discretionary sector
RegressionData = df[['CurveShortName', 'rets', 'estu', 'B_MKT', 'B_Q1', 'B_Q2', 'B_Q3', 'B_Q4', 'B_AUTOS', 'B_CDRTL', 'B_CNSDA', 'B_CNSSV','B_MEDIA','B_SIZE']]

# write each date data to csv
date = df.index.unique()
for jDate in date[0:5]:
    data = RegressionData[RegressionData.index == jDate]
    data.to_csv('data ' + jDate + '.csv', index = False)


# ExposureMatrix.to_csv('ExposureMatrix.csv')