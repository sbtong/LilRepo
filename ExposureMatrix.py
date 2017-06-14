import numpy as np
import pandas as pd
import macpy.utils.database as db

# CurveAverage
sql="""
use marketdata 
SELECT c.[AxiomaDataId]
   ,u.[CurveShortName]
      ,c.[TradeDate]
      ,[Quote]
      ,p.[SumAmtOutstanding]
   , RE.CountryOfRisk
 , region.Region
 , region.RegionGroup
 , IndGrp.IndustryGroup
 , IndMap.Market
 , IndMap.Sector
 , IndMap.SectorGroup
from xcIssuerCurveAverage c
join xcIssuerProperty p on p.AxiomaDataId = c.AxiomaDataId
      and p.TradeDate = c.TradeDate
join xcIssuerCurveUniverse u on u.AxiomaDataId = c.AxiomaDataId
left join [MarketData].[dbo].[RiskEntity]   RE  on RE.RiskEntityId = u.RiskEntityId
                and RE.ToDate = '31-Dec-9999'
left join [MarketData].[dbo].[AL_IssuerIndustryGroup] IndGrp on u.RiskEntityId = IndGrp.IssuerId
left join [MarketData].[dbo].[AL_IndustryMap]  IndMap on IndGrp.IndustryGroup = IndMap.IndustryGroup
left join [MarketData].[dbo].[AL_CountryToRegionMap]  region on region.iso_scntry_cd = RE.CountryOfRisk
where IndGrp.IndustryGroup = 'AUTOS'
order by c.TradeDate, c.AxiomaDataId
"""
df_namesAvg = db.MSSQL.extract_dataframe(sql, environment='DEV')

# df = df_namesAvg[df_namesAvg.TradeDate > '2012-12-31']
df = df_namesAvg.set_index('TradeDate')

# normalize size exposure
mean = np.mean(df['SumAmtOutstanding'])
sigma = np.std(df['SumAmtOutstanding'])
df['SumAmtOutstanding'] = (df['SumAmtOutstanding'] - mean) / sigma

# Rank and quantile for each day
appended_data = []
# a: date set
a = df.index.unique ()
# loop for each tradedate
for iDate in a[0:(len(a) - 1)]:
    df_sub = df.loc[df.index == iDate]
    df_sub.loc[:, 'RANK'] = df_sub.loc[:, 'Quote'].rank ()
    df_sub.loc[:, 'B_Q'] = pd.qcut (df_sub.loc[:, 'RANK'], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    appended_data.append (df_sub)
appended_data = pd.concat (appended_data, axis=0)

# Factor Exposure Matrix
# Market Factor
df['B_MKT'] = df['Market'].mask(df.Market == 'CR', 1.)
# Quality Factor
df['B_Q1'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q1', 1.)
df['B_Q2'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q2', 1.)
df['B_Q3'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q3', 1.)
df['B_Q4'] = appended_data['B_Q'].mask(appended_data['B_Q'] == 'Q4', 1.)
# Industry Factors
df['B_AUTOS'] = df['IndustryGroup'].mask(df.IndustryGroup == 'AUTOS', 1.)
df['B_CDRTL'] = df['IndustryGroup'].mask(df.IndustryGroup == 'CDRTL', 1.)
df['B_CNSDA'] = df['IndustryGroup'].mask(df.IndustryGroup == 'CNSDA', 1.)
df['B_CNSSV'] = df['IndustryGroup'].mask(df.IndustryGroup == 'CNSSV', 1.)
df['B_MEDIA'] = df['IndustryGroup'].mask(df.IndustryGroup == 'MEDIA', 1.)
# size factor
df['B_SIZE'] = df['SumAmtOutstanding']

# convert str to NaN and fill in 0
df['B_Q1'] = pd.to_numeric(df['B_Q1'],'coerce')
df['B_Q2'] = pd.to_numeric(df['B_Q2'],'coerce')
df['B_Q3'] = pd.to_numeric(df['B_Q3'],'coerce')
df['B_Q4'] = pd.to_numeric(df['B_Q4'],'coerce')
df['B_AUTOS'] = pd.to_numeric(df['B_AUTOS'],'coerce')
df['B_CDRTL'] = pd.to_numeric(df['B_CDRTL'],'coerce')
df['B_CNSDA'] = pd.to_numeric(df['B_CNSDA'],'coerce')
df['B_CNSSV'] = pd.to_numeric(df['B_CNSSV'],'coerce')
df['B_MEDIA'] = pd.to_numeric(df['B_MEDIA'],'coerce')
df = df.fillna(0.0)

# exposure matrix for consumer discretionary sector
ExposureMatrix = df[['AxiomaDataId','B_MKT', 'B_Q1', 'B_Q2', 'B_Q3', 'B_Q4', 'B_AUTOS', 'B_CDRTL', 'B_CNSDA', 'B_CNSSV','B_MEDIA','B_SIZE']]

# ExposureMatrix.to_csv('ExposureMatrix.csv')