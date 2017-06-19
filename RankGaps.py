import numpy as np
import pandas as pd
import macpy.utils.database as db
import macpy.utils.ngc_utils as ut
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
df = df.loc[~(df.AxiomaDataId == 207231847)]
df = df[df.TradeDate >= '2013-01-01']
df = df[df.TradeDate <= '2016-12-31']

df_ing = df[df['IndustryGroup'] == 'AUTOS']
issuer = df_ing.CurveShortName.unique()
df_ing.loc['Average'] = df_ing.Average.abs()

rankIssuer = []
for icurve in issuer:
    df1 = df_ing[df_ing.CurveShortName == icurve]
    df_gap = df1[df1.Average == 0]
    numgap = len(df_gap)
    rankIssuer.append(numgap)
ranking = pd.Series(rankIssuer, index = issuer)


date = df.TradeDate.unique()
for idate in date:
    df_day = df[df.TradeDate == idate]
    a = ut.adjBoxplotStats(df_day.Average)['fence']
    df[df.TradeDate == idate] = df_day.loc[(df_day.Average>=a[0]) & (df_day.Average<=a[1])]
df = df.dropna(axis = 0, how = 'any')