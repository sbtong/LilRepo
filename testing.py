import numpy as np
import pandas as pd
import collections
import macpy.utils.database as db

df = pd.read_csv('AUTODATA.csv')

# df_namesReturn = db.MSSQL.extract_dataframe(sql, environment='DEV')
# df = df_namesReturn
# df=df.rename(columns = {'Average':'Quote'})


# drop dates without adequate observations
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