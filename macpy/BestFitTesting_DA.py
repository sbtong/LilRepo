import time
import finance
import string
import datetime
import macpy.utils.database as db
import pandas as pd
import matplotlib.pyplot as plt
import macpy.bondPortfolio as bondPortfolio
import numpy as np
from optparse import OptionParser
import statsmodels.api as sm
import macpy.bond as bond
import macpy.nelsonsiegel as ns
from scipy.optimize import curve_fit

SQL = """
    SELECT [CurveId]
          ,d.TradeDate
          ,d.InstrCode
          ,NomTermToMat = DateDiff(d, px.TradeDate, sec.MatDate)/365.25
		  ,sec.MatDate
		  ,sec.InitAccDate
		  ,FrstCpnRate = sec.CurrCpnRate
		  ,CleanPrice = px.Prc
		  ,MatStdYld
          ,[Lud]
          ,[Lub]
      FROM [MarketData].[dbo].[DerivCurveFilteredBond] d
      join qai..FIEJVSecInfo sec on sec.InstrCode = d.InstrCode
      join qai..FIEJVPrcDly px on px.InstrCode = d.InstrCode and px.TradeDate = d.TradeDate

      where CurveId = 200302039

    and ItemId = 2
	--and sec.CurrCpnRate > 0

	order by d.TradeDate
"""

#df = db.MSSQL.extract_dataframe(SQL, environment='DEV')

tradeDate = '2015-01-'

df = pd.read_csv('C:\\Users\\dantonio\\Documents\\Projects\\BestFit\\US_20140101-20150601.csv')
df.TradeDate = df.TradeDate.apply(lambda x: pd.to_datetime(x))
df.MatDate = df.MatDate.apply(lambda x: pd.to_datetime(x))



df.Amt = np.where(pd.isnull(df.Amt),1000,df.Amt)

for i, row in df.iterrows():
    if row['FrstCpnRate'] == 0:
        row['InitAccDate'] = tradeDate

dates = df.TradeDate.unique()

fitting_results=[]
for d in dates: #loop over dates performing a curve fit on each
    print 'running %s' % pd.to_datetime(d).strftime('%Y-%m-%d')
    df_date = df[df.TradeDate == d]
    df_date.InitAccDate = np.where(pd.isnull(df_date.InitAccDate),d,df_date.InitAccDate)
    df_date.InitAccDate = df_date.InitAccDate.apply(lambda x: pd.to_datetime(x))
    df_fitting_data=[]
    for i, row in df_date.iterrows():
        pricer = bond.BondPricer.create_from_row(row)
        # test = pricer.cashflows.values
        # test2=pricer.timeToCashflowInYears.values
        # check = test*test2

        tmp={}
        tmp.update({'Bond':row['InstrCode'],
                    'CashFlows':pricer.cashflows.values[1:],
                    'CashFlowTimes':pricer.timeToCashflowInYears.values[1:],
                    'AccruedInterest':pricer.compute_accrued_interest(),
                    'DirtyPrice':row['CleanPrice']+pricer.compute_accrued_interest(),
                    'Amt':row['Amt']})
        df_fitting_data.append(tmp)

    df_fitting_data = pd.DataFrame(df_fitting_data)
    df_fitting_data.to_clipboard()

    # test price
    L=0.33
    b1=0.01
    b2=0.002
    b3=-0.02
    price =  ns.get_NS_bond_price(df_fitting_data,b1,b2,b3)

    #df_fitting_data['DirtyPrice'] = price

    ydata = df_fitting_data.DirtyPrice.values
    xdata = df_fitting_data[['CashFlowTimes', 'CashFlows']]
    sigma = 1000000.0/df_fitting_data.Amt.values #this corresponds to relative weighting by notional amount.

    popt, pcov = curve_fit(ns.get_NS_bond_price, xdata, ydata, sigma=sigma, absolute_sigma=False)

    print popt

    t=np.arange(0.5,30,0.5)
    zc = ns.zeroYield(L,popt[0],popt[1],popt[2],t)

    z1 = ns.zeroYield(L,popt[0],popt[1],popt[2],1.0)
    z2 = ns.zeroYield(L,popt[0],popt[1],popt[2],2.0)
    z5 = ns.zeroYield(L,popt[0],popt[1],popt[2],5.0)
    z10 = ns.zeroYield(L,popt[0],popt[1],popt[2],10.0)
    z30 = ns.zeroYield(L,popt[0],popt[1],popt[2],30.0)



    tmp={}
    tmp.update({'TradeDate':d,
                'b1':popt[0],
                'b2':popt[1],
                'b3':popt[2],
                'z1':z1,
                'z2':z2,
                'z5':z5,
                'z10':z10,
                'z30':z30})
    fitting_results.append(tmp)

fitting_results = pd.DataFrame(fitting_results)
fitting_results.to_csv('fitting_results_weighted.csv')


    #print popt, pcov


