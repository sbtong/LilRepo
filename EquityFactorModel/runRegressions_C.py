#! /usr/bin/python
import macpy.utils.database as db
import macpy.utils.ngc_utils as u

import sys
from optparse import OptionParser
import glob
import os.path
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
import constrainedmodel as cm


def get_curves(Sector, start_date, end_date):
    sql = """
    use marketdata
    SELECT c.[AxiomaDataId]
          ,c.[TradeDate]
          ,[Quote]
    from xcIssuerCurveAverage c
    join xcIssuerProperty p on p.AxiomaDataId = c.AxiomaDataId
          and p.TradeDate = c.TradeDate
    join xcIssuerCurveUniverse u on u.AxiomaDataId = c.AxiomaDataId
    left join [MarketData].[dbo].[RiskEntity]   RE  on RE.RiskEntityId = u.RiskEntityId
                    and RE.ToDate = '31-Dec-9999'
    left join [MarketData].[dbo].[AL_IssuerIndustryGroup] IndGrp on u.RiskEntityId = IndGrp.IssuerId
    left join [MarketData].[dbo].[AL_IndustryMap]  IndMap on IndGrp.IndustryGroup = IndMap.IndustryGroup
    left join [MarketData].[dbo].[AL_CountryToRegionMap]  region on region.iso_scntry_cd = RE.CountryOfRisk
    where IndMap.Sector = '%s'
    and c.TradeDate >= '%s'
    and c.TradeDate <= '%s'
    order by c.TradeDate, c.AxiomaDataId
    """ % (Sector, start_date, end_date)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df


def get_properties(Sector, start_date, end_date):
    sql = """
    use marketdata
    SELECT u.[AxiomaDataId]
   ,u.[CurveShortName]
      ,p.[TradeDate]
    ,p.SumAmtOutstanding
   , RE.CountryOfRisk
 , region.Region
 , region.RegionGroup
 , IndGrp.IndustryGroup
 , IndMap.Market
 , IndMap.Sector
 , IndMap.SectorGroup
    from xcIssuerCurveUniverse u
    join xcIssuerProperty p on p.AxiomaDataId = u.AxiomaDataId
    left join [MarketData].[dbo].[RiskEntity]   RE  on RE.RiskEntityId = u.RiskEntityId
                    and RE.ToDate = '31-Dec-9999'
    left join [MarketData].[dbo].[AL_IssuerIndustryGroup] IndGrp on u.RiskEntityId = IndGrp.IssuerId
    left join [MarketData].[dbo].[AL_IndustryMap]  IndMap on IndGrp.IndustryGroup = IndMap.IndustryGroup
    left join [MarketData].[dbo].[AL_CountryToRegionMap]  region on region.iso_scntry_cd = RE.CountryOfRisk
    where IndMap.Sector = '%s'
    and p.TradeDate >= '%s'
    and p.TradeDate <= '%s'
    order by p.TradeDate, u.AxiomaDataId
    """ % (Sector, start_date, end_date)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def computeFactorReturns(df0, df1):

    IndustryGroups = ['Software',
                      'Communications Equipment',
                      'Technology Hardware, Storage & Peripherals',
                      'Electronic Equipment, Instruments & Components',
                      'Semiconductors & Semiconductor Equipment',
                      'Diversified Telecommunication Services',
                      'Wireless Telecommunication Services',
                      'Electric Utilities',
                      'Gas Utilities',
                      'Multi-Utilities',
                      'Water Utilities',
                      'Independent Power and Renewable Electricity Producers',
                      'Equity Real Estate Investment Trusts (REITs)',
                      'Real Estate Management & Development',
                      'Energy Equipment & Services']


    r = df1['rets'].fillna(0.0)
    estu = df1['estu']
    estu = estu[estu == 1.0].index
    estu = df0.index.intersection(estu)
    B = df0.drop(['rets', 'mcap', 'estu'], axis=1).fillna(0.0)

    #add constraints
    C = pd.DataFrame(0.0, index=['IndustryGroup'], columns=B.columns)
    C.loc['IndustryGroup', C.columns.intersection(IndustryGroups)] = 1.0
    model = cm.ConstrainedLinearModel(r, B, C)
    fr = model.params

    # result = sm.WLS(r[estu], B.loc[estu], weights=np.sqrt(df0['mcap'].loc[estu])).fit()
    # fr = result.params
    assets = r.index.intersection(B.index)
    resid = r[assets] - B.loc[assets].dot(fr)
    return (fr, resid)

def runmain(argv=None):
    if argv == None:
        argv = sys.argv

    usage = 'usage: %prog [options]\n'
    parser = OptionParser(usage=usage)
    parser.add_option("--in", dest="inputdir", default='rmm_data',
            help='Name of input directory (default: %default)')
    parser.add_option("--out", dest="outputdir", default='rmm_result',
            help='Name of input directory (default: %default)')
    (cmdoptions, args) = parser.parse_args(argv)


    #1. pull in spread data
    #2. calc returns
    #3. build exposure matrix
    #4. perform regression
    Sector = 'CNSD'
    start_date = '2016-07-13'
    end_date = '2016-07-14'

    df_s = get_curves(Sector,start_date, end_date)
    df_s.Quote = df_s.Quote.apply(lambda x: u.mLog_(x))

    df_s_r = df_s.pivot('TradeDate','AxiomaDataId', 'Quote')
    ret = df_s_r - df_s_r.shift()

    #df0 = ret.loc['2016-07-13']
    #df1 = ret.loc['2016-07-14']

    #df_p = get_properties(Sector, start_date, end_date)




    files = glob.glob(os.path.join(cmdoptions.inputdir, 'data*.csv'))
    
    if not os.path.exists(cmdoptions.outputdir):
        os.makedirs(cmdoptions.outputdir)
    
    dates = sorted(pd.to_datetime(x[-14:-4]).date() for x in files)
    for d0, d1 in itertools.izip(dates[:-1],dates[1:]):
        print d0, d1
        df0 = pd.DataFrame.from_csv(os.path.join(cmdoptions.inputdir, 'data' + str(d0) + '.csv'))
        df1 = pd.DataFrame.from_csv(os.path.join(cmdoptions.inputdir, 'data' + str(d1) + '.csv'))
        (fr, sr) = computeFactorReturns(df0, df1)
        fr.to_csv(os.path.join(cmdoptions.outputdir, 'fr' + str(d1) + '.csv'))
        sr.to_csv(os.path.join(cmdoptions.outputdir, 'sr' + str(d1) + '.csv'))



if __name__ == "__main__":
    runmain()

