import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import weighted as wq # this is from the package 'wquantiles', for calculating a weighted median.
import datetime
import CurveQueries as cq
import utils.Utilities as U
import curve_utility as cu
import os
import concurrent.futures
import argparse
import time
import utils.database as db

def get_current_time():
    return str(datetime.datetime.now())[0:-3]

#calling R from Python
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()   #ensures easy translation of arrays between R and Python

import macpy.utils.ngc_utils as u
import macpy.utils.ngc_queries as q
import visualization.ngc_plot as p

def get_curves(IndustryGroup, start_date, end_date):
    sql = """
    use marketdata
    SELECT c.[AxiomaDataId]
       --,u.[CurveShortName]
          ,c.[TradeDate]
          ,c.[TenorEnum]
          ,[Quote]
          ,[category]
       ,t.InYears
       ,p.[SumAmtOutstanding]
       --, RE.CountryOfRisk
     --, region.Region
     --, region.RegionGroup
     --, IndGrp.IndustryGroup
     --, IndMap.Market
     --, IndMap.Sector
     --, IndMap.SectorGroup
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
    and IndGrp.IndustryGroup = '%s'
    and c.TradeDate >= '%s'
    and c.TradeDate <= '%s'
    order by c.TradeDate, c.AxiomaDataId, c.category, t.InYears
    """ % (IndustryGroup, start_date, end_date)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def write_curves_to_db(connection, delete, level, average, chunkSize=1000):
    #connection = U.createMSSQLConnection(macDBInfo)
    curs = connection.cursor()
    xcIssuerCurveCombinedDelete = 'delete from [MarketData].[dbo].[xcIssuerCurve] where AxiomaDataId = %s and category = %s'
    # xcIssuerCurveAverageDelete = 'delete from [MarketData].[dbo].[xcIssuerCurveAverage] where AxiomaDataId = %s --and TradeDate = %s'
    for chunk in U.grouper(delete, chunkSize):
        curs.executemany(xcIssuerCurveCombinedDelete, chunk)
        #curs.executemany(xcIssuerCurveAverageDelete, chunk)
        connection.commit()

    xcIssuerCurveCombinedQuery = 'insert into [MarketData].[dbo].[xcIssuerCurve] values (%s,%s,%s,%d,%s,%s, GETDATE())'
    for chunk in U.grouper(level, chunkSize):
        print 'Writing %s records', len(chunk)
        curs.executemany(xcIssuerCurveCombinedQuery, chunk)
        connection.commit()

    #xcIssuerCurveAverage = 'insert into [MarketData].[dbo].[xcIssuerCurveAverage] values (%s,%s,%d,%d,%s, GETDATE())'
    xcIssuerCurveAverage = 'UPDATE [MarketData].[dbo].[xcIssuerProperty] SET Average = %d, Alpha = %d WHERE AxiomaDataId = %s AND TradeDate = %s'
    for chunk in U.grouper(average, chunkSize):
        curs.executemany(xcIssuerCurveAverage, chunk)
        connection.commit()


def calc_curve(df_both, alpha, tenor):
    #alpha = level/change weighting coefficient
    #tenor = choice of tenor
    #df_both = merged dataframe of levels and changes for each tenor by TradeDate
    tenor=str(tenor)
    curve=[]
    first_iteration=True
    for i,r in df_both[[tenor + '_l', tenor + '_c']].iterrows():
        if first_iteration:
            c=r[tenor + '_l']
        else:
            c=(1.-alpha)*r[tenor + '_l'] + alpha*(c_prev+ r[tenor + '_c'])
        #store results
        tmp={}
        tmp.update({'TradeDate':i, 'Curve':c})
        curve.append(tmp)
        #get ready for next iteration
        c_prev = c
        first_iteration=False
    df_curve = pd.DataFrame(curve)
    return df_curve

def main():

    #1. Get histories for a particular sector: s_l, s_c, w
    #2. Calculate s_combined using an input alpha
    #3. Calculate a weighted average s_combined
    #4. Store s_combined, average(s_combined)



    pwd = os.path.dirname(os.path.realpath(__file__)) + os.sep
    max_workers = 2
    configFile = open(pwd + "/production.config", 'r')
    configuration = U.loadConfigFile(configFile)
    environment  = 'DEV'

    sectionID = cu.getAnalysisConfigSection(environment)
    envInfoMap = U.getConfigSectionAsMap(configuration, sectionID)
    macDBInfo = cu.getMSSQLDatabaseInfo(U.getConfigSectionAsMap(
        configuration, envInfoMap.get('macdb', None)))

    connection = U.createMSSQLConnection(macDBInfo)
    curs = connection.cursor()

    start_date = '2012-12-31'
    end_date = '2016-12-31'

    names = ['CNSDA','MEDIA']
    for iname in names:
        IndustryGroup = iname
        df_s = get_curves(IndustryGroup, start_date, end_date)

        df_s_level = df_s.loc[df_s.category == 'll']
        df_s_change = df_s.loc[df_s.category == 'lc']
        df_s_weight = df_s.loc[df_s.category == 'w']
        df_s_coupon = df_s.loc[df_s.category == 'cn']

        df_s_change = df_s_change.replace(to_replace=-10, value=0)

        #loop over curves:
        for c in df_s.AxiomaDataId.unique():
            print 'Curve', c
            df_s_level_c = df_s_level.loc[df_s_level.AxiomaDataId == c]
            df_s_change_c = df_s_change.loc[df_s_change.AxiomaDataId == c]
            df_s_weight_c = df_s_weight.loc[df_s_weight.AxiomaDataId == c]
            df_sumAmtOut = df_s.loc[df_s.AxiomaDataId == c][['TradeDate','SumAmtOutstanding']]

            df_s_level_c = df_s_level_c.pivot('TradeDate', 'TenorEnum', 'Quote')
            df_s_change_c = df_s_change_c.pivot('TradeDate', 'TenorEnum', 'Quote')
            df_s_weight_c = df_s_weight_c.pivot('TradeDate', 'TenorEnum', 'Quote')

            results=[]
            results_av=[]
            delete=[]
            first_iteration = True
            alpha=0.85
            comb_prev = 0
            sbar_prev = 0
            print 'starting loop over dates'
            for d in df_s_level_c.index:
                if first_iteration:
                    comb = df_s_level_c.loc[d,:]
                else:
                    comb = (1-alpha)*df_s_level_c.loc[d, :] + alpha*(df_s_change_c.loc[d,:] + comb_prev)

                comb_prev = comb

                tenor = np.array(comb.index)
                #level = u.mExp(np.array(comb.values))
                level = np.array(comb.values)

                #calculate average combined curve and return
                #mExp_level = u.mExp(level)
                w = np.array(df_s_weight_c.loc[d,:].values)
                sbar = np.inner(level, w)/w.sum()
                if first_iteration:
                    ret = 0
                else:
                    ret = sbar - sbar_prev
                sbar_prev = sbar

                first_iteration = False

                # collect results for write: #TODO: there must be a better way than looping through the array?
                for i in range(0, level.size):
                    results.append((int(c), d, tenor[i], level[i], 'cb', 'dantonio'))
                results_av.append((ret, sbar, int(c), d))
            delete.append((int(c),'cb'))
            print 'completed loop over dates'


            write_curves_to_db(connection,delete, results, results_av)





if __name__ == '__main__':
    t1 = time.clock()
    main()
    t2 = time.clock()
    print t2-t1