import os
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))
import next_gen_curve as ngc
from string import Template
import macpy.utils.database as db
import logging
import utils.ngc_utils as u
import utils.ngc_queries as q
import time

def main():
    params = {'specific_curve': 'USD-JPM-SEN',
              'start_date': '2017-02-16',
              'end_date': '2017-02-17',
              'Currency': 'USD',
              'alpha': 0.5,
              'gamma': 0.5,
              'smfitCompressionShort': 0.1,
              'smfitCompressionLong': 0.1,
              'sigmaScaleLevels': 0.2,
              'sigmaScaleChanges': 0.2,
              'smfitSpldf': 4,
              'write_data': False,
              'debugplot': False,
              'debugplot_ixlist': [-1],
              'plot_f': False,
              'sswidthLong': .05,
              'sswidthShort': 0.8,
              'maxIter': 10,
              'numOutlierIterations': 3,
              'numIterLevelGuess': 2,
              'numBondsLevels': 200,
              'numBondsChanges': 200,
              'overwriteStart': False,
              'fitSingle': False,
              'IndustryGroup': 'BANKS',
              'Sector': 'FINS',
              'SectorGroup': 'FIN',
              'Market': 'CR',
              'Region': 'NAMR',
              'RegionGroup': 'DVL',
              'PricingTier': 'SEN',
              'RiskEntityId': 100054757}

    ssc = ngc.SmoothSplineCurve(**params)
    s_prev = pd.read_csv('test/data/test_ngc_USD_20170216.csv')
    s_prev['MatDate'] = s_prev.MatDate.apply(lambda x: pd.to_datetime(x))
    s_prev = ssc.process_data(s_prev)
    s_i = pd.read_csv('test/data/test_ngc_USD_20170217.csv')
    s_i['MatDate'] = s_i.MatDate.apply(lambda x: pd.to_datetime(x))
    s_i = ssc.process_data(s_i)

    curve_list = pd.DataFrame(
        s_i[[ssc.curveCol, 'IndustryGroup', 'RiskEntityId', 'PricingTier', 'Region',
             'RegionGroup', 'Sector', 'SectorGroup', 'Market']].groupby(
            ssc.curveCol).first())  # list of all curves
    count = 0
    for i, row in curve_list.iterrows():  # loop over issuer curves
        ssc.specific_curve = i
        ssc.IndustryGroup = row['IndustryGroup']
        ssc.RiskEntityId = int(row['RiskEntityId'])
        ssc.PricingTier = row['PricingTier']
        ssc.Region = row['Region']
        ssc.RegionGroup = row['RegionGroup']
        ssc.Sector = row['Sector']
        ssc.SectorGroup = row['SectorGroup']
        ssc.Market = row['Market']
        CurveId = q.get_curveId(CurveShortName=ssc.specific_curve, RiskEntityId=ssc.RiskEntityId,
                                Currency=ssc.Currency, PricingTier=ssc.PricingTier)
        try:
            print 'Curve #', count, ssc.specific_curve  # , 'Curve ', c, dates_ts.ix[i_dt], self.IndustryGroup
            res_f, res_d = ssc.fit_date_pair(ssc.end_date, s_i, s_prev)

            # if self.write_data:
            #     logging.info('Writing data to DB started')
            #     u.write_curve_to_db(res_f, res_d, CurveId, date_i, self.tenorGridEnum)
            #     logging.info('Writing to DB complete')

            count += 1
        except Exception, e:
            print 'ERROR curve', i

if __name__ == '__main__':
    t1 = time.clock()
    main()
    t2 = time.clock()
    print t2-t1
