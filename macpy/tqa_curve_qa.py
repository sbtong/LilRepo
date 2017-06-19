import pandas as pd
import numpy as np

from pandas.tseries.offsets import BDay

import macpy.utils.database as db

from macpy.curve_stats import std_dev_bounds, TQACurveStats

def _print_header():
    print 'curve_id, agg_std_dev, bounds'

def run_chain_rics(trade_date, database, config, curve_short_name):
    curve_df = db.TQAChainRics(database, config).extract_from_db()
    if curve_short_name is not None:
        curves = [curve_short_name]
    else:
        curves = np.unique(curve_df['curve_id'])
    end_date = pd.to_datetime(trade_date)
    start_date = end_date - BDay(120)
    start_date = pd.datetime.strftime(start_date, '%Y-%m-%d')
    end_date = pd.datetime.strftime(end_date, '%Y-%m-%d')
    for curve in curves:
        ric_df = curve_df[curve_df.curve_id == curve]
        dbconn = db.ReutersCurvesQuotes(start_date, end_date, ric_df, database=None, config=None)
        df = dbconn.extract_from_db()
        df = df.merge(ric_df, on="Ric", how="left")
        ric = curve_df[curve_df.curve_id == curve]['chain_ric'].values[0]
        if df.empty:
            print 'No data found for curve: {}, {} to {}'.format(ric, start_date, end_date)
            continue
        else:
            wt_std_dev, wt_std_dev_bound = std_dev_bounds(df)
            print '{}, {}, {}'.format(ric ,wt_std_dev, wt_std_dev_bound)
            cs = TQACurveStats(trade_date, ric, enviroment=database, config=config)
            cs.upload_curve_stats(wt_std_dev, 5)
            cs.upload_curve_stats(wt_std_dev_bound, 6)
    return

def run_basis_curves(trade_date, database, config, curve_short_name):
    curve_df = db.BasisSwapCurves(database, config).extract_from_db()
    if curve_short_name is not None:
        curves = [curve_short_name]
    else:
        curves = curve_df['BasisSwapShortName']
    end_date = pd.to_datetime(trade_date)
    start_date = end_date - BDay(120)
    start_date = pd.datetime.strftime(start_date, '%Y-%m-%d')
    end_date = pd.datetime.strftime(end_date, '%Y-%m-%d')
    for curve in curves:
        try:
            dbconn = db.BasisSwapQuote(start_date, end_date, curve, database=None, config=None)
            df = dbconn.extract_from_db()
            df = df.merge(dbconn._df, on="Ric", how="left")
        except:
            df = pd.DataFrame()
        if df.empty:
            print 'No data found for curve: {}, {} to {}'.format(curve, start_date, end_date)
            continue
        else:
            wt_std_dev, wt_std_dev_bound = std_dev_bounds(df)
            print '{}, {}, {}'.format(curve, wt_std_dev, wt_std_dev_bound)
            cs = TQACurveStats(trade_date, curve, enviroment=database, config=config)
            cs.upload_curve_stats(wt_std_dev, 5)
            cs.upload_curve_stats(wt_std_dev_bound, 6)
    return

def run_ois_curves(trade_date, database, config, curve_short_name):
    curve_df = db.OISCurves(database, config).extract_from_db()
    if curve_short_name is not None:
        curves = [curve_short_name]
    else:
        curves = curve_df['OISShortName']
    end_date = pd.to_datetime(trade_date)
    start_date = end_date - BDay(120)
    start_date = pd.datetime.strftime(start_date, '%Y-%m-%d')
    end_date = pd.datetime.strftime(end_date, '%Y-%m-%d')
    for curve in curves:
        dbconn = db.OISCurveQuote(start_date, end_date, curve, database=None, config=None)
        df = dbconn.extract_from_db()
        df = df.merge(dbconn._df, on="Ric", how="left")
        if df.empty:
            print 'No data found for curve: {}, {} to {}'.format(curve, start_date, end_date)
            continue
        else:
            wt_std_dev, wt_std_dev_bound = std_dev_bounds(df)
            print '{}, {}, {}'.format(curve, wt_std_dev, wt_std_dev_bound)
            cs = TQACurveStats(trade_date, curve, enviroment=database, config=config)
            cs.upload_curve_stats(wt_std_dev, 5)
            cs.upload_curve_stats(wt_std_dev_bound, 6)
    return

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-s", "--start_date", dest="start_date", help="Starting trade date to run curve stats", metavar="2014-05-01 example")
    parser.add_option("-e", "--end_date", dest="end_date", help="Starting trade date to run curve stats", metavar="2014-05-01 example")
    parser.add_option('-d', '--database', dest='database', help='database name', metavar='example DEV')
    parser.add_option('-g', '--config', dest='config', help='configuration file ncame', metavar='example database.config')
    parser.add_option("-c", "--curve", dest="curve_type", help="curve type", metavar=" example OIS, BASIS")
    parser.add_option("-n", "--curve_short_name", dest="curve_short_name", help="curve id", metavar=" example =#USDRZ01", default=None)
    (options, args) = parser.parse_args()
    database = options.database
    config = options.config
    start_date = options.start_date
    end_date = options.end_date
    timerange = pd.bdate_range (start_date, end_date)
    curve_short_name = options.curve_short_name
    for trade_date in timerange:
        _print_header()
        if options.curve_type == 'OIS':
            run_ois_curves(trade_date, database, config, curve_short_name)
        elif options.curve_type == 'BASIS':
            run_basis_curves(trade_date, database, config, curve_short_name)
        elif options.curve_type == 'RIC':
            run_chain_rics(trade_date, database, config, curve_short_name)