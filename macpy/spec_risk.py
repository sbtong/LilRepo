import numpy as np
import pandas as pd
import datetime
import functools
import getpass
import math

# import macpy
import macpy.utils.database as db
import macpy.bond as bond

from scipy.interpolate import interp1d

import string


class SpecRisk(object):
    def __init__(self, *args, **kwargs):
        self.df_deriv = kwargs.pop('df_deriv')
        self.trade_date = kwargs.pop('trade_date')
        self.country = kwargs.pop('country')
        self.currency = kwargs.get('currency')
        self.min_spread = kwargs.get('min_spread', 0.05)
        self.max_spread = kwargs.get('max_spread', -0.05)
        self.database = kwargs.get('database', 'DEV')
        self.config = kwargs.get('config', None)
        self.freq = kwargs.get('freq', 'weekly')
        self.curve_id = kwargs.pop('curve_id', None)
        self.debug = kwargs.get('config', False)
        self.medians = []
        self.empty = False
        self.median = None
        if self.freq == 'daily':
            self.resample_freq = 'B'
            self.resample_how = 'first'
            self.halflife = 126
            self.horizon = 252
            self.lookback = 365*4 #lookback in days
            self.min_periods = 252
        if self.freq == 'weekly':
            self.resample_freq = [0,1,2,3,4]
            self.resample_how = 'first'
            self.halflife = 26
            self.horizon = 52
            self.lookback = 365*4 #lookback in days
            self.min_periods = 52

    def run(self):
        self.process_data()
        if self.empty:
            return
        for resample_freq in self.resample_freq:
            if self.currency == 'USD':
                self._calc_hard_res_vol(resample_freq)
            elif self.currency == 'EUR':
                self._calc_hard_res_vol(resample_freq)
            else:
                self._calc_local_res_vol(resample_freq)
        self.median = np.nanmedian(self.medians)
        return

    def process_data(self):
        self.df_deriv.AnalysisDate = self.df_deriv.AnalysisDate.apply(lambda x: pd.to_datetime(x))
        self.df_deriv = self.df_deriv[self.df_deriv['AnalysisDate'] <= pd.to_datetime(self.trade_date)]
        self.df_deriv = self.df_deriv[self.df_deriv['AnalysisDate'] > pd.to_datetime(self.trade_date) - datetime.timedelta(days = self.lookback)]
        self.df_deriv = self.df_deriv[self.df_deriv.Maturity > 1.01]
        self.df_deriv['Weekday'] = self.df_deriv.AnalysisDate.apply(lambda x: x.weekday())
        # if self.trade_date not in self.df_deriv.AnalysisDate:
        #     self.empty = True
        if self.df_deriv.empty:
            self.empty = True

    def _filter_trade_date(self):
        itemId = 1
        dbconn = db.BondOverides(self.curve_id, self.trade_date, itemId, self.database, self.config)
        df = dbconn.extract_from_db()
        self._bond_overides(df, 'Trade date removed: ')

    def _bond_overides(self, df, msg):
        if not df.empty:
            for instr_code in df['InstrCode']:
                if np.any((self.df_deriv['InstrCode'] == instr_code) & (self.df_deriv['AnalysisDate'] == self.trade_date)):
                    if self.debug:
                        print msg +  str(self.trade_date) + ' ' + str(instr_code) 
                    self.df_deriv = self.df_deriv[self.df_deriv['InstrCode'] != instr_code]

    def _calc_hard_res_vol(self, resample_freq):
        df_res = self.df_deriv[self.df_deriv.Weekday == resample_freq].pivot('AnalysisDate', 'InstrCode', 'Residual')

        # limit maximum and minimum spread
        df_res = np.minimum(df_res, self.min_spread)
        df_res = np.maximum(df_res, self.max_spread)

        if len(df_res) < 3:
            self.medians.append(np.array([np.nan]))
            return
        # interpolate gaps (temporary measure for Cap Group documentation)
        df_maturity = self.df_deriv[self.df_deriv.Weekday == resample_freq].pivot('AnalysisDate', 'InstrCode', 'Maturity')
        # Calculate spread residual returns
        df_res_ret = df_res - df_res.shift(1)
        # Cap the residual returns at +/- 2%
        df_res_ret = np.minimum(df_res_ret, 0.02)
        df_res_ret = np.maximum(df_res_ret, -0.02)
        # calculate the rolling volatility of residual spread returns.
        df_res_vol = pd.ewmvar(df_res_ret, halflife=self.halflife, min_periods=self.min_periods).apply(np.sqrt) * np.sqrt(self.horizon)
        df_res_vol = df_res_vol[df_maturity.notnull()]

        self.medians.append(df_res_vol.median(axis=1, numeric_only=True).values[-1:])

    def _calc_local_res_vol(self, resample_freq):
        df_oas = self.df_deriv[self.df_deriv.Weekday == resample_freq].pivot('AnalysisDate', 'InstrCode', 'OAS_Govt')

        df_oas = np.minimum(df_oas, 0.02)
        df_oas = np.maximum(df_oas, -0.02)

        if len(df_oas) < 3:
            self.medians.append(np.array([np.nan]))
            return
        # Calculate the residual spread volatilities
        df_oas_ret = (df_oas - df_oas.shift(1))
        df_oas_ret = np.minimum(df_oas_ret, 0.01)
        df_oas_ret = np.maximum(df_oas_ret, -0.01)
        df_oas_vol = pd.ewmvar(df_oas_ret, halflife=self.halflife, min_periods=self.min_periods).apply(np.sqrt) * np.sqrt(self.horizon)
        df_maturity = self.df_deriv[self.df_deriv.Weekday == resample_freq].pivot('AnalysisDate','InstrCode','Maturity')
        df_oas_vol = df_oas_vol[df_maturity.notnull()]
        self.medians.append(df_oas_vol.median(axis=1, numeric_only=True).values[-1:])

def get_history(country, currency, curve_id, end_date):
    curve_short_name = '{}.{}.SPRVOL'.format(country, currency)
    if currency in ['USD', 'EUR']:
        dbconn = db.EMHardSvSprCurves(country, currency, end_date, database='DEV')
    else:
        dbconn = db.EMLocalSvSprCurves(curve_short_name, database='DEV')
    df = dbconn.extract_from_db()
    if (currency != 'USD') and (currency != 'EUR'):
        df = df[df['AnalysisDate'] > '2014-01-01']
    if currency == 'KRW' and country == 'KR':
        df = df[df.Maturity > 6.01]
    return df

class SpecRiskDB(object):
    def __init__(self, trade_date, curve_id, median, enviroment=None, config=None, production_table=False):
        self.tradeDate = trade_date
        self.curve_id = curve_id
        self.median = median
        self.enviroment = enviroment
        self.config = config
        self.production_table = production_table

    def write_to_db(self):
        self.delete_sql = self.delete_query()
        user_name = getpass.getuser()
        dt = str(datetime.datetime.now())[0:-3]
        try:
            db.MSSQL.execute_commit(self.delete_sql, self.enviroment, self.config)
        except Exception as e:
            print "Exception thrown commit:", e, self.delete_sql
            return 1
        d = {}
        d['CurveNodeId'] = self.curve_id
        d['TradeDate'] = self.tradeDate
        d['IsCorrected'] = ''
        d['Quote'] = self.median
        d['Lud'] = dt
        d['Lub'] = user_name

        query = self.create_insert_sql(d)
        try:
            db.MSSQL.execute_commit(query, self.enviroment, self.config)
        except Exception as e:
            print "Exception thrown commit:", e.message, query
            return 1
        return 0

    def delete_query(self):
        if self.production_table:
            sqlstatement = string.Template("""
            DELETE from marketdata.dbo.CurveNodeQuote where CurveNodeId = '$CurveId'  and TradeDate = '$TradeDate' 
            """).substitute({'CurveId': self.curve_id, 'TradeDate': self.tradeDate})
        else:
            sqlstatement = string.Template("""
            DELETE from marketdata.dbo.CurveNodeQuote_Research where CurveNodeId = '$CurveId'  and TradeDate = '$TradeDate'
            """).substitute({'CurveId': self.curve_id, 'TradeDate': self.tradeDate})
        return sqlstatement

    def create_insert_sql(self, d):
        if self.production_table:
            sqlstatement = string.Template(
                """INSERT INTO marketdata.dbo.CurveNodeQuote VALUES('$CurveNodeId', '$TradeDate', '$Quote', '$IsCorrected', '$Lud', '$Lub')""").substitute(
                d)
        else:
            sqlstatement = string.Template(
                """INSERT INTO marketdata.dbo.CurveNodeQuote_Research VALUES('$CurveNodeId', '$TradeDate', '$Quote', '$IsCorrected', '$Lud', '$Lub')""").substitute(
                d)
        return sqlstatement

if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-s", "--startDate", dest="startDate", help="Starting trade date to run curve generator",
                      metavar="2014-05-01 example")
    parser.add_option("-e", "--endDate", dest="endDate", help="End trade date to run curve generator",
                      metavar="2014-05-02 example")
    parser.add_option("-c", "--currency", dest="currency", help="currency to run curve generator",
                      metavar="example USD")
    parser.add_option("-q", "--country", dest="country", help="countryName to run curve generator",
                      metavar="example US")
    parser.add_option('-d', '--database', dest='database', help='enviroment name', metavar='example DEV')
    parser.add_option('-g', '--config', dest='config', help='configuration file name',
                      metavar='example database.config')
    parser.add_option('-t', '--test', dest='test', help='save spec_risk to file instead of db',
                      action="store_true")
    parser.add_option('-r', '--production_table', dest='production_table', help='save to CurveNodeQuote or CurveNodeQuote_Research',
                      action="store_true")
    (options, args) = parser.parse_args()

    params = {}
    params['currency'] = options.currency
    params['country'] = options.country
    params['database'] = options.database
    params['config'] = options.config
    params['production_table'] = options.production_table
    params['debug'] = False
    if options.production_table is None:
        production_table = False
    else:
        production_table = True
    if options.currency == 'USD':
        curve_type = 'RsdSprVol.EmgHrdCcy'
    else:
        curve_type = 'RsdSprVol.EmgLclCcy'

    timerange = pd.bdate_range(options.startDate, options.endDate)

    dbconn = db.AxiomaSpreadID(options.currency, options.country, options.database, options.config)
    df_hist = None
    curve_node_id = dbconn.extract_from_db().values[0][0]
    dbconn = db.AxiomaCurveID(curve_type=curve_type, database=options.database, config=options.config)
    df_curves = dbconn.extract_from_db()
    curve = df_curves[df_curves['CountryEnum'] == options.country]
    params['curve_id'] = curve['CurveId'].values[0]
    print 'Running Residual Spread Vol calc for Country:{}, Currency:{}, CurveNodeID: {}'.format(options.country, options.currency, curve_node_id)
    res_vol = []
    dates = []
    for trade_date in timerange:
        print trade_date
        if df_hist is None:
            df_hist = get_history(options.country, options.currency, params['curve_id'], options.endDate)
        params['df_deriv'] = df_hist.copy(deep=True)
        params['trade_date'] = trade_date
        sr = SpecRisk(**params)
        sr.run()
        if sr.median is None:
            continue
        if math.isnan(sr.median):
            continue
        if not options.test:
            sp = SpecRiskDB(trade_date.strftime('%Y-%m-%d'),
                            curve_node_id, 
                            sr.median,
                            production_table=production_table)
            sp.write_to_db()
        else:
            res_vol.append(sr.median)
            dates.append(trade_date)
    if options.test:
        df_test = pd.DataFrame()
        df_test['res_vol'] = res_vol
        df_test['trade_date'] = dates
        df_test.to_csv('{}.csv'.format(curve_node_id))

