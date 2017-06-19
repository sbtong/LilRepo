import numpy as np
import pandas as pd
import macpy.bond as bond
import macpy.utils.database as db
import macpy.bond_utils as bu
import getpass
import string
import datetime

from pandas.tseries.offsets import BDay

from macpy.curve_functions import create_govt_zero_curve


def _chisqg(ydata, ymod, sd=None):  
    # Chi-square statistic (Bevington, eq. 6.9)  
    if sd==None:  
       chisq = np.sum((ydata-ymod)**2)  
    else:  
       chisq = np.sum( ((ydata-ymod)/sd)**2 )  

    return chisq 

def compute_redchisqg(ydata, ymod, deg=2, sd=None):  
    # reduced Chi-square statistic  
    x = _chisqg(ydata, ymod, sd)
    # Number of degrees of freedom assuming 2 free parameters  
    nu = ydata.size - 1 - deg  

    return x/nu  

def compute_stats(df, gvtCurve):
    metrics = {}
    errors_price = []
    zero_price = []
    for i, x in df.iterrows():
        issueDate, maturityDate, valuationDate, marketPrice, settlement_adj, coupon, MarketStandardYield = bu.process_reuters_market_data_row(x)
        # freq = x['CompFreqCode']
        freq = 2.0
        first_cpn_dt = x['FrstCpnDate']
        last_cpn_dt = x['LastCpnDate']
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        clean_price = pricer.compute_clean_price(gvtCurve)
        # dirty_price = pricer.compute_dirty_price(gvtCurve)
        zero_price.append(clean_price)
        errors_price.append(np.power(marketPrice - clean_price, 2))
    df['errors_price'] = errors_price
    df['ZeroPrice'] = zero_price
    metrics['rmse_price'] = np.sqrt(np.sum(errors_price) / df.shape[0]) #ItemId 1
    metrics['redchisqg_price'] = compute_redchisqg(df['Price'], df['ZeroPrice']) #ItemId 2
    metrics['real_std'] = np.std(df['MatStdYld']/100.0) #ItemId 3
    metrics['pred_std'] = np.std(gvtCurve._curveDataFrame['Quote']) #ItemId 4
    return metrics

def compute_sov_curve_stats(valuationDate, currency, country, window, database=None):
    dbconn = db.ReutersZCurveCodes(database=database)
    df_curves = dbconn.extract_from_db()
    curveId = df_curves[df_curves['ISOCurCode'] == currency]['CurveId'].values[0]

    dbconn = db.DerivedDataExtractSovereignCurveBenchmarkBondsSQL(valuationDate, curveId, database=database)
    df = dbconn.extract_from_db()
    if df.empty:
        return None
    try:
        gvtCurve = create_govt_zero_curve(valuationDate, currency, database, False)
    except:
        return None
    metrics = compute_stats(df, gvtCurve)
    df_hist = load_history(valuationDate, currency, window, database)
    metrics['agg'], metrics['std_bound'] = std_dev_bounds(df_hist)
    return metrics

def load_history(valuationDate, currency, window=60, database=None):
    df = pd.DataFrame()
    start_date = valuationDate - BDay(window+5)
    for dt in pd.bdate_range(start_date, valuationDate):
        try:
            curve = create_govt_zero_curve(str(dt).split(' ')[0], 'USD', 'PROD')
            df = df.append(curve._curveDataFrame)
        except:
            pass
    return df

def std_dev_bounds(df, window=60):
    to_date =  df.TradeDate.max()
    Lambda = np.exp(-np.log(2) / 30.0)
    df.TradeDate = pd.to_datetime(df.TradeDate)
    df_diff = pd.merge(df, df.shift(), left_index=True, right_index=True, how='outer')
    df_diff['Days'] = pd.to_datetime(to_date) - df_diff.TradeDate_x
    df_diff['YieldDiff'] = df_diff['Quote_x'] - df_diff['Quote_y']
    df_diff['YieldDiff'].fillna(value = 0,inplace=True)
    df_diff['wt'] = Lambda ** df_diff['Days'].dt.days
    df_diff['YieldDiffSq'] = df_diff['YieldDiff'] ** 2
    df_agg = df_diff.groupby(['TradeDate_x']).sum()
    df_agg['Date'] = pd.to_datetime(df_agg.index)
    df_agg['Days'] = pd.to_datetime(to_date) - df_agg['Date']
    df_agg['wt'] = Lambda ** df_agg['Days'].dt.days
    df_agg['L2_i'] = np.sqrt(df_agg['YieldDiffSq'])
    df_agg['Sgn'] = np.sign(df_agg['YieldDiff'])
    df_agg['X_i'] = df_agg['Sgn']*df_agg['L2_i']
    df_agg['wtX_i2'] = df_agg['wt']*df_agg['X_i']**2
    # df_agg['rSum'] = pd.DataFrame.rolling(df_agg[['wtX_i2']], window=60).sum()
    df_agg['rSum'] = pd.rolling_sum(df_agg[['wtX_i2']], window=60)
    # df_agg['rWtSum'] = pd.DataFrame.rolling(df_agg[['wt']], window=60).sum()
    df_agg['rWtSum'] = pd.rolling_sum(df_agg[['wt']], window=60)
    df_agg['wtStDev'] = np.sqrt(df_agg['rSum']/df_agg['rWtSum'])
    df_agg = df_agg.reset_index()
    df_agg['z'] = (pd.merge(df_agg,df_agg.shift(), on=df_agg.Date)['X_i_x']
             /pd.merge(df_agg,df_agg.shift(), on=df_agg.Date)['wtStDev_y'])
    return df_agg['X_i'].values[-1], df_agg['wtStDev'].values[-1]


class CurveStats(object):
    def __init__(self, tradeDate, curveId, enviroment=None, config=None, production_table=False):
        self.tradeDate = tradeDate
        self.curveId = curveId
        self.enviroment = enviroment
        self.config = config
        self.production_table = production_table

    def upload_curve_stats(self, ItemValue, ItemId):
        self.delete_sql = self._create_delete_sql(ItemId)
        try:
            db.MSSQL.execute_commit(self.delete_sql, self.enviroment, self.config)
        except Exception as e:
            print "Exception thrown commit:", e, self.delete_sql
            return 1
        d = {}
        d['Lub'] = getpass.getuser()
        d['Lud'] = str(datetime.datetime.now())[0:-3]
        d['CurveId'] = self.curveId
        d['TradeDate'] = self.tradeDate
        d['ItemId'] = ItemId
        d['ItemValue'] = ItemValue
        self.insert_sql = self._create_insert_sql(d)
        try:
            db.MSSQL.execute_commit(self.insert_sql, self.enviroment, self.config)        
        except Exception as e:
            print "Exception thrown commit:", e.message, self.insert_sql
            return 1
        return 0

    def _create_delete_sql(self, ItemId):
        if self.production_table:
            sqlstatement = string.Template("""
            DELETE FROM [MarketData].[dbo].CurveStats where CurveId = '$CurveId'  and TradeDate = '$TradeDate' and ItemId = '$ItemId' 
            """).substitute({'CurveId':self.curveId, 'TradeDate': self.tradeDate, 'ItemId':ItemId})
        else:
            sqlstatement = string.Template("""
            DELETE FROM [MarketData].[dbo].CurveStats_Research where CurveId = '$CurveId'  and TradeDate = '$TradeDate' and ItemId = '$ItemId'
            """).substitute({'CurveId':self.curveId, 'TradeDate': self.tradeDate, 'ItemId':ItemId})
        return sqlstatement

    def _create_insert_sql(self, d):
        if self.production_table:
            sql = string.Template("""INSERT INTO [MarketData].[dbo].CurveStats VALUES ('$CurveId', '$TradeDate', '$ItemId', '$ItemValue', '$Lud', '$Lub')""").substitute(d)
        else:
            sql = string.Template("""INSERT INTO [MarketData].[dbo].CurveStats_Research VALUES ('$CurveId', '$TradeDate', '$ItemId', '$ItemValue', '$Lud', '$Lub')""").substitute(d)
        return sql

class TQACurveStats(object):
    def __init__(self, tradeDate, CurveIdentifier, enviroment=None, config=None, production_table=False):
        self.tradeDate = tradeDate
        self.CurveIdentifier = CurveIdentifier
        self.enviroment = enviroment
        self.config = config
        self.production_table = production_table

    def upload_curve_stats(self, ItemValue, ItemId):
        self.delete_sql = self._create_delete_sql(ItemId)
        try:
            db.MSSQL.execute_commit(self.delete_sql, self.enviroment, self.config)
        except Exception as e:
            print "Exception thrown commit:", e, self.delete_sql
            return 1
        d = {}
        d['Lub'] = getpass.getuser()
        d['Lud'] = str(datetime.datetime.now())[0:-3]
        d['CurveIdentifier'] = self.CurveIdentifier
        d['TradeDate'] = self.tradeDate
        d['ItemId'] = ItemId
        d['ItemValue'] = ItemValue
        self.insert_sql = self._create_insert_sql(d)
        try:
            db.MSSQL.execute_commit(self.insert_sql, self.enviroment, self.config)        
        except Exception as e:
            print "Exception thrown commit:", e.message, self.insert_sql
            return 1
        return 0

    def _create_delete_sql(self, ItemId):
        if self.production_table:
            sqlstatement = string.Template("""
            DELETE FROM [MarketData].[dbo].TQACurveStats where CurveIdentifier = '$CurveIdentifier'  and TradeDate = '$TradeDate' and ItemId = '$ItemId' 
            """).substitute({'CurveIdentifier':self.CurveIdentifier, 'TradeDate': self.tradeDate, 'ItemId':ItemId})
        else:
            sqlstatement = string.Template("""
            DELETE FROM [MarketData].[dbo].TQACurveStats_Research where CurveIdentifier = '$CurveIdentifier'  and TradeDate = '$TradeDate' and ItemId = '$ItemId'
            """).substitute({'CurveIdentifier':self.CurveIdentifier, 'TradeDate': self.tradeDate, 'ItemId':ItemId})
        return sqlstatement

    def _create_insert_sql(self, d):
        if self.production_table:
            sql = string.Template("""INSERT INTO [MarketData].[dbo].TQACurveStats VALUES ('$CurveIdentifier', '$TradeDate', '$ItemId', '$ItemValue', '$Lud', '$Lub')""").substitute(d)
        else:
            sql = string.Template("""INSERT INTO [MarketData].[dbo].TQACurveStats_Research VALUES ('$CurveIdentifier', '$TradeDate', '$ItemId', '$ItemValue', '$Lud', '$Lub')""").substitute(d)
        return sql

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-s", "--startDate", dest="startDate", help="Starting trade date to run curve stats", metavar="2014-05-01 example")
    parser.add_option("-e", "--endDate", dest="endDate", help="End trade date to run curve stats", metavar="2014-05-02 example")
    parser.add_option('-d', '--enviroment', dest='enviroment', help='enviroment name', metavar='example DEV')
    parser.add_option('-g', '--config', dest='config', help='configuration file name', metavar='example database.config')
    parser.add_option("-c", "--currency", dest="currency", help="currency to run curve stats", metavar=" example USD")
    parser.add_option("-q", "--country", dest="country", help="country to run curve stats", metavar=" example US")
    parser.add_option('-t', '--production_table ', dest='production_table', help='save to CurveStats or CurveStats_Research', action="store_true")
    # parser.add_option('-w', '--window ', dest='window', help='lookback window', metavar="example 60")
    (options, args) = parser.parse_args()
    timerange = pd.bdate_range(options.startDate, options.endDate)
    dbconn = db.AxiomaCurveID(curve_type='Sov.Zero', database=options.enviroment)
    df = dbconn.extract_from_db()
    axioma_curve_id = df[(df['CountryEnum'] == options.country) & (df['CurrencyEnum'] == options.currency)]['CurveId'].values[0]
    if options.production_table:
        production_table = True
    else:
        production_table = False
    window = 60
    for trade_date in timerange:
        metrics = compute_sov_curve_stats(trade_date, options.currency, options.country, window, options.enviroment)
        if not metrics:
            print 'No data for: {} {}.{}'.format(trade_date, options.currency, options.country)
            continue
        cs = CurveStats(trade_date, axioma_curve_id, enviroment=options.enviroment, production_table=production_table)
        cs.upload_curve_stats(metrics['rmse_price'], 1)
        cs.upload_curve_stats(metrics['redchisqg_price'], 2)
        cs.upload_curve_stats(metrics['real_std'], 3)
        cs.upload_curve_stats(metrics['pred_std'], 4)
        cs.upload_curve_stats(metrics['agg'], 5)
        cs.upload_curve_stats(metrics['std_bound'], 6)
        print 'Success: {} {}.{}'.format(trade_date, options.currency, options.country)
