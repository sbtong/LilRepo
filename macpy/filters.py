from macpy.utils import database as db
import pandas as pd
import numpy as np
import macpy.bond as bond
import macpy.bond_utils as bond_utils
import datetime
import getpass
import string
import inspect

import sys
from scipy.interpolate import interp1d

# pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# pd.set_option('display.precision', 2)
pd.set_option('display.expand_frame_repr', False)


class BondFilter(object):
    def __init__(self, currency, country, tradeDate, write_to_db=False, database=None, debug=False, logger=None):
        self.currency = currency
        self.tradeDate = tradeDate
        self.country = country
        self.write_to_db = write_to_db
        self.database = database
        self.debug = debug
        self.logger = logger
        self.apply_prc_filter = False
        self.get_bonds()
        self.get_axioma_curveid()

    def _log(self, s):
        if self.debug:
            if self.logger:
                self.logger.info(s)
            else:
                print s
        else:
            pass

    def _concat(self, l):
        s = ''
        for i in l:
            s = s + str(i) + ','
        return s[:-1]

    def get_bonds(self):
        self.get_reuters_curveid()
        if self.currency in ['CAD']:
            self.dbconn = db.UnfilteredBonds(self.currency, self.country, self.tradeDate, self.database)
            self.raw_df = self.dbconn.extract_from_db()
            self.process_data()
        elif self.currency in ['USD']:
            self.dbconn = db.UnfilteredBonds(self.currency, self.country, self.tradeDate, self.database)
            self.raw_df = self.dbconn.extract_from_db()
            self.process_reuters_filtered_bonds()
            self.bill_matrurity_filter()
            self.dbconn = db.DerivedDataExtractSovereignCurveBenchmarkBondsSQL(self.tradeDate, self.reuters_curveid, self.database)
            self.raw_df = self.dbconn.extract_from_db()
            self.process_reuters_filtered_bonds()
            self.bonds_df = pd.concat([self.bills_df, self.bonds_df])
            self.bonds_df = self.bonds_df.drop_duplicates('InstrCode', take_last=True)
        else:
            if self.currency in ['EUR', 'SEK']:
                self.dbconn = db.DerivedDataExtractSovereignCurveBenchmarkBondsSQL(self.tradeDate, self.reuters_curveid, self.database)
            else:
                self.dbconn =  db.DerivedDataExtractSovereignCurveBenchmarkBonds(self.tradeDate, self.reuters_curveid, self.database)
                # self.instr_codes = self._concat(list(dbconn.extract_from_db()['InstrCode']))
                # self.dbconn = db.DerivedDataExtractSoverignCurveByInstrCodes(self.tradeDate, self.instr_codes)
                self.apply_prc_filter = True
            self.raw_df = self.dbconn.extract_from_db()

            self.process_reuters_filtered_bonds()

        self._log('Benchmarks Count: ' + str(len(self.raw_df)))
        self._log(repr(self.raw_df))

    def get_axioma_curveid(self):
        dbconn = db.AxiomaCurveID(database=self.database)
        df = dbconn.extract_from_db()
        self.curveId = df[(df['CountryEnum'] == self.country) & (df['CurrencyEnum'] == self.currency)]['CurveId'].values[0]

    def get_reuters_curveid(self):
        dbconn = db.ReutersBenchmarkCurveId(currency=self.currency, country=self.country, database=self.database)
        df = dbconn.extract_from_db()
        if len(df[df['PriCtryFl'] == 1]) > 0:  # bonds can have PriCtryFl set as 1 or 0 (Primary Country Flag)
            self.reuters_curveid = df[df['PriCtryFl'] == 1]['CurveId'].values[0]
        else:
            self.reuters_curveid = df[df['PriCtryFl'] == 0]['CurveId'].values[0]


    def process_data(self):
        data_list = []
        for i, data in self.raw_df.iterrows():
            data_list.append({'InstrCode':data[0], 'OffgMktCode':data[1], 'DebtISOCurCode':data[2], 'IssTypeCode':data[3], 'DebtIssTypeCode':data[7], 'IssName':data[15],
              'IssNameTypeCode':data[16], 'IssClassTypeCode':data[18], 'IssSovTypeCode':data[19], 'FrstCpnRate':data[28],
              'Price':data[38], 'MatCorpYld':data[41], 'WrstCorpYld':data[42], 'MatStdYld':data[43], 'WrstStdYld':data[44], 
              'MatDate':data[22],  'TradeDate': data[36], 'InitAccDate':data[27], 'tenor':1.0*(data[22]-data[36]).days/365.25, 
              'DayCountCode':'ACT/360' if data[28]==0 else 'ACT/ACT', 'FrstCpnDate':data[71], 'LastCpnDate':data[72], 'DomSettDays':data[79]})
        # self.country_code = self.raw_df['IncISOCtryCode'].values[0]
        self.unfiltered_df = pd.DataFrame(data_list)
        self.bonds_df = self.unfiltered_df.copy(deep=True)
        self.bonds_df = self.bonds_df.drop_duplicates('InstrCode', take_last=True)

    def process_reuters_filtered_bonds(self):
        self.unfiltered_df = self.raw_df.copy(deep=True)
        # self.country_code = self.raw_df['ISOCtryCode'].values[0]
        self.unfiltered_df['tenor'] = [x/np.timedelta64(1, 'D')/365.25 for x in self.unfiltered_df['MatDate'] - self.unfiltered_df['TradeDate']]
        self.bonds_df = self.unfiltered_df.copy(deep=True)
        #drop duplicates
        self.bonds_df = self.bonds_df.drop_duplicates('InstrCode', take_last=True)

    def run_filter(self):
        # if self.currency in ['USD', 'CAD']:
        #     self.bill_matrurity_filter()
        self.banned_instruments()
        self.negative_price_filter()
        self.maturity_filter()
        self.compute_ytm()
        self.apply_filters()
        self.yield_overrides()
        self.tradeDate_overrides()
        self.ban_entire_trade()
        self.min_bond_rule()
        if self.write_to_db:
            self.write_bonds_to_db()
        self._log('Filtered Count: ' + str(len(self.bonds_df)))
        self._log(repr(self.bonds_df))
	return self.bonds_df

    def apply_filters(self):
        self.compute_yield_filter()
        self.yield_outlier_filter()
        self.short_end_filter()
        if self.currency == 'CAD':
            self.cad_short_end_filter()
        #     List<BondItem> removedDueToCNShortEndFilter = ApplyFilter(CreateChinaShortEndFilter(), "China Short-end Filter", applyMinimumBonds:false);
        if self.currency == 'BRL':
            self.brl_short_end_filter()
        if self.currency == 'IDR':
            self.idr_short_end_filter()
        if self.currency == 'CLP':
            self.clp_short_end_filter()
        if self.currency == 'PHP':
            self.php_bond_price_filter()
        self.min_distance_filter()
        self.reuters_comparisson_filter()

    def maturity_filter(self):
        remove_df = self.bonds_df[self.bonds_df['TradeDate'] == self.bonds_df['MatDate']]
        self._remove_bonds(remove_df)

    def compute_yield_filter(self):
        if self.bonds_df.empty:
            return

        if len(self.bonds_df) == 1:
            return

        reuters_yld = 'MatCorpYld' if 'MatCorpYld' in self.bonds_df.columns else 'MatStdYld'

        remove_df = self.bonds_df[(np.abs(self.bonds_df[reuters_yld])) > (np.abs(self.bonds_df['ytm']) * 2.0)]
        self._remove_bonds(remove_df)

        remove_df = self.bonds_df[(np.abs(self.bonds_df[reuters_yld])) < (np.abs(self.bonds_df['ytm']) * 0.5)]
        self._remove_bonds(remove_df)

    def yield_outlier_filter(self):
        if self.bonds_df.empty:
            return
        #CreateWildYieldFilter
        self.bonds_df = self.bonds_df[ (self.bonds_df['ytm']/100 < 200.0 * 0.01) & (self.bonds_df['ytm']/100 > -25.0 * 0.01)]

    def short_end_filter(self):
        threshold = 3
        tdf = self.bonds_df[self.bonds_df['tenor'] < 1.0]
        if tdf.empty:
            return
        avg_yield = tdf['ytm'].mean()
        std_dev = np.sqrt( (np.power(tdf['ytm'] - avg_yield, 2)).mean())
        self._log('Average Yield (First 10 years): {}'.format(avg_yield))
        self._log('StdDev Yield (First 10 years): {}'.format(std_dev))
        tdf = tdf[ tdf['tenor'] > 5]
        remove_df = tdf[np.abs(tdf['ytm'] - avg_yield) >  threshold * std_dev]
        self._remove_bonds(remove_df)

    def cad_short_end_filter(self):
        bonds = self.bonds_df[self.bonds_df['tenor'] < 0.05]
        self._remove_bonds(bonds)

    def brl_short_end_filter(self):
        bonds = self.bonds_df[self.bonds_df['tenor'] < 0.5]
        self._remove_bonds(bonds)

    def idr_short_end_filter(self):
        bonds = self.bonds_df[self.bonds_df['tenor'] < 0.5]
        self._remove_bonds(bonds)

    def clp_short_end_filter(self):
        bonds = self.bonds_df[self.bonds_df['tenor'] < 0.25]
        self._remove_bonds(bonds)

    def php_bond_price_filter(self):
        bonds = self.bonds_df[self.bonds_df['Price'] < 20.0]
        self._remove_bonds(bonds)

    def reuters_comparisson_filter(self):
        if self.bonds_df.empty:
            return
        dbconn = db.RiggsZeroCurve(self.country, self.tradeDate, self.database)
        df_riggs = dbconn.extract_from_db()
        if df_riggs.empty:
            return
        f = interp1d(df_riggs['term'], df_riggs['spot'], bounds_error=False)
        tdf = self.bonds_df[self.bonds_df['tenor'] < 1.0]
        tdf['zero'] = f(tdf['tenor']) 
        if tdf.shape[0] >= 4:
            tdf['yield_diffs'] = np.abs(tdf['MatStdYld'] - tdf['zero'])
            remove_df = pd.DataFrame(tdf.sort(columns=['yield_diffs'], ascending=False).irow(0)).T
            self._remove_bonds(remove_df)

    def bill_matrurity_filter(self):
        # only get bonds where bond maturity is greater than bill maturity
        self.bills_df = self.bonds_df[(self.bonds_df['FrstCpnRate'] == 0.0) 
                         & (self.bonds_df['Price'] > 0.5)]
                         #& (self.unfiltered_df['DebtIssTypeCode'] != 'PRIN' ) # Remove Prins
                         #& (self.unfiltered_df['DebtIssTypeCode'] != 'INT' )] # Remove Strips

        if self.bills_df.empty:
            self.max_bill_maturity = datetime.datetime(1990, 1, 1)
        else:
            self.max_bill_maturity = max(self.bills_df['MatDate'])
        # self.bonds_df = self.bonds_df[(self.bonds_df['FrstCpnRate'] > 0.0) & (self.bonds_df['MatDate'] < self.max_bill_maturity)]
        # self.bonds_df = pd.concat([self.bills_df, self.bonds_df])

    def _bond_overides(self, df, msg):
        if not df.empty:
            for instr_code in df['InstrCode']:
                if np.any(self.bonds_df['InstrCode'] == instr_code):
                    self._log(msg + str(instr_code))
                    self.bonds_df = self.bonds_df[self.bonds_df['InstrCode'] != instr_code]

    def negative_price_filter(self):
        itemId = 1
        conn = db.BondOverides(self.curveId, self.tradeDate, itemId, self.database)
        df = conn.extract_from_db()
        self._bond_overides(df, 'Removed due to negative price filter: ')

    def banned_instruments(self):
        itemId = 14
        conn = db.BondOverides(self.curveId, self.tradeDate, itemId, self.database)
        df = conn.extract_from_db()
        self._bond_overides(df, 'Removed due to banned instruments: ')

    def ban_entire_trade(self):
        itemId = 50
        conn = db.BondOverides(self.curveId, self.tradeDate, itemId, self.database)
        df = conn.extract_from_db()
        if len(df) == 0:
            return
        self._bond_overides(self.bonds_df, 'All bonds removed due to banned trade date:')

    def yield_overrides(self):
        itemId = 3
        conn = db.BondOverides(self.curveId, self.tradeDate, itemId, self.database)
        df = conn.extract_from_db()
        self._bond_overides(df, 'Removed due to yield overides: ')

    def tradeDate_overrides(self):
        itemId = 13
        conn = db.BondOverides(self.curveId, self.tradeDate, itemId, self.database)
        df = conn.extract_from_db()
        instr_codes = self.bonds_df['InstrCode']
        for instr_code in df['InstrCode']:
            if instr_code not in instr_codes:
                self._force_add_bond(instr_code)

    def min_bond_rule(self):
        g7 = "CA|US|NO|SE|AU|JP|CH|GB|EP|MT|SK|EP|ES|FI|IE|DE|BE|AT|GR|NL|PT"
        if (self.bonds_df.shape[0] < 4 and self.country in g7):
            self._log("min_bond_rule: Less than 4 bonds remaining")

    def _date_to_str(self, s):
        if type(s) == pd.tslib.Timestamp:
            s = s.strftime('%Y-%m-%d')
        if type(s) == datetime.datetime:
            s = datetime.datetime.strftime(s, '%Y-%m-%d')
        return s

    def _force_add_bond(self, instr_code):
        s = self.unfiltered_df[self.unfiltered_df.InstrCode == instr_code]
        if s.empty:
            s = self._find_missing_bond(instr_code)
        else:
            ytm = self.compute_ytm_row(s)
            s['ytm'] = ytm
        if s.empty:
            return
        self._log('Tradedate override(Force adding bond): {}'.format(instr_code))
        self._log(s)
        s['row_num'] = self.bonds_df.shape[0] + 1
        s['lub'] = None
        s['lud'] = None
        s['Action'] = None
        #s = s[self.bonds_df.columns]
        # print s.columns
        # print self.bonds_df.columns
        self.bonds_df = self.bonds_df.append(s, verify_integrity=False)
        self.bonds_df = self.bonds_df.sort(columns='tenor')

    def _remove_bonds(self, df):
        if df.empty:
            return
        name = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
        self._log('{}: removed'.format(name))
        self._log(df)
        for i, x in df.iterrows():
            drop_instr = x['InstrCode']
            self.bonds_df = self.bonds_df[ self.bonds_df.InstrCode != drop_instr ]

    def _find_missing_bond(self, instr_code):
        df = db.DerivedDataExtractSoverignCurveByInstrCodes(self.tradeDate, str(instr_code), self.database).extract_from_db()
        if df.empty:
            return pd.DataFrame()
        #df.rename(columns={'IssName':'InstrDesc'}, inplace=True)
        if ('FrstCpnDate' in df.columns) and ('FrstCpnDate' not in self.bonds_df.columns):
            del df['FrstCpnDate']
            del df['LastCpnDate']
        df['tenor'] = [x/np.timedelta64(1, 'D')/365.25 for x in df['MatDate'] - df['TradeDate']]
        ytm = self.compute_ytm_row(df.irow(0))
        df['ytm'] = ytm
        df.drop_duplicates('InstrCode', take_last=True, inplace=True)
        return df 

    def compute_ytm_row(self, row):
        #print type(row)
	if isinstance(row, pd.DataFrame):
	    row = row.iloc[0,:]
	issueDate, maturityDate, valuationDate, marketPrice, settlement_adj, coupon, MarketStandardYield = bond_utils.process_reuters_market_data_row(row)
        try:
            first_cpn_dt = row.FrstCpnDate
            last_cpn_dt = row.LastCpnDate
        except:
            first_cpn_dt = None
            last_cpn_dt = None
        # bondPricer = bond.create_pricer(issueDate, maturityDate, coupon, valuationDate, 2, settlement_adj, first_cpn_dt, last_cpn_dt)
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, 2, settlement_adj, first_cpn_dt, last_cpn_dt)
        #accured_intersest = bondPricer.compute_accrued_interest()

        ytm = -1
        try:
            ytm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=True)*100.0
        except RuntimeError as e:
            pass
        except IndexError as e:
            pass
        except TypeError as e:
            pass
        return ytm

    def compute_ytm(self):
        if self.bonds_df.empty:
            return

        self.bonds_df['ytm'] = self.bonds_df.apply(self.compute_ytm_row, axis=1)
        # self.test_df = self.bonds_df

    def min_distance_filter(self, min_dist=0.08):
        remove_codes = []
        bonds = self.bonds_df.sort(columns='tenor')
        for i, x in bonds.iterrows():
            cur_bond = x
            if 'prev_bond' not in locals():
                prev_bond = cur_bond
            distance = cur_bond['tenor'] - prev_bond['tenor']
            if cur_bond['InstrCode'] == prev_bond['InstrCode']:
                continue
            if distance < min_dist:
                remove_codes.append(cur_bond['InstrCode'])
            else:
                prev_bond = cur_bond
        if len(remove_codes) == 0:
            return
        else:
            for instr_code in remove_codes:
                self._log('min_distance_filter: removed ' + str(instr_code))
                self.bonds_df = self.bonds_df[self.bonds_df.InstrCode != instr_code ]

class BondFilterDbWriter(object):

    def __init__(self, tradeDate, curveId, df, enviroment=None, config=None, production_table=False):
        self.tradeDate = tradeDate
        self.curveId = curveId
        self.df = df
        self.enviroment = enviroment
        self.config = config
        self.production_table = production_table

    def write_to_dbs(self):
        self.delete_sql = self.delete_query()
        try:
            db.MSSQL.execute_commit(self.delete_sql, self.enviroment, self.config)
        except Exception as e:
            print "Exception thrown commit:", e, self.delete_sql
            return 1
        user_name = getpass.getuser()
        dt = str(datetime.datetime.now())[0:-3] 
        queries = []
        for i, x in self.df.iterrows():
            queries.append( self.create_insert_sql({'CurveId':self.curveId, 
                                               'TradeDate':self.tradeDate, 
                                               'InstrCode':x['InstrCode'], 
                                               'ItemId':1, 
                                               'ItemValue':x['Price'], 
                                               'Lud': dt, 
                                               'Lub': user_name}))

            queries.append( self.create_insert_sql({'CurveId':self.curveId,
                                               'TradeDate':self.tradeDate,
                                               'InstrCode':x['InstrCode'],
                                               'ItemId':2,
                                               'ItemValue':'-100' if x['MatStdYld'] is None or np.isnan(x['MatStdYld'])
                                               else x['MatStdYld'],
                                               'Lud':dt,
                                               'Lub': user_name}))

            queries.append( self.create_insert_sql({'CurveId':self.curveId,
                                               'TradeDate':self.tradeDate,
                                               'InstrCode':x['InstrCode'],
                                               'ItemId':4,
                                               'ItemValue':x['ytm']/100.0,
                                               'Lud':dt,
                                               'Lub': user_name}))
        self.sql_inserts = [query + '\n' for query in queries]
        try:
            db.MSSQL.execute_commit(''.join(self.sql_inserts), self.enviroment, self.config)        
        except Exception as e:
            print "Exception thrown commit:", e.message, self.sql_inserts, str(e)
            return 1
        return 0

    def delete_query(self):
        if self.production_table:
            sqlstatement = string.Template("""
            DELETE from marketdata.dbo.DerivCurveFilteredBond where CurveId = '$CurveId'  and TradeDate = '$TradeDate' and (ItemId = 1 or ItemId = 2 or ItemId = 4) 
            """).substitute({'CurveId':self.curveId, 'TradeDate': self.tradeDate})
        else:
            sqlstatement = string.Template("""
            DELETE from marketdata.dbo.DerivCurveFilteredBond_Research where CurveId = '$CurveId'  and TradeDate = '$TradeDate' and (ItemId = 1 or ItemId = 2 or ItemId = 4)
            """).substitute({'CurveId': self.curveId, 'TradeDate': self.tradeDate})
        return sqlstatement

    def create_insert_sql(self, d):
        if self.production_table:
            sqlstatement = string.Template("""INSERT INTO marketdata.dbo.DerivCurveFilteredBond VALUES('$CurveId', '$TradeDate', '$InstrCode', '$ItemId', '$ItemValue', '$Lud', '$Lub')""").substitute(d)
        else:
            sqlstatement = string.Template("""INSERT INTO marketdata.dbo.DerivCurveFilteredBond_Research VALUES('$CurveId', '$TradeDate', '$InstrCode', '$ItemId', '$ItemValue', '$Lud', '$Lub')""").substitute(d)        
        return sqlstatement
































