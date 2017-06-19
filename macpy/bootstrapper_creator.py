
import macpy.bond_utils as bond_utils
import macpy.utils.database as db
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
from dateutil.relativedelta import relativedelta


from macpy.forward_rate import ForwardRate

from macpy.curve_wrappers import adjusted_yield_curve
import macpy.bei_curve as bei_curve

class BootStrapper:

    def __init__(self, startDate, endDate, currency, conventions, enviroment=None, config=None, dataframe=None):
        self.startDate=startDate
        self.endDate = endDate
        self.currency=currency
        self.enviroment = enviroment
        self.config = config
        self.dataframe = dataframe
        self.conventions = conventions
        self.timeInYearExtension = [40.00,50.000,60.00]
        if not isinstance(self.dataframe, pd.DataFrame):
            benchmarkGovernmentBonds = db.DerivedFilteredBootstrap(self.currency, self.startDate, self.endDate, database=self.enviroment, config=self.config)
            self.dataframe = benchmarkGovernmentBonds.extract_from_db()

    def createYieldCurve(self):
        # benchmarkGovernmentBonds = db.DerivedFilteredBootstrap(self.currency, self.startDate, self.endDate, self.enviroment, self.config)
        # dataframe = benchmarkGovernmentBonds.extract_from_db()

        holidays = None
        if self.currency == 'ISK':
            asset_id = self.dataframe['EJVAssetID'].values[0]
            holidays = db.EJVHolidays(asset_id, 'DEV').extract_from_db()

        timeInYearsInterp = self.conventions['timeInYearsInterp']
        grouped = self.dataframe.groupby('TradeDate')

        for dataframe in grouped:
            forwardRateList=[0]
            maturityInYearsList=[0]
            maturityDateList=[pd.to_datetime(dataframe[0]).strftime('%Y-%m-%d')]
            yieldList=[0] # start yields at zero - helps stabilize bootstrap and interpolations:w

            dataframe_NewIssDate = dataframe[1].apply(lambda row: self.adjusted_issue_date(row),axis=1)
            CurveShortName = str(dataframe_NewIssDate['CurveShortName'].values[0])

            dirtyPriceFlag = False
            if CurveShortName in self.conventions['dirtyPriceFlag']:
                dirtyPriceFlag = True

            use_clean_price_for_short_end_bond = False
            if CurveShortName in self.conventions['use_clean_price_for_short_end_bond']:
                use_clean_price_for_short_end_bond = True

            adjust_Market_Price=False

            if 'MarketPriceAdjustList' not in self.conventions:
                raise Exception('MarketPriceAdjustList key is not in conventions dict')

            if self.currency in self.conventions['MarketPriceAdjustList']:
                adjust_Market_Price=True

            args=[forwardRateList, maturityDateList, maturityInYearsList, yieldList, dirtyPriceFlag,
                  use_clean_price_for_short_end_bond, adjust_Market_Price, holidays]
            # Call to func_create_yield_curve computes Recursive Bootstrap

            #dataframe_NewIssDate['AxiomaYTMPrecomputed'] = self.apply(lambda row: self.(row, args), axis=1)
            dataframe_NewIssDate.apply(lambda row: self.func_create_yield_curve(row, args), axis=1)

            tradeDate = dataframe_NewIssDate['TradeDate'].values[0]
            dataframe_NewIssDate['forward_rate'] = forwardRateList[1:]
            #dataframe_NewIssDate['zero_rate'] = yieldList[1:]
            self.upload_qa_data(pd.to_datetime(tradeDate).strftime('%Y-%m-%d'), CurveShortName, dataframe_NewIssDate, self.enviroment, self.config)
            #yieldList.insert(0, yieldList[0])
            maturityInYearsList.extend(self.timeInYearExtension)
            yieldList.extend([yieldList[-1], yieldList[-1], yieldList[-1]])
            LastMaturityInYears = maturityInYearsList[-4]

            if self.currency in self.conventions['frontEndAdjustList']:
                yieldList[0] = yieldList[1]

            customYieldCurve = interp1d(maturityInYearsList, yieldList, kind='linear')
            #print yieldList
            customYieldRates = customYieldCurve(timeInYearsInterp)

            if self.currency in self.conventions['linearInterpList']:
                if CurveShortName in self.conventions['EurCubicInterpList']:
                    interpMethodPostBootstrap = 'cubic'
                else:
                    interpMethodPostBootstrap = 'linear'
            else:
                interpMethodPostBootstrap = 'cubic'

            yieldCurve = interp1d(timeInYearsInterp, customYieldRates, kind=interpMethodPostBootstrap)

            if self.currency in self.conventions['FlatExtrapList']:
                ExtrapFlag = 'True'
            else:
                ExtrapFlag = 'False'

            adjustedCurve = adjusted_yield_curve(timeInYearsInterp, yieldCurve, LastMaturityInYears, ExtrapFlag)
            return adjustedCurve

    def func_create_yield_curve(self, row, args):
        issueDate, maturityDate, valuationDate, marketPrice, settlement_adj, coupon, MarketStandardYield  = bond_utils.process_reuters_market_data_row(row)
        freq=2.0 #Coupon Frequency are set equal to 2.0 manually #
        # freq = row['CompFreqCode']
        first_cpn_dt = row['FrstCpnDate']
        last_cpn_dt = row['LastCpnDate']
        notional = 100.0

        if str(row['DebtISOCurCode']) in ['INR']:
            PriceInclAccrIntFlg = 'y'
        else:
            PriceInclAccrIntFlg = str(row['PriceInclAccrIntFlg'])

        forwardRateList=args[0]
        maturityDateList=args[1]
        maturityInYearsList=args[2]
        yieldList=args[3]
        dirtyPriceFlag = args[4]
        use_clean_price_for_first_bond = args[5]
        adjust_market_price = args[6]
        holidays = args[7]


        forwardRateGenerator = ForwardRate(forwardRateList,
                                           maturityInYearsList,
                                           maturityDateList,
                                           yieldList,
                                           issueDate,
                                           maturityDate,
                                           marketPrice,
                                           valuationDate,
                                           coupon,
                                           MarketStandardYield,
                                           freq,
                                           settlement_adj,
                                           first_cpn_dt,
                                           last_cpn_dt,
                                           use_clean_price_for_first_bond,
                                           adjust_market_price,
                                           PriceInclAccrIntFlg,
                                           notional,
                                           holidays)

        forwardRateGenerator.compute_forward_rate(dirtyPriceFlag)

    def adjusted_issue_date(self,row):
        if not pd.isnull(row['InitAccDate']):
            diff = (pd.to_datetime(row['MatDate']) - pd.to_datetime(row['InitAccDate'])).days/365.25
            diff_decimal = diff - int(diff)
            newDate_one = pd.to_datetime(row['MatDate'])-relativedelta(years=int(diff), months=6)
            newDate_two = pd.to_datetime(row['MatDate'])-relativedelta(years=int(diff)+1)
            newDate_three = pd.to_datetime(row['MatDate'])-relativedelta(years=int(diff))
            if diff_decimal<=0.5:
                issDate_new = pd.to_datetime(row['MatDate'])-relativedelta(years=int(diff), months=6) if (row['InitAccDate']!=newDate_one and row['InitAccDate']!=newDate_three) else row['InitAccDate']
            else:
                issDate_new = pd.to_datetime(row['MatDate'])-relativedelta(years=int(diff)+1) if (row['InitAccDate']!=newDate_one and row['InitAccDate']!=newDate_two) else row['InitAccDate']

            row['InitAccDate']=issDate_new

        return row

    def upload_qa_data(self, tradeDate, curveName, df, enviroment, config):
        print 'tradeDate=' + tradeDate
        print 'CurveShortName=' + curveName
        print 'env=' + str(enviroment)
        print 'config=' + str(config)
        print 'Input bonds'
        print df
        db.BondFilterDbWriter(tradeDate, curveName, df, enviroment, config, production_table=True).write_to_dbs()


class Bootstrapper_BEI(BootStrapper):

    def __init__(self, startDate, endDate, currency, conventions, enviroment=None, config=None, dataframe=None):
        BootStrapper.__init__(self, startDate, endDate, currency, conventions, enviroment, config, dataframe)
        if dataframe is None:
            benchmark_bond_query = bei_curve.BEI(country_code=currency[0:2], start_date = self.startDate, end_date = self.endDate, environment = self.enviroment)
            self.dataframe = benchmark_bond_query.extract_from_db()[0]

    def func_create_yield_curve(self, row, args):
        issueDate, maturityDate, valuationDate, marketPrice, settlement_adj, coupon, MarketStandardYield  = bond_utils.process_reuters_market_data_row(row)
        freq=2.0 #Coupon Frequency are set equal to 2.0 manually #
        # freq = row['CompFreqCode']
        first_cpn_dt = row['FrstCpnDate']
        last_cpn_dt = row['LastCpnDate']
        notional = 100.0

        if self.currency in self.conventions['IndexFactorAdjustList']:
            index_factor = 1.0
        else:
            index_factor = row['index_factor']

        notional = 100.0*index_factor
        #marketPrice = marketPrice/row['index_factor']
        PriceInclAccrIntFlg = str(row['PriceInclAccrIntFlg'])

        forwardRateList=args[0]
        maturityDateList=args[1]
        maturityInYearsList=args[2]
        yieldList=args[3]
        dirtyPriceFlag = args[4]
        use_clean_price_for_first_bond = args[5]
        adjust_market_price = args[6]
        holidays = args[7]


        forwardRateGenerator = ForwardRate(forwardRateList,
                                           maturityInYearsList,
                                           maturityDateList,
                                           yieldList,
                                           issueDate,
                                           maturityDate,
                                           marketPrice,
                                           valuationDate,
                                           coupon,
                                           MarketStandardYield,
                                           freq,
                                           settlement_adj,
                                           first_cpn_dt,
                                           last_cpn_dt,
                                           use_clean_price_for_first_bond,
                                           adjust_market_price,
                                           PriceInclAccrIntFlg,
                                           notional,
                                           holidays)

        forwardRateGenerator.compute_forward_rate(dirtyPriceFlag)

