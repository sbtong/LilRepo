import string
import datetime
import macpy.utils.database as db
from macpy.utils.database import CurveNodeQuoteFinal
import pandas as pd
import time
#import matplotlib.pyplot as plt
import macpy.bondPortfolio as bondPortfolio
import numpy as np
from scipy.interpolate import interp1d
from irr import  compute_irr, compute_macaulay_duration, compute_npv
from optparse import OptionParser
from dateflow import DateFlow
import getpass
import traceback
import macpy.hazardrates as h

from curve_functions import convert_gvt_curve, convert_swp_sprd_curve

def get_current_time():
    return str(datetime.datetime.now())[0:-3] 

def split_list(mylist, step=30):
    i=0
    splitted_list=[]
    while i<len(mylist):
        splitted_list.append(mylist[i:i+step])
        i=i+step
                       
    return splitted_list


class BondPricer:

    @classmethod
    def create_from_row(cls, row):
        """
        Function takes a row from dataframe, depend on these fields:
        """
        trade_date = row['TradeDate'].strftime('%Y-%m-%d')
        coupon = row['FrstCpnRate'] / 100.0
        issue_date = row['InitAccDate'].strftime('%Y-%m-%d')
        maturity_date = row['MatDate'].strftime('%Y-%m-%d')
        pricer = BondPricer(issue_date, maturity_date, coupon, trade_date)
        return pricer



    def __init__(self, issueDate, maturityDate, coupon, valuationDate, freq=2.0, settlement_adj=0.0, first_cpn_dt=None, last_cpn_dt=None, notional = 100.0, holidays=None):
        self.issueDate = issueDate if issueDate is not None else valuationDate
        self.maturityDate = maturityDate
        self.coupon = np.amax([float(coupon),0.0000000001])
        self.valuationDate = valuationDate
        self.first_cpn_dt = first_cpn_dt
        self.last_cpn_dt = last_cpn_dt
        self.notional = notional
        self.couponsPerYear = freq
        self.date_flow = DateFlow(self.coupon/self.couponsPerYear, self.issueDate, self.maturityDate, self.valuationDate, self.couponsPerYear, first_cpn_dt=self.first_cpn_dt, last_cpn_dt=self.last_cpn_dt, notionalAmount = self.notional)
        self.cashflowDates, self.cashflows, self.timeToCashflowInYears = self.date_flow.get_dateflow()
        self.discountFactors = None
        self.discountedCashflows = None
        self.settlement_adj = settlement_adj
        self.holidays = holidays

    def coupon_payment_steps(self, freq):
        if freq == 1.0:
            step = '12m'
        elif freq == 2.0:
            step = '6m'
        elif freq == 4.0:
            step = '3m'
        elif freq == 12.0:
            step = '1m'
        else:
            step = '6m'
        return step

    def compute_accrued_interest(self):
        if self.valuationDate<self.issueDate:
            accruedInterest=0.0
        else:
            timeToFirstCoupon = self.timeToCashflowInYears.values[1]
            accruedInterest = self.adjust_accrued_interest(self.couponsPerYear, timeToFirstCoupon, self.coupon)
        return accruedInterest

    def adjust_accrued_interest(self, couponsPerYear, timeToFirstCoupon, coupon):
        SettlementDay = self.get_settlement_date()
        if isinstance(self.holidays, pd.DataFrame):
            holiday_adj = 0
            holidays = self.holidays[(self.holidays['holiday_dt'] > self.ValuationDay) &  (self.holidays['holiday_dt'] <= SettlementDay)]['holiday_dt']
            for holiday in holidays.values:
                holiday = datetime.datetime.utcfromtimestamp(holiday.tolist()/1e9)
                if holiday.isoweekday() < 6:
                    holiday_adj += 1
            if holiday_adj > 0:
                SettlementAdj = datetime.timedelta(days=holiday_adj)
                SettlementDay = SettlementDay + SettlementAdj
                self.settlement_adj -= holiday_adj
        AdjustedCouponPerYear = 1.0 if timeToFirstCoupon > 0.8 else self.couponsPerYear
        accruedInterest = (1.0/AdjustedCouponPerYear-(timeToFirstCoupon - (self.settlement_adj)/360.0))*self.coupon*100.0
        return accruedInterest

    def get_settlement_date(self, SettlementDay=None):
        weekend_adj = 0
        if not SettlementDay:
            self.ValuationDay = datetime.datetime.strptime(self.valuationDate, '%Y-%m-%d')
            SettlementAdj = datetime.timedelta(days=self.settlement_adj)
            SettlementDay = self.ValuationDay + SettlementAdj
        if SettlementDay.isoweekday() == 6:
            weekend_adj = 2
        if SettlementDay.isoweekday() == 7:
            weekend_adj = 1
        self.settlement_adj += weekend_adj
        if weekend_adj > 0:
            SettlementDay = SettlementDay + datetime.timedelta(days=weekend_adj)
            self.settlement_adj = self.settlement_adj - weekend_adj
        return SettlementDay

    def generate_discounted_flows(self, yieldCurve=None):
        self.discountFactors =  [yieldCurve.compute_discount_factor(float(x)) for x in self.timeToCashflowInYears]  if (yieldCurve is not None) else [1 for x in self.timeToCashflowInYears]
        self.discountedCashflows = np.array([cashflow*discountFactor for (discountFactor, cashflow) in zip(self.discountFactors, self.cashflows)])

    def generate_discount_factors(self, yieldCurve=None):
        self.discountFactors =  [yieldCurve.compute_discount_factor(float(x)) for x in self.timeToCashflowInYears]  if (yieldCurve is not None) else [1 for x in self.timeToCashflowInYears]

    def get_diagnostics(self, yieldCurve=None, marketPrice=None):
        self.generate_discounted_flows(yieldCurve)
        columnsDictionary = {}
        columnsDictionary["TimeToCashflowInYears"] = list(self.timeToCashflowInYears)
        columnsDictionary["Cashflows"] = list(self.cashflows)
        columnsDictionary["CashflowDates"] = list(self.cashflowDates)
        if(self.discountedCashflows is not None): columnsDictionary["DiscountedCashflows"] = list(self.discountedCashflows)
        if(self.discountFactors is not None): columnsDictionary["DiscountFactors"] = list(self.discountFactors)

        diagnostics = {}
        diagnostics["ValuationDate"] = self.valuationDate
        diagnostics["Coupon"] = self.coupon
        diagnostics["Notional"] = self.notional
        diagnostics["AccruedInterest"] = self.compute_accrued_interest()

        if(marketPrice is not None): diagnostics["YieldToMaturity"] = self.compute_yield_to_maturity(marketPrice)


        df = pd.DataFrame(data=columnsDictionary)
        if(yieldCurve is not None):
            diagnostics["YieldCurve"] = yieldCurve.get_diagnostics()
            df["ZeroCouponYield"] = df.apply(lambda row: yieldCurve(float(row['TimeToCashflowInYears'])), axis=1)

        if(yieldCurve is not None and marketPrice is not None):
            diagnostics["Spread"] = self.compute_spread(yieldCurve, marketPrice)
            diagnostics["SpreadDuration"] = self.compute_spread_duration_numerical(yieldCurve, marketPrice)

        diagnostics["CashflowDataFrame"] = df

        return diagnostics


    def compute_clean_price(self, yieldCurve, spreadValue=0.0):
        return self.compute_dirty_price(yieldCurve, spreadValue) - self.compute_accrued_interest()


    def compute_dirty_price(self, yieldCurve, spreadValue=0.0):
        self.generate_discounted_flows(yieldCurve)
        dirty_price = compute_npv(self.timeToCashflowInYears, self.discountedCashflows, spreadValue)
        # dirty_price = float(self.discountedTimeFlow.npv_value(spreadValue)) slow method
        return dirty_price

    def compute_yield_to_maturity(self, marketPrice, priceIsClean=True, max_iter=100):
        if priceIsClean:
            accruedInterest = self.compute_accrued_interest()
        else:
            accruedInterest = 0.0
        dirty_price = marketPrice + accruedInterest

        spread_value = compute_irr(self.timeToCashflowInYears, self.cashflows, market_price=dirty_price, max_iter=max_iter)
        # yieldToMaturity = float(self.timeFlow.npv_spread(dirtyPrice)) slow method
        return spread_value

    def compute_HazardUkq(self,yieldCurve,RecoveryRate,marketPrice,priceIsClean=True,alpha=0.03):
        if priceIsClean:
            accruedInterest = self.compute_accrued_interest()
        else:
            accruedInterest = 0.0
        dirty_price = marketPrice + accruedInterest

        self.discountFactors =  [yieldCurve.compute_discount_factor(float(x)) for x in self.timeToCashflowInYears]  if (yieldCurve is not None) else [1 for x in self.timeToCashflowInYears]
        HazardUkq, Vq = h.HazardUk(self.coupon,self.couponsPerYear,self.discountFactors,self.timeToCashflowInYears,RecoveryRate,alpha,dirty_price)
        return HazardUkq, Vq


    def compute_spread(self, yieldCurve, cleanMarketPrice):
        self.generate_discounted_flows(yieldCurve)
        dirty_price = cleanMarketPrice + self.compute_accrued_interest()
        #spreadValue= float(self.discountedTimeFlow.npv_spread(dirtyPrice)) #slow method
        spread_value = compute_irr(self.timeToCashflowInYears, self.discountedCashflows, market_price=dirty_price)
        return spread_value

    def compute_spread_duration_analytical(self, yieldCurve, cleanMarketPrice):
        self.generate_discounted_flows(yieldCurve)
        spreadValue = self.compute_spread(yieldCurve, cleanMarketPrice)
        #spreadDuration = self.discountedTimeFlow.modified_duration(spreadValue) slow method
        spreadDuration = compute_macaulay_duration(self.timeToCashflowInYears, self.discountedCashflows, spreadValue)
        return spreadDuration

    def compute_spread_duration_numerical(self, yieldCurve, cleanMarketPrice, shiftSize=0.001):
        sprvalue = self.compute_spread(yieldCurve, cleanMarketPrice)
        spreadValueUpShift = sprvalue + shiftSize
        spreadValueDownShift = sprvalue - shiftSize
        priceAtUpShift =  self.compute_clean_price(yieldCurve, spreadValueUpShift)
        priceAtDownShift = self.compute_clean_price(yieldCurve, spreadValueDownShift)
        spreadDuration = (priceAtDownShift - priceAtUpShift) / (2*cleanMarketPrice*shiftSize)
        return spreadDuration
            
    def compute_spread_over_ytm(self, yieldCurve, yieldToMaturity):
        timeToMaturityInYears = self.timeToCashflowInYears.values[-1]
        spread = yieldToMaturity - yieldCurve(timeToMaturityInYears)
        return spread

    def __str__(self):
        return 'BondPricer(IssueDate:%s, MaturityDate:%s, Coupon:%s)' % (
            self.issueDate, self.maturityDate, self.coupon)
            
    def __repr__(self):
        return self.__str__()

class BondStatDatabaseWriter:
    def __init__(self, startDate, endDate, currency, discountCurveName, spreadCurveName, issuerId, instrumentCode, rating, sector, filename, database):
        self.startDate = startDate
        self.endDate= endDate
        self.currency = currency
        self.discountCurveName = discountCurveName
        self.spreadCurveName = spreadCurveName
        self.issuerId = issuerId
        self.instrumentCode = instrumentCode
        self.rating = rating
        self.sector = sector
        self.filename = filename
        self.database = database

    def retry_on_failed_database_connection(self, command, numberOfRetries=3):
        retryCount = 0
        result = None
        while retryCount < numberOfRetries:
            try:
                result = command() # will make long running database calls
                return result
            except Exception as e:
                print "Failure most likely cause by database connection disruption ", "\nErrorMessage: ",  e.message, \
                " Corporate Stored Procedure failed.", \
                "StartDate:",  self.startDate,\
                "EndDate:", self.endDate
                time.sleep(10) # sleep for 5 seconds to let network recover
                retryCount += 1
            continue
        return result

    def retry_on_failed_execute_commit(self, sql_statement, numberOfRetries=3):
        retryCount = 0
        result = None
        while retryCount < numberOfRetries:
            try:
                print 'sql commit'
                db.MSSQL.execute_commit(sql_statement)
                break
            except Exception as e:
                print 'Exception on execute_commit(sql_statement). Try= %d of %d' % (retryCount+1, numberOfRetries)
                print e.message
                #print sql_statement
                time.sleep(5) # sleep to let network recover
                retryCount += 1
            continue




    def write_to_db(self):

        corporate_bond_query = db.DerivedDataCorpBondPriceForCcyResearch(valuationDate=self.startDate,
                                                                         currency = self.currency,
                                                                         endTradeDate=self.endDate,
                                                                         issuerId=self.issuerId,
                                                                         instrumentCode=self.instrumentCode,
                                                                         rating=self.rating,
                                                                         sector=self.sector,
                                                                         suffix='',
                                                                         database=self.database,
                                                                         config=self.filename)
        print corporate_bond_query.sqlstatement



        bondDataFrame =  corporate_bond_query.extract_from_db()
        grouped = bondDataFrame.groupby('TradeDate')

        swapCurves = dict()
        govtCurves = dict()
        def retrieve_yield_curves():
            def return_none_on_failure(command, failureMessage):
                result = None
                try:
                    result = command()
                except Exception as e:
                    print "StartDate:", self.startDate, "EndDate:", self.endDate, "failureMessage", " -> ", failureMessage
                return result
            for tradeDate, group in grouped:
                govtCurves[tradeDate] = return_none_on_failure(lambda:
                                                               create_govt_zero_curve(tradeDate, self.currency, self.database),
                                                               "{} not found for govt curve".format(tradeDate))
                swapCurves[tradeDate] = return_none_on_failure(lambda:
                                                               create_swap_zero_curve(tradeDate, self.currency, self.database),
                                                               "{} not found for swap curve".format(tradeDate))
        retrieve_yield_curves()

        bondDataFrameAvailCurves = bondDataFrame[
            bondDataFrame.TradeDate.apply(lambda x: True if govtCurves.has_key(x) and swapCurves.has_key(x) else False)
        ]

        #Generate SQL for inserting statistics
        insertSqlStatisticAccumulator =[]
        bondDataFrameAvailCurves.apply(
            lambda row: self.compute_stats(row, insertSqlStatisticAccumulator, swapCurves, govtCurves),
            axis=1
        )
        sql_statement=';\n'.join(insertSqlStatisticAccumulator)

        return sql_statement

    def compute_stats(self, row, sqlList, swapCurves, govtCurves):
        valuationDate = pd.to_datetime(row.TradeDate).strftime('%Y-%m-%d')
        issueDate=pd.to_datetime(row.InitAccDate).strftime('%Y-%m-%d')
        maturityDate=pd.to_datetime(row.MatDate).strftime('%Y-%m-%d')
        coupon= row.FrstCpnRate/100
        marketPrice=row.Prc
        instrCode = int(row.InstrCode)
        issuerId=row.UltParIsrId
        rateScore= str(row.CompositeRatingEnum)
        currency = str(row.DebtISOCurCode)
        sector= str(row.GicsLevel1)
        corpMatYield= row.MatCorpYld
        couponfreq = 2.0
        issuerIdInt = 0
        try:
            FrstCpnDate = row.FrstCpnDate
        except:
            FrstCpnDate = None

        try:
            LastCpnDate = row.LastCpnDate
        except:
            LastCpnDate = None

        if FrstCpnDate is not None and type(FrstCpnDate) != pd.tslib.NaTType:
            frstCpnDate = pd.to_datetime(FrstCpnDate).strftime('%Y-%m-%d')
            #print 'FrstCpnDate', frstCpnDate
        else:
            frstCpnDate = None

        if LastCpnDate is not None and type(LastCpnDate) != pd.tslib.NaTType:
            lastCpnDate = pd.to_datetime(LastCpnDate).strftime('%Y-%m-%d')
            #print 'LastCpnDate', lastCpnDate
        else:
            lastCpnDate = None

        swapCurve = swapCurves[row.TradeDate]
        gov_zero_curve = govtCurves[row.TradeDate]

        try:
            float_coupon_freq=float(couponfreq)
        except:
            float_coupon_freq = 2.0

        if float_coupon_freq ==0:
            float_coupon_freq=2.0


        if marketPrice < 40:
            print  "MarketPrice is less than 40" , issueDate, ' ', maturityDate, ' ', instrCode, ' ', issuerId, ' ', coupon, ' ', marketPrice, ' ', rateScore
            return None
        else:
            pass

        try:
            issuerIdInt = issuerIdInt if np.isnan(issuerId) else int(issuerId)
            sql_statement_delete = self.delete_tradeDate_sql(valuationDate,
                                                             currency,
                                                             issuerIdInt,
                                                             instrCode,
                                                             rateScore,
                                                             sector)

            sqlList.append(sql_statement_delete + "\n")

            bondPricer = BondPricer(issueDate, maturityDate, coupon, valuationDate, float_coupon_freq, first_cpn_dt=frstCpnDate, last_cpn_dt=lastCpnDate)

            ratecode = self.convert_RatingScore(rateScore)
            sectorcode = self.convert_sectorCode(sector)

            spreadAnalytic = bondPricer.compute_spread(gov_zero_curve, marketPrice)
            user_name = getpass.getuser()
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': spreadAnalytic,'StatCode': 1 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))

            spreadDurationAnalytic = bondPricer.compute_spread_duration_analytical(gov_zero_curve, marketPrice)
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': spreadDurationAnalytic,'StatCode': 2 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))

            spread = bondPricer.compute_spread(swapCurve, marketPrice)
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': spread,'StatCode': 3 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))

            spreadDuration = bondPricer.compute_HazardUkq(swapCurve, marketPrice)
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': spreadDuration,'StatCode': 4 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))

            yieldToMaturity = bondPricer.compute_yield_to_maturity(marketPrice)
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': yieldToMaturity,'StatCode': 5 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))

            #print 'Running: ', instrCode, '  Valuation Date', valuationDate, ' ', issuerId, ' ', coupon, ' ', marketPrice, ' spread duration:', spreadDuration

            Uk, V = bondPricer.compute_HazardUkq(swapCurve,0.4,marketPrice)
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': Uk[0],'StatCode': 6 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': Uk[1],'StatCode': 7 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': Uk[2],'StatCode':8 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))
            sqlList.append(self.create_bond_stats_sql({'InstrCode': instrCode,'StatValue': V,'StatCode': 9 ,'TradeDate': valuationDate,'Currency': currency,'InstrDesc': issuerIdInt, 'RatingCode': ratecode, 'SectorCode':sectorcode, 'Lud': get_current_time(), 'Lub': user_name}))

        except Exception as e:
            print "Failed computing stats:" , e.message, ' ', valuationDate, ' ', maturityDate, ' ', instrCode, ' ', issuerId, ' ', coupon, ' ', marketPrice

                 
    def delete_tradeDate_sql(self, valuationDate, currency, issuerId, instrumentCode, rating, sector):

        sectorCode = self.convert_sectorCode(sector)
        ratingCode = self.convert_RatingScore(rating)

        sqlStatementList= ['DELETE FROM [MarketData].[dbo].[ResearchBondStatsDetail]']
        sqlStatementList.append("WHERE TradeDate='$valuationDate' ")
        sqlStatementList.append("and Currency='$currency' ")
        if issuerId is not None:
            sqlStatementList.append("and IssuerId=$issuerId ")
        if instrumentCode is not None:
            sqlStatementList.append("and InstrCode=$instrumentCode ")
        if rating is not None:
            sqlStatementList.append("and RatingCode=$rating ")
        if sector is not None:
            sqlStatementList.append("and SectorCode=$sector ")        

        sqlStatement = "\n".join(sqlStatementList)
        sqlResult = string.Template(sqlStatement).substitute({'valuationDate':valuationDate, 'currency':currency, 'issuerId':issuerId, 'instrumentCode':instrumentCode , 'rating':ratingCode, 'sector':sectorCode })

        return sqlResult

                
    def convert_RatingScore(self, ratinglevel):
        ratingCode = 0
        if ratinglevel == 'AAA':
            ratingCode = 1
        elif ratinglevel == 'AA':
            ratingCode = 2
        elif ratinglevel == 'A':
            ratingCode = 3
        elif ratinglevel == 'BBB':
            ratingCode = 4
        elif ratinglevel == 'SUB-IG':
            ratingCode =5
        return ratingCode 
    
    def convert_sectorCode (self, sectorName):

        sectorCode = 0
        if sectorName == 'Financials':
            sectorCode = 1
        elif sectorName == 'Energy':
            sectorCode = 2
        elif sectorName == 'Utilities':
            sectorCode = 3
        elif sectorName == 'Health Care':
            sectorCode = 4
        elif sectorName == 'Telecommunication Services':
            sectorCode = 5
        elif sectorName == 'Consumer Discretionary':
            sectorCode = 6
        elif sectorName == 'Information Technology':
            sectorCode = 7
        elif sectorName == 'Industrials':
            sectorCode = 8
        elif sectorName == 'Materials':
            sectorCode = 9
        elif sectorName == 'Consumer Staples':
            sectorCode = 10
        elif sectorName == 'UNKNOWN':
            sectorCode = 11                                                                        
        return sectorCode              
                                                  
    
    def create_bond_stats_sql(self, bondstatDic):
        sqlStatement = string.Template ("""
        INSERT INTO MarketData.dbo.ResearchBondStatsDetail
        VALUES($InstrCode, $StatValue, $StatCode, '$TradeDate', '$Currency', $InstrDesc, $RatingCode, $SectorCode, '$Lud', '$Lub' )
        """).substitute(bondstatDic)
        return sqlStatement
        
    def __str__(self):
        return 'BondPricer(IssueDate:%s, MaturityDate:%s, Coupon:%s)' % (
            self.issueDate, self.maturityDate, self.coupon)
            
    def __repr__(self):
        return self.__str__()
        
def write_stats_to_db(args):
    startDate = args[0]
    endDate = args[1]
    currency = args[2]
    govtCurveName = args[3]
    swpSpreadCurveName = args[4]
    rating = args[5]
    sector = args[6]
    issuerId = args[7]     
    instrumentCode = args[8]
    database = args[9]
    config = args[10]

    bondStatsWriter = BondStatDatabaseWriter(startDate, endDate, currency, govtCurveName, swpSpreadCurveName, issuerId, instrumentCode, rating, sector, config, database)
    sql_statement = bondStatsWriter.write_to_db()
    return sql_statement

if __name__ == '__main__':
    import bond
    import concurrent.futures
    import cProfile
    parser = OptionParser()
    parser.add_option("-s", "--startDate", dest="startDate", help="Starting trade date to run bond stats", metavar="2014-05-01 example")
    parser.add_option("-e", "--endDate", dest="endDate", help="End trade date to run bond stats", metavar="2014-05-02 example")
    parser.add_option('-d', '--enviroment', dest='enviroment', help='enviroment name', metavar='example DEV')
    parser.add_option('-g', '--config', dest='config', help='configuration file name', metavar='example database.config')
    parser.add_option("-m", "--instrumentCode", dest="instrumentCode", help="instrumentCode to run bond stats", metavar="1158953 example")
    parser.add_option("-i", "--issuerId", dest="issuerId", help="issuerId to run bond stats", metavar=" example 1000825")
    parser.add_option("-c", "--currency", dest="currency", help="currency to run bond stats", metavar=" example USD")
    parser.add_option("-r", "--rating", dest="rating", help="rating to run bond stats", metavar=" example AAA")
    parser.add_option("-t", "--sector", dest="sector", help="sector to run bond stats", metavar=" example Consumer Discretionary")
    parser.add_option("-p", "--parallel", action="store_true", dest="parallelize", help="runs computation across all cores", metavar=" example -p off")
    (options, args) = parser.parse_args()

    currency = options.currency
    govtCurveName = convert_gvt_curve(currency)
    swpSpreadCurveName = convert_swp_sprd_curve(currency)
    rating = options.rating
    sector = options.sector
    startDate = options.startDate
    endDate = options.endDate
    enviroment = options.enviroment
    config = options.config  
    issuerId = options.issuerId     
    instrumentCode = options.instrumentCode

    timerange = pd.bdate_range(startDate, endDate)
    timerangelist = timerange.tolist()
    splitted_list = split_list(timerangelist)
    arg_list = [(x[0].strftime('%Y-%m-%d'), x[-1].strftime('%Y-%m-%d'), currency, 
                 govtCurveName, swpSpreadCurveName, rating, sector, issuerId, 
                 instrumentCode, enviroment, config) for x in splitted_list]

    startTime = get_current_time()
    print "startTime: ", startTime
    sql_collection = []
    #cProfile.run('map(write_stats_to_db, arg_list)')
    if options.parallelize:
        print "running single process"
        sql_to_execute = map(write_stats_to_db, arg_list)
        sql_collection.append([sql_to_execute, startDate, endDate])
    else:
        print "running multi-process"
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for (date_range, sql_to_execute) in zip(arg_list, executor.map(write_stats_to_db, arg_list)):
                print "Ran: ", date_range[0], date_range[1]
                sql_collection.append([sql_to_execute,date_range[0],date_range[1]])

    for sql_value in sql_collection:
        try:
            db.MSSQL.execute_commit("BEGIN TRAN T1;\n" + sql_value[0] + "\nCOMMIT TRAN T1;")
            print 'SQL Commit Success: start=', sql_value[1], ', end=', sql_value[2]
        except Exception as e:
            print 'Exception on execute SQL commit!: start=', sql_value[1], ', end=', sql_value[2]
            print e.message, str(e)
            print sql_value[1]

    endTime = get_current_time()

    print "startTime: ", startTime, "endTime: ", endTime
        





    
        
    

    
