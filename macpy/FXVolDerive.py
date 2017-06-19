import pandas as pd
import numpy as np
import macpy.utils.database as db
import macpy.dateflow as df
import unittest
import datetime as dt
import string
import getpass
import bond as bond
from optparse import OptionParser
from scipy.stats import norm
from scipy.optimize import brentq, fsolve
import logging
from utils import Utilities



def get_current_time():
    return str(dt.datetime.now())[0:-3]


def split_list(mylist, step=30):
    i=0
    splitted_list=[]
    while i<len(mylist):
        splitted_list.append(mylist[i:i+step])
        i=i+step

    return splitted_list


class FXVolDerive:

    def __init__(self, startDate, endDate, strikeSwitch = 'on', levelSwitch = 'off', database='PROD'):
        self.startDate = startDate
        self.endDate = endDate
        self.strikeSwitch = strikeSwitch
        self.levelSwitch = levelSwitch
        self.database = database

    def derive_FXVol(self):

        volSource = db.FXVolDerive(self.startDate, self.endDate, database=self.database)
        dataframeExtracted = volSource.extract_from_db()
        sqlList = []

        sql_statement_delete = self.delete_tradeDate_sql(self.startDate, self.endDate)
        print "processing: ", self.startDate, self.endDate
        sqlList.append(sql_statement_delete+"\n")
        computeTimeStart = dt.datetime.now()
        grouped = dataframeExtracted.groupby(['TradeDate', 'VolSurfaceId', 'TenorEnum'])

        for df in grouped:
            try:
                newdf = df[1]

                VolSurfaceDataId = str(newdf['VolSurfaceDataId'].values[0])
                SettleCurrency = str(newdf['SettleCurrency'].values[0])
                ForeignCurrency = str(newdf['ForeignCurrency'].values[0])
                VolSurfaceId=newdf['VolSurfaceId'].values[0]
                TenorEnum = str(newdf['TenorEnum'].values[0])
                TradeDate = dt.datetime.strptime(newdf['TradeDate'].values[0],"%Y-%m-%d")
                username = getpass.getuser()
                TenorInYears = self.convert_tenor(TenorEnum)
                if TenorInYears>2.0:
                    DeltaConv = 'ForwardDelta'
                else:
                    DeltaConv = 'SpotDelta'


                if 'ATM' in list(newdf['Delta']):
                    VolATM = list(newdf[(newdf['Delta']=='ATM')].Quote)[0]
                    StrikeATM = self.compute_strike(TenorInYears,0.5,TradeDate,SettleCurrency,ForeignCurrency,VolATM/100,self.strikeSwitch)
                    sqlList.append(self.create_sql_statement({'VolSurfaceDataId': VolSurfaceDataId, 'SettleCurrency':SettleCurrency, 'ForeignCurrency': ForeignCurrency, 'VolSurfaceId': VolSurfaceId, 'TenorEnum': TenorEnum, 'TradeDate': TradeDate, 'Delta': '0.5', 'DeltaConvention': DeltaConv, 'premiumAdjust': StrikeATM[3], 'ImpliedStrike': StrikeATM[0], 'Quote': VolATM, 'kmin':StrikeATM[1], 'kmax':StrikeATM[2], 'FXSpot': StrikeATM[4], 'r_d': StrikeATM[5], 'r_f': StrikeATM[6], 'Lud': get_current_time(), 'Lub': username}))

                    if self.levelSwitch == 'on':
                        if 'B10' in list(newdf['Delta']) and 'R10' in list(newdf['Delta']):
                            VolB10 = list(newdf[(newdf['Delta']=='B10')].Quote)[0]
                            VolR10 = list(newdf[(newdf['Delta']=='R10')].Quote)[0]
                            Vol10 = (2*VolB10+2*VolATM+VolR10)/2
                            Vol90 = (2*VolB10+2*VolATM-VolR10)/2
                            Strike10 = self.compute_strike(TenorInYears,0.1,TradeDate,SettleCurrency,ForeignCurrency,Vol10/100, self.strikeSwitch)
                            Strike90 = self.compute_strike(TenorInYears,0.9,TradeDate,SettleCurrency,ForeignCurrency,Vol90/100, self.strikeSwitch)
                            sqlList.append(self.create_sql_statement({'VolSurfaceDataId': VolSurfaceDataId, 'SettleCurrency':SettleCurrency, 'ForeignCurrency': ForeignCurrency, 'VolSurfaceId': VolSurfaceId, 'TenorEnum': TenorEnum, 'TradeDate': TradeDate, 'Delta': '0.1', 'DeltaConvention': DeltaConv, 'premiumAdjust': Strike10[3],'ImpliedStrike': Strike10[0], 'Quote': Vol10, 'kmin': Strike10[1], 'kmax': Strike10[2], 'FXSpot': Strike10[4], 'r_d': Strike10[5], 'r_f': Strike10[6], 'Lud': get_current_time(), 'Lub': username}))
                            sqlList.append(self.create_sql_statement({'VolSurfaceDataId': VolSurfaceDataId, 'SettleCurrency':SettleCurrency, 'ForeignCurrency': ForeignCurrency, 'VolSurfaceId': VolSurfaceId, 'TenorEnum': TenorEnum, 'TradeDate': TradeDate, 'Delta': '0.9', 'DeltaConvention': DeltaConv, 'premiumAdjust': Strike90[3], 'ImpliedStrike': Strike90[0], 'Quote': Vol90, 'kmin': Strike90[1], 'kmax': Strike90[2], 'FXSpot': Strike90[4], 'r_d': Strike90[5], 'r_f': Strike90[6], 'Lud': get_current_time(), 'Lub': username}))


                    if 'B25' in list(newdf['Delta']) and 'R25' in list(newdf['Delta']):
                        VolB25 = list(newdf[(newdf['Delta']=='B25')].Quote)[0]
                        VolR25 = list(newdf[(newdf['Delta']=='R25')].Quote)[0]
                        Vol25 = (2*VolB25+2*VolATM+VolR25)/2
                        Vol75 = (2*VolB25+2*VolATM-VolR25)/2
                        Strike25 = self.compute_strike(TenorInYears,0.25,TradeDate,SettleCurrency,ForeignCurrency,Vol25/100, self.strikeSwitch)
                        Strike75 = self.compute_strike(TenorInYears,0.75,TradeDate,SettleCurrency,ForeignCurrency,Vol75/100, self.strikeSwitch)
                        DVol10 = VolATM - VolR25*(0.1-0.5)+16*VolB25*np.square((0.1-0.5))
                        DVol90 = VolATM - VolR25*(0.9-0.5)+16*VolB25*np.square((0.9-0.5))
                        DStrike10 = self.compute_strike(TenorInYears,0.1,TradeDate,SettleCurrency,ForeignCurrency,DVol10/100, self.strikeSwitch)
                        DStrike90 = self.compute_strike(TenorInYears,0.9,TradeDate,SettleCurrency,ForeignCurrency,DVol90/100, self.strikeSwitch)
                        sqlList.append(self.create_sql_statement({'VolSurfaceDataId': VolSurfaceDataId, 'SettleCurrency':SettleCurrency, 'ForeignCurrency': ForeignCurrency, 'VolSurfaceId': VolSurfaceId, 'TenorEnum': TenorEnum, 'TradeDate': TradeDate, 'Delta': '0.25', 'DeltaConvention': DeltaConv, 'premiumAdjust': Strike25[3],'ImpliedStrike': Strike25[0], 'Quote': Vol25, 'kmin': Strike25[1], 'kmax': Strike25[2],'FXSpot': Strike25[4], 'r_d': Strike25[5], 'r_f': Strike25[6], 'Lud': get_current_time(), 'Lub': username}))
                        sqlList.append(self.create_sql_statement({'VolSurfaceDataId': VolSurfaceDataId, 'SettleCurrency':SettleCurrency, 'ForeignCurrency': ForeignCurrency, 'VolSurfaceId': VolSurfaceId, 'TenorEnum': TenorEnum, 'TradeDate': TradeDate, 'Delta': '0.75', 'DeltaConvention': DeltaConv, 'premiumAdjust': Strike75[3],'ImpliedStrike': Strike75[0], 'Quote': Vol75, 'kmin': Strike75[1], 'kmax': Strike75[2], 'FXSpot': Strike75[4], 'r_d': Strike75[5], 'r_f': Strike75[6],'Lud': get_current_time(), 'Lub': username}))
                        if self.levelSwitch == 'on':
                            sqlList.append(self.create_sql_statement({'VolSurfaceDataId': VolSurfaceDataId, 'SettleCurrency':SettleCurrency, 'ForeignCurrency': ForeignCurrency, 'VolSurfaceId': VolSurfaceId, 'TenorEnum': TenorEnum, 'TradeDate': TradeDate, 'Delta': '0.1EXTRAP', 'DeltaConvention': DeltaConv, 'premiumAdjust': DStrike10[3], 'ImpliedStrike': DStrike10[0], 'Quote': DVol10, 'kmin': DStrike10[1], 'kmax': DStrike10[2],'FXSpot': DStrike10[4], 'r_d': DStrike10[5], 'r_f': DStrike10[6], 'Lud': get_current_time(), 'Lub': username}))
                            sqlList.append(self.create_sql_statement({'VolSurfaceDataId': VolSurfaceDataId, 'SettleCurrency':SettleCurrency, 'ForeignCurrency': ForeignCurrency, 'VolSurfaceId': VolSurfaceId, 'TenorEnum': TenorEnum, 'TradeDate': TradeDate, 'Delta': '0.9EXTRAP', 'DeltaConvention': DeltaConv, 'premiumAdjust': DStrike90[3], 'ImpliedStrike': DStrike90[0], 'Quote': DVol90, 'kmin': DStrike90[1], 'kmax': DStrike90[2], 'FXSpot': DStrike90[4], 'r_d': DStrike90[5], 'r_f': DStrike90[6],'Lud': get_current_time(), 'Lub': username}))
            except:
                continue

        computeTimeEnd = dt.datetime.now()
        print "Finished processing: ", self.startDate, self.endDate,  "Seconds spent: ", round((computeTimeEnd - computeTimeStart).total_seconds(), 5)
        sql_statement = ';\n'.join(sqlList)

        return sql_statement
        #try:
         #   db.MSSQL.execute_commit(sql_statement)
        #except Exception as e:
         #   print 'Exception on execute_commit(sql_statement)'
          #  print e.message

    def premium_adjust_flag_select(self, settleCurrency, foreignCurrency):
        currencyPairs = foreignCurrency+settleCurrency
        nonPaList = ['GBPUSD', 'EURUSD', 'AUDUSD', 'NZDUSD']
        if currencyPairs in nonPaList:
            adjustflag = 'N'
        else:
            adjustflag = 'Y'
        return adjustflag

    def FXSpotRateDerive(self, tradeDate, currency):
        DataIdDF = db.FXCurrencyDataId(currency)
        IDdf = DataIdDF.extract_from_db()
        AxiomaDataId = IDdf['AxiomaDataId'].values[0]

        FXSpotDF = db.FXSpotRate(tradeDate,AxiomaDataId)
        df = FXSpotDF.extract_from_db()
        filterdf = df[(df['TradeDate']==tradeDate)]
        FXSpotRate = list(filterdf['FXRate'])[0]

        return FXSpotRate

    def compute_strike(self, tenorInYears, delta,tradeDate, settleCurrency, foreignCurrency, vol, strikeSwitch):

        if strikeSwitch == 'off':
            r_d = 'NULL'
            r_f = 'NULL'
            FXSpotRate = 'NULL'
            adjustflag = self.premium_adjust_flag_select(settleCurrency, foreignCurrency)

        else:
            domesticCurveName = bond.convert_gvt_curve(settleCurrency)
            foreignCurveName = bond.convert_gvt_curve(foreignCurrency)
            domesticCurve = bond.create_yield_curve(tradeDate, domesticCurveName)
            foreignCurve = bond.create_yield_curve(tradeDate, foreignCurveName)
            tradedt = pd.to_datetime(tradeDate).strftime('%Y%m%d')
            r_d = domesticCurve(tenorInYears)
            r_f = foreignCurve(tenorInYears)


            if settleCurrency == 'USD':
                FXSpotRate = self.FXSpotRateDerive(tradedt, foreignCurrency)
                adjustflag = self.premium_adjust_flag_select(settleCurrency, foreignCurrency)
            elif foreignCurrency == 'USD':
                FXSpotRate = 1/self.FXSpotRateDerive(tradedt, settleCurrency)
                adjustflag = self.premium_adjust_flag_select(settleCurrency, foreignCurrency)
            else:
                FXSpotRateForeign = self.FXSpotRateDerive(tradedt, foreignCurrency)
                FXSpotRateDomestic = self.FXSpotRateDerive(tradedt, settleCurrency)
                FXSpotRate = FXSpotRateForeign/FXSpotRateDomestic
                adjustflag = self.premium_adjust_flag_select(settleCurrency, foreignCurrency)

        implied_strike_analysis_list = self.func_implied_strike(tenorInYears, FXSpotRate, r_d, r_f, delta,vol, adjustflag, strikeSwitch)
        return implied_strike_analysis_list

    def func_implied_strike(self, tenorInYears, FXSpotRate, r_d, r_f, delta, vol, adjustflag, strikeSwitch):

        if strikeSwitch == 'off':
            k_min = 'N/A'
            k_max = 'N/A'
            implied_strike = 'NULL'
        else:
            FXForwardRate = FXSpotRate*np.exp((r_d-r_f)*tenorInYears)
            if adjustflag == 'False':
                if tenorInYears >2.0:
                    implied_strike = FXForwardRate*np.exp(-vol*np.sqrt(tenorInYears)*norm.ppf(delta)+0.5*tenorInYears*np.square(vol))
                    k_min = 'N/A'
                    k_max = 'N/A'
                else:
                    normInverse = delta/np.exp(-r_f*tenorInYears)
                    if normInverse <= 0 or normInverse >1:
                        print "Inverse Nomral Function failed for Value ", normInverse
                        implied_strike = 'NULL'
                    else:
                        implied_strike = FXForwardRate*np.exp(-vol*np.sqrt(tenorInYears)*norm.ppf(normInverse)+0.5*tenorInYears*np.square(vol))
                    k_min = 'N/A'
                    k_max = 'N/A'
            else:
                if tenorInYears > 2.0:
                    k_max = FXForwardRate*np.exp(-vol*np.sqrt(tenorInYears)*norm.ppf(delta)+0.5*tenorInYears*np.square(vol))
                    #strikeImplied = k_max
                    #d_plus = (np.log(FXForwardRate/strikeImplied)+0.5*np.square(vol)*tenorInYears)/(vol*np.sqrt(tenorInYears))
                    #d_minus = d_plus - vol*np.sqrt(tenorInYears)
                    #callValue = np.exp(-r_d*tenorInYears)*(FXForwardRate*norm.cdf(d_plus)-strikeImplied*norm.cdf(d_minus))
                    #adjustedDelta = delta - callValue/FXSpotRate
                    implicit_func_strike = lambda k: k/FXForwardRate*norm.cdf((np.log(FXForwardRate/k)-0.5*np.square(vol)*tenorInYears)/(vol*np.sqrt(tenorInYears)))-delta
                else:
                    normInverse = delta/np.exp(-r_f*tenorInYears)
                    k_max = FXForwardRate*np.exp(-vol*np.sqrt(tenorInYears)*norm.ppf(normInverse)+0.5*tenorInYears*np.square(vol))
                    #strikeImplied = k_max
                    #d_plus = (np.log(FXForwardRate/strikeImplied)+0.5*np.square(vol)*tenorInYears)/(vol*np.sqrt(tenorInYears))
                    #d_minus = d_plus - vol*np.sqrt(tenorInYears)
                    #callValue = np.exp(-r_d*tenorInYears)*(FXForwardRate*norm.cdf(d_plus)-strikeImplied*norm.cdf(d_minus))
                    #adjustedDelta = delta - callValue/FXSpotRate
                    implicit_func_strike = lambda k: np.exp(-r_f*tenorInYears)*k/FXForwardRate*norm.cdf((np.log(FXForwardRate/k)-0.5*np.square(vol)*tenorInYears)/(vol*np.sqrt(tenorInYears)))-delta

                implict_func_min = lambda x: vol*np.sqrt(tenorInYears)*norm.cdf(x)-norm.pdf(x)
                implied_d = fsolve(implict_func_min,0.5)
                k_min = FXForwardRate*np.exp(-vol*np.sqrt(tenorInYears)*implied_d-0.5*tenorInYears*np.square(vol))[0]

                try:
                    implied_strike = brentq(implicit_func_strike, k_min, k_max)
                except:
                    implied_strike = 'NULL'
                    print 'Fail to compute Implied Strike for tenor ', tenorInYears, ' at Delta level ', delta

        Implied_strike_diagnose_list = [implied_strike, k_min, k_max, adjustflag, FXSpotRate, r_d, r_f]

        return Implied_strike_diagnose_list


    def convert_tenor(self,tenor):
        tenorInYears = 1.0
        if tenor == '1W':
            tenorInYears = 1.0/52
        elif tenor == '1M':
            tenorInYears = 1.0/12
        elif tenor == '2M':
            tenorInYears = 2.0/12
        elif tenor == '3M':
            tenorInYears = 3.0/12
        elif tenor == '6M':
            tenorInYears = 6.0/12
        elif tenor == '1Y':
            tenorInYears = 1.0
        elif tenor == '2Y':
            tenorInYears = 2.0
        elif tenor == '3Y':
            tenorInYears = 3.0
        elif tenor == '5Y':
            tenorInYears = 5.0
        elif tenor == '10Y':
            tenorInYears = 10.0
        return tenorInYears


    def create_sql_statement(self, volDictionary):

        sqlstatement = string.Template("""
        INSERT INTO MarketData.dbo.FXVolDerive
        VALUES ('$VolSurfaceDataId','$SettleCurrency', '$ForeignCurrency', '$VolSurfaceId','$TenorEnum','$TradeDate', '$Delta', '$DeltaConvention', '$premiumAdjust', $ImpliedStrike, '$Quote', '$kmin', '$kmax', '$FXSpot', '$r_d', '$r_f', '$Lud', '$Lub')
        """).substitute(volDictionary)

        return sqlstatement

    def delete_tradeDate_sql(self, startDate, endDate):
        sqlstatement = string.Template("""
        DELETE FROM MarketData.dbo.FXVolDerive
        WHERE TradeDate >= '$startDate'
        and TradeDate <= '$endDate'
        """).substitute({'startDate': startDate, 'endDate':endDate })

        return sqlstatement

def write_vol_to_db(args):
    startDate = args[0]
    endDate = args[1]
    strikeSwitch = args[2]
    levelSwitch = args[3]
    enviroment = args[4]

    FXVolStatsWriter = FXVolDerive(startDate, endDate, strikeSwitch, levelSwitch, enviroment)
    sql_statement = FXVolStatsWriter.derive_FXVol()

    return sql_statement




if __name__=='__main__':
    import FXVolDerive as fv
    import concurrent.futures
    import os


    parser = OptionParser()
    parser.add_option("-s", "--startDate", dest="startDate", help="Starting trade date to run FXVolDerive", metavar="2014-05-01 example")
    parser.add_option("-e", "--endDate", dest="endDate", help="End trade date to run curve FXVolDerive", metavar="2014-05-02 example")
    parser.add_option('-d', '--enviroment', dest='enviroment', help='enviroment name', metavar='example DEV')
    parser.add_option('-k',"--strikeSwitch", dest="strikeSwitch", help="turn off strike inference", metavar="off example")
    parser.add_option('-l',"--levelSwitch", dest="levelSwitch", help="turn off vol derivation at 10 and 90", metavar="off example")
    parser.add_option("-p", "--parallel", action="store_true", dest="parallelize", help="runs computation across all cores", metavar=" example -p off")
    (options, args) = parser.parse_args()

    startDate = options.startDate
    endDate = options.endDate
    enviroment = options.enviroment
    strikeSwitch = options.strikeSwitch
    levelSwitch = options.levelSwitch

    timerange = pd.bdate_range(startDate, endDate)
    timerangelist = timerange.tolist()
    splitted_list = split_list(timerangelist)

    arg_list = [ (x[0].strftime('%Y-%m-%d'), x[-1].strftime('%Y-%m-%d'),strikeSwitch, levelSwitch, enviroment) for x in splitted_list]

    log_level = logging.INFO
    logging.basicConfig(level=log_level)

    startTime = get_current_time()
    logging.info("startTime %s"%startTime)
    #print "startTime: ", startTime
    computeTimeStart = dt.datetime.now()
    sql_collection = []

    if options.parallelize:
        logging.info("running single process")
        for (date_range, sql_to_execute) in zip(arg_list, map(write_vol_to_db, arg_list)):
            print "Completed SQL Generation: ", date_range[0], date_range[1]
            sql_collection.append([sql_to_execute,date_range[0],date_range[1]])
    else:
        logging.info("running multi-process")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for (date_range, sql_to_execute) in zip(arg_list, executor.map(write_vol_to_db, arg_list)):
                print "Completed SQL Generation: ", date_range[0], date_range[1]
                sql_collection.append([sql_to_execute,date_range[0],date_range[1]])
    logging.info("Finished computing results")
    print "Starting db inserts at time: ",get_current_time()

    for sql_value in sql_collection:
        try:
            db.MSSQL.execute_commit(sql_value[0],enviroment)
            print 'SQL Insert success: start=', sql_value[1], ', end=', sql_value[2]
        except Exception as e:
            print 'Exception on Insert SQL commit!: start=', sql_value[1], ', end=', sql_value[2]
            print e.message, str(e)

    voldataframe = db.FXVolDerivedCount(startDate, endDate, database=enviroment)
    dfVol = voldataframe.extract_from_db()
    logging.info("Number of Records Inserted: %r"%dfVol['Quote'].count())

    computeTimeEnd = dt.datetime.now()
    endTime = get_current_time()
    print "startTime: ", startTime
    print "endTime: ", endTime
    print "Total Time in Hours: ", round((computeTimeEnd - computeTimeStart).total_seconds()/(60*60), 6)