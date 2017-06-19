import macpy.bond as bond
import macpy.bond_utils as bond_utils
import string
import macpy.utils.database as db
import pandas as pd
import traceback
import datetime
from optparse import OptionParser
import getpass


import macpy.bei_curve as bei_curve
from bootstrapper_creator import BootStrapper, Bootstrapper_BEI


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# pd.set_option('display.precision', 2)
pd.set_option('display.expand_frame_repr', False)

def get_current_time():
    return str(datetime.datetime.now())[0:-3]

def split_list(mylist, step=30):                
    i=0
    splitted_list=[]
    while i<len(mylist):
        splitted_list.append(mylist[i:i+step])
        i=i+step
                       
    return splitted_list


class CurveType:

    def __init__(self, startDate, endDate, currency, countryName=None, enviroment=None, config=None, production_table=False, input_bond_table=False):
        self.startDate = startDate
        self.endDate = endDate
        self.currency = currency
        self.countryName = countryName
        self.enviroment = enviroment
        #self.filteredbondDatabase = ''#'[PROD_MAC_MKT_DB].' if enviroment is None or enviroment.upper() == 'DEV' else ''
        self.config = config

        self.production_table = production_table
        self.input_bond_table = input_bond_table
        self.curveName = bond.convert_gvt_curve(self.currency, self.countryName)

        self.curve_node_id_cache = {}

        return

    @classmethod
    def log_no_benchmarks(cls, dataframe, sql_statement):
            err_message = "Cannot bootstrap - there are no benchmark bonds found for query: \n "\
                          + sql_statement
            print err_message

    def delete_curve(self):
        sql_delete_list = self.delete_yield_sql(self.startDate, self.endDate, self.curveName)
        try:
            # Delete existing records if present
            [db.MSSQL.execute_commit(sql, self.enviroment, self.config) for sql in sql_delete_list]
        except Exception as e:
            print "Could not delete prior data: ", e

    def create_yield_sql(self, dict):
        if self.production_table:
            sqlstatement = string.Template("""
                INSERT INTO MarketData.dbo.CurveNodeQuote VALUES('$CurveNodeId', '$TradeDate', '$Quote', '$IsCorrected', '$Lud', '$Lub')
                """).substitute(dict)
        else:
            sqlstatement = string.Template("""
            INSERT INTO MarketData.dbo.ResearchCurves VALUES('$CurveName', '$Category', '$TimeInYears', '$Level', '$TradeDate', '$Lud', '$Lub')
            """).substitute(dict)
        #print sqlstatement
        return sqlstatement

    def delete_yield_sql(self, startDate, endDate, curveName):
        if self.production_table:
            sqllist = []
            curve_node_df = self.get_curve_node_id(self.curveName)
            dictionary = [{'StartDate': startDate, 'EndDate':endDate, 'CurveNodeId': node_id} for node_id in curve_node_df['CurveNodeId'].values]
            for d in dictionary:
                sqlstatement_delete = string.Template("""
                DELETE FROM [MarketData].[dbo].[CurveNodeQuote]
                WHERE TradeDate>='$StartDate' and TradeDate<='$EndDate'
                and CurveNodeId = '$CurveNodeId'
                """).substitute(d)
                sqllist.append(sqlstatement_delete)
            return sqllist
        else:
            sqlstatement_delete = string.Template("""
            DELETE FROM [MarketData].[dbo].[ResearchCurves]
            WHERE TradeDate>='$StartDate' and TradeDate<='$EndDate'
            and CurveName = '$curveName' and Category = 'AxiomaBootStrapper'
            """).substitute({'StartDate':startDate, 'EndDate':endDate, 'curveName': curveName})

        return [sqlstatement_delete]

    def get_curve_node_id(self, curveName):
        if curveName in self.curve_node_id_cache:
            return self.curve_node_id_cache[curveName]
        sqlstatement = r"""
                select cv.CurveShortName, cv.CurveId, te.InYears, cn.CurveNodeId
                from MarketData.dbo.curve cv
                join MarketData.dbo.curvenodes cn on cv.curveid = cn.curveid
                join MarketData.dbo.TenorEnum te on te.TenorEnum = cn.TenorEnum
                where cv.curveshortname in ('{}')
                order by te.InYears
                """.format(curveName)
        self.curve_node_id_cache[curveName] = db.MSSQL.extract_dataframe(sqlstatement, self.enviroment, self.config)
        return self.curve_node_id_cache[curveName]

class SovereignCurve(CurveType):
    
    def __init__(self, startDate, endDate, currency, countryName=None, enviroment=None, config=None, production_table=False, input_bond_table=False):
        CurveType.__init__(self, startDate, endDate, currency, countryName, enviroment, config, production_table, input_bond_table)

    def write_to_dbs(self):

        benchmark_bond_query = db.DerivedFilteredBootstrap(self.currency, self.startDate, self.endDate,
                                                           self.countryName,
                                                           database=self.enviroment,
                                                           config=self.config,
                                                           input_bond_table=self.input_bond_table)
        dataframe = benchmark_bond_query.extract_from_db()
        if len(dataframe) == 0:
            self.log_no_benchmarks(dataframe, benchmark_bond_query.sqlstatement)
            self.delete_curve()
            return
        if type(dataframe) == list:
            for i, curveName in enumerate(db.EURO_CURVES):
                self.curveName = curveName
                self.write_curve(dataframe[i])
        else:
            self.write_curve(dataframe)

    def write_curve(self, dataframe):

        self.delete_curve()
        grouped = dataframe.groupby('TradeDate')
        sqlList=[]
        curve_node_df = self.get_curve_node_id(self.curveName)
        CustomTimeInYearsList=curve_node_df['InYears'].tolist()

        for name, df in grouped:
            # try:
            tradeDate = pd.to_datetime(name).strftime('%Y-%m-%d')
            conventions = bond_utils.get_sovereign_conventions(self.currency)
            bootstrapper = BootStrapper(tradeDate, tradeDate, self.currency, conventions, self.enviroment, self.config, df)
            yieldcurve = bootstrapper.createYieldCurve()
            yieldList = yieldcurve(CustomTimeInYearsList)

            if self.production_table:
                dictionary = []
                for ttm, level in zip(CustomTimeInYearsList[0:], yieldList[0:]):
                    try:
                        dictionary.append({'CurveNodeId': curve_node_df[curve_node_df.InYears == ttm]['CurveNodeId'].values[0],
                                    'TradeDate': tradeDate,
                                    'Quote': level,
                                    'IsCorrected': '',
                                    'Lud': str(datetime.datetime.now())[0:-3],
                                    'Lub': getpass.getuser()})
                    except Exception as e:
                        print "Exception thrown on append:", str(e)
                        continue
            else:
                dictionary = [{'CurveName':self.curveName, 
                               'Category':'AxiomaBootStrapper', 
                               'TimeInYears':ttm, 
                               'Level':level, 
                               'TradeDate':tradeDate, 
                               'Lud': str(datetime.datetime.now())[0:-3], 
                               'Lub': getpass.getuser()} for ttm, level in zip(CustomTimeInYearsList[0:],yieldList[0:])]
            sqlList.extend([self.create_yield_sql(x) for x in dictionary])                        
            # except Exception as e:
            #     print "Yield Curve Failed on ", name
            #     print e
            #     # print e.message
            #     continue
                
        sqlstatement='\n'.join(sqlList)                       
        try:            
            db.MSSQL.execute_commit(sqlstatement, self.enviroment, self.config)
        except Exception as e:
            print "Exception thrown commit:", e.message, "SQL execute commit:" , sqlstatement         
        # return        

class BEI_Curve(CurveType):

    def __init__(self, startDate, endDate, currency, countryName=None, enviroment=None, config=None, production_table=False, input_bond_table=False):
        CurveType.__init__(self, startDate, endDate, currency, countryName, enviroment, config, production_table, input_bond_table)

    def write_to_dbs(self):

        benchmark_bond_query = bei_curve.BEI(country_code=self.countryName, start_date = self.startDate, end_date = self.endDate, environment = self.enviroment, use_research_table=not self.production_table)
        dataframe = benchmark_bond_query.extract_from_db()
        if len(dataframe) == 0:
            self.log_no_benchmarks(dataframe, benchmark_bond_query.sqlstatement)
            self.delete_curve()
            return
        for sub_dataframe in dataframe:
            self.curveName = str(sub_dataframe['CurveShortName'].values[0])
            self.write_curve(sub_dataframe)

    def write_curve(self, dataframe):
        self.delete_curve()
        grouped = dataframe.groupby('TradeDate')
        sqlList=[]
        curve_node_df = self.get_curve_node_id(self.curveName)
        CustomTimeInYearsList=curve_node_df['InYears'].tolist()

        for name, df in grouped:
            # try:
            tradeDate = pd.to_datetime(name).strftime('%Y-%m-%d')
            conventions = bond_utils.get_BEI_conventions(self.currency)
            bootstrapper = Bootstrapper_BEI(tradeDate, tradeDate, self.currency, conventions, self.enviroment, self.config, df)
            yieldcurve = bootstrapper.createYieldCurve()

            nominalCurveName = bond.convert_gvt_curve(self.currency, self.countryName)
            nominal_curveNodeQuote_df = db.CurveNodeQuoteFinal(curveName = nominalCurveName, startDate = tradeDate,database =self.enviroment, isFinalTable=False)
            nominal_curveNodeQuote = nominal_curveNodeQuote_df.extract_from_db()
            nominal_curveNodeQuote_adjusted = nominal_curveNodeQuote[nominal_curveNodeQuote['InYears'].isin(CustomTimeInYearsList)]
            nominal_yield_list = nominal_curveNodeQuote_adjusted['Quote'].values[0:]
            real_yield_list = yieldcurve(CustomTimeInYearsList)
            yieldList = [x-y for x,y in zip(nominal_yield_list, real_yield_list)]

            if self.production_table:
                dictionary = []
                for ttm, level in zip(CustomTimeInYearsList[0:], yieldList[0:]):
                    try:
                        dictionary.append({'CurveNodeId': curve_node_df[curve_node_df.InYears == ttm]['CurveNodeId'].values[0],
                                    'TradeDate': tradeDate,
                                    'Quote': level,
                                    'IsCorrected': '',
                                    'Lud': str(datetime.datetime.now())[0:-3],
                                    'Lub': getpass.getuser()})
                    except Exception as e:
                        print "Exception thrown on append:", str(e)
                        continue
            else:
                dictionary = [{'CurveName':self.curveName,
                               'Category':'AxiomaBootStrapper',
                               'TimeInYears':ttm,
                               'Level':level,
                               'TradeDate':tradeDate,
                               'Lud': str(datetime.datetime.now())[0:-3],
                               'Lub': getpass.getuser()} for ttm, level in zip(CustomTimeInYearsList[0:],yieldList[0:])]
            sqlList.extend([self.create_yield_sql(x) for x in dictionary])
            # except Exception as e:
            #     print "Yield Curve Failed on ", name
            #     print e
            #     # print e.message
            #     continue

        sqlstatement='\n'.join(sqlList)
        try:
            db.MSSQL.execute_commit(sqlstatement, self.enviroment, self.config)
        except Exception as e:
            print "Exception thrown commit:", e.message, "SQL execute commit:" , sqlstatement
        # return

def write_yields_to_dbs(args):
    
    startDate = args[0]
    endDate = args[1]
    currency = args[2]
    country = args[3]
    enviroment = args[4]
    config = args[5]
    production_table = args[6]
    input_bond_table = args[7]
    bei = args[8]

    if bei == True:
        yieldsWriter = BEI_Curve(startDate, endDate, currency,country, enviroment, config, production_table, input_bond_table)
    else:
        yieldsWriter = SovereignCurve(startDate, endDate, currency,country, enviroment, config, production_table, input_bond_table)
    try:
        yieldsWriter.write_to_dbs()
    except Exception as e:
        print traceback.print_exc()
        print 'Bootstrap failed for tradeDate:{}, currency:{}, country:{}'.format(startDate + ':' + endDate,
                                                                                  currency, country)
    return None

if __name__=='__main__':
    import concurrent.futures   
    import cProfile

    parser = OptionParser()
    parser.add_option("-s", "--startDate", dest="startDate", help="Starting trade date to run curve generator", metavar="2014-05-01 example")
    parser.add_option("-e", "--endDate", dest="endDate", help="End trade date to run curve generator", metavar="2014-05-02 example")
    parser.add_option("-c", "--currency", dest="currency", help="currency to run curve generator", metavar="example USD")
    parser.add_option("-q", "--country", dest="country", help="countryName to run curve generator", metavar="example US")
    parser.add_option('-d', '--environment', dest='environment', help='enviroment name', metavar='example DEV')
    parser.add_option('-g', '--config', dest='config', help='configuration file name', metavar='example database.config')
    parser.add_option('-b', '--bei', dest='bei',help='BEI Curve Bootstrapping',metavar='example -b on')
    parser.add_option("-p", "--parallel", action="store_true", dest="parallelize", help="runs computation across all cores", metavar=" example -p off")
    parser.add_option("-t", "--production_table", action="store_true", dest="production_table", help="store results in Curve Node Quotes instead of research")
    parser.add_option("-y", "--input_bond_table", action="store_true", dest="input_bond_table", help="Use the research input bond set or production")
    (options, args) = parser.parse_args()
    
    currency = options.currency
    country = options.country
    startDate = options.startDate
    endDate = options.endDate
    environment = options.environment
    config = options.config
    if options.production_table:
        production_table = True
    else:
        production_table = False
    if options.input_bond_table:
        input_bond_table = False
    else:
        input_bond_table = True
    if options.bei:
        bei = True
    else:
        bei = False

    timerange = pd.bdate_range (startDate, endDate)
    timerangelist = timerange.tolist()
    splitted_list = split_list(timerangelist)
    arg_list = [(x[0].strftime('%Y-%m-%d'),
                 x[-1].strftime('%Y-%m-%d'),
                 currency,
                 country,
                 environment,
                 config,
                 production_table,
                 input_bond_table,
                 bei
                 ) for x in splitted_list]
    
    startTime = get_current_time()
    print "startTime: ", startTime
    if options.parallelize:
        print "running single process"
        map(write_yields_to_dbs, arg_list)
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for (x, y) in zip(arg_list, executor.map(write_yields_to_dbs, arg_list)):
                    print "Ran: ", x[0], x[1]        
        except Exception as e:
            print "Uncaught exception in main executor loop"
            print traceback.print_exc()
    
    endTime = get_current_time()
    print "startTime: ", startTime, "endTime: ", endTime
