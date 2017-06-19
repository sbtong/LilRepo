
"""database.py

This module offers database connectivity to Axioma's primary database services: Oracle, MSSQL Server 
Example:

    import maclab.utils.database as db
    sqlstatement = 'select top 1 * from MarketData.dbo.Curve'
    results = db.MSSQL.extract_dataframe(sqlstatement)
    results.head()



"""
import os
import sys
import Utilities
from macpy.curve_utility import getMSSQLDatabaseInfo, getAnalysisConfigSection
import pandas.io.sql as sql
import string
import os
import inspect
import getpass
import datetime
from contextlib import contextmanager
import pymssql

EURO_CURVES = [ 
    'AT.EUR.GVT.ZC',
    'BE.EUR.GVT.ZC',
    'CY.EUR.GVT.ZC',
    'DE.EUR.GVT.ZC',
    'EP.EUR.GVT.ZC',
    'ES.EUR.GVT.ZC',
    'FI.EUR.GVT.ZC',
    'FR.EUR.GVT.ZC',
    'GR.EUR.GVT.ZC',
    'IE.EUR.GVT.ZC',
    'IT.EUR.GVT.ZC',
    'LT.EUR.GVT.ZC',
    'MT.EUR.GVT.ZC',
    'NL.EUR.GVT.ZC',
    'PT.EUR.GVT.ZC',
    'SK.EUR.GVT.ZC']

class DatabaseExtract(object):
    """
    Mixin class for database extraction
    """
    def __init__(self):
        self.sqlstatement = None

    def extract_from_db(self):
        if hasattr(self, 'sqlstatements'):
            return [MSSQL.extract_dataframe(sql, self.database, self.config) for sql in self.sqlstatements]
        return MSSQL.extract_dataframe(self.sqlstatement, self.database, self.config)

class MSSQL(object):

    @classmethod
    def extract_dataframe(cls, sqlstatement, environment=None, filename=None):
        current_module = sys.modules[__name__]
        module_path = inspect.getabsfile(current_module)
        module_dir = os.path.dirname(module_path)
        #print module_dir
        if filename is None:
            filename = os.path.join(module_dir, 'database.config')
            filenamefull = os.path.realpath(filename)
        else:
            if os.path.isfile(filename):
                filenamefull = filename
            else:
                filenamefull = os.path.realpath(os.path.join(module_dir, filename))
        if not os.path.isfile(filenamefull):
            raise ValueError(filenamefull + ' is not a path that resolves to a file. Hello')
        environ = 'DEV' 
        if environment is not None:
            environ = environment
        dbconnection = DatabaseConfigurator.create_database_connection(filenamefull, environ)
        df = dbconnection.create_dataframe_mssql(sqlstatement)
        return df

    @classmethod
    def execute_commit(cls, sqlstatement, environment=None, filename=None):
        current_module = sys.modules[__name__]
        module_path = inspect.getabsfile(current_module)
        module_dir = os.path.dirname(module_path)
        if filename is None:
            filename = os.path.join(module_dir, 'test.config')
            filenamefull = os.path.realpath(filename)
        else:
            if os.path.isfile(filename):
                filenamefull = filename
            else:
                filenamefull = os.path.realpath(os.path.join(module_dir, filename))
        if not os.path.isfile(filenamefull) :
            raise ValueError(filenamefull + ' is not a path that resolves to a file.')
        environ = 'DEV' 
        if environment is not None:
            environ = environment
        dbconnection = DatabaseConfigurator.create_database_connection(filenamefull, environ)
        dbconnection.execute_sql(sqlstatement)
        #dbconnection.commit()



class DatabaseConfigurator:
    """
    Adds databaseConfig and environment to an option parser
    """
    @classmethod
    def add_option_argument(cls, optionParser):
        optionParser.add_argument('-e', '--environment', help="Environment to use. Available options: PROD, UAT, DEV. Default is DEV", action="store", default='DEV')
        optionParser.add_argument("databaseConfig", help="Input configuration file containing database connection info.", action="store")
    
    """
    Configure Logger
    """
    @classmethod
    def create_database_connection(cls, filename, environment):
        configFile = open(filename, 'r')
        configuration = Utilities.loadConfigFile(configFile)
        configFile.close()
        cursor = DatabaseConnection(configuration, environment)
        return cursor 

    @classmethod
    def create_database_connection_from_commandline(cls, cmdlineargs):
        return cls.create_database_connection(cmdlineargs.databaseConfig, cmdlineargs.environment)


class DatabaseConnection(object):
    def __init__(self, configuration, environment):
        self.configuration = configuration
        self.environment = environment.replace('\'', '').replace('"', '').strip().upper()
        self.sectionID = 'CurvesDB%s'%(self.environment)
        if(self.environment == 'DEV'):
            self.envInfoMap = Utilities.getConfigSectionAsMap(self.configuration, self.sectionID)        
            self.host = self.envInfoMap.get('host', None)
            self.user = self.envInfoMap.get('user', None) 
            self.pwd = self.envInfoMap.get('password', None)
            self.database = self.envInfoMap.get('database', None)
            self.dbInfo = Utilities.DatabaseInfo(self.host, self.user, self.pwd, self.database)
        else:
            self.envInfoMap = Utilities.getConfigSectionAsMap(self.configuration, self.sectionID)        
            self.macdb = self.envInfoMap.get('macdb', None)
            self.infoMap = Utilities.getConfigSectionAsMap(self.configuration, self.macdb)
            self.host = self.infoMap.get('host', None)
            self.user = self.infoMap.get('user', None) 
            self.pwd = self.infoMap.get('password', None)
            self.database = self.infoMap.get('database', None)
            self.dbInfo = Utilities.DatabaseInfo(self.host, self.user, self.pwd, self.database)
    
    def create_dataframe_mssql(self, sqlstatement):
        connection = Utilities.createMSSQLConnection(self.dbInfo)
        try:
            df = sql.read_frame(sqlstatement, connection)
        except:
            df = sql.read_sql(sqlstatement, connection)
        if pymssql.__version__ == '2.1.3':
            for dtype, col in zip(df.dtypes, df.columns):
                if dtype == 'object':
                    #UnicodeEncodeError: 'ascii' codec can't encode character u'\xa0' in position 36: ordinal not in range(128)
                    try:
                        df[col] = df[col].astype(str)
                    except:
                        df[col] = df[col].str.encode('utf-8')
        connection.commit()
        connection.close()
        return df

    def execute_sql(self, sqlstatement):
        connection = Utilities.createMSSQLConnection(self.dbInfo) 
        cursor = connection.cursor()
        cursor.execute(sqlstatement)
        connection.commit()        
        connection.close()

    def create_dictionary_mssql(self, sqlstatement):
        connection = Utilities.createMSSQLConnection(self.dbInfo) 
        cursor = connection.cursor()
        cursor.execute(sqlstatement)
        resultsDB = cursor.fetchall()
        connection.close()
        return resultsDB


class CurveNodeQuoteFinal(DatabaseExtract):
    """
    YieldCurve wraps database calls to retrieve Quotes
    """
    def __init__(self, curveName, startDate, endDate = None, database = 'Dev', config=None, isFinalTable=True):
        self.startDate = startDate
        self.endDate = startDate if endDate is None else endDate
        self.curveName = curveName
        self.database = database
        self.config = config
        self.quoteTable = 'CurveNodeQuoteFinal' if isFinalTable else 'CurveNodeQuote'
        self.sqlstatement = string.Template("""
select cq.TradeDate, te.InYears, cq.Quote
from MarketData.dbo.Curve cv
join MarketData.dbo.CurveNodes cn on cv.curveid=cn.CurveId
join MarketData.dbo.$quoteTable cq on cq.CurveNodeId = cn.CurveNodeId
join MarketData.dbo.TenorEnum te on cn.TenorEnum = te.TenorEnum
where cv.CurveShortName = '$curveName'
and cq.TradeDate >= '$startDate'
and cq.TradeDate <= '$endDate'
order by te.InYears""").substitute({'curveName': curveName, 'startDate': self.startDate, 'endDate': self.endDate, 'quoteTable': self.quoteTable})

    def __str__(self):
        return  "CurveQuote(startDate='%s', curveName='%s') \n%s" % (
            self.startDate, self.curveName, self.sqlstatement)

    def __repr__(self):
        return self.__str__()

class CurveNodeQuoteHist(DatabaseExtract):
    def __init__(self, trade_date, curve_node_id, database='DEV', config=None):
        self.trade_date = trade_date
        self.curve_node_id = curve_node_id
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
            SELECT TradeDate, Quote from marketdata.dbo.CurveNodeQuote 
            WHERE CurveNodeID = $curve_node_id 
            AND TradeDate > '$trade_date'""").substitute({'curve_node_id': curve_node_id, 
                                                          'trade_date': trade_date})

class CurveNodeQuote(DatabaseExtract):
    def __init__(self, startDate, endDate, curveName, database = 'Dev',config=None):
        self.startDate = startDate
        self.endDate = endDate
        self.curveName = curveName
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
select cq.TradeDate, te.InYears, cq.Quote
from MarketData.dbo.Curve cv
join MarketData.dbo.CurveNodes cn on cv.curveid=cn.CurveId
join MarketData.dbo.CurveNodeQuoteFinal cq on cq.CurveNodeId = cn.CurveNodeId
join MarketData.dbo.TenorEnum te on cn.TenorEnum = te.TenorEnum
where cv.CurveShortName = '$curveName'
and cq.TradeDate >= '$startDate' and  cq.TradeDate <= '$endDate'
order by te.InYears""").substitute({'curveName': curveName, 'startDate': startDate, 'endDate':endDate})

    def __str__(self):
        return  "CurveQuote(valuationDate='%s', curveName='%s') \n%s" % (
            self.valuationDate, self.curveName, self.sqlstatement)

    def __repr__(self):
        return self.__str__()

class DerivedCurveFilteredBond(DatabaseExtract):

    def __init__(self, TradeDate, CurveName, database=None, config=None):
        self.TradeDate = TradeDate
        self.CurveName = CurveName
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
select cv.CurveShortName, fb.curveid, fb.InstrCode, fb.TradeDate, prc.Prc, prc.MatCorpYld, md.MatDate
from DerivCurveFilteredBond fb
join Curve cv on cv.curveid = fb.CurveId and cv.CurveShortName= '$CurveName'
join [QAI].[dbo].FIEJVPRCDly prc on prc.TradeDate = fb.TradeDate and prc.InstrCode = fb.InstrCode
join [QAI].[dbo].FIEJVSecInfo md on md.InstrCode = fb.InstrCode
and fb.itemid=2
and fb.TradeDate = '$TradeDate'
        """ ).substitute({'CurveName':CurveName, 'TradeDate':TradeDate})

class DerivedCurveFilteredBondResearch(DatabaseExtract):

    def __init__(self, currency, trade_date,  database=None, config=None):
        self.trade_date = trade_date
        self.currency = currency
        self.database = database
        self.config = config
        self.curve_name = convert_currency_to_curveName(currency)
        self.sqlstatement = string.Template("""
            select * from MarketData.dbo.DerivCurveFilteredBond_Research fb
            join Curve cv on cv.curveid = fb.CurveId and cv.CurveShortName= '$curve_name'
            where cv.CurveId= fb.CurveId and TradeDate = '$trade_date'
        """ ).substitute({'curve_name':self.curve_name, 'trade_date':self.trade_date})

class RiskEntity(DatabaseExtract):
    def __init__(self, issuer_id=None, ISIN=None, database=None, config=None):
        self.issuer_id = issuer_id
        self.ISIN = ISIN
        self.database = database
        self.config = config
        if self.issuer_id:
            self.sqlstatement = string.Template("""exec GetEntityHierarchy @IssuerId = $issuer_id""").substitute({'issuer_id':self.issuer_id})
        if self.ISIN:
            self.sqlstatement = string.Template("""exec GetEntityHierarchy @ISIN = '$ISIN'""").substitute({'ISIN':self.ISIN})

class ResearchCurves(DatabaseExtract):

    def __init__(self, startDate, endDate, curveName, database = None, config=None):
        self.startDate = startDate
        self.endDate = endDate
        self.curveName = curveName
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
select TradeDate, CurveName, Category, TimeInYears, Level, Lud, Lub
from [MarketData].[dbo].[ResearchCurves]
where '$StartDate' <= TradeDate and '$EndDate' >= TradeDate
and CurveName = '$CurveName'
order by TradeDate, Category, CurveName, TimeInYears
        """ ).substitute({'CurveName':self.curveName, 'StartDate':self.startDate, 'EndDate':self.endDate})

class ReutersCurves(DatabaseExtract):
    def __init__(self,startDate, endDate, curveName, database=None, config=None):
        self.startDate = startDate
        self.endDate = endDate
        self.curveName = curveName
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
        SELECT date AS TradeDate, term AS InYears, spot/100.0 AS Quote, hist.curve_id, c.curve_id,c.long_name
        from ejv_rigs..curves c
        join ejv_ejv_common..analytic_yield_curve_hist hist on hist.curve_id = c.curve_id
        WHERE
        c.[curve_sub_cd] = 'GVBM' and
        c.cntry_cd ='$Country'  and
		c.currency_cd='$Currency' and
        date >= '$StartDate' and
        date <= '$EndDate'
        """).substitute({'StartDate':self.startDate, 'EndDate':self.endDate, 'Country':self.curveName[0:2], 'Currency':self.curveName[3:6]})

class TQAChainRics(DatabaseExtract):
    def __init__(self, database=None, config=None):
        self.database = database
        self.config = config
        self.sqlstatement = """
            select c.curve_id, c.chain_ric, p.rt_ric as Ric, Te.InYears, c.long_name
            from EJV_rigs.dbo.curves c
            JOIN EJV_rigs.dbo.points p on p.curve_id = c.curve_id
            JOIN MarketData.dbo.TenorEnum te on te.TenorEnum = p.short_name
            WHERE DATALENGTH(p.rt_ric) > 0
            AND DATALENGTH(c.chain_ric) > 0"""

class ReutersCurvesQuotes(DatabaseExtract):
    def __init__(self,startDate, endDate, ric_df, database=None, config=None):
        self.startDate = startDate
        self.endDate = endDate
        self.database = database
        self.config = config
        self.vendor_ids = [str(s.__repr__()) + ',' for s in list(ric_df['Ric'])]
        self.vendor_ids[-1] = self.vendor_ids[-1][:-1]
        self.vendor_ids = ''.join(self.vendor_ids)
        self.sqlstatement = string.Template("""
    select q.[Trade Date] as TradeDate, q.Yield/100.0 as Quote, q.ric as Ric
            FROM [PROD_VNDR_DB].DataScopeSelect.dbo.TRCurveNodeTS q
            where q.ric in ($vendor_ids)
            and q.[Trade Date] between '$StartDate' and '$EndDate'
            and q.Yield not in (-9999401,-9999402)
        """).substitute({'StartDate':self.startDate, 'EndDate':self.endDate, 'vendor_ids':self.vendor_ids})

class OISCurves(DatabaseExtract):
    def __init__(self, database=None, config=None):
        self.database = database
        self.config = config
        self.sqlstatement = 'select * from MarketData.dbo.OvernightIndexSwap'

class _OISCurvesInternalID(DatabaseExtract):
    def __init__(self, curve_name, database=None, config=None):
        self.database = database
        self.config = config
        self.curve_name = curve_name
        self.sqlstatement = string.Template("""
            SELECT oisn.VendorDataInternalID as Ric, Te.InYears
            FROM MarketData.dbo.OvernightIndexSwap ois
            JOIN MarketData.dbo.OvernightIndexSwapNodes oisn on oisn.OISID = ois.OISID
            JOIN MarketData.dbo.TenorEnum te on te.TenorEnum = oisn.TenorEnum
            where ois.OISShortName = '$CurveName'""").substitute({'CurveName':self.curve_name})

class OISCurveQuote(DatabaseExtract):
        def __init__(self, start_date, end_date, curve_name, database=None, config=None):
            self.database = database
            self.config = config
            self.start_date = start_date
            self.end_date = end_date
            self._dbconn = _OISCurvesInternalID(curve_name, database, config)
            self._df = self._dbconn.extract_from_db()
            self.vendor_ids = [str(s.__repr__()) + ',' for s in list(self._df['Ric'])]
            self.vendor_ids[-1] = self.vendor_ids[-1][:-1]
            self.vendor_ids = ''.join(self.vendor_ids)
            self.sqlstatement = string.Template("""
                select q.[Trade Date] as TradeDate, "Bid Yield" as Quote, q.Ric
                from [PROD_VNDR_DB].DataScopeSelect.dbo.ois_zc_yield q
                where q.[Trade Date] between '$StartDate' and '$EndDate'
                and q.RIC in ($vendor_ids)
                and q.[Bid Yield] not in (-9999401,-9999402)
                """).substitute({'StartDate':self.start_date, 'EndDate':self.end_date, 'vendor_ids':self.vendor_ids})

class BasisSwapCurves(DatabaseExtract):
    def __init__(self, database=None, config=None):
        self.database = database
        self.config = config
        self.sqlstatement = 'select * from MarketData.dbo.BasisSwap'

class _BasisSwapInternalID(DatabaseExtract):
    def __init__(self, curve_name, database=None, config=None):
        self.database = database
        self.config = config
        self.curve_name = curve_name
        self.sqlstatement = string.Template("""
            SELECT bsn.VendorDataInternalID as Ric, Te.InYears
            FROM MarketData.dbo.BasisSwap bs
            JOIN MarketData.dbo.BasisSwapNodes bsn on bsn.BasisSwapID = bs.BasisSwapID
            JOIN MarketData.dbo.TenorEnum te on te.TenorEnum = bsn.TenorEnum
            where bs.BasisSwapShortName = '$CurveName'""").substitute({'CurveName':self.curve_name})

class BasisSwapQuote(DatabaseExtract):
        def __init__(self, start_date, end_date, curve_name, database=None, config=None):
            self.database = database
            self.config = config
            self.start_date = start_date
            self.end_date = end_date
            self._dbconn = _BasisSwapInternalID(curve_name, database, config)
            self._df = self._dbconn.extract_from_db()
            self.vendor_ids = [str(s.__repr__()) + ',' for s in list(self._df['Ric'])]
            self.vendor_ids[-1] = self.vendor_ids[-1][:-1]
            self.vendor_ids = ''.join(self.vendor_ids)
            self.sqlstatement = string.Template("""
                select [Trade Date] as TradeDate, "Bid Price" as Quote, RIC as Ric
                FROM [PROD_VNDR_DB].DataScopeSelect.dbo.BasisSwaps 
                WHERE [Trade Date] between '$StartDate' and '$EndDate'
                and RIC in ($vendor_ids)
                --and q.[Bid Price] not in (-9999401,-9999402)
                """).substitute({'StartDate':self.start_date, 'EndDate':self.end_date, 'vendor_ids':self.vendor_ids})
            
class EMHardSvSprCurves(DatabaseExtract):
    # Hard CCY
    def __init__(self, country, currency, TradeDate, database = None, config=None):
        self.country = country
        self.currency = currency
        self.database = database
        self.config = config
        self.TradeDate = TradeDate
        self.curve_short_name = '{}.{}.SOVSPR'.format(self.country, self.currency)
        if self.currency == 'USD':
            self.country_type_enum = 'SvSprEmg'
        elif self.currency == 'EUR':
            self.country_type_enum = 'SVSPR'
        self.sqlstatement = string.Template("""
select s.[AnalysisDate],
       s.[ISIN],
       s.[CalibratedPrice],
       s.[OAS],
       s.[AccruedInterest],
       s.[OAS_Govt],
       s.[OAS_Swap],
       s.[OAS_Riskfree],
       s.[OAS_Discount],
       s.[ValuationType],
       s.[Elapsed],
       s.[Effective_Duration],
       s.[Spread_Duration],
       s.InstrCode,
       s.Maturity,
       s.ISOCtryCode,
       s.Residual,
       s.Quote,
       Amt = amt.value_,
       RemoveOnDate = NULL,
       RemoveForward = NULL
into #ddd
from
     (
      select
         o.AnalysisDate,
         o.ISIN,
         o.CalibratedPrice,
         o.OAS,
         o.AccruedInterest,
         o.OAS_Govt,
         o.OAS_Swap,
         o.OAS_Riskfree,
         o.OAS_Discount,
         o.ValuationType,
         o.Elapsed,
         o.Effective_Duration,
         o.Spread_Duration,
         o.InstrCode,
         Maturity = DateDiff(d, o.AnalysisDate, sec.MatDate)/365.25,
         org.ISOCtryCode,
         o.OAS_Govt - c.Quote as Residual,
         c.Quote
      from
          (select cv.CurveId, cq.TradeDate, cq.Quote
           from MarketData..Curve cv,
                MarketData..curvenodequote cq ,
                MarketData..CurveNodes cn
           where  cv.CurveId = cn.CurveId
             and cn.CurveNodeId     = cq.CurveNodeId
             and cv.CurveShortName  = '$curve_short_name'
             and cv.curvetypeenum   = '$country_type_enum'
             and cv.CurrencyEnum    = '$currency'
          ) c,

           [MarketData].[dbo].[DerivCurveBondStatistics] o,
           qai.dbo.FIEJVSecInfo sec ,
           [QAI].[dbo].FIEJVOrgInfo org

      where sec.InstrCode      = o.InstrCode
        and   org.GemOrgId       = sec.IssrGEMOrgId
        and   sec.DebtISOCurCode = '$currency'
        and   c.TradeDate        = o.AnalysisDate
        and not exists (select 1 from [MarketData].[dbo].DerivCurveFilteredBondCorrection fbday
                        where  fbday.InstrCode = o.InstrCode and c.CurveId = fbday.CurveId
                        and    o.AnalysisDate = fbday.TradeDate and fbday.ItemId=1)
        and not exists (select 1 from [MarketData].[dbo].DerivCurveFilteredBondCorrection fbfwd
                        where  fbfwd.InstrCode = o.InstrCode and c.CurveId = fbfwd.CurveId
                        and    o.AnalysisDate > fbfwd.TradeDate and fbfwd.ItemId=14)
        and org.ISOCtryCode = '$country'
        and o.AnalysisDate >= '2006-01-01'
        and o.AnalysisDate <= '$TradeDate'
         ) s,
       [QAI].[dbo].FIEJVSecAmt amt
where amt.Item       = 191
  and amt.InstrCode  = s.InstrCode
  and amt.AmtDate    = (select max(AmtDate)
                        from [QAI].[dbo].FIEJVSecAmt amt2
                         where amt2.InstrCode  = s.InstrCode
                           and amt2.Item         = 191
                           and amt2.AmtDate      <= s.AnalysisDate)

select * from #ddd order by AnalysisDate""").substitute({'curve_short_name':self.curve_short_name, 
                                                         'country': self.country, 
                                                         'TradeDate':self.TradeDate, 
                                                         'currency':self.currency,
                                                         'country_type_enum':self.country_type_enum})

class EMLocalSvSprCurves(DatabaseExtract):
    def __init__(self, curve_short_name, database = None, config=None):
        self.curve_short_name = curve_short_name
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""SELECT [AnalysisDate]
  , cv.CurveId
  , o.CurveShortName
  ,[ISIN]
  ,o.InstrCode
  ,[CalibratedPrice]
  ,[OAS]
  ,[AccruedInterest]
  ,[OAS_Govt]
  ,[Effective_Duration]
  ,TermInYears as Maturity
  ,[OAS_Swap]
  ,[OAS_Riskfree]
  ,[OAS_Discount]
  ,o.[Lud]
  ,o.[Lub]
  ,[ValuationType]
  ,[Elapsed]
  ,[Spread_Duration]
  ,[SpreadKeyDuration]
  ,[KeyRate1M]
  ,[KeyRate6M]
  ,[KeyRate1Y]
  ,[KeyRate2Y]
  ,[KeyRate5Y]
  ,[KeyRate10Y]
  ,[KeyRate30Y]
  ,[DebtIsoCurCode]
  , RemoveOnDate = fbday.InstrCode 
  , RemoveForward = fbfwd.InstrCode
  FROM [MarketData].[dbo].[DerivCurveBondStatistics] o
  LEFT JOIN [MarketData].[dbo].Curve cv on cv.CurveShortName = o.CurveShortName
  LEFT JOIN [MarketData].[dbo].DerivCurveFilteredBondCorrection fbday on fbday.InstrCode = o.InstrCode and cv.CurveId = fbday.CurveId and o.AnalysisDate = fbday.TradeDate and fbday.ItemId=1
  LEFT JOIN [MarketData].[dbo].DerivCurveFilteredBondCorrection fbfwd on fbfwd.InstrCode = o.InstrCode and cv.CurveId = fbfwd.CurveId and o.AnalysisDate > fbfwd.TradeDate and fbfwd.ItemId=14
  WHERE fbday.InstrCode is null
  AND fbfwd.InstrCode is null 
  AND o.CurveShortName = '$curve_short_name'
  ORDER BY AnalysisDate, InstrCode""").substitute({'curve_short_name': self.curve_short_name})


class ResearchBondStatsTimeSeries(DatabaseExtract):
    def __init__(self, tradeStartDate, tradeEndDate, database = None, config=None):
        self.tradeStartDate = tradeStartDate
        self.tradeEndDate = tradeEndDate
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
[MarketData].[dbo].[ResearchBondStatsTimeSeries] @tradeDateBegin='$tradeBeginDate', @tradeDateEnd='$tradeEndDate',@statcode='5'
                    """).substitute({'tradeBeginDate':self.tradeStartDate, 'tradeEndDate':self.tradeEndDate})

class DerivedDataCorpBondPriceForCcy(DatabaseExtract):
    def __init__(self, valuationDate, currency, endTradeDate=None, issuerId=None, instrumentCode=None, rating=None, sector=None, suffix='', database = None, config=None):
        self.valuationDate = valuationDate
        self.currency = currency
        self.database = database
        self.config = config
        self.endTradeDate = endTradeDate if endTradeDate is not None else self.valuationDate
        self.sector = 'null' if sector is None else "'" + sector + "'"
        self.rating = 'null' if rating is None else "'" + rating + "'"
        self.issuerId = 'null' if issuerId is None else "'" + issuerId + "'"
        self.instrumentCode = 'null' if instrumentCode is None else "'" + instrumentCode + "'"
        self.suffix = suffix
        self.sqlstatement = string.Template ("""
exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcy$suffix] @tradeDateBegin='$valuationDate',  @tradeDateEnd='$endTradeDate',  @currencyISO='$currency', @issuerId=$issuerid, @InstrCode=$instrumentCode, @RatingEnum = $rating, @GicsLevel1 = $sector
                """).substitute({'valuationDate': valuationDate, 'currency': currency, 'endTradeDate': self.endTradeDate, 'issuerid':self.issuerId, 'instrumentCode': self.instrumentCode, 'rating': self.rating, 'sector': self.sector, 'suffix': self.suffix})

class DerivedDataCorpBondPriceForCcyResearch(DatabaseExtract):
    def __init__(self, valuationDate, currency, endTradeDate=None, issuerId=None, instrumentCode=None, rating=None, sector=None, suffix='', database = None, config=None):
        self.valuationDate = valuationDate
        self.currency = currency
        self.database = database
        self.config = config
        self.endTradeDate = endTradeDate if endTradeDate is not None else self.valuationDate
        self.sector = 'null' if sector is None else "'" + sector + "'"
        self.rating = 'null' if rating is None else "'" + rating + "'"
        self.issuerId = 'null' if issuerId is None else "'" + issuerId + "'"
        self.instrumentCode = 'null' if instrumentCode is None else "'" + instrumentCode + "'"
        self.suffix = suffix
        self.sqlstatement = string.Template ("""
exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyResearch$suffix] @tradeDateBegin='$valuationDate',  @tradeDateEnd='$endTradeDate',  @currencyISO='$currency', @issuerId=$issuerid, @InstrCode=$instrumentCode, @RatingEnum = $rating, @GicsLevel1 = $sector
                """).substitute({'valuationDate': valuationDate, 'currency': currency, 'endTradeDate': self.endTradeDate, 'issuerid':self.issuerId, 'instrumentCode': self.instrumentCode, 'rating': self.rating, 'sector': self.sector, 'suffix': self.suffix})

class DerivedDataExtractSovereignCurveBenchmarkBonds(DatabaseExtract):
    def __init__(self, valuationDate, curveId, database=None, config=None):
        self.valuationDate = valuationDate
        self.database = database
        self.config = config
        self.curveId = curveId #convert_currency_to_government_benchmark_curveid(currency)
        self.sqlstatement = string.Template("""
exec [MarketData].[dbo].[DerivedDataExtractSovereignCurveBenchmarkBonds] @CurveIdIn=$curveId, @TradeDateIn='$valuationDate'
            """).substitute({'valuationDate': self.valuationDate, 'curveId': self.curveId})

class ExtractCurveIdforForeignCurrencyDominatedSovereignCurve(DatabaseExtract):
    def __init__(self, CurveName, database = None, config=None):
        self.curveName = CurveName
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
        SELECT * FROM MarketData.dbo.Curve where CurveShortName = '$CurveShortName'
        """).substitute({'CurveShortName':self.curveName})


class DerivedDataExtractSoverignCurveByInstrCodes(DatabaseExtract):
    def __init__(self, valuationDate, InstrCodes, database = None, config=None):
        self.valuationDate = valuationDate
        self.InstrCodes = InstrCodes
        self.database = 'DEV' if database is None else database
        self.config = config
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' if self.database.upper() == 'DEV' else 'sec.FrstCpnRate'
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' 
        self.sqlstatement = string.Template("""
            select  distinct(sec.InstrCode),       
                    sec.IssName,
                    cal.DomSettDays,
                    cal.DomSettCalCode,
                    p.TradeDate,
                    p.Prc as Price,
                    p.MatStdYld,
                    CompFreqCode,
                    cp.NomTermToMat,
                    $cpnFieldName,
                    sec.MatDate,
                    cpn.FrstCpnDate,
                    cpn.LastCpnDate,
                    DayCountCode = ( select d.Desc_ from   [QAI].dbo.FIEJVCode d
                                     where  d.Type_ = 1
                                     and d.Code = cpiv.Value_),
                sec.DebtISOCurCode,
                org.ISOCtryCode,
                sec.InitAccDate,
                sec.DebtIssTypeCode 
                from [QAI].[dbo].FIEJVPrcDly p
                inner join [QAI].[dbo].FIEJVCurveBenchmark cb on p.InstrCode = cb.InstrCode
                inner join [QAI].[dbo].FIEJVCurve cv on cv.CurveId = cb.CurveId
                inner join [QAI].[dbo].FIEJVCurvePoint cp on cp.CurveId = cb.CurveId and cp.PointId = cb.PointId
                inner join [QAI].[dbo].FIEJVCpnHdr as cpn on p.InstrCode = cpn.InstrCode
                inner join [QAI].[dbo].FIEJVSecInfo sec on sec.InstrCode = cb.InstrCode
                inner join [QAI].[dbo].FIEJVCpn cpiv on cpiv.instrcode = cb.InstrCode
                and cpiv.Item = 120
                inner join [QAI].[dbo].FIEJVSecCalendar cal on cal.InstrCode = sec.InstrCode
                inner join [QAI].[dbo].FIEJVOrgInfo org on org.GemOrgId=sec.IssrGEMOrgId
                left outer join MarketData.dbo.DerivedDataExcludedNodes exnodes on cv.ISOCtryCode = exnodes.IsoCtryCode                   
                and cv.ISOCurCode = exnodes.IsoCurCode
                where p.TradeDate = '$valuationDate'
                    and p.InstrCode in ($InstrCodes)
                    and NomTermToMat is not null
                    and ISNULL(exnodes.MinNomTermToMat, 0.0) < cp.NomTermToMat                        
                order by p.TradeDate, NomTermToMat""").substitute(
            {'valuationDate': self.valuationDate,
             'InstrCodes': self.InstrCodes,
             'cpnFieldName': self.cpn_field_name})


class DerivedDataExtractSovereignCurveBenchmarkBondsSQL(DatabaseExtract):
    def __init__(self, valuationDate, curveId, database=None, config=None):
        self.valuationDate = valuationDate
        self.database = 'DEV' if database is None else database
        self.config = config
        self.curveId = curveId
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' if self.database.upper() == 'DEV' else 'sec.FrstCpnRate'
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' 
        self.sqlstatement = string.Template("""
            select  sec.InstrCode,       
                    sec.IssName,
                    cal.DomSettDays,
                    cal.DomSettCalCode,
                    p.TradeDate,
                    p.Prc as Price,
                    p.MatStdYld,
                    CompFreqCode,
                    cp.NomTermToMat,
                    $CpnFieldName,
                    sec.MatDate,
                    cpn.FrstCpnDate,
                    cpn.LastCpnDate,
                    DayCountCode = ( select d.Desc_ from   [QAI].dbo.FIEJVCode d
                                     where  d.Type_ = 1
                                     and d.Code = cpiv.Value_),
                sec.DebtISOCurCode,
                org.ISOCtryCode,
                sec.InitAccDate,
                sec.DebtIssTypeCode 
                from [QAI].[dbo].FIEJVCurveBenchmark cb
                inner join [QAI].[dbo].FIEJVCurve cv on cv.CurveId = cb.CurveId
                inner join [QAI].[dbo].FIEJVCurvePoint cp on cp.CurveId = cb.CurveId and cp.PointId = cb.PointId
                inner join [QAI].[dbo].FIEJVPrcDly p on  p.InstrCode = cb.InstrCode
                inner join [QAI].[dbo].FIEJVCpnHdr as cpn on p.InstrCode = cpn.InstrCode
                and p.TradeDate = '$valuationDate'
                inner join [QAI].[dbo].FIEJVSecInfo sec on sec.InstrCode = cb.InstrCode
                and cb.CurveID = $curveId
                inner join [QAI].[dbo].FIEJVCpn cpiv on cpiv.instrcode = cb.InstrCode
                and cpiv.Item = 120
                inner join [QAI].[dbo].FIEJVSecCalendar cal on cal.InstrCode = sec.InstrCode
                inner join [QAI].[dbo].FIEJVOrgInfo org on org.GemOrgId=sec.IssrGEMOrgId
                left outer join dbo.DerivedDataExcludedNodes exnodes on cv.ISOCtryCode = exnodes.IsoCtryCode                   
                and cv.ISOCurCode = exnodes.IsoCurCode
                where ( cb.EndDate >= '$valuationDate' or cb.EndDate is null)
                    and cb.StartDate <= '$valuationDate'
                    and NomTermToMat is not null
                    and ISNULL(exnodes.MinNomTermToMat, 0.0) < cp.NomTermToMat                        
                order by p.TradeDate, NomTermToMat""").substitute(
            {'valuationDate': self.valuationDate,
             'curveId': self.curveId,
             'CpnFieldName': self.cpn_field_name})


class DerivedFilteredBootstrap(DatabaseExtract):
    def __init__(self, currency, startDate, endDate, countryName = None, curveName = None, database=None, config=None, input_bond_table=True):
        self.currency = currency
        self.countryName = countryName
        self.startDate = startDate
        self.endDate = endDate
        self.database = 'DEV' if database is None else database
        self.config = config
        self.input_bond_table = input_bond_table
        if curveName is not None:
            self.curveName = curveName
        else:
            self.curveName = convert_currency_to_curveName(currency, countryName)
        if self.input_bond_table is True:
            self.bond_table = 'DerivCurveFilteredBond'
        else:
            self.bond_table = 'DerivCurveFilteredBond_Research'
        template = string.Template("""
            select distinct InstrCode= sec.InstrCode, NomTermToMat = DateDiff(d, fb.TradeDate, sec.MatDate)/365.25, sec.IssName, p.TradeDate, Price= p.Prc, PriceInclAccrIntFlg = ii.px_incl_accr_int_fl, p.MatStdYld, sec.DebtISOCurCode, cv.CurveShortName,
            $CpnFieldName, cal.DomSettDays, sec.MatDate, sec.InitAccDate, sec.DebtIssTypeCode, CompFreqCode = cpn.Value_, c.FrstCpnDate, c.LastCpnDate, sec.EJVAssetID
            from [MarketData].[dbo].[$bond_table] fb
            join [MarketData].[dbo].[Curve] cv on cv.curveid = fb.curveid
            join [QAI].[dbo].FIEJVPrcDly p on p.InstrCode = fb.InstrCode and p.TradeDate = fb.TradeDate
            join [QAI].[dbo].FIEJVSecInfo sec on sec.InstrCode = fb.InstrCode
            join [QAI].[dbo].FIEJVCpnHdr c on c.InstrCode = fb.InstrCode
            left join [QAI].[dbo].FIEJVCpn cpn  on cpn.InstrCode = fb.InstrCode
                                                                and cpn.Item = 129 --CouponPaymentFrequencyCode
                                                                and cpn.CpnLegNum = 1
            left join [QAI].[dbo].FIEJVSecCalendar cal on  cal.InstrCode = fb.InstrCode
            LEFT  JOIN EJV_GovCorp.dbo.orig_iss_info ii on convert(varbinary(max), sec.EJVAssetId, 1) = ii.asset_id
            where fb.TradeDate >= '$StartDate' and fb.TradeDate <= '$EndDate'
            and cv.CurveShortName = '$CurveName'
            and fb.ItemID = 1
            order by TradeDate, NomTermToMat""")
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' if self.database.upper() == 'DEV' else 'sec.FrstCpnRate'
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' 
        self.sqlstatement = template.substitute(
            {'bond_table': self.bond_table, 'CurveName': self.curveName, 'StartDate': self.startDate,
             'EndDate': self.endDate, 'CpnFieldName': self.cpn_field_name})

        if self.curveName == 'EP.EUR.GVT.ZC':
            self.sqlstatements = []
            self.curveName = EURO_CURVES
            for curve in self.curveName:
                self.sqlstatements.append(template.substitute({'bond_table': self.bond_table,
                                                               'CurveName': curve,
                                                               'StartDate': self.startDate,
                                                               'EndDate': self.endDate,
                                                               'CpnFieldName': self.cpn_field_name}))


class ExtractBootstrappedCurve(DatabaseExtract):
    def __init__(self, startDate, endDate, curveName, database='PROD', config=None):
        self.startDate = startDate
        self.endDate = endDate
        self.curveName = curveName
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
        select * from Marketdata.dbo.ResearchCurves
         where TradeDate >='$startDate' and TradeDate <='$endDate'
         and CurveName = '$curveName'
         --and Category = 'AxiomaBootStrapper'
         --and Level>2.0 or Level <-2.0
         order by TradeDate""").substitute({'startDate':self.startDate, 'endDate':self.endDate, 'curveName':self.curveName})


class FXVolDerive(DatabaseExtract):
    def __init__(self, startDate, endDate, database=None, config=None):
        self.startDate = startDate
        self.endDate = endDate
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
    select vs.VolSurfaceDataId, vsn.VolSurfaceNodeDataId, vsn.VendorDataInternalId, vs.SettleCurrency, vs.ForeignCurrency, vsn.VolSurfaceId, vsn.TenorEnum, vsn.Delta, TradeDate=ic.[Trade Date], Quote=ic.[Last Price]
    from [MarketData].dbo.FXVolSurfaceNode vsn
    join [PROD_VNDR_DB].DataScopeSelect.dbo.FXoptionVols ic
    on ic.RIC = vsn.VendorDataInternalId
    join [MarketData].dbo.FXVolSurface vs
    on vs.VolSurfaceId = vsn.VolSurfaceId
    where vs.VolSurfaceDataId in ( 'RefType|Name=FXOptionVolSurface|AUDJPY.FXO', 'RefType|Name=FXOptionVolSurface|AUDSGD.FXO', 'RefType|Name=FXOptionVolSurface|AUDUSD.FXO', 'RefType|Name=FXOptionVolSurface|EURCHF.FXO','RefType|Name=FXOptionVolSurface|EURCNY.FXO', 'RefType|Name=FXOptionVolSurface|EURCZK.FXO', 'RefType|Name=FXOptionVolSurface|EURGBP.FXO', 'RefType|Name=FXOptionVolSurface|EURHUF.FXO', 'RefType|Name=FXOptionVolSurface|EURJPY.FXO', 'RefType|Name=FXOptionVolSurface|EURKRW.FXO', 'RefType|Name=FXOptionVolSurface|EURNOK.FXO', 'RefType|Name=FXOptionVolSurface|EURPLN.FXO', 'RefType|Name=FXOptionVolSurface|EURRUB.FXO',
    'RefType|Name=FXOptionVolSurface|EURSEK.FXO', 'RefType|Name=FXOptionVolSurface|EURSGD.FXO','RefType|Name=FXOptionVolSurface|EURTRY.FXO', 'RefType|Name=FXOptionVolSurface|EURUSD.FXO', 'RefType|Name=FXOptionVolSurface|GBPCHF.FXO', 'RefType|Name=FXOptionVolSurface|GBPJPY.FXO', 'RefType|Name=FXOptionVolSurface|GBPNOK.FXO', 'RefType|Name=FXOptionVolSurface|GBPPLN.FXO', 'RefType|Name=FXOptionVolSurface|GBPUSD.FXO', 'RefType|Name=FXOptionVolSurface|GBPZAR.FXO', 'RefType|Name=FXOptionVolSurface|JPYKRW.FXO', 'RefType|Name=FXOptionVolSurface|NOKSEK.FXO', 'RefType|Name=FXOptionVolSurface|NZDUSD.FXO', 'RefType|Name=FXOptionVolSurface|SGDJPY.FXO', 'RefType|Name=FXOptionVolSurface|USDBRL.FXO', 'RefType|Name=FXOptionVolSurface|USDCAD.FXO', 'RefType|Name=FXOptionVolSurface|USDCHF.FXO',
    'RefType|Name=FXOptionVolSurface|USDCNH.FXO', 'RefType|Name=FXOptionVolSurface|USDCNY.FXO', 'RefType|Name=FXOptionVolSurface|USDCZK.FXO', 'RefType|Name=FXOptionVolSurface|USDHKD.FXO', 'RefType|Name=FXOptionVolSurface|USDHUF.FXO', 'RefType|Name=FXOptionVolSurface|USDIDR.FXO','RefType|Name=FXOptionVolSurface|USDILS.FXO', 'RefType|Name=FXOptionVolSurface|USDINR.FXO', 'RefType|Name=FXOptionVolSurface|USDJPY.FXO', 'RefType|Name=FXOptionVolSurface|USDKRW.FXO', 'RefType|Name=FXOptionVolSurface|USDMXN.FXO', 'RefType|Name=FXOptionVolSurface|USDMYR.FXO',  'RefType|Name=FXOptionVolSurface|USDPHP.FXO', 'RefType|Name=FXOptionVolSurface|USDPLN.FXO',
    'RefType|Name=FXOptionVolSurface|USDRUB.FXO', 'RefType|Name=FXOptionVolSurface|USDSGD.FXO', 'RefType|Name=FXOptionVolSurface|USDTHB.FXO', 'RefType|Name=FXOptionVolSurface|USDTRY.FXO', 'RefType|Name=FXOptionVolSurface|USDTWD.FXO', 'RefType|Name=FXOptionVolSurface|USDZAR.FXO' )
    and ic.[Trade Date] >= '$StartDate' and ic.[Trade Date] <= '$EndDate'
    --and vs.VolSurfaceDataId = 'RefType|Name=FXOptionVolSurface|USDJPY.FXO'
    --and vsn.Delta='ATM'
    --and vsn.TenorEnum = '10Y'
    order by TradeDate
       """ ).substitute({'StartDate': self.startDate, 'EndDate':self.endDate})


class FXSpotRate(DatabaseExtract):
    def __init__(self, tradeDate, AxiomaDataId, database=None, config=None):
        self.tradeDate = tradeDate
        self.AxiomaId = AxiomaDataId
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
        exec [MarketData].dbo.[IC_CurrencyTS] @AsOfDate = '$TradeDate', @AxiomaDataIds = '$DataId'
        """).substitute({'TradeDate':self.tradeDate, 'DataId':self.AxiomaId})

class FXCurrencyDataId(DatabaseExtract):
    def __init__(self, Currency, database=None, config=None):
        self.currency = Currency
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
        select CurrencyEnum, AxiomaDataId from Metadata.dbo.CurrencyEnum
        where CurrencyEnum = '$Currency'
        """).substitute({'Currency':self.currency})

class ExtractDerivedFXVol(DatabaseExtract):
    def __init__(self, startDate, endDate, tenor, SettleCurrency =None, ForeignCurrency =None, database = None, config = None):
        self.startDate = startDate
        self.endDate = endDate
        self.tenor = tenor
        self.SettleCurrency = SettleCurrency
        self.ForeignCurrency = ForeignCurrency
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
        select * from marketdata.dbo.FXVolDerive
        where TradeDate >= '$startDate' and TradeDate <= '$endDate'
        and TenorEnum = '$tenor'
        and SettleCurrency = '$SettleCurrency' and ForeignCurrency = '$ForeignCurrency'
        """).substitute({'startDate':self.startDate, 'endDate':self.endDate, 'tenor':self.tenor, 'SettleCurrency':self.SettleCurrency, 'ForeignCurrency':self.ForeignCurrency})

class FXVolDerivedCount(DatabaseExtract):
    def __init__(self, startDate, endDate,  database = None, config = None):
        self.startDate = startDate
        self.endDate = endDate
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
        select * from marketdata.dbo.FXVolDerive
        where TradeDate >= '$startDate' and TradeDate <= '$endDate'
        """).substitute({'startDate':self.startDate, 'endDate':self.endDate})

class UnfilteredBonds(DatabaseExtract):
    def __init__(self, currency, country, tradeDate, database=None, config=None):
        self.tradeDate = tradeDate
        self.database = database
        self.config = config
        self.currency = currency
        self.country = country
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' if self.database.upper() == 'DEV' else 'sec.FrstCpnRate'
        self.cpn_field_name = 'FrstCpnRate = sec.CurrCpnRate' 
        self.sqlstatement = self._get_sql_statement()

    def _get_sql_statement(self):
        if self.currency == 'USD':
            return string.Template("""
            select p.InstrCode,       
                    sec.IssName,
                    cal.DomSettDays,
                    cal.DomSettCalCode,
                    p.TradeDate,
                    p.Prc as Price,
                    p.MatStdYld,
                    CompFreqCode = cpiv.Value_,
                    NomTermToMat = DateDiff(d, p.TradeDate, sec.MatDate)/365.25,
                    FrstCpnRate = sec.CurrCpnRate,
                    sec.MatDate,
                    cpn.FrstCpnDate,
                    cpn.LastCpnDate,
                    DayCountCode = ( select d.Desc_ from   [QAI].dbo.FIEJVCode d
                                     where  d.Type_ = 1
                                     and d.Code = cpiv.Value_),
                sec.DebtISOCurCode,
                org.ISOCtryCode,
                sec.InitAccDate,
                sec.DebtIssTypeCode  from QAI.dbo.FIEJVPRCDly as p
        join [QAI].[dbo].FIEJVSecInfo sec on sec.InstrCode = p.InstrCode
        inner join QAI.dbo.FIEJVOrgInfo as org on sec.IssrGEMOrgId = org.GemOrgID
        inner join [QAI].[dbo].FIEJVCpnHdr as cpn on p.InstrCode = cpn.InstrCode
        inner join [QAI].[dbo].FIEJVCpn cpiv on cpiv.instrcode = p.InstrCode
        and cpiv.Item = 120
        and cpiv.Value_ NOT LIKE '%[^0-9]%'
        left join [QAI].[dbo].FIEJVSecCalendar cal on sec.InstrCode = cal.InstrCode
        where sec.IssTypeCode = 'GOVT' 
        and p.TradeDate =  '$tradeDate'
        and ((sec.DebtIssTypeCode != 'BL' and sec.MoodysSnrtyCode = 'SU') or (sec.DebtIssTypeCode != 'BND' or sec.DebtIssTypeCode != 'NT'))
        and (sec.DebtIssTypeCode = 'BL' or sec.DebtIssTypeCode = 'BND' or sec.DebtIssTypeCode = 'NT') 
        and sec.MatDate is not null 
        and (select count(*) from QAI.dbo.FIEJVPrincipalHdr as prin
        where sec.InstrCode = prin.InstrCode 
        and prin.Item = 150 
        and prin.Value_ = '1' ) = 0 
        and sec.DebtFlngStaCode = 2 
        and org.ISOCtryCode = 'US' 
        and sec.DebtISOCurCode = 'USD' 
        and p.MatCorpYld > 0.80 * p.WrstCorpYld 
        and p.MatCorpYld < 1.20 * p.WrstCorpYld 
        and sec.SecInfoDFlag <> 16781504
        order by NomTermToMat
            """).substitute({'tradeDate':self.tradeDate})
        if self.currency in ['CAD', 'EUR', 'SEK']:
            return string.Template("""
        select * from QAI.dbo.FIEJVSecInfo as s
        join QAI.dbo.FIEJVPRCDly as p on s.InstrCode = p.InstrCode
        join QAI.dbo.FIEJVOrgInfo as org on s.IssrGEMOrgId = org.GemOrgID
        join QAI.dbo.FIEJVCpnHdr as cpn on s.InstrCode = cpn.InstrCode
        left join [QAI].[dbo].FIEJVSecCalendar cal on s.InstrCode = cal.InstrCode
        where s.IssTypeCode = 'GOVT'
        and (select count(*) from QAI.dbo.FIEJVPrincipalHdr as prin where s.InstrCode = prin.InstrCode
        and prin.Item = 150 
        and prin.Value_ = '1' ) = 0
        and (select count(*) from QAI.dbo.FIEJVCpnFltRate as fltnt where s.InstrCode = fltnt.InstrCode and fltnt.CpnRateSrcCode is not null) = 0
        and p.TradeDate = '$tradeDate'
        and (s.DebtIssTypeCode = 'BUBL' or s.DebtIssTypeCode = 'BL' or  s.DebtIssTypeCode = 'BND' or  s.DebtIssTypeCode = 'BUND' 
        or s.DebtIssTypeCode = 'GILTTSY' or s.DebtIssTypeCode = 'OT' or s.DebtIssTypeCode = 'DISNT' or s.DebtIssTypeCode = 'BONOS'
        or s.DebtIssTypeCode = 'OBLIGAC' or s.DebtIssTypeCode = 'OAT' or s.DebtIssTypeCode = 'IGB' or s.DebtIssTypeCode = 'DUTTRE' 
        or s.DebtIssTypeCode = 'DSL')
        and s.MatDate is not null  
        and (cpn.StripCpnFl  =  0 or org.ISOCtryCode = 'DE')
        and (cpn.CpnSubTypeCode = 'FXDI' or cpn.CpnSubTypeCode = 'FXPV')
        and (s.ProsDflag  =  0 or (org.ISOCtryCode = 'AU' or org.ISOCtryCode = 'LU')) 
        and org.ISOCtryCode = '$ISOCtryCode'
        and s.DebtISOCurCode = '$DebtISOCurCode'
        and ((org.ISOCtryCode = 'FR'  and s.WithhldgTaxRate is not null) or (org.ISOCtryCode!='FR'))
        and (((s.DebtIssTypeCode = 'BND' or s.DebtIssTypeCode = 'SCHATZE')  and s.MoodysSnrtyCode = 'SU') or s.DebtIssTypeCode!='BND')
        and p.PrcgMetCode = 'PR'
        and (p.MatCorpYld > 0.97 * p.WrstCorpYld)  and (p.MatCorpYld < 1.03 * p.WrstCorpYld)
        and ((org.ISOCtryCode = 'DE'  and s.DebtFlngStaCode = 2) or (org.ISOCtryCode!='DE'))
        and ((org.ISOCtryCode = 'JP'  and s.PhyFormCode = 'CTF') or org.ISOCtryCode!='JP')
        and ((org.ISOCtryCode  =  'BE'  and s.PhyFormCode  =  'EJV') or org.ISOCtryCode != 'BE')
        and ((org.ISOCtryCode  =  'FR'  and s.WithhldgTaxRate  =  29) or org.ISOCtryCode != 'FR')
        and ((org.ISOCtryCode =  'FR'  and (s.DebtFlngStaCode = 1 or s.DebtFlngStaCode = 2) ) or (org.ISOCtryCode!='FR'))
        and ((org.ISOCtryCode  =  'NL'  and (s.DebtFlngStaCode  =  1 or s.DebtFlngStaCode  =  2)) or (org.ISOCtryCode != 'NL'))
                """).substitute({'tradeDate':self.tradeDate, 
                                 'DebtISOCurCode': self.currency,
                                 'ISOCtryCode':self.country})

class ReutersZCurveCodes(DatabaseExtract):
    def __init__(self, database=None, config=None):
        self.database = database
        self.config = config
        self.sqlstatement = """
    select * from qai.dbo.FIEJVCurve 
    where YldCurveIndCode = 'SOV'
    and YldCurveSubCode = 'GVBM' 
    and PriCurFl=1"""

class BondOverides(DatabaseExtract):
    def __init__(self, curveId, tradeDate, itemId, database=None, config=None):
        self.curveId = curveId
        self.itemId = itemId
        self.database = database
        self.config = config
        self.tradeDate = tradeDate
        if itemId == 14:
            self.sqlstatement = string.Template("""
            select cb.InstrCode from [MarketData].[dbo].[DerivCurveFilteredBondCorrection] as cb
            where cb.CurveId = '$curveId' 
            AND cb.TradeDate <= '$tradeDate' 
            AND cb.ItemId = '$itemId'
                """).substitute({'curveId': self.curveId, 
                                 'tradeDate': self.tradeDate,
                                 'itemId': self.itemId})
        elif itemId == 3 or 13 or 50:
            self.sqlstatement = string.Template("""
            select cb.InstrCode from [MarketData].[dbo].[DerivCurveFilteredBondCorrection] as cb
            where cb.CurveId = '$curveId' 
            AND cb.TradeDate = '$tradeDate' 
            AND cb.ItemId = '$itemId'
                """).substitute({'curveId': self.curveId, 
                                 'tradeDate': self.tradeDate,
                                 'itemId': self.itemId})

        else:
            self.sqlstatement = string.Template("""
            select cb.InstrCode from [MarketData].[dbo].[DerivCurveFilteredBondCorrection] as cb
            where cb.CurveId = '$curveId' 
            AND cb.TradeDate = '$tradeDate' 
            AND cb.ItemId = '$itemId'
            AND cb.ItemValue < 0
                """).substitute({'curveId': self.curveId, 
                                 'tradeDate': self.tradeDate,
                                 'itemId': self.itemId})

class RiggsZeroCurve(DatabaseExtract):
    def __init__(self, country_code, tradeDate, database=None, config=None):
        self.country_code = country_code
        self.database = database
        self.config = config
        self.tradeDate = tradeDate
        self.sqlstatement = string.Template("""
            select * from EJV_ejv_common.dbo.analytic_yield_curve_hist
            where curve_id = (select cv.curve_id from EJV_rigs.dbo.curves cv
            where cv.cntry_cd = '$country_code'
            and cv.curve_cd = 'FI'
            and cv.ayc_fl = 'y'
            and (cv.cntry_cd = 'US' or cntry_primary_fl = 'y')
            and cv.curve_sub_cd = 'GVBM'
            and cv.status_cd = 'AOK')
            and date = '$tradeDate'
                """).substitute({'country_code': self.country_code, 
                                 'tradeDate': self.tradeDate})

def convert_currency_to_curveName(currency, countryName = None):
    if countryName == None:
        country = currency[0:2]
    else:
        country = countryName
    curveName = string.Template("$country.$currency.GVT.ZC").substitute({'country':country, 'currency':currency})
    return 'EP.EUR.GVT.ZC' if 'EU' in curveName else curveName

class AxiomaSpreadID(DatabaseExtract):
    def __init__(self ,currency, country, database=None, config=None):
        self.currency = currency
        self.country = country
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
            select CurveNodeId from marketdata.dbo.CurveNodes where CurveId = 
            (SELECT CurveId FROM MarketData.dbo.Curve 
            WHERE CurveTypeEnum LIKE 'RsdSprVol%'
            AND CountryEnum = '$country'
            AND CurrencyEnum = '$currency')""").substitute({'country':self.country,
                                                            'currency':self.currency})
class AxiomaCurveID(DatabaseExtract):
    def __init__(self, curve_type='Sov.Zero', database=None, config=None):
        self.curve_type = curve_type
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
            select * from MarketData.dbo.Curve 
            where CurveTypeEnum='$curve_type' 
            and activefromdate<getdate() 
            and activetodate>getdate()""").substitute({'curve_type': self.curve_type})

class AxiomaCurveMeta(DatabaseExtract):
    def __init__(self, database=None, config=None):
        self.database = database
        self.config = config
        self.sqlstatement = 'SELECT * from [MarketData].[dbo].Curve'

class ReutersBenchmarkCurveId(DatabaseExtract):
     def __init__(self, currency, country, database=None, config=None):
        self.currency = currency
        self.country = country
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
            SELECT * FROM [QAI].[dbo].FIEJVCurve
            WHERE ISOCurCode = isnull('$currency', ISOCurCode) 
            AND ISOCtryCode =isnull('$country', ISOCtryCode)
            AND YldCurveIndCode = isnull('SOV', YldCurveIndCode)""").substitute({'country':country, 
                                                                                'currency':currency})

class BondFilterDbWriter(object):

    def __init__(self, tradeDate, curveShortName, df, enviroment=None, config=None, production_table=False):
        self.tradeDate = tradeDate
        self.curveShortName = curveShortName
        self.df = df
        self.enviroment = enviroment
        self.config = config
        self.production_table = production_table

    def write_to_dbs(self):
        delete_sql = self.delete_query()
        MSSQL.execute_commit(delete_sql, self.enviroment, self.config)
        user_name = getpass.getuser()
        dt = str(datetime.datetime.now())[0:-3] 
        queries = []
        for i, x in self.df.iterrows():
            queries.append( self.create_insert_sql({'curveShortName':self.curveShortName, 
                                               'TradeDate':self.tradeDate, 
                                               'InstrCode':x['InstrCode'], 
                                               'ItemId':3, 
                                               'ItemValue':x['forward_rate'], 
                                               'Lud': dt, 
                                               'Lub': user_name}))
        sql_inserts = [query + '\n' for query in queries]
        MSSQL.execute_commit(''.join(sql_inserts), self.enviroment, self.config)        

    def delete_query(self):
        if self.production_table:
            sqlstatement = string.Template("""
            DELETE from MarketData.dbo.DerivCurveFilteredBond
            where CurveId = (select curveId from  [MarketData].[dbo].[Curve] cv where cv.CurveShortName = '$curveShortName')  
            and TradeDate = '$TradeDate' and ItemId = 3
            """).substitute({'curveShortName':self.curveShortName, 'TradeDate': self.tradeDate})
        else:
            sqlstatement = string.Template("""
            DELETE from MarketData.dbo.DerivCurveFilteredBond_Research
            where CurveId = (select curveId from  [MarketData].[dbo].[Curve] cv where cv.CurveShortName = '$curveShortName') 
            and TradeDate = '$TradeDate' and ItemId = 3
            """).substitute({'curveShortName':self.curveShortName, 'TradeDate': self.tradeDate})
        return sqlstatement

    def create_insert_sql(self, d):
        if self.production_table:
            sqlstatement = string.Template("""
                INSERT INTO MarketData.dbo.DerivCurveFilteredBond VALUES((select curveId from  [MarketData].[dbo].[Curve] cv where cv.CurveShortName = '$curveShortName'),
                 '$TradeDate', '$InstrCode', '$ItemId', '$ItemValue', '$Lud', '$Lub')
                """).substitute(d)
        else:
            sqlstatement = string.Template("""
            INSERT INTO MarketData.dbo.DerivCurveFilteredBond_Research VALUES((select curveId from  [MarketData].[dbo].[Curve] cv where cv.CurveShortName = '$curveShortName'),
             '$TradeDate', '$InstrCode', '$ItemId', '$ItemValue', '$Lud', '$Lub')
            """).substitute(d)        
        return sqlstatement

class EJVHolidays(DatabaseExtract):
    def __init__(self, asset_id, database=None, config=None):
        self.asset_id = asset_id
        self.database = database
        self.config = config
        self.sqlstatement = string.Template("""
            select h.holiday_dt from EJV_assumptions.dbo.holiday_v01 h, EJV_govcorp.dbo.asset a
            WHERE a.settle_bus_ctr_cd = h.holiday_cal_cd
            AND a.asset_id = $asset_id """).substitute({'asset_id':asset_id})

class MacDatabase(object):
    def __init__(self, config_file_path, environment='DEV'):
        self.env_info = 'macdb'
        self.config_file = open(config_file_path, 'r')
        self.configuration = Utilities.loadConfigFile(self.config_file)
        self.environment = environment.replace('"', '').replace('\'', '').strip().upper()
        self.section_id = getAnalysisConfigSection(environment)
        self.env_info_map = Utilities.getConfigSectionAsMap(self.configuration, self.section_id)
        self.mac_db_info = getMSSQLDatabaseInfo(Utilities.getConfigSectionAsMap(
            self.configuration, self.env_info_map.get(self.env_info, None)))

    @contextmanager
    def create_connection(self):
        connection = Utilities.createMSSQLConnection(self.mac_db_info)
        yield connection
        connection.commit()
        connection.close()

    def extract_dataframe(self, sql_query, logger_callback=lambda x: None):
        with self.create_connection() as connection:
            logger_callback(sql_query)
            result = sql.read_frame(sql_query, connection)
            logger_callback("dataframe results rows: " + str(len(result)))
            logger_callback("dataframe results columns: " + str(len(result.columns)))
            return result

    @contextmanager
    def executemany(self, sql_list, logger_callback=lambda x: None, records_per_chunk=100):
        with self.create_connection() as connection:
            cursor = connection.cursor()
            for chunked_sql_list in Utilities.grouper(sql_list, records_per_chunk):
                chunked_sql = (';' + os.linesep).join(chunked_sql_list) + ';'

                try:
                    cursor.execute(chunked_sql)
                    connection.commit()
                except Exception as e:
                    logger_callback("Failed to execute sql")
                    logger_callback(chunked_sql)
                    raise e

    @contextmanager
    def execute(self, sql_command, logger_callback=lambda x: None):
        with self.create_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(sql_command)
            logger_callback(sql_command)
            connection.commit()


lub = 'DataGenOpsFl_Content'


def main():
    pass

if __name__ == '__main__':
    main()
