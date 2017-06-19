import os
from prototype import *
from string import Template
import macpy.utils.database as db

@constructor
def SQLCreator(self, sqltemplate):
    """
    Converts templated SQL into SQL statement and runs SQL against database returning a pandas dataframe
    """
    self.sqltemplate = sqltemplate;
    
    def create_sql(self, **kwargs):
        template = Template(self.sqltemplate)
        subst = template.safe_substitute(kwargs)
        return subst
    
    self.create_sql = create_sql

class CurveNodeQuote(object):
    """
    Wraps execution of stored proc DerivedDataYieldCurveTS  
    import macpy.extraction as extraction     

    args = {
    'CurveShortName':'US.USD.GVT.ZC',
    'StartDate':'2014-01-01'
    'EndDate' :'2014-02-01'
    'InterimTable':0}

    cnq = extract.CurveNodeQuote()
    df = cnq.extract_dataframe(args)
    
    Returns a dataframe with fields: 
    CurveShortName and TradeDate form a pd.MultiIndex on the DataFrame
    CurveShortName TradeDate	Quote1D	..	Quote40Y
    CA.CAD.GVT.ZC	11/5/2014	0.008810701	..	0.026064869
    """
 
    def __init__(self):        
        self.sql_template = r"""
declare 
    @CurveShortNames varchar(max) = '$CurveShortName',
    @StartDate    varchar(max)     = '$StartDate',
    @EndDate      varchar(max)     = '$EndDate',
    @InterimTable int              = $InterimTable
    
declare @AxiomaDataIds varchar(max)= 
(SELECT Marketdata.dbo.strsum(curveid,'|') 
FROM Marketdata.dbo.Curve
WHERE CurveShortName  in (SELECT * from MarketData.dbo.StrToTable(@CurveShortNames,'|')))
exec [MarketData].[dbo].[DerivedDataYieldCurveTS] 
    @AxiomaDataIds=@AxiomaDataIds,
    @LookBackStartDate=@StartDate,
    @AsOfDate=@EndDate,
    @InterimTable=@InterimTable
"""
        self.sql_creator = SQLCreator(self.sql_template)
    
    def create_sql(self, **kwargs):
        return self.sql_creator.create_sql(**kwargs)
     
    def extract_dataframe(self, environment=None, **kwargs):
        """
        environment = 'PROD|DEV|UAT'
        """

        sqlstatement = self.create_sql(**kwargs) 
        df = db.MSSQL.extract_dataframe(sqlstatement, environment)
        df = df.set_index(['CurveShortName','TradeDate'], drop=True)
        return df    
    
class DerivCurveFilteredBonds(object):
    """
    Wraps execution of stored proc DerivedDataYieldCurveTS       
    import macpy.extraction as extract
    args = {
    'CurveShortName':'CA.CAD.GVT.ZC',
    'StartDate':'2014-01-01'
    'EndDate' :'2014-02-01'}

    fb = extract.DerivCurveFilteredBonds()
    df = cnq.extract_dataframe(args)
    
    Returns a dataframe with fields: 
    CurveId    CurveShortName    CurveTypeEnum    TradeDate    InstrCode    ISIN    TermInYears    Yield
    200302044    CA.CAD.GVT.ZC    Sov.Zero    12/4/2014    3515799    CA1350Z7VQ46    0.153319    0.893768

    CurveShortName and TradeDate form a pd.MultiIndex on the DataFrame
    """
 
    def __init__(self):        
        self.sql_template = r"""
exec [MarketData].[dbo].[DerivedCurveInstrumentDetail]
    @tradeDateBegin     = '$StartDate',
    @tradeDateEnd       = '$EndDate',
    @curveShortName     = '$CurveShortName'
"""
        self.sql_creator = SQLCreator(self.sql_template)
    
    def create_sql(self, **kwargs):
        return self.sql_creator.create_sql(**kwargs)
     
    def extract_dataframe(self, environment=None, **kwargs):
        """
        environment = 'PROD|DEV|UAT'
        """

        sqlstatement = self.create_sql(**kwargs) 
        df = db.MSSQL.extract_dataframe(sqlstatement, environment)
        df = df.set_index(['CurveShortName','TradeDate','ISIN'], drop=True)
        return df    
    
