import itertools

import pymssql
from pyutils.simple_struct import Struct
import pandas

daymap = {'Y': 365, 'M': 31, 'W': 7, 'D': 1}

def nodeNameKey(node):
    if isinstance(node, basestring):
        nodename = node
    else:
        nodename = node.name
    tenoridx = nodename.rfind('.')
    if tenoridx == -1:
        raise Exception('Could not find separator')
    tenor = nodename[tenoridx+1:]
    unit = tenor[-1].upper()
    if unit not in daymap:
        raise Exception('Could not find unit %s in daymap' % unit)
    days = daymap[unit] * int(tenor[:-1])
    return (nodename[:tenoridx], days)

class CurveLoader(object):
    def __init__(self):
        port = '1433'
        host='prod_mac_mkt_db'
        database='MarketData'
        dbUser='MarketDataLoader'
        dbPass='mdl1234'
        self.cn_ = pymssql.connect(database=database, host=host, user=dbUser, password=dbPass, port = port)
        self.cursor_ = self.cn_.cursor()

    def _buildResult(self, cols):
        result = []
        for row in self.cursor_:
            result.append(Struct(dict(itertools.izip(cols, row))))
        return result

    def getAllCurves(self):
        query = """
        SELECT
        CurveShortName, CurveLongName, CurveTypeEnum, CountryEnum, CurrencyEnum 
        from marketdata.dbo.Curve C
        """

        self.cursor_.execute(query)
        cols = ['name', 'description', 'type', 'country', 'currency']
        return self._buildResult(cols)

    def getAllCurveNodes(self, name):
        query = """
        SELECT
        DISTINCT cn.NodeShortName, cn.tenorenum
        from marketdata.dbo.Curve C 
        JOIN marketdata.dbo.CurveNodes cn ON C.CurveId=cn.CurveId
        where C.CurveShortName=%s
        """
        
        self.cursor_.execute(query, (name,))
        cols = ['name', 'tenor']
        return self._buildResult(cols)

    def getAllFactorNodes(self):
        query = """
        SELECT
        DISTINCT C.CurveShortName, cn.tenorenum
        FROM marketdata.dbo.DerivCurveUniversalModel dc
        JOIN marketdata.dbo.Curve C ON dc.CurveShortName = C.CurveShortName
        JOIN marketdata.dbo.CurveNodes cn ON C.CurveId=cn.CurveId
        WHERE cn.TenorEnum IN ('6M', '1Y', '2Y', '5Y', '10Y', '30Y') and dc.SpecificRisk is null
        """

        self.cursor_.execute(query)
        cols = ['name', 'tenor']
        return self._buildResult(cols)

    def getNodesHistory(self, names):
        query = """SELECT
            cn.NodeShortName, cq.TradeDate, cq.Quote
            FROM marketdata.dbo.CurveNodeQuoteFinal cq join marketdata.dbo.CurveNodes cn on cq.CurveNodeId=cn.CurveNodeId
            WHERE cn.NodeShortName in %s
        """
        orignames = names
        if not isinstance(names[0], basestring):
            names = [n.name for n in names]
        self.cursor_.execute(query, (tuple(names),))
        df = pandas.DataFrame(self.cursor_.fetchall(), columns=['name', 'dt', 'value']).pivot('dt', 'name', 'value').reindex(columns=names)
        df.columns = orignames
        return df
