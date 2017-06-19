import unittest
import macpy.utils.database as db



class Test_DerivedDataCorpBondPriceCCY(unittest.TestCase):
    """
    This suite tests extraction calls to database which results in a slow test suite - it can take several minutes to run
    """

    def test_inconsistent_results(self):
        """
        Demonstrates bug in stored procedure where larger date ranges fail
        """

    # MarketData.[dbo].[DerivedDataCorpBondPriceForCcy]
    # @tradeDateBegin = '2010-01-16',
    # @tradeDateEnd = '2010-01-18',
    # @currencyISO = 'GBP'
    # returns a quote for instrument code 1851459 on trade date 2010-01-18 but:
    # MarketData.[dbo].[DerivedDataCorpBondPriceForCcy]
    # @tradeDateBegin = '2010-01-16',
    # @tradeDateEnd = '2012-12-31',
    # @currencyISO = 'GBP'

        currency = 'GBP'
        startTradeDate = '2010-01-16'
        endTradeDate = '2010-01-18'
        suffix = '_RS'
        bondPortfolio = db.DerivedDataCorpBondPriceForCcyResearch(startTradeDate, currency, endTradeDate, suffix=suffix)
        print bondPortfolio.sqlstatement
        df = bondPortfolio.extract_from_db()
        numberOfInstruments = len(df.InstrCode[df.InstrCode == 1851459])

        endTradeDate = '2012-12-31'
        storedProc = db.DerivedDataCorpBondPriceForCcyResearch(startTradeDate, currency, endTradeDate)
        print storedProc.sqlstatement
        df = storedProc.extract_from_db()
        numberOfInstruments = len(df.InstrCode[df.InstrCode == 1851459])

        self.assertGreater(numberOfInstruments, 0)


    def test_ensure_no_dupe_quotes(self):
        """
        Ensures results do not have duplicated
        """
        currency = 'USD'
        startTradeDate = '2011-01-16'
        endTradeDate = '2012-12-31'
        suffix = '_RS'
        storedProc = db.DerivedDataCorpBondPriceForCcyResearch(startTradeDate, currency, endTradeDate, suffix=suffix)
        df = storedProc.extract_from_db()
        counts = df.groupby(['TradeDate','InstrCode'])
        counts = counts.size().unique()
        self.assertAlmostEqual(counts, 1, "Duplicate quotes found")



class Test_StoredProc_GovernmentBenchmark(unittest.TestCase):
    """
    This suite tests extraction calls to database which results in a slow test suite - it can take several minutes to run
    """
    def test_government_benchmark_(self):
        currency = 'USD'
        valuationDate = '2014-01-01'
        benchmarkGovernmentBonds = db.DerivedDataExtractSovereignCurveBenchmarkBonds(valuationDate, currency)
        df = benchmarkGovernmentBonds.extract_from_db()
        self.assertGreater(len(df), 0)

class Test_StoredProc_GovernmentBenchmark(unittest.TestCase):
    """
    This suite tests extraction calls to database which results in a slow test suite - it can take several minutes to run
    """
    def test_government_benchmark_(self):
        currency = 'USD'
        valuationDate = '2014-01-01'
        benchmarkGovernmentBonds = db.DerivedDataExtractSovereignCurveBenchmarkBonds(valuationDate, currency)
        df = benchmarkGovernmentBonds.extract_from_db()
        self.assertGreater(len(df), 0)


class Test_ResearchCurves(unittest.TestCase):
    """
    Tests ResearchCurves
    """
    def test_main_use_research_curves(self):
        curveName = 'US.USD.GVT.ZC'
        valuationDate = '2014-01-01'
        #database = 'Prod'
        rc = db.ResearchCurves(valuationDate, valuationDate, curveName)
        df = rc.extract_from_db()
        self.assertGreater(len(df), 0)


class Test_CurveNodeQuoteFinal(unittest.TestCase):
    """

    """
    def test_main_use_prod(self):
        curveName = 'US.USD.GVT.ZC'
        startDate = '2014-01-01'
        database = 'Prod'
        cnq = db.CurveNodeQuoteFinal(curveName, startDate, database=database)
        df = cnq.extract_from_db()
        self.assertGreater(len(df), 0)


class Test_MSSQL(unittest.TestCase):
    """
    
    """
    def test_main_use_prod(self):
        sqlStatement = "Select top 1 * from MarketData.dbo.Curve"
        environment = 'Prod'
        df = db.MSSQL.extract_dataframe(sqlStatement, environment)
        self.assertGreater(len(df), 0)


class Test_DerivedFilteredBootstrap(unittest.TestCase):
    """

    """
    def test_main_use_prod(self):
        currency = 'USD'
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        fb = db.DerivedFilteredBootstrap(currency, startDate, endDate)
        df = fb.extract_from_db()
        self.assertGreater(len(df), 0)
