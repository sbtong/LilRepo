import unittest
import numpy as np
import macpy.bootstrapper as bs
import macpy.bond as bond
import macpy.utils.database as db
import pandas as pd
import datetime
from macpy import bond_utils
from dateutil.relativedelta import relativedelta

from macpy.curve_functions import create_yield_curve

from macpy.curve_wrappers import InterpolatedCurve, InterplatedCurve


class Test_bootstrapper(unittest.TestCase):

    def test_bootstrapper(self):
        currency='JPY'
        startDate='2015-01-02'
        endDate = '2015-01-02'
        conventions = bond_utils.get_sovereign_conventions(currency)
        bootstrapper = bs.BootStrapper(startDate, endDate, currency, conventions)
        result = bootstrapper.createYieldCurve()
        expectedValue = .01356
        actualValue = result(30.0)
        self.assertAlmostEquals(expectedValue, actualValue, delta=1e-2)

    def test_YieldsDatabaseWriter(self):
        startDate = '2005-06-30'
        endDate = '2005-06-30'
        currency = 'BRL'
        yieldsWriter = bs.SovereignCurve(startDate, endDate, currency, enviroment='DEV')
        yieldsWriter.write_to_dbs()

    def test_YieldsDatabaseWriter_EUR(self):
        startDate = '2005-06-30'
        endDate = '2005-06-30'
        currency = 'EUR'
        yieldsWriter = bs.SovereignCurve(startDate, endDate, currency, enviroment='DEV')
        yieldsWriter.write_to_dbs()


    def test_YieldsDatabaseWriter_KRW(self):
        startDate = '2016-07-19'
        endDate = '2016-07-19'
        currency = 'KRW'
        yieldsWriter = bs.SovereignCurve(startDate, endDate, currency, enviroment='PROD')
        yieldsWriter.write_to_dbs()

    def test_write_yields_to_dbs(self):
        startDate = '2015-10-08'
        endDate = '2015-10-08'
        currency = 'USD'
        args=[startDate, endDate, currency, 'PROD', 'database.config', False]
        #bs.write_yields_to_dbs(args)


    def test_benchmarkJPY(self):
        startDate='2016-01-20'
        endDate = '2016-01-20'
        currency = 'JPY'

        #actual values:
        conventions = bond_utils.get_sovereign_conventions(currency)
        bootstrapper = bs.BootStrapper(startDate, startDate, currency, conventions)
        modelCurve = InterplatedCurve(bootstrapper)

        #expected values:
        expectedValue = {}
        expectedValue[0.25] = -0.000178525301582
        expectedValue[0.5] = -0.000352864240659
        expectedValue[1] = -0.0003139854078
        expectedValue[2] = -0.000179828978578
        expectedValue[5] = -0.000106645408598
        expectedValue[7] = 0.000210088625204
        expectedValue[10] = 0.00220943861676
        expectedValue[20] = 0.00973270143265
        expectedValue[30] = 0.0131333830853

        for i in expectedValue.keys():
            message = 'JPY curve fails benchmark test at point %f years'%i
            self.assertAlmostEquals(expectedValue[i], modelCurve(i),msg=message, delta=1e-3)

    def test_benchmarkCAD(self):
        startDate='2015-01-08'
        endDate = '2015-01-08'
        currency = 'CAD'

        #actual values:
        conventions = bond_utils.get_sovereign_conventions(currency)
        bootstrapper = bs.BootStrapper(startDate, startDate, currency, conventions)
        modelCurve = InterplatedCurve(bootstrapper)
        print modelCurve(2.0)
        #expected values:
        expectedValue = {}
        expectedValue[0.25] =0.00892031589526
        expectedValue[0.5] =0.00924071512027
        expectedValue[1] =0.00949588548778
        expectedValue[2] =0.00963202881169
        expectedValue[5] =0.0126179892905
        expectedValue[7] =0.0148903574644
        expectedValue[10] =0.0181241499673
        expectedValue[20] =0.0226691470335
        expectedValue[30] =0.0234875022471

        for i in expectedValue.keys():
            message = 'CAD curve fails benchmark test at point %f years' % i
            self.assertAlmostEquals(expectedValue[i], modelCurve(i),msg=message,delta=1e-3)

    def test_model_error(self):

        startDate='2015-01-08'
        endDate = '2015-01-08'
        currency = 'CAD'
        conventions = bond_utils.get_sovereign_conventions(currency)
        curveName = bond.convert_gvt_curve(currency)        
        bootstrapper = bs.BootStrapper(startDate, startDate, currency, conventions)
        modelCurve = InterplatedCurve(bootstrapper)
        gvtCurve = InterpolatedCurve(startDate, curveName)
        
        filtered_bond_query = db.DerivedFilteredBootstrap(currency, startDate, endDate)
        dataframeExtracted = filtered_bond_query.extract_from_db()
        dataframeExtracted_New = dataframeExtracted.apply(lambda row: self.adjusted_maturity_date(row),axis=1)

        modelError=[]
        numerixError=[]        
        modelargs = [modelCurve, modelError]
        NumerixArgs = [gvtCurve, numerixError]

        dataframeExtracted_New.apply(lambda row:self.compute_model_error(row,modelargs), axis=1)
        dataframeExtracted_New.apply(lambda row:self.compute_model_error(row,NumerixArgs), axis=1)
        
        totalError = sum([i*i for i in modelError])

        NumerixError = sum([i*i for i in numerixError])
        
        print 'ModelError: ', totalError, '\n', 'NumerixError: ', NumerixError


    def compute_model_error(self, row, args):

        yieldCurve = args[0]
        errorlist = args[1]

        issueDate = pd.to_datetime(row.InitAccDate).strftime('%Y-%m-%d') if not pd.isnull(row.InitAccDate) else datetime.datetime(row.MatDate.year-1, row.MatDate.month, row.MatDate.day).strftime('%Y-%m-%d')
        maturityDate=pd.to_datetime(row.MatDate).strftime('%Y-%m-%d')
        coupon= row.FrstCpnRate/100
        valuationDate = pd.to_datetime(row.TradeDate).strftime('%Y-%m-%d')

        bondpricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)        
        dirtyPrice = bondpricer.compute_dirty_price(yieldCurve)
        yieldToMaturity = bondpricer.compute_yield_to_maturity(row.Price)
        print row.Price, ' ', dirtyPrice, ' ', row.MatDate, ' ', yieldToMaturity, ' ',row.MatStdYld/100.0, ' ', row.NomTermToMat, ' ', row.FrstCpnRate

        error = dirtyPrice-row.Price
        errorlist.append(error)


    def adjusted_maturity_date(self, row):
        if not pd.isnull(row['InitAccDate']):
            diff = (pd.to_datetime(row['MatDate']) - pd.to_datetime(row['InitAccDate'])).days/365.25
            diff_decimal = diff - int(diff)
            maturityDate_new = pd.to_datetime(row['InitAccDate'])+relativedelta(years=int(diff)) if diff_decimal<0.495 else pd.to_datetime(row['InitAccDate'])+relativedelta(years=int(diff), months=6)
            row['MatDate'] = maturityDate_new
        return row

    def test_JPY_short_end(self):
        """
        Does the short-end (inside of 3 months) of JPY curve stay flat from 0 to .25 years
        We pin the value of the short-end at .25 years to reduce volatility of the short-end
        """
        currency = 'JPY'
        startDate = '2016-01-12'
        endDate = '2016-01-12'
        conventions = bond_utils.get_sovereign_conventions(currency)
        bootstrapper = bs.BootStrapper(startDate, endDate, currency, conventions)
        yieldCurveOutput = bootstrapper.createYieldCurve()
        actualValue = yieldCurveOutput(0.00001)
        expectedValue = yieldCurveOutput(.25)
        self.assertAlmostEquals(expectedValue, actualValue, delta=1e-12)

    def test_CAD_short_end(self):
        """
        Does the short-end (inside of 3 months) of CAD curve stay flat from 0 to .25 years
        We pin the value of the short-end at .25 years to reduce volatility of the short-end
        """
        currency = 'CAD'
        startDate = '2016-01-14'
        endDate = '2016-01-14'
        conventions = bond_utils.get_sovereign_conventions(currency)
        bootstrapper = bs.BootStrapper(startDate, endDate, currency, conventions)
        yieldCurveOutput = bootstrapper.createYieldCurve()
        actualValue = yieldCurveOutput(0.000000)
        expectedValue = yieldCurveOutput(.25)
        self.assertAlmostEquals(expectedValue, actualValue, delta=1e-12)

    def test_CHF_short_end(self):
        """
        Is the shore-end sufficiently negative?
        """
        currency = 'CHF'
        startDate = '2016-01-18'
        endDate = '2016-01-18'
        conventions = bond_utils.get_sovereign_conventions(currency)
        bootstrapper = bs.BootStrapper(startDate, endDate, currency, conventions)
        yieldCurveOutput = bootstrapper.createYieldCurve()
        actualValue = yieldCurveOutput(0.000000)
        expectedValue = -.005
        self.assertGreater(expectedValue, actualValue)

    def test_LT_3152626(self):
        currency = 'EUR'
        curveName = 'FR.EUR.GVT.ZC'
        startDate = '2016-05-17'
        endDate = '2016-05-17'

    def test_FR_short_end(self):
        currency = 'EUR'
        curveName = 'FR.EUR.GVT.ZC'
        startDate = '2016-05-17'
        endDate = '2016-05-17'

        filtered_bond_query = db.DerivedFilteredBootstrap(currency, startDate, endDate, curveName = curveName)
        dfFR = filtered_bond_query.extract_from_db()
        conventions = bond_utils.get_sovereign_conventions(currency)
        bootstrapper = bs.BootStrapper(startDate, endDate, currency, conventions, dataframe = dfFR)
        yieldcurve = bootstrapper.createYieldCurve()
        actualValue = yieldcurve(0.0001)
        expectedValue = 0.000118
        self.assertAlmostEqual(actualValue,expectedValue, delta = 0.001)

    def test_bond_pricer_constructor_from_dataframe(self):
        currency = 'ISK'
        curve_name = 'IS.ISK.GVT.ZC'
        environment = 'PROD'
        start_date = '2016-08-01'
        end_date = '2016-08-01'
        filtered_bond_query = db.DerivedFilteredBootstrap(currency, start_date, end_date, curveName=curve_name, database=environment)
        df = filtered_bond_query.extract_from_db()
        df['BondPricer'] = df.apply(lambda row:  bond.BondPricer.create_from_row(row), axis=1)
        bond_pricer = df['BondPricer'][0]
        df['YTM'] = df.apply(lambda row: row['BondPricer'].compute_yield_to_maturity(row['Price']), axis=1)
        expected_value = .01


    def test_Iceland_short_end_short_end_jump(self):
        currency = 'ISK'
        curve_name = 'IS.ISK.GVT.ZC'
        environment = 'PROD'
        conventions = bond_utils.get_sovereign_conventions(currency)
        start_date_t0 = '2016-07-29'
        end_date_t0 = '2016-07-29'
        filtered_bond_query_t0 = db.DerivedFilteredBootstrap(currency, start_date_t0, end_date_t0, curveName = curve_name,database=environment)
        df_t0 = filtered_bond_query_t0.extract_from_db()
        bootstrapper_t0 = bs.BootStrapper(start_date_t0, end_date_t0, currency, conventions, dataframe=df_t0)
        yield_curve_t0 = bootstrapper_t0.createYieldCurve()
        actual_value_t0 = yield_curve_t0(0.0001)

        start_date_t1 = '2016-08-01'
        end_date_t1 = '2016-08-01'
        filtered_bond_query_t1 = db.DerivedFilteredBootstrap(currency, start_date_t1, end_date_t1, curveName = curve_name,database=environment)
        df_t1 = filtered_bond_query_t1.extract_from_db()
        bootstrapper_t1 = bs.BootStrapper(start_date_t1, end_date_t1, currency, conventions, dataframe=df_t1)
        yield_curve_t1 = bootstrapper_t1.createYieldCurve()
        actual_value_t1 = yield_curve_t1(0.0001)

        self.assertAlmostEqual(actual_value_t0, actual_value_t1, delta=0.005)








