import unittest
import macpy.bond as bond
import macpy.finance as finance
import macpy.utils.database as db
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from macpy.curve_wrappers import InterpolatedCurve, NSCurve, SwapCurve, CustomCurve
from macpy.curve_functions import create_swap_zero_curve, create_govt_zero_curve, create_yield_curve

class TestBond(unittest.TestCase):
    def test_accrued_interest_with_no_accrued_days(self):
        issueDate = '2010-06-03'
        valuationDate = '2014-06-03'
        maturityDate = '2017-06-03'
        coupon = 0.0425

        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
        actualAccruedInterest = bondPricer.compute_accrued_interest()
        expectedAccruedInterest = 0.0

        self.assertAlmostEquals(actualAccruedInterest, expectedAccruedInterest, places=4)

    def test_accrued_interest_main_use(self):
        valuationDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28'
        coupon = 0.0475

        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
        actualAccruedInterest = bondPricer.compute_accrued_interest()
        expectedAccruedInterest = 0.83124
        self.assertAlmostEquals(actualAccruedInterest, expectedAccruedInterest, places=4)
        # np.testing.assert_almost_equal(expectedSpreadDuration, float(actualSpreadDuration))

    def test_accrued_interest_CHINA(self):
        valuationDate = '2016-06-24'
        issueDate = '2016-05-05'
        maturityDate = '2017-05-05'
        coupon = 0.023

        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0, settlement_adj=1.0, first_cpn_dt='2017-05-05', last_cpn_dt=None)
        actualAccruedInterest = bondPricer.compute_accrued_interest()
        print actualAccruedInterest



    def test_clean_vs_dirty_market_price(self):
        valuationDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28'
        currency = 'USD'
        environment = 'Prod'
        coupon = 0.0475
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
        swapCurve = create_swap_zero_curve(valuationDate, currency, environment)

        actualCleanPrice = pricer.compute_clean_price(swapCurve)
        expectedCleanPrice = pricer.compute_dirty_price(swapCurve) - pricer.compute_accrued_interest()
        self.assertAlmostEquals(actualCleanPrice, expectedCleanPrice, places=6)


    def yield_sov_curve(self):
        usd_curve = 'US.USD.GVT.ZC'
        sov_spr_curve = 'MX.USD.SOVSPR'
        database='Dev'
        valuationDate = '2010-01-01'
        #yield_curve = bond.SwapCurve(valuationDate, usd_curve, sov_spr_curve, database)
        #curve_5y = yield_curve(.5)

    def test_yield_to_maturity_and_spread_are_consistent(self):
        valuationDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28'
        currency = 'USD'
        coupon = 0.0475

        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
        swapCurve = create_swap_zero_curve(valuationDate, currency)

        actualCleanPrice = pricer.compute_clean_price(swapCurve)
        expectedCleanPrice = 115.0
        self.assertAlmostEqual(actualCleanPrice, expectedCleanPrice, delta=.1)

        #make sure coupon and ytm are not very different
        actualYieldToMaturity = pricer.compute_yield_to_maturity(actualCleanPrice)
        expectedYieldToMaturity = coupon
        self.assertAlmostEqual(actualYieldToMaturity, expectedYieldToMaturity, delta=.05)

        actualSpread = pricer.compute_spread(swapCurve, actualCleanPrice)
        expectedSpread = 0.0
        self.assertAlmostEqual(actualSpread, expectedSpread, delta=1e-8)

    def test_ytm_instrcode(self):
        valuationDate = '2016-06-07'
        issueDate = '2016-05-15'
        maturityDate = '2046-05-15'
        currency = 'USD'
        coupon = 2.5 / 100
        cleanPrice = 99.281
        #cleanPricePar = 100
        #swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice)
        diagnostics = pricer.get_diagnostics()

        expectedYtm =2.534361/ 100
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-2) #tolerance of 10bps

    def test_ytm_instrcode_4350902(self):
        valuationDate = '2016-06-06'
        issueDate = '2016-05-05'
        maturityDate = '2017-05-05'
        currency = 'CNY'
        coupon = 2.3/100
        cleanPrice = 99.9357
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice)

        expectedYtm = 2.368/100
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-2)

    def test_ytm_instrcode_2679869(self):
        valuationDate = '2016-09-08'
        issueDate = '2012-01-26'
        maturityDate = '2062-01-26'
        coupon = 3.8/100
        cleanPrice = 200.12
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0,first_cpn_dt='2013-01-26', last_cpn_dt='2061-01-26')
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice)

        expectedYtm = 1.0277/100
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-2)

    def test_ytm_instrcode_2169328(self):
        valuationDate = '2016-09-08'
        issueDate = '2010-06-30'
        maturityDate = '2040-12-07'
        coupon = 4.25/100
        cleanPrice = 159.12
        first_cpn_dt = '2010-12-07'
        last_cpn_dt = '2040-06-07'
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0,first_cpn_dt=first_cpn_dt, last_cpn_dt=last_cpn_dt)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice)

        expectedYtm = 1.3742/100
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-2)


    def test_ytm_instrcode_3152626(self):
        #test to check the bond spread calculation
        #demonstrates issue with dataflow.py where cashflows are not positioned correctly through time.
        valuationDate = '2016-08-29'
        issueDate = '2013-03-04'
        maturityDate = '2019-08-30'
        currency = 'EUR'
        coupon = 3.1 / 100.0
        cleanPrice = 109.108
        first_cpn_dt = '2013-08-30'
        last_cpn_dt = '2018-08-30'
        freq = 2.0
        settlement_adj = 3.0 
        #cleanPricePar = 100
        #swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice)
        diagnostics = pricer.get_diagnostics()

        expectedYtm = 0.046070 / 100
        delta = 0.0005  #5 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)              


    def test_ytm_instrcode_1919643(self):
        #test to check the bond spread calculation
        #demonstrates issue with dataflow.py where cashflows are not positioned correctly through time.
        valuationDate = '2009-11-17'
        issueDate = '2009-08-24'
        maturityDate = '2015-11-04'
        currency = 'GBP'
        coupon = 7 / 100.0
        cleanPrice = 116.214404
        #cleanPricePar = 100
        #swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=1.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice)
        diagnostics = pricer.get_diagnostics()

        expectedYtm =3.851111 / 100
        delta = 0.0005  #5 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_sov_jpy_3802013(self):
        valuationDate = '2016-01-12'
        issueDate = '2015-03-15'
        maturityDate = '2017-03-15'
        currency = 'JPY'
        coupon = 0.001
        freq = 2.0
        settlement_adj = 2.0
        marketPrice = 100.184
        first_cpn_dt = '2015-09-15'
        last_cpn_dt = '2016-09-15'
        dirtyPriceFlag = True
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=dirtyPriceFlag)*100.0

        expectedYtm = -0.058
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_2051042(self):
        valuationDate = '2016-07-05'
        issueDate = '2010-01-01'
        maturityDate = '2021-01-01'
        currency = 'BRL'
        coupon = 0.1
        freq = 2.0
        settlement_adj = 1.0
        marketPrice = 93.1616
        first_cpn_dt = '2010-07-01'
        last_cpn_dt = '2020-07-01'
        dirtyPriceFlag = True
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=dirtyPriceFlag)*100.0
        print actualYtm

    def test_ytm_instrcode_4350902(self):
        valuationDate = '2016-07-07'
        issueDate = '2016-05-05'
        maturityDate = '2017-05-05'
        currency = 'CNY'
        coupon = 0.023
        freq = 2.0
        settlement_adj = 1.0
        marketPrice = 99.96
        first_cpn_dt = '2017-05-05'
        last_cpn_dt = None
        dirtyPriceFlag = True
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=dirtyPriceFlag)
        expectedYtm = 0.0234
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)


    def test_ytm_instrcode_4346690(self):
        valuationDate = '2016-07-08'
        issueDate = '2015-08-03'
        maturityDate = '2016-08-03'
        currency = 'HKD'
        coupon = 0.0
        freq = 2.0
        settlement_adj = 1.0
        marketPrice = 99.988
        first_cpn_dt = None
        last_cpn_dt = None
        dirtyPriceFlag = True
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=dirtyPriceFlag)
        expectedYtm = 0.190458 / 100.0 
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)


    def test_ytm_instrcode_2876537(self):
        valuationDate = '2016-10-04'
        issueDate = '2012-09-10'
        maturityDate = '2017-09-10'
        currency = 'KRW'
        coupon = 2.750 / 100.0
        freq = 2.0
        settlement_adj = 1.0
        marketStandardYield = 1.345096 / 100.0
        #marketPrice = 10148.45 / 100.0
        first_cpn_dt = '2013-03-10'
        last_cpn_dt = '2017-03-10'
        dirtyPriceFlag = True
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketStandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(cleanMarketPrice, priceIsClean=dirtyPriceFlag)
        expectedYtm = 1.345096 / 100.0 
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_4179692(self):
        valuationDate = '2016-10-04'
        issueDate = '2016-06-10'
        maturityDate = '2019-06-10'
        currency = 'KRW'
        coupon = 1.5 / 100.0
        freq = 2.0
        settlement_adj = 1.0
        marketstandardYield = 1.281759/100
        #marketPrice = 10105.17 / 100.0
        first_cpn_dt = '2016-12-10'
        last_cpn_dt = '2018-12-10'
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(cleanMarketPrice, priceIsClean=True)
        expectedYtm = 1.281759 / 100.0 
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_4205094(self):
        valuationDate = '2016-10-04'
        issueDate = '2016-01-01'
        maturityDate = '2027-01-01'
        currency = 'BRL'
        coupon = 10.0 / 100.0
        freq = 2.0
        settlement_adj = 1.0
        marketstandardYield = 11.57/100
        marketPrice = 93.878872
        first_cpn_dt = '2016-07-01'
        last_cpn_dt = '2026-07-01'
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        dirtyMarketPrice = bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(cleanMarketPrice, priceIsClean=True)
        expectedYtm = 11.57 / 100.0
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_3730179(self):
        valuationDate = '2016-10-04'
        issueDate = '2015-10-13'
        maturityDate = '2016-10-13'
        currency = 'BE.EUR'
        coupon = 0.0 / 100.0
        freq = 2.0
        settlement_adj = 2.0
        marketstandardYield = -0.822726/100
        marketPrice = 100.016
        first_cpn_dt = None
        last_cpn_dt = None
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        dirtyMarketPrice = bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(cleanMarketPrice, priceIsClean=True)
        expectedYtm = -0.822726 / 100.0
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_3031491(self):
        valuationDate = '2016-10-04'
        issueDate = '2013-02-26'
        maturityDate = '2018-06-22'
        currency = 'BE.EUR'
        coupon = 1.25 / 100.0
        freq = 1.0
        settlement_adj = 2.0
        marketstandardYield = -0.651188/100
        marketPrice = 103.28
        first_cpn_dt = '2013-06-22'
        last_cpn_dt = '2017-06-22'
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        dirtyMarketPrice = bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=True)
        expectedYtm = -0.651188 / 100.0
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_3730166(self):
        valuationDate = '2016-10-04'
        issueDate = '2015-12-15'
        maturityDate = '2016-12-15'
        currency = 'BE.EUR'
        coupon = 0.0 / 100.0
        freq = 2.0
        settlement_adj = 2.0
        marketstandardYield = -0.708736/100
        marketPrice = 100.138
        first_cpn_dt = None
        last_cpn_dt = None
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        dirtyMarketPrice = bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=True)
        expectedYtm = -0.708736 / 100.0
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_4340963(self):
        valuationDate = '2016-10-04'
        issueDate = '2016-01-05'
        maturityDate = '2017-01-05'
        currency = 'USD'
        coupon = 0.0 / 100.0
        freq = 2.0
        settlement_adj = 1.0
        marketstandardYield = 0.335782/100
        marketPrice = 99.915667
        first_cpn_dt = None
        last_cpn_dt = None
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        dirtyMarketPrice = bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=True)
        expectedYtm = 0.335782 / 100.0
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_4340963(self):
        valuationDate = '2016-10-04'
        issueDate = '2016-09-30'
        maturityDate = '2018-09-30'
        currency = 'USD'
        coupon = 0.75 / 100.0
        freq = 2.0
        settlement_adj = 1.0
        marketstandardYield = 0.821513/100
        marketPrice = 99.859375
        first_cpn_dt = '2017-03-31'
        last_cpn_dt = '2018-03-31'
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        dirtyMarketPrice = bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=True)
        expectedYtm = 0.821513 / 100.0
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    def test_ytm_instrcode_1760194(self):
        valuationDate = '2016-05-02'
        issueDate = '2009-01-15'
        maturityDate = '2029-01-15'
        currency = 'USD'
        coupon = 2.5 / 100.0
        freq = 2.0
        settlement_adj = 1.0
        marketstandardYield = 0.464318/100
        marketPrice = 125.078125
        first_cpn_dt = '2009-07-15'
        last_cpn_dt = '2028-07-15'
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        cleanMarketPrice = bondPricer.compute_clean_price(yieldCurve=None, spreadValue=marketstandardYield)
        dirtyMarketPrice = bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=marketstandardYield)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=True)
        expectedYtm = 0.464318 / 100.0
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)


    def test_yield_to_maturity_corp_instrcode_1688764(self):
        #this test should show a difference in Axioma YTM vs. TR YTM calcs.
        #The cause of the difference is an incorrect price quote from TR.
        valuationDate = '2010-10-18'
        issueDate = '2008-12-12'
        maturityDate = '2014-12-24'
        currency = 'GBP'
        coupon = 1.75 / 100.0
        cleanPrice = 116.1982
        #cleanPricePar = 100
        #swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=4.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice,True)
        diagnostics = pricer.get_diagnostics()

        expectedYtm = -0.019610798238973538
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-3) #tolerance of 10bps

    #def test_yield_to_maturity_corp_instrcode_1276266(self):
    #    valuationDate = '2014-01-07'
    #    issueDate = '2007-01-12'
    #    maturityDate = '2014-05-12'
    #    coupon = 0.0525
    #    #cleanPrice = 101.046424
    #    cleanPrice = 100.316964
    #    firstCpnDate = '2007-07-12'
    #    lastCpnDate = '2014-01-12'
    #    bondpricer = bond.BondPricer(issueDate,maturityDate, coupon,valuationDate,first_cpn_dt=firstCpnDate, last_cpn_dt=lastCpnDate)
    #    yieldToMaturity = bondpricer.compute_yield_to_maturity(cleanPrice,True)
    #    expectedYtm = 0.02171455
    #    self.assertAlmostEqual(yieldToMaturity, expectedYtm, delta=10e-3)

    def test_yield_to_maturity_corp_instrcode_119449(self):
        #this test demonstrates that the accrued interest is begin handled correctly
        valuationDate = '2009-07-08'
        issueDate = '1982-07-15'
        maturityDate = '2010-07-15'
        currency = 'GBP'
        coupon = 12 / 100.0
        cleanPrice = 109.834381
        #cleanPricePar = 100
        #swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice,True)
        diagnostics = pricer.get_diagnostics()

        expectedYtm = 0.02123535
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-3) #tolerance of 10bps

    def test_yield_to_maturity_corp_instrcode_2044327(self):
        #this test should show a difference in Axioma YTM vs. TR YTM calcs.
        #The cause of the difference is an incorrect price quote from TR.
        valuationDate = '2010-11-08'
        issueDate = '2010-03-12'
        maturityDate = '2015-03-09'
        currency = 'GBP'
        coupon = 4 / 100.0
        cleanPrice = 121.545588
        #cleanPricePar = 100
        #swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=1.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice,True)
        diagnostics = pricer.get_diagnostics()

        expectedYtm = -0.0085885615306637748
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-3) #tolerance of 10bps

    def test_yield_to_maturity_corp_instrcode_1911767(self):
        #test accuracy of YTM calculation versus TR data vendor
        valuationDate = '2010-07-06'
        issueDate = '2009-08-17'
        maturityDate = '2021-08-17'
        currency = 'GBP'
        coupon = 5.75 / 100.0
        cleanPrice = 102.354419
        #cleanPricePar = 100
        #swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=1.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice,True)
        diagnostics = pricer.get_diagnostics()

        expectedYtm = 0.05459924
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-3) #tolerance of 10bps

    def test_YTM_corp_instrcode_113751(self):
        #test accuracy of swap spread calculation versus TR data vendor
        valuationDate = '2015-01-29'
        issueDate = '1996-06-20'
        maturityDate = '2016-06-20'
        currency = 'GBP'
        coupon = 9.5 / 100.0
        cleanPrice = 111.284062
        #cleanPricePar = 100
        swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=1.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice,True)
        diagnostics = pricer.get_diagnostics(swapCurve)

        expectedYtm = 1.211063/100
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10e-3) #tolerance of 10bps

    def test_YTM_corp_instrcode_2643615(self):
        #[TODO: test last coupon date]
        valuationDate = '2014-01-08'
        issueDate = '2011-12-15'
        maturityDate = '2028-12-15'
        FirstCpnDate='2012-01-15'
        currency = 'GBP'
        coupon = 5.5 / 100.0
        cleanPrice = 98.087834
        #cleanPricePar = 100
        swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=12.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice,True)
        diagnostics = pricer.get_diagnostics(swapCurve)

        expectedYtm = 1.211063/100
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10) #tolerance of 10bps


    def test_YTM_corp_instrcode_3205076(self):
        valuationDate = '2016-07-18'
        issueDate = '2013-09-10'
        maturityDate = '2018-09-10'
        FirstCpnDate='2014-03-10'
        currency = 'KRW'
        coupon = 3.25 / 100.0
        cleanPrice = 105.3553
        #cleanPricePar = 100
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0)
        actualYtm = pricer.compute_yield_to_maturity(cleanPrice, True)
        expectedYtm = 0.012584
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=10)

    def test_YTM_sov_instrcode_3782109(self):
        valuationDate = '2016-08-01'
        issueDate = '2015-02-06'
        maturityDate = '2017-02-06'
        currency = 'ISK'
        coupon = 5.0 / 100.0
        freq = 2.0
        settlement_adj = 2.0
        marketPrice = 99.46
        first_cpn_dt = '2016-02-06'
        last_cpn_dt = '2016-02-06'
        dirtyPriceFlag = False
        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
        actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=dirtyPriceFlag)
        expectedYtm = 5.943651 / 100.0 
        delta = 0.0010  #10 bps
        self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    # def test_YTM_sov_instrcode_3782109_2(self):
    #     valuationDate = '2016-12-22'
    #     issueDate = '2015-02-06'
    #     maturityDate = '2017-02-06'
    #     currency = 'ISK'
    #     coupon = 5.0 / 100.0
    #     freq = 1.0
    #     settlement_adj = 2.0
    #     marketPrice = 99.94
    #     first_cpn_dt = '2016-02-06'
    #     last_cpn_dt = '2016-02-06'
    #     dirtyPriceFlag = True
    #     bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
    #     actualYtm = bondPricer.compute_yield_to_maturity(marketPrice, priceIsClean=dirtyPriceFlag)
    #     expectedYtm = 0.053033
    #     delta = 0.0010  #10 bps
    #     self.assertAlmostEqual(actualYtm, expectedYtm, delta=delta)

    # def test_sov_instrcode_3782109(self):
    #     valuationDate = '2016-12-22'
    #     issueDate = '2015-02-06'
    #     maturityDate = '2017-02-06'
    #     currency = 'ISK'
    #     coupon = 5.0 / 100.0
    #     freq = 1.0
    #     settlement_adj = 2.0
    #     marketPrice = 99.94
    #     first_cpn_dt = '2016-02-06'
    #     last_cpn_dt = '2016-02-06'
    #     dirtyPriceFlag = True
    #     bp = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
    #     accured_interest = bp.compute_accrued_interest()

    def test_bond_diagnostics(self):
        valuationDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28'
        coupon = 0.0475

        bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
        diags = bondPricer.get_diagnostics()
        # print diags
        actualItems = len(diags)
        expectedItems = 5
        self.assertEqual(actualItems, expectedItems)

    def test_bond_diagnostics_with_discount_factors(self):
        valuationDate = '2014-02-27'
        issueDate = '2000-06-12'
        maturityDate = '2030-06-12'
        coupon = 2.85 / 100.0
        currency = 'JPY'

        swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
        diagnostics = pricer.get_diagnostics(swapCurve)
        actualItems = len(diagnostics.keys())
        expectedItems = 6
        self.assertEqual(actualItems, expectedItems)

    def test_bond_diagnostics_with_discount_factors(self):
        analysis_date = '2016-02-15'
        analysis_date2 = '2016-02-14'
        issue_date = '2015-08-15'
        maturity_date = '2045-08-15'
        coupon = 2.875 / 100.0
        currency = 'USD'

        govt_curve = create_govt_zero_curve(analysis_date, currency)
        pricer = bond.BondPricer(issue_date, maturity_date, coupon, analysis_date2)
        diagnostics = pricer.get_diagnostics(govt_curve)



    def test_yield_curve_discount_factor(self):
        valuationDate = '2014-05-01'
        curveName = 'US.USD.GVT.ZC'
        yieldCurve = create_yield_curve(valuationDate, curveName)
        timeInYears = 10.0
        actualDiscountFactor = yieldCurve.compute_discount_factor(timeInYears)
        expectedDiscountFactor = 0.7627
        self.assertAlmostEqual(actualDiscountFactor, expectedDiscountFactor, places=3)

        diagnostics = yieldCurve.get_diagnostics()
        self.assertTrue(len(diagnostics) > 0)


    def portfolio(self):
        valuationDate = '2014-05-01'
        curveName = 'US.USD.GVT.ZC'
        yieldCurve = create_yield_curve(valuationDate, curveName)

        marketPrices = [100.0, 100.0, 100.0]
        issueDates = ['2014-05-01', '2014-06-01', '2014-07-01']
        maturityDates = ['2020-06-15', '2020-07-15', '2020-08-15']
        coupons = [0.05, 0.05, 0.05]

        combinedTerms = zip(issueDates, maturityDates, coupons, marketPrices)
        for issueDate, maturityDate, coupon, marketPrice in combinedTerms:
            print "bond_terms: ", issueDate, maturityDate, coupon
            bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
            actualSpread = bondPricer.compute_spread(yieldCurve, marketPrice)
            spreadDurationAnalytic = bondPricer.compute_spread_duration_analytical(yieldCurve, marketPrice)
            spreadDurationNumerical = bondPricer.compute_spread_duration_numerical(yieldCurve, marketPrice)


    def bond_stat_write_to_database(self):
        currency = 'USD'
        issuerIds = ['100061304', '100081877', '100048688', '100061200', '100003112']
        curveName = 'US.USD.GVT.ZC'
        startDate = '2014-01-01'
        endDate = '2015-03-23'

        bondStatsWriter = bond.BondStatDatabaseWriter(startDate, endDate, issuerIds, currency, curveName)
        bondStatsWriter.write_to_db()


    def test_bond_stats_write_to_db(self):
        currency = 'GBP'
        yieldcurveName = 'GB.GBP.GVT.ZC'
        spreadcurveName = 'GB.GBP.SWP.ZCS'
        startDate = '2007-01-01'
        endDate = '2007-01-01'
        issuerId = '100082278'

        #bondStatsWriter = bond.BondStatDatabaseWriter(startDate, endDate, currency, yieldcurveName, spreadcurveName, issuerId, None, None, None)
        #bondStatsWriter.write_to_db()


    def assertInterpolationWorks(self, curve, timeInYears, expectedLevel):
        actualLevel = curve(timeInYears)
        self.assertAlmostEquals(actualLevel, expectedLevel, places=3)


    def test_InterpolatedCurve_main_use(self):
        valuationDate = '2014-01-01'
        curveName = 'US.USD.GVT.ZC'
        curve = InterpolatedCurve(valuationDate, curveName)
        df = curve.get_diagnostics()
        print df

        # test interpolation for 1 year
        timeInYears = 1.0
        expectedLevel = 0.001331
        self.assertInterpolationWorks(curve, timeInYears, expectedLevel)

        # test extrapolation (left hand side)
        timeInYears = 0.0
        expectedLevel = 0.000731
        self.assertInterpolationWorks(curve, timeInYears, expectedLevel)

        # test extrapolation (right hand side)
        timeInYears = 50.0
        expectedLevel = 0.0419
        self.assertInterpolationWorks(curve, timeInYears, expectedLevel)


    def test_InterpolatedCurve_Prod(self):
        valuationDate = '2014-01-01'
        curveName = 'US.USD.GVT.ZC'
        database = 'Prod'
        curve = InterpolatedCurve(valuationDate, curveName, database)

        # test interpolation for 1 year
        timeInYears = 1.0
        expectedLevel = 0.001331
        self.assertInterpolationWorks(curve, timeInYears, expectedLevel)

        # test extrapolation (left hand side)
        timeInYears = 0.0
        expectedLevel = 0.000731
        self.assertInterpolationWorks(curve, timeInYears, expectedLevel)

        # test extrapolation (right hand side)
        timeInYears = 50.0
        expectedLevel = 0.0419
        self.assertInterpolationWorks(curve, timeInYears, expectedLevel)


    def test_swap_curve_main_use(self):
        valuationDate = '2014-01-01'
        currency = 'USD'
        enviroment ='Prod'
        swapCurve = create_swap_zero_curve(valuationDate, currency, enviroment)
        timeInYears = 25.0  # 25 years
        actualSwapLevel = swapCurve(timeInYears)
        expectedSwapLevel = .04
        self.assertAlmostEqual(actualSwapLevel, expectedSwapLevel, places=2)

        actualDiscountFactor = swapCurve.compute_discount_factor(timeInYears)
        expectedDiscountFactor = .357
        self.assertAlmostEqual(actualDiscountFactor, expectedDiscountFactor, places=2)


    def test_bond_stats_main_use(self):
        # Bond Terms & Conditions| Instrument Code: 2119447
        valuationDate = '2014-01-01'
        currency = 'USD'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28'
        marketPrice = 100.392372
        coupon = 0.0475
        yieldToMaturity = 4.676294 / 100.0
        # ___________________________________________________

        swapCurve = create_swap_zero_curve(valuationDate, currency)
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
        actualSpreadOverYTM = pricer.compute_spread_over_ytm(swapCurve, yieldToMaturity)
        expectedSpreadOverYTM = 0.023
        self.assertAlmostEqual(actualSpreadOverYTM, expectedSpreadOverYTM, places=2)

        actualSpread = pricer.compute_spread(swapCurve, marketPrice)
        expectedSpread = 0.0248
        self.assertAlmostEqual(actualSpread, expectedSpread, places=3)

        actualMarketPrice = pricer.compute_clean_price(swapCurve, actualSpread)
        expectedMarketPrice = marketPrice
        self.assertAlmostEqual(actualMarketPrice, expectedMarketPrice, places=4)

        actualSpreadDuration = pricer.compute_spread_duration_numerical(swapCurve, marketPrice)
        expectedSpreadDuration = 5.51992811905
        self.assertAlmostEqual(actualSpreadDuration, expectedSpreadDuration, places=2)


    def test_BondStats_to_dbs(self):
        currency = 'USD'
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issuerId = '100082278'

        #bondStatsWriter = bond.BondStatDatabaseWriter(startDate, endDate, currency, yieldcurveName, spreadcurveName, issuerId, None, None, None)
        #bondStatsWriter.write_to_db()


    def test_split_bonds_writer(self):
        currency = 'USD'
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'
        startDate = '2014-03-01'
        endDate = '2014-03-05'
        issuerId = '100082278'
        timerange = pd.bdate_range (startDate, endDate)
        timerangelist = timerange.tolist()
        splitted_list = bond.split_list(timerangelist)
        #for interval in splitted_list:
            #bondStatsWriter = bond.BondStatDatabaseWriter(interval[0], interval[-1], currency, yieldcurveName, spreadcurveName, issuerId, None, None, None)
            #bondStatsWriter.write_to_db()


    def test_split_func(self):
        
        
        startDate = '2014-01-01'
        endDate = '2014-02-12'
        timerange = pd.bdate_range (startDate, endDate)
        timerangelist = timerange.tolist()
        splitted_list = bond.split_list(timerangelist)
        print splitted_list


    def test_dataframe(self):
        currency = 'USD'
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issuerId = '100082278'
        bondpf = db.DerivedDataCorpBondPriceForCcyResearch(valuationDate=startDate, currency=currency, issuerId=issuerId, instrumentCode=None, rating=None, sector=None, suffix='_RS')       
        bondDataFrame = bondpf.extract_from_db()


    def plot_entire_corporate_bonds(self):
        tradeStartDate = '2015-04-08'
        tradeEndDate = '2015-04-08'
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'
        bondStatsPlot = bond.bondStatsPlot(tradeStartDate, tradeEndDate, yieldcurveName, spreadcurveName)
        bondStatsPlot.bondstats_plotting()


    def test_bond_pricer_instrument_code(self):
        val = """
        exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyResearch] @tradeDateBegin='01/01/2014 12:00:00 AM',  @tradeDateEnd='01/01/2014 12:00:00 AM',  @currencyISO='USD', @IssuerId='100082278'
        InstrCode	IssName	Value_	FrstCpnRate	InitAccDate	MatDate	OrgCode	DebtISOCurCode	ISOCtryCode	NIACSLevel4	DebtIssTypeCode	IssTypeCode	SnrtyCode	OrgTypeCode	MoodyRating	S&PRating	TradeDate	Prc	WrstStdYld	WrstCorpYld	MatCorpYld	MatStdYld	Amt	Isin	MaxAmtDate	OrgName	MoodyCompositeScore	S&PCompositeScore	CompRatScore	CompositeRatingEnum	GicsLevel1	UltParIsrId	UltParIsrShortName
        2119447	4.75% Fxd Rt MTN Due 2020	52211	4.75	2010-04-28 00:00:00.000	2020-04-28 00:00:00.000	2371	USD	GB	Commercial Banking	NT	CORP	SR	COM	A2	A	2014-01-01 00:00:00.000	100.392372	4.676294	4.676294	4.676294	4.676294	1678000	US06738JAY01	2010-04-28 00:00:00.000	BARCLAYS BANK PLC	6	6	6	A	Financials	100082278	BARCLAYS
        """

        startDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28'
        cleanMarketPrice = 100.392372
        coupon = 0.0475

        pricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate)
        actualYTM = pricer.compute_yield_to_maturity(cleanMarketPrice, priceIsClean=True) * 100
        expectedYTM = 4.62
        delta = .05
        self.assertAlmostEqual(actualYTM, expectedYTM, delta=delta)


    def assertYieldToMaturityIsExpected(self, pricer, marketPrice, expectedYTM, delta=.05):
        actualYTM = pricer.compute_yield_to_maturity(marketPrice)
        self.assertAlmostEqual(actualYTM, expectedYTM, delta=delta)


    def test_bond_pricer_instrument_code_2119447(self):
        val = """
        exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyResearch] @tradeDateBegin='01/01/2014 12:00:00 AM',  @tradeDateEnd='01/01/2014 12:00:00 AM',  @currencyISO='USD', @IssuerId='100082278'
        InstrCode	IssName	Value_	FrstCpnRate	InitAccDate	MatDate	OrgCode	DebtISOCurCode	ISOCtryCode	NIACSLevel4	DebtIssTypeCode	IssTypeCode	SnrtyCode	OrgTypeCode	MoodyRating	S&PRating	TradeDate	Prc	WrstStdYld	WrstCorpYld	MatCorpYld	MatStdYld	Amt	Isin	MaxAmtDate	OrgName	MoodyCompositeScore	S&PCompositeScore	CompRatScore	CompositeRatingEnum	GicsLevel1	UltParIsrId	UltParIsrShortName
        2119447	4.75% Fxd Rt MTN Due 2020	52211	4.75	2010-04-28 00:00:00.000	2020-04-28 00:00:00.000	2371	USD	GB	Commercial Banking	NT	CORP	SR	COM	A2	A	2014-01-01 00:00:00.000	100.392372	4.676294	4.676294	4.676294	4.676294	1678000	US06738JAY01	2010-04-28 00:00:00.000	BARCLAYS BANK PLC	6	6	6	A	Financials	100082278	BARCLAYS
        """
        startDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28'
        marketPrice = 100.392372
        coupon = 0.0475
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate)
        expectedYTM = 0.04731
        delta = .05
        self.assertYieldToMaturityIsExpected(pricer, marketPrice, expectedYTM, delta=delta)


    def test_bond_pricer(self):
        issueDate = '2014-12-11'
        tradeDate = '2014-01-01'
        maturityDate = '2015-12-31'
        coupon = 0.01
        freq = 2.0
        dateflow = finance.dateflow_generator(coupon, enddate_or_integer=2, start_date=issueDate, step='6m', cashflowtype='bullit', profile='payment')
        valuationDateToTime = finance.DateToTime(valuation_date = tradeDate,daycount_method='act/365')
        timeFlow = finance.TimeFlow(date_to_time=valuationDateToTime, dateflow=dateflow)        
        print dateflow

    def test_cashflow_generator_Frequency_Quaterly(self):
        issDate = '2014-01-15'
        tradeDate = '2014-01-20'
        maturityDate = '2015-01-15'
        coupon = 0.01
        freq = 4.0
        bondpricer = bond.BondPricer(issDate,maturityDate,coupon,tradeDate,freq)
        expectedCashFlowNumber = 5
        print 'cashflowNumber:', len(bondpricer.cashflows)
        self.assertAlmostEqual(expectedCashFlowNumber, len(bondpricer.cashflows))



    def test_bond_pricer_instrument_code_2354480(self):
        val = """
        exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyResearch] @tradeDateBegin='01/01/2014 12:00:00 AM',  @tradeDateEnd='01/01/2014 12:00:00 AM',  @currencyISO='USD', @IssuerId='100082278'
        InstrCode	IssName	Value_	FrstCpnRate	InitAccDate	MatDate	OrgCode	DebtISOCurCode	ISOCtryCode	NIACSLevel4	DebtIssTypeCode	IssTypeCode	SnrtyCode	OrgTypeCode	MoodyRating	S&PRating	TradeDate	Prc	WrstStdYld	WrstCorpYld	MatCorpYld	MatStdYld	Amt	Isin	MaxAmtDate	OrgName	MoodyCompositeScore	S&PCompositeScore	CompRatScore	CompositeRatingEnum	GicsLevel1	UltParIsrId	UltParIsrShortName
        2354480	3.50% Fxd Rt MTN Due 2018	52211	3.5	2011-01-27 00:00:00.000	2018-01-27 00:00:00.000	2371	USD	GB	Commercial Banking	NT	CORP	SR	COM	A2	A	2014-01-01 00:00:00.000	100.866757	3.269814	3.269814	3.269814	3.269814	216000	US06738J7F56	2011-01-27 00:00:00.000	BARCLAYS BANK PLC	6	6	6	A	Financials	100082278	BARCLAYS
        """
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issueDate = '2011-01-27'
        maturityDate = '2018-01-27'
        cleanMarketPrice = 100.866757
        coupon = 0.035

        pricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate)
        actualYTM = pricer.compute_yield_to_maturity(cleanMarketPrice, priceIsClean=True) * 100
        expectedYTM = 3.269814
        delta = .05
        self.assertAlmostEqual(actualYTM, expectedYTM, delta=delta)


    def test_custom_curve_main_use(self):
        c = [1,1,1]
        def func(t):
            return c[0] + c[1]*t + c[2]*(t**2)

        t = [0, 0.5, 200.0]

        curve = CustomCurve(func, t)
        yieldLevel = curve(.5)

    def test_curve_with_array_input(self):
        t = [0, 0.5, 200.0]
        c = [1,1,1]
        def func(t):
            return c[0] + c[1]*t + c[2]*(t**2)
        curve = CustomCurve(func, t)
        yieldLevel = curve.compute_discount_factor(t)

    def test_bondpricer_with_frequency(self):
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issueDate = '2011-01-27'
        maturityDate = '2018-01-27'
        frequency = 1.0
        coupon = 0.035
        pricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate, frequency)
        pricer.get_diagnostics()
        print pricer.cashflowDates 

    def test_command_line(self):
        startDate = '2015-10-14'
        endDate = '2015-10-14'
        currency = 'USD'
        govtCurveName = 'US.USD.GVT.ZC'
        swpSpreadCurveName = 'US.USD.SWP.ZCS'
        rating = None
        sector = None
        issuerId = '100061200'
        instrumentCode = None
        enviroment = 'DEV' 
        config = None
        arg_list = [(startDate, endDate, currency, govtCurveName, swpSpreadCurveName, rating, sector, issuerId,
                    instrumentCode, enviroment, config)]
        map(bond.write_stats_to_db, arg_list)
 
    def test_NSCurve(self):
        L=0.33
        b1=0.01
        b2=0.002
        b3=-0.02

        t = np.array([1, 2, 3,4,5,10,20,30])

        curve = NSCurve(L,b1,b2,b3)

        y = curve.compute_discount_factor(t)

    def test_NS_bond_pricer(self):
        analysis_date = '2016-02-15'
        issue_date = '2015-08-15'
        maturity_date = '2045-08-15'
        coupon = 2.875 / 100.0
        currency = 'USD'

        L=0.33
        b1=0.01
        b2=0.002
        b3=-0.02

        #govt_curve = create_govt_zero_curve(analysis_date, currency)
        curve = NSCurve(L,b1,b2,b3)
        pricer = bond.BondPricer(issue_date, maturity_date, coupon, analysis_date)
        prc = pricer.compute_dirty_price(yieldCurve=curve)

if __name__ == '__main__':
    unittest.main()
