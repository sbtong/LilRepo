import macpy.bond as bond 
import macpy.bondPortfolio as bondPortfolio
import macpy.nelsonsiegel as ns
import pandas as pd
import string
import datetime
import macpy.utils.database as db
import numpy as np
import unittest


class Test_NelsonSeigel(unittest.TestCase):

    def test_nelsonSeigel(self):
        lambdaValue = .01
        b1 = .01
        b2 = .02
        b3 = .04
        tau = [.03, .04, 0.5]

        yieldPricer = ns.NelsonSiegel(lambdaValue, b1, b2, b3, tau)

        actualYieldValues = yieldPricer.compute_zero_yield()
        expectedYieldValues = np.array([ 0.030003  ,  0.030004  ,  0.03004975])
        np.testing.assert_almost_equal(actualYieldValues, expectedYieldValues, 5)
        


    def test_accruedInterest(self):
        valuationDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28' 
        coupon = 0.0475

        bondPricer = bond.create_pricer(issueDate, maturityDate, coupon, valuationDate)
        actualAccruedInterest = bondPricer.compute_accrued_interest()
        expectedAccruedInterest = 0.8523972602739726
        self.assertAlmostEquals(actualAccruedInterest, expectedAccruedInterest, places=4)


    def test_bond_price(self):
        issueDate = '2014-01-05'
        maturityDate = '2020-06-15'
        valuationDate = '2014-05-01'
        coupon = 0.05
        spreadvalue=0.0646
   
        bondPricer = bond.create_pricer(issueDate, maturityDate, coupon, valuationDate)

        curveName = 'US.USD.GVT.ZC'    
        yieldCurve = bond.create_yield_curve(valuationDate, curveName)

        actualModelPrice = bondPricer.compute_price(yieldCurve, spreadvalue)
        print "modelPrice:", actualModelPrice
        #expectedModelPrice = 100.00    
        #np.testing.assert_almost_equal(expectedModelPrice, actualModelPrice)        
    
    
        marketPrice = 100.0
        actualSpread = bondPricer.compute_spread(yieldCurve, marketPrice)
        print "spreadVal:", actualSpread
        #expectedSpread = 0
        #np.testing.assert_almost_equal(expectedSpread, float(actualSpread))    

        #expectedSpreadDuration = 4
        actualSpreadDuration_Analytical = bondPricer.compute_spread_duration_analytical(yieldCurve, marketPrice)
        SpreadDuration_Numerical = bondPricer.compute_spread_duration_numerical(yieldCurve, marketPrice)
        print "Analytical_SprDur:", actualSpreadDuration_Analytical, "Numerical_SprDur:", SpreadDuration_Numerical
        #np.testing.assert_almost_equal(expectedSpreadDuration, float(actualSpreadDuration)) 
    
    def test_bond_diagnostics(self):
        valuationDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28' 
        marketPrice = 100.392372
        coupon = 0.0475

   
        bondPricer = bond.create_pricer(issueDate, maturityDate, coupon, valuationDate)
        diags = bondPricer.get_diagnostics()
        #print diags
        actualItems = len(diags)
        expectedItems = 4
        self.assertEqual(actualItems, expectedItems)

    def test_bond_vipin(self):
        valuationDate = '2008-10-29'
        startDate = '2008-09-30'
        maturityDate = '2013-09-30'
        marketPrice =101.718748 
        coupon = 0.03125
        
        bondPricer = bond.BondPricer(startDate, maturityDate, coupon, valuationDate)
        corpYield = bondPricer.compute_CorpYield(marketPrice)
        accruedInterest = bondPricer.compute_accrued_interest()
        
        print corpYield, accruedInterest        

    def test_yield_curve(self):
        valuationDate = '2014-05-01'
        curveName = 'US.USD.GVT.ZC'
        yieldCurve = bond.create_yield_curve(valuationDate, curveName)
        timeInYears = 10.0
        actualDiscountFactor = yieldCurve.compute_discount_factor(timeInYears)
        print "DF:", actualDiscountFactor
        #expectedDiscountFactor = 0.7627588025148106472297310834
        #np.testing.assert_almost_equal(expectedDiscountFactor, actualDiscountFactor)
   
    def test_portfolio(self):
        valuationDate = '2014-05-01'
        curveName = 'US.USD.GVT.ZC'
        yieldCurve = bond.create_yield_curve(valuationDate, curveName)    
    
        marketPrices = [100.0, 100.0, 100.0, 100.0]
        issueDates = ['2014-05-01','2014-06-01','2014-07-01']
        maturityDates = ['2020-06-15','2020-07-15','2020-08-15']
        coupons = [0.05, 0.05, 0.05]
    
        combinedTerms = zip(issueDates, maturityDates, coupons, marketPrices)
        for issueDate, maturityDate, coupon, marketPrice in combinedTerms:
            print "bond_terms: ", issueDate, maturityDate, coupon        
            bondPricer = bond.BondPricer(issueDate, maturityDate, coupon, valuationDate)
            actualSpread = bondPricer.compute_spread(yieldCurve, marketPrice)
            SD_Analytical = bondPricer.compute_spread_duration_analytical (yieldCurve, marketPrice)
            SD_Numerical = bondPricer.compute_spread_duration_numerical (yieldCurve, marketPrice)
            print "actualSpread:", actualSpread, "SD_Analytical:", SD_Analytical, "SD_Numerical:", SD_Numerical
        
    def test_bond_stat_write_to_database(self):
        currency = 'USD'
        issuerIds = ['100061304','100081877','100048688','100061200','100003112']
        curveName = 'US.USD.GVT.ZC'
        startDate = '2014-01-01'
        endDate = '2015-03-23'

        bondStatsWriter = bond.BondStatDatabaseWriter(startDate, endDate, issuerIds, currency, curveName)
        bondStatsWriter.write_to_db()

    def test_yieldCurve_for_2015(self):
        curveName = 'US.USD.GVT.ZC'
        startDate = '2015-01-29'
        yieldCurve = bond.create_yield_curve(startDate, curveName)

    def test_swap_curve(self):
        valuationDate = '2014-01-01'
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'
        swapcurveTest = bond.SwapCurve(valuationDate, yieldcurveName, spreadcurveName)
        swapcurveTest.create_linear_swapCurve()
        swapcurveTest.get_continuous_forwardRate(1)
        swapcurveTest.compute_discount_factor(0.6)
    
    def test_spread_over_swap(self):
        issueDate = '2014-05-01'
        MaturityDate ='2020-06-15'
        coupon = 0.05
        CorpYield = 0.07
        valuationDate = '2014-05-01'
        swapcurve = bond.create_swap_curve(valuationDate, 'US.USD.GVT.ZC','US.USD.SWP.ZCS')
        bondpricer = bond.BondPricer(issueDate, MaturityDate, coupon, valuationDate)
        spread_over_swap = bondpricer.compute_spread_over_swap_noTermstructure(valuationDate, MaturityDate, swapcurve, CorpYield)
        spread_over_swap_withTermStructure = bondpricer.compute_spread_over_swap_withTermStr(swapcurve, 101)
        spreadDuration = bondpricer.compute_spreadDuration_over_swap(swapcurve, 101)
        print spreadDuration

    def testBondStats_to_dbs(self):
        currency = 'USD'       
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issuerId = '100082278'     
        
        bondStatsWriter = bond.BondStatDatabaseWriter(startDate, endDate, currency, yieldcurveName, spreadcurveName, issuerId)
        bondStatsWriter.write_to_db()



    def test_CorpYield(self):
        issueDate = '2014-05-01'
        MaturityDate ='2020-06-15'
        coupon = 0.05
        valuationDate = '2014-05-01'
        marketPrice = 97
        bondpricer = bond.BondPricer(issueDate, MaturityDate, coupon, valuationDate)
        CorpYield = bondpricer.compute_CorpYield(marketPrice)
        print CorpYield

    def test_plot(self):

        tradeStartDate ='2015-04-08'
        tradeEndDate = '2015-04-08'
        currency = 'USD'
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'

        bondstatsplot = bond.bondStatsPlot(tradeStartDate, tradeEndDate)
        bondstatsplot.bondstats_plotting()

    def test_bond_pricer_instrument_code(self):
        val="""
        exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyResearch] @tradeDateBegin='01/01/2014 12:00:00 AM',  @tradeDateEnd='01/01/2014 12:00:00 AM',  @currencyISO='USD', @IssuerId='100082278'
        InstrCode	IssName	Value_	FrstCpnRate	InitAccDate	MatDate	OrgCode	DebtISOCurCode	ISOCtryCode	NIACSLevel4	DebtIssTypeCode	IssTypeCode	SnrtyCode	OrgTypeCode	MoodyRating	S&PRating	TradeDate	Prc	WrstStdYld	WrstCorpYld	MatCorpYld	MatStdYld	Amt	Isin	MaxAmtDate	OrgName	MoodyCompositeScore	S&PCompositeScore	CompRatScore	CompositeRatingEnum	GicsLevel1	UltParIsrId	UltParIsrShortName
        2119447	4.75% Fxd Rt MTN Due 2020	52211	4.75	2010-04-28 00:00:00.000	2020-04-28 00:00:00.000	2371	USD	GB	Commercial Banking	NT	CORP	SR	COM	A2	A	2014-01-01 00:00:00.000	100.392372	4.676294	4.676294	4.676294	4.676294	1678000	US06738JAY01	2010-04-28 00:00:00.000	BARCLAYS BANK PLC	6	6	6	A	Financials	100082278	BARCLAYS
        """
       
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28' 
        marketPrice = 100.392372
        coupon = 0.0475

        bondpricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate)  
        actualCorpYield = bondpricer.compute_CorpYield(marketPrice)*100

        expectedYield = 4.676294
        toleranceDigits = 4 
        self.assertAlmostEqual(actualCorpYield, expectedYield, places=toleranceDigits)
        

    def test_bond_pricer_instrument_code_2119447(self):
        val="""
        exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyResearch] @tradeDateBegin='01/01/2014 12:00:00 AM',  @tradeDateEnd='01/01/2014 12:00:00 AM',  @currencyISO='USD', @IssuerId='100082278'
        InstrCode	IssName	Value_	FrstCpnRate	InitAccDate	MatDate	OrgCode	DebtISOCurCode	ISOCtryCode	NIACSLevel4	DebtIssTypeCode	IssTypeCode	SnrtyCode	OrgTypeCode	MoodyRating	S&PRating	TradeDate	Prc	WrstStdYld	WrstCorpYld	MatCorpYld	MatStdYld	Amt	Isin	MaxAmtDate	OrgName	MoodyCompositeScore	S&PCompositeScore	CompRatScore	CompositeRatingEnum	GicsLevel1	UltParIsrId	UltParIsrShortName
        2119447	4.75% Fxd Rt MTN Due 2020	52211	4.75	2010-04-28 00:00:00.000	2020-04-28 00:00:00.000	2371	USD	GB	Commercial Banking	NT	CORP	SR	COM	A2	A	2014-01-01 00:00:00.000	100.392372	4.676294	4.676294	4.676294	4.676294	1678000	US06738JAY01	2010-04-28 00:00:00.000	BARCLAYS BANK PLC	6	6	6	A	Financials	100082278	BARCLAYS
        """
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issueDate = '2010-04-28'
        maturityDate = '2020-04-28' 
        marketPrice = 100.392372
        coupon = 0.0475

        bondpricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate)  
        actualCorpYield = bondpricer.compute_CorpYield(marketPrice)*100

        expectedYield = 4.676294
        toleranceDigits = 4 
        self.assertAlmostEqual(actualCorpYield, expectedYield, places=toleranceDigits)


    def test_bond_pricer_instrument_code_2354480(self):
        val="""
        exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyResearch] @tradeDateBegin='01/01/2014 12:00:00 AM',  @tradeDateEnd='01/01/2014 12:00:00 AM',  @currencyISO='USD', @IssuerId='100082278'
        InstrCode	IssName	Value_	FrstCpnRate	InitAccDate	MatDate	OrgCode	DebtISOCurCode	ISOCtryCode	NIACSLevel4	DebtIssTypeCode	IssTypeCode	SnrtyCode	OrgTypeCode	MoodyRating	S&PRating	TradeDate	Prc	WrstStdYld	WrstCorpYld	MatCorpYld	MatStdYld	Amt	Isin	MaxAmtDate	OrgName	MoodyCompositeScore	S&PCompositeScore	CompRatScore	CompositeRatingEnum	GicsLevel1	UltParIsrId	UltParIsrShortName
        2354480	3.50% Fxd Rt MTN Due 2018	52211	3.5	2011-01-27 00:00:00.000	2018-01-27 00:00:00.000	2371	USD	GB	Commercial Banking	NT	CORP	SR	COM	A2	A	2014-01-01 00:00:00.000	100.866757	3.269814	3.269814	3.269814	3.269814	216000	US06738J7F56	2011-01-27 00:00:00.000	BARCLAYS BANK PLC	6	6	6	A	Financials	100082278	BARCLAYS
        """
        startDate = '2014-01-01'
        endDate = '2014-01-01'
        issueDate = '2011-01-27'
        maturityDate = '2018-01-27' 
        marketPrice = 100.866757
        coupon = 0.035

        bondpricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate)  
        actualCorpYield = bondpricer.compute_CorpYield(marketPrice)*100

        expectedYield =3.269814
        toleranceDigits = 1 
        self.assertAlmostEqual(float(actualCorpYield), expectedYield, places=toleranceDigits)

    def test_bond_ytm_big_data_set(self):
        currency = 'USD'       
        yieldcurveName = 'US.USD.GVT.ZC'
        spreadcurveName = 'US.USD.SWP.ZCS'
        valuationDate = '2014-01-01'
        issuerId = '100082278'

        bondpf = bondPortfolio.BondPortfolio(valuationDate, currency, issuerId)       
        bondpf.extract_from_db()        

        #bondpricer = bond.BondPricer(issueDate, maturityDate, coupon, startDate)  
        #actualCorpYield = bondpricer.compute_CorpYield(marketPrice)*100

        #expectedYield =3.269814
        #toleranceDigits = 1 
        #self.assertAlmostEqual(float(actualCorpYield), expectedYield, places=toleranceDigits)

            
  
    
            
if __name__ == '__main__':
    unittest.main()
