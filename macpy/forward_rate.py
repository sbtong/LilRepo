import pandas as pd
import numpy as np

from irr import compute_irr, compute_npv
import macpy.bond as bond
import datetime
import dateutil

from macpy.curve_wrappers import CustomYieldCurve 

class ForwardRate:       
    def __init__(self,
                 forwardRateList,
                 maturityInYearsList,
                 maturityDateList,
                 YieldList,
                 issueDate,
                 maturityDate,
                 marketPrice,
                 valuationDate,
                 coupon,
                 MarketStandardYield,
                 freq=2.0,
                 settlement_adj=0.0,
                 first_cpn_dt=None,
                 last_cpn_dt=None,
                 use_clean_price_for_first_bond=False,
                 adjust_market_price=False,
                 PriceInclAccrIntFlg='n',
                 notional=100.0,
                 holidays=None):
        self.issueDate = issueDate
        self.LastMaturityDate = maturityDateList[-1]
        self.valuationDate = valuationDate
        self.coupon = coupon
        self.notional = notional
        self.first_cpn_dt = first_cpn_dt
        self.last_cpn_dt = last_cpn_dt
        self.freq =float(freq) if freq is not None else 2.0
        self.settlement_adj = settlement_adj
        self.maturityDate=maturityDate
        self.holidays = holidays
        self.termToMaturity= (dateutil.parser.parse(self.maturityDate)-dateutil.parser.parse(self.valuationDate)).total_seconds()/(365.25*24*3600)
        self.bondPricer = bond.BondPricer(self.issueDate, self.maturityDate, self.coupon, self.valuationDate, self.freq, self.settlement_adj, self.first_cpn_dt, self.last_cpn_dt, self.notional)
        self.accruedInterest = self.bondPricer.compute_accrued_interest()
        self.MarketStandardYield = MarketStandardYield
        try:
            if adjust_market_price == True and PriceInclAccrIntFlg=='y':
                self.dirtyMarketPrice = self.bondPricer.compute_dirty_price(yieldCurve=None, spreadValue=self.MarketStandardYield)
                self.cleanMarketPrice = self.bondPricer.compute_clean_price(yieldCurve=None, spreadValue=self.MarketStandardYield)
            else:
                self.cleanMarketPrice = marketPrice
                self.dirtyMarketPrice = marketPrice + self.accruedInterest
        except:
            print "market price adjustment error for stardardYield at ", self.MarketStandardYield
        self.use_clean_price_for_first_bond = use_clean_price_for_first_bond
        self.forwardRateList = forwardRateList
        self.maturityInYearsList = maturityInYearsList
        self.maturityDateList = maturityDateList
        self.YieldList = YieldList
        self.yieldCurve = CustomYieldCurve(self.forwardRateList, self.maturityInYearsList)

    def compute_forward_rate(self, priceIsClean = False):
        use_dirty_market_price = True
        # Turn off accrued interest for forward bonds except
        # when there is a currency override for short end bonds (remaining term less than 1)
        # some countries observe this convention
        if self.use_clean_price_for_first_bond and self.termToMaturity < 1.0:
            priceIsClean = True
            use_dirty_market_price = False

        forwardPrice = self.compute_forward_price(use_dirty_market_price=use_dirty_market_price)
        bondPricer = bond.BondPricer(self.issueDate,
                                     self.maturityDate,
                                     self.coupon,
                                     self.LastMaturityDate,
                                     self.freq,
                                     self.settlement_adj,
                                     self.first_cpn_dt,
                                     self.last_cpn_dt,
                                     self.notional, 
                                     self.holidays)

        forwardRate = bondPricer.compute_yield_to_maturity(forwardPrice, priceIsClean=priceIsClean, max_iter=1000)
        #print 'Maturity Date: ', self.maturityDate, 'forwardRate: ', forwardRate
        #print 'Cashflow dates: ', bondPricer.cashflowDates, 'forward cashflows: ', bondPricer.cashflows
        self.forwardRateList.append(forwardRate)
        self.maturityDateList.append(self.maturityDate)
        self.maturityInYearsList.append(self.termToMaturity)
        yieldCurveNew = CustomYieldCurve(self.forwardRateList, self.maturityInYearsList)
        yieldRate = yieldCurveNew(self.maturityInYearsList[-1])
        self.YieldList.append(yieldRate)

    def compute_forward_price(self, use_dirty_market_price=True):
        timeToMaturity = (dateutil.parser.parse(self.LastMaturityDate)-dateutil.parser.parse(self.valuationDate)).total_seconds()/(365.25*24*3600)
        cashflowPrice = self.compute_earlyCashflow_price()
        discountedForwardPrice = (self.dirtyMarketPrice if use_dirty_market_price else self.cleanMarketPrice) - float(cashflowPrice)
        yieldRate = self.yieldCurve(timeToMaturity)
        ForwardPrice = discountedForwardPrice*np.exp(yieldRate*timeToMaturity)        
        return ForwardPrice

    def compute_earlyCashflow_price(self):                              
        cashflowDates = self.bondPricer.cashflowDates
        cashflows = self.bondPricer.cashflows
        cashflowInYears = self.bondPricer.timeToCashflowInYears
        earlyCashFlows=[(x,y,z) for (x,y,z) in zip(cashflowDates, cashflows, cashflowInYears) if x <= datetime.datetime.strptime(self.LastMaturityDate, '%Y-%m-%d')]
        earlyDiscountFactors = pd.Series([self.yieldCurve.compute_discount_factor(x[2]) for x in earlyCashFlows])
        earlyFlowInYears = pd.Series([x[2] for x in earlyCashFlows])
        earlyCashFlows = pd.Series([x[1] for x in earlyCashFlows])
        earlyCashflowPrice = compute_npv(earlyFlowInYears, earlyCashFlows*earlyDiscountFactors, 0.0)             
        return earlyCashflowPrice
