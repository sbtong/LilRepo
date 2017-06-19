
import unittest
import numpy as np
import macpy.bootstrapper as bs
import macpy.bond as bond
import macpy.utils.database as db
import pandas as pd
import datetime
from macpy import bond_utils
from dateutil.relativedelta import relativedelta
import macpy.bootstrapper_creator as bc


class TestBEI(unittest.TestCase):

    def test_BEI_Generation(self):
        country = 'TH'
        currency = 'THB'
        startDate = '2016-10-28'
        endDate = '2016-10-28'
        yieldsWriter = bs.BEI_Curve(startDate, endDate, currency, countryName = country, enviroment='DEV')
        yieldsWriter.write_to_dbs()

    def test_instrcode_2512773(self):
        issueDate = '2011-07-14'
        maturityDate = '2021-07-14'
        firstCpnDate = '2012-01-14'
        lastCpnDate = '2021-07-14'
        coupon = 1.20/100
        cleanPrice = 98.192
        tradeDate = '2016-10-28'
        bondpricer = bond.BondPricer(issueDate,maturityDate,coupon,tradeDate,first_cpn_dt=firstCpnDate,last_cpn_dt=lastCpnDate)
        realyield = bondpricer.compute_yield_to_maturity(cleanPrice)
        expectedYield = 1.5989/100.0
        self.assertAlmostEqual(realyield,expectedYield, delta =10e-3)

    def test_real_yields_TH(self):
        country = 'TH'
        currency = 'THB'
        startDate = '2016-10-28'
        endDate = '2016-10-28'

        CustomTimeInYearsList=[1,2,5,10,30]

        conventions = bond_utils.get_BEI_conventions(currency)
        bootstrapper = bc.Bootstrapper_BEI(startDate, endDate, currency, conventions, 'DEV')
        yieldCurve = bootstrapper.createYieldCurve()
        real_yield_list = yieldCurve(CustomTimeInYearsList)
        expected5Y = 1.3698/100
        self.assertAlmostEqual(real_yield_list[2], expected5Y, delta=10e-3)

    def test_real_yields_SE(self):
        currency = 'SEK'
        startDate = '2016-03-10'
        endDate = '2016-03-10'

        CustomTimeInYearsList=[1,2,5,10,30]

        conventions = bond_utils.get_BEI_conventions(currency)
        bootstrapper = bc.Bootstrapper_BEI(startDate, endDate, currency, conventions, 'DEV')
        yieldCurve = bootstrapper.createYieldCurve()
        real_yield_list = yieldCurve(CustomTimeInYearsList)
        print real_yield_list[0]


