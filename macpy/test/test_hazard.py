import unittest
from macpy.dateflow import DateFlow
import datetime
import pandas as pd
from pandas.util.testing import assert_series_equal
import macpy.bond as bond


class Test_hazard(unittest.TestCase):
    def test_Uqk(self):
        issueDate = '2008-06-30'
        valuationDate = '2014-06-30'
        maturityDate = '2017-06-30'
        coupon = 0.05
        RecoveryRate=0.2
        Alpha=0.2
        Price=102.0
        currency='USD'
        environment='Dev'

        pricer = bond.create_pricer(issueDate, maturityDate, coupon, valuationDate)
        swapCurve = bond.create_swap_zero_curve(valuationDate, currency, environment)

        diagnostics = pricer.get_diagnostics(swapCurve)

        Ukq, V = pricer.compute_HazardUkq(swapCurve,RecoveryRate,Price,True,Alpha)

        self.assertAlmostEqual(Ukq[0],0.528920043,places=4)
        self.assertAlmostEqual(Ukq[1],0.309571492,places=4)
        self.assertAlmostEqual(Ukq[2],0.185647582,places=4)
        self.assertAlmostEqual(V,0.820326533,places=4)

    def test_Uqk_zeroRecoveryRate(self):
        issueDate = '2008-06-30'
        valuationDate = '2014-06-30'
        maturityDate = '2017-06-30'
        coupon = 0.05
        RecoveryRate=0.0
        Alpha=0.0
        Spread=0.0
        currency='USD'
        environment='Dev'

        pricer = bond.create_pricer(issueDate, maturityDate, coupon, valuationDate)
        swapCurve = bond.create_swap_zero_curve(valuationDate, currency, environment)
        DirtyPrice = pricer.compute_dirty_price(swapCurve,spreadValue=Spread)

        diagnostics = pricer.get_diagnostics(swapCurve)

        Ukq, V = pricer.compute_HazardUkq(swapCurve,RecoveryRate,DirtyPrice,False,Alpha)

        self.assertAlmostEqual(Ukq[0],DirtyPrice/100,places=4)
        self.assertAlmostEqual(Ukq[1],DirtyPrice/100,places=4)
        self.assertAlmostEqual(Ukq[2],DirtyPrice/100,places=4)
        self.assertAlmostEqual(V,DirtyPrice/100,places=4)