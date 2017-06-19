from macpy.spec_risk import SpecRisk
import unittest
import pandas as pd

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class TestSpecRisk(unittest.TestCase):
    def test_spec_risk_HC_AR_two_dates(self):
        df = pd.read_csv(dir_path + '/data/AR.USD.OAS.csv')
        params = {}
        params['country'] = 'AR'
        params['currency'] = 'USD'

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2016-08-15'
        sr = SpecRisk(**params)
        sr.run()
        median_1 = sr.median

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2016-08-17'
        sr = SpecRisk(**params)
        sr.run()
        median_2 = sr.median

        self.assertEqual(median_1, median_2)

    def test_spec_risk_HC_AR_20160819(self):
        df = pd.read_csv(dir_path + '/data/AR.USD.OAS.csv')
        params = {}
        params['country'] = 'AR'
        params['currency'] = 'USD'

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2016-08-19'
        sr = SpecRisk(**params)
        sr.run()
        median = float(sr.median)

        actual_value = 0.0323842717668

        delta = 0.0001  #1 bps
        self.assertAlmostEqual(median, actual_value, delta=delta)

    def test_spec_risk_HC_AR_two_dates_diff(self):
        df = pd.read_csv(dir_path + '/data/AR.USD.OAS.csv')
        params = {}
        params['country'] = 'AR'
        params['currency'] = 'USD'

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2016-08-08'
        sr = SpecRisk(**params)
        sr.run()
        median_1 = float(sr.median)

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2016-08-22'
        sr = SpecRisk(**params)
        sr.run()
        median_2 = float(sr.median)

        actual_value_1 = 0.0331470046343
        actual_value_2 = 0.0323842717668

        delta = 0.0005  #5 bps
        self.assertNotAlmostEqual(median_1, median_2, delta=delta)

        delta = 0.0001
        self.assertAlmostEqual(median_1, actual_value_1, delta=delta)
        self.assertAlmostEqual(median_2, actual_value_2, delta=delta)

    def test_spec_risk_local_ZAR_20150223(self):
        df = pd.read_csv(dir_path + '/data/ZA.ZAR.OAS.csv')
        params = {}
        params['country'] = 'ZA'
        params['currency'] = 'ZAR'

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2015-02-23'
        sr = SpecRisk(**params)
        sr.run()
        median = float(sr.median)

        actual_value = 0.00314059304105

        delta = 0.0001  #1 bps
        self.assertAlmostEqual(median, actual_value, delta=delta)

    def test_spec_risk_local_ZAR_20160727(self):
        df = pd.read_csv(dir_path + '/data/ZA.ZAR.OAS.csv')
        params = {}
        params['country'] = 'ZA'
        params['currency'] = 'ZAR'

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2016-07-27'
        sr = SpecRisk(**params)
        sr.run()
        median = float(sr.median)

        actual_value = 0.00167741301594

        delta = 0.0001  #1 bps
        self.assertAlmostEqual(median, actual_value, delta=delta)

    def test_spec_risk_local_GR_20120806(self):
        df = pd.read_csv(dir_path + '/data/GR.EUR.OAS.csv')
        params = {}
        params['country'] = 'GR'
        params['currency'] = 'EUR'

        params['df_deriv'] = df.copy(deep=True)
        params['trade_date'] = '2012-08-06'
        sr = SpecRisk(**params)
        sr.run()
        median = float(sr.median)

        actual_value = 0.0270492725642

        delta = 0.0001  #1 bps
        self.assertAlmostEqual(median, actual_value, delta=delta)