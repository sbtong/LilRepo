import unittest
from macpy.dateflow import DateFlow
import datetime
import pandas as pd
from pandas.util.testing import assert_series_equal
import macpy.bond as bond

class Test_Date_Flow(unittest.TestCase):
    def test_basic(self):
        issue = '2014-10-29'
        maturity = '2015-10-29'
        valuation = '2015-10-08'
        freq = 2.0
        coupon = 0.0000000001/freq
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2015, 10, 8), datetime.datetime(2015, 10, 29)])

    def test_basic_qtr(self):
        issue = '2014-10-31'
        maturity = '2015-10-29'
        valuation = '2014-10-31'
        frstCpnDate = '2015-1-29'
        freq = 4.0
        coupon = 0.0000000001/freq
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2014, 10, 31, 0, 0), datetime.datetime(2015, 1, 29, 0, 0), datetime.datetime(2015, 4, 29, 0, 0),
                                       datetime.datetime(2015, 7, 29, 0, 0), datetime.datetime(2015, 10, 29, 0, 0)])

    def test_basic_qtr2(self):
        issue = '2014-10-31'
        frstCpnDate = '2015-01-29'
        maturity = '2015-10-29'
        valuation = '2014-11-01'
        freq = 4.0
        coupon = 0.0000000001/freq
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2014, 11, 1, 0, 0), datetime.datetime(2015, 1, 29, 0, 0), datetime.datetime(2015, 4, 29, 0, 0),
                                       datetime.datetime(2015, 7, 29, 0, 0), datetime.datetime(2015, 10, 29, 0, 0)])

    def test_leap_year(self):
        maturity = '2012-02-29'
        issue = '2011-02-28'
        valuation = '2011-03-21'
        frstCpnDate = '2011-8-29'
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(0.0255, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2011, 3, 21, 0, 0), datetime.datetime(2011, 8, 29, 0, 0), datetime.datetime(2012, 2, 29, 0, 0)])
        assert_series_equal(cash_flows, pd.Series([0.0, 2.55, 102.55]))

    def test_annualCashflows(self):
        maturity = '2018-04-30'
        issue = '2013-04-30'
        valuation = '2015-04-15'
        frstCpnDate = '2014-4-30'
        freq=1.0
        coupon=0.07
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2015, 4, 15, 0, 0),
                                       datetime.datetime(2015, 4, 30, 0, 0),
                                       datetime.datetime(2016, 4, 30, 0, 0),
                                       datetime.datetime(2017, 4, 30, 0, 0),
                                       datetime.datetime(2018, 4, 30, 0, 0)]
                         )

    def test_annualCashflows2(self):
        maturity = '2018-04-22'
        issue = '2013-04-22'
        valuation = '2015-04-15'
        frstCpnDate = '2014-4-22'
        freq=1.0
        coupon=0.07
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2015, 4, 15, 0, 0),
                                       datetime.datetime(2015, 4, 22, 0, 0),
                                       datetime.datetime(2016, 4, 22, 0, 0),
                                       datetime.datetime(2017, 4, 22, 0, 0),
                                       datetime.datetime(2018, 4, 22, 0, 0)]
                         )

    def test_monthlyCashflows(self):
        maturity = '2018-04-22'
        issue = '2017-04-15'
        valuation = '2017-04-16'
        freq=12.0
        coupon=7
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2017, 4, 16, 0, 0),
                                       datetime.datetime(2017, 4, 22, 0, 0),
                                       datetime.datetime(2017, 5, 22, 0, 0),
                                       datetime.datetime(2017, 6, 22, 0, 0),
                                       datetime.datetime(2017, 7, 22, 0, 0),
                                       datetime.datetime(2017, 8, 22, 0, 0),
                                       datetime.datetime(2017, 9, 22, 0, 0),
                                       datetime.datetime(2017, 10, 22, 0, 0),
                                       datetime.datetime(2017, 11, 22, 0, 0),
                                       datetime.datetime(2017, 12, 22, 0, 0),
                                       datetime.datetime(2018, 1, 22, 0, 0),
                                       datetime.datetime(2018, 2, 22, 0, 0),
                                       datetime.datetime(2018, 3, 22, 0, 0),
                                       datetime.datetime(2018, 4, 22, 0, 0)]
                         )

    def test_monthlyCashflows2(self):
        maturity = '2018-04-22'
        issue = '2017-04-15'
        valuation = '2017-04-16'
        firstCpnDate='2017-05-22'
        freq=12.0
        coupon=7
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq,first_cpn_dt=firstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2017, 4, 16, 0, 0),
                                       datetime.datetime(2017, 5, 22, 0, 0),
                                       datetime.datetime(2017, 6, 22, 0, 0),
                                       datetime.datetime(2017, 7, 22, 0, 0),
                                       datetime.datetime(2017, 8, 22, 0, 0),
                                       datetime.datetime(2017, 9, 22, 0, 0),
                                       datetime.datetime(2017, 10, 22, 0, 0),
                                       datetime.datetime(2017, 11, 22, 0, 0),
                                       datetime.datetime(2017, 12, 22, 0, 0),
                                       datetime.datetime(2018, 1, 22, 0, 0),
                                       datetime.datetime(2018, 2, 22, 0, 0),
                                       datetime.datetime(2018, 3, 22, 0, 0),
                                       datetime.datetime(2018, 4, 22, 0, 0)]
                         )

    def test_monthlyCashflows2(self):
        maturity = '2018-04-30'
        issue = '2017-04-15'
        valuation = '2017-04-16'
        freq=12.0
        coupon=7
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2017, 4, 16, 0, 0),
                                       datetime.datetime(2017, 4, 30, 0, 0),
                                       datetime.datetime(2017, 5, 30, 0, 0),
                                       datetime.datetime(2017, 6, 30, 0, 0),
                                       datetime.datetime(2017, 7, 30, 0, 0),
                                       datetime.datetime(2017, 8, 30, 0, 0),
                                       datetime.datetime(2017, 9, 30, 0, 0),
                                       datetime.datetime(2017, 10, 30, 0, 0),
                                       datetime.datetime(2017, 11, 30, 0, 0),
                                       datetime.datetime(2017, 12, 30, 0, 0),
                                       datetime.datetime(2018, 1, 30, 0, 0),
                                       datetime.datetime(2018, 2, 28, 0, 0),
                                       datetime.datetime(2018, 3, 30, 0, 0),
                                       datetime.datetime(2018, 4, 30, 0, 0)]
                         )

    def test_samemonth(self):
        maturity = '2016-02-29'
        issue = '2015-02-28'
        valuation = '2016-02-01'
        frstCpnDate = '2015-8-29'
        freq=2.0
        coupon=0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2016, 2, 1, 0, 0), datetime.datetime(2016, 2, 29, 0, 0)])

    def test_samemonth2(self):
        maturity = '2016-02-17'
        issue = '2015-02-17'
        valuation = '2016-02-01'
        frstCpnDate = '2015-8-17'
        freq=2.0
        coupon=0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2016, 2, 1, 0, 0), datetime.datetime(2016, 2, 17, 0, 0)])

    def test_samemonth3(self):
        maturity = '2016-04-29'
        issue = '2015-02-17'
        valuation = '2016-02-01'
        frstCpnDate = '2015-02-28'
        freq=12.0
        coupon=0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2016, 2, 1, 0, 0), datetime.datetime(2016, 2, 29, 0, 0),datetime.datetime(2016, 3, 29, 0, 0),datetime.datetime(2016, 4, 29, 0, 0)])

    def test_samemonth4(self):
        maturity = '2016-04-17'
        issue = '2015-02-17'
        valuation = '2016-02-01'
        frstCpnDate = '2015-02-17'
        freq=12.0
        coupon=0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2016, 2, 1, 0, 0), datetime.datetime(2016, 2, 17, 0, 0),datetime.datetime(2016, 3, 17, 0, 0),datetime.datetime(2016, 4, 17, 0, 0)])

    def test_samemonth5(self):
        maturity = '2016-05-17'
        issue = '2015-02-17'
        valuation = '2016-02-01'
        frstCpnDate = '2015-02-17'
        freq=4.0
        coupon=0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt=frstCpnDate).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2016, 2, 1, 0, 0), datetime.datetime(2016, 2, 17, 0, 0),datetime.datetime(2016, 5, 17, 0, 0)])

    def test_basic_yearly(self):
        issue = '2014-10-31'
        maturity = '2015-10-29'
        valuation = '2014-10-31'
        freq = 1.0
        coupon = 0.0000000001/freq
        df = DateFlow(coupon, issue, maturity, valuation, freq)
        future_dates, cash_flows, time_in_years = df.get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2014, 10, 31, 0, 0), datetime.datetime(2015, 10, 29, 0, 0)])
        self.assertEqual(df.cash_flows, [0.0, 100.00000001])

    def test_basic2(self):
        maturity = '2013-11-01'
        issue = '2016-05-01'
        valuation = '2016-02-01'
        coupon = 0.0255
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()

    def test_date_range(self):
        issue = '2015-09-30'
        maturity = '2022-09-30'
        valuation = '2020-09-30'
        coupon = 0.00875
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2020,9,30), datetime.datetime(2021,3,30), datetime.datetime(2021,9,30), datetime.datetime(2022,3,30), datetime.datetime(2022,9,30)] )

    def test_date_range2(self):
        issue = '2012-09-30'
        maturity = '2017-09-30'
        valuation = '2015-10-14'
        coupon = 0.003125
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2015, 10, 14), datetime.datetime(2016, 3, 30), datetime.datetime(2016,9,30), datetime.datetime(2017,3,30), datetime.datetime(2017,9,30)])

    def test_date_range3(self):
        issue = '2015-09-30'
        maturity = '2020-09-30'
        valuation = '2015-10-14'
        coupon = 0.006875
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2015,10,14), datetime.datetime(2016,3,30), datetime.datetime(2016,9,30), datetime.datetime(2017,3,30), datetime.datetime(2017,9,30), datetime.datetime(2018,3,30), datetime.datetime(2018,9,30), datetime.datetime(2019,3,30), datetime.datetime(2019,9,30), datetime.datetime(2020,3,30), datetime.datetime(2020,9,30)])

    def test_missing_cashflow_sov_instrcode_3782109(self):
        valuationDate = '2016-08-01'
        issueDate = '2015-02-06'
        maturityDate = '2017-02-06'
        currency = 'ISK'
        coupon = 5.0 / 100.0
        freq = 1.0
        first_cpn_dt = '2016-02-06'
        last_cpn_dt = '2016-02-06'
        date_flow = DateFlow(coupon/freq, issueDate, maturityDate, valuationDate, freq, first_cpn_dt=first_cpn_dt, last_cpn_dt=last_cpn_dt)
        future_dates, cash_flows, time_in_years = date_flow.get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2016, 8, 1, 0, 0),
                                        datetime.datetime(2017, 2, 6, 0, 0)])

    def test_dateflow_BondPricer(self):
        valuationDate = '2017-09-14'
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
        date_flow = DateFlow(coupon/freq, issueDate, maturityDate, valuationDate, freq, first_cpn_dt=first_cpn_dt, last_cpn_dt=last_cpn_dt)
        future_dates, cash_flows, time_in_years = date_flow.get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2017, 9, 14, 0, 0),
                                        datetime.datetime(2017, 9, 30, 0, 0),
                                        datetime.datetime(2018, 3, 30, 0, 0),
                                        datetime.datetime(2018, 9, 30, 0, 0)])

    def test_leap_year2(self):
        maturity = '2013-03-15'
        issue = '2010-09-15'
        valuation = '2012-02-29'
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(0.0255, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates,[datetime.datetime(2012, 2, 29, 0, 0), datetime.datetime(2012, 3, 15, 0, 0), datetime.datetime(2012, 9, 15, 0, 0), datetime.datetime(2013, 3, 15, 0, 0)])
        assert_series_equal(cash_flows, pd.Series([0.0, 2.55, 2.55, 102.55]))

    def test_issue_date(self):
        coupon = 5e-11
        issue = '2016-09-15'
        maturity = '2017-09-15'
        valuation = '2016-09-16'
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime(2016, 9, 16), datetime.datetime(2017, 3, 15), datetime.datetime(2017, 9, 15)])

    def test_extra_cashflow(self):
        coupon = 5e-11
        issue = '2016-09-15'
        maturity = '2017-09-15'
        valuation = '2016-09-16'
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq).get_dateflow()
        assert_series_equal(cash_flows, pd.Series([0.0, 5e-09, 100.000000005]))

    def test_required_extra_cash(self):
        coupon = 0.00475
        issue = '2014-11-12'
        maturity = '2016-11-14'
        valuation = '2015-10-14'
        first_cpn_dt = '2015-05-14'
        last_cpn_dt = '2016-05-14'
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt, last_cpn_dt).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime.strptime(dt, '%Y-%m-%d') for dt in ['2015-10-14', '2015-11-14', '2016-05-14', '2016-11-14']])

    def test_cad_1399240(self):
        coupon = 0.02125
        issue= '2007-10-29'
        maturity = '2018-06-01'
        valuation = '2016-02-08'
        first_cpn_dt = '2007-12-01'
        last_cpn_dt = '2017-12-01'
        freq = 2.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt, last_cpn_dt).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime.strptime(dt, '%Y-%m-%d') for dt in ['2016-02-08', '2016-06-01', '2016-12-01', '2017-06-01', '2017-12-01', '2018-06-01']])

    def test_cop_1998857(self):
        coupon = 0.0725
        issue = '2009-06-15'
        maturity = '2016-06-15'
        valuation = '2016-06-07'
        first_cpn_dt = '2010-06-15'
        last_cpn_dt = '2015-06-15'
        freq = 1.0
        future_dates, cash_flows, time_in_years = DateFlow(coupon, issue, maturity, valuation, freq, first_cpn_dt, last_cpn_dt).get_dateflow()
        self.assertEqual(future_dates, [datetime.datetime.strptime(dt, '%Y-%m-%d') for dt in ['2016-06-07', '2016-06-15']])



if __name__ == '__main__':
    unittest.main()
