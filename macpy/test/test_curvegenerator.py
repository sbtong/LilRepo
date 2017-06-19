import unittest
from macpy.dateflow import DateFlow
import datetime
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
import macpy.bond as bond
from macpy.CurveGenerator import CurveGenerator, EconomyDateRange, CurveSpec
from macpy.utils import Utilities
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def filter_by_tenor(tenor, quote_list):
    """args tenor:double
     returns quote
     raises exception if tenor not found
    """
    quote_value = filter(lambda x: x['tenor'] == tenor, quote_list)[0]['quotes']
    return quote_value

class TestCurveGenerator(unittest.TestCase):
    """
    Environment to use. Available options: PROD, DEVTEST, DEV. Default is DEV
    """

    def test_brazil_svspr_emg(self):

        curve_short_name = 'BR.USD.SOVSPR'
        currency_code = 'USD'
        country_iso = 'BR'
        trade_start_date = '2011-03-17'
        trade_end_date = '2011-03-17'

        curve_generator = CurveGenerator.create_curve_generator()

        curve_quotes = curve_generator.get_sovereign_spread_quotes_emg(
            curve_short_name,
            currency_code,
            country_iso,
            trade_start_date,
            trade_end_date)

        sov_spr = curve_quotes['2011-03-17']['sovereign_spread']

        expected_sov_spr = 0.0104379316347532

        self.assertAlmostEqual(sov_spr, expected_sov_spr, delta=1e-8)

    def test_chile_svspr_emg(self):

        curve_short_name = 'CL.USD.SOVSPR'
        currency_code = 'USD'
        country_iso = 'CL'
        trade_start_date = '2011-03-17'
        trade_end_date = '2011-03-17'

        curve_generator = CurveGenerator.create_curve_generator()

        curve_quotes = curve_generator.get_sovereign_spread_quotes_emg(
            curve_short_name,
            currency_code,
            country_iso,
            trade_start_date,
            trade_end_date)

        sov_spr = curve_quotes['2011-03-17']['sovereign_spread']

        expected_sov_spr = 0.009745120334925

        self.assertAlmostEqual(sov_spr, expected_sov_spr, delta=1e-8)

    def test_swap_spread_usd_v2(self):

        curve_short_name = 'US.USD.SWP.ZCS'
        country_iso = 'US'
        currency_code = 'USD'
        curve_id = 200302039 # in DEV
        trade_start_date = '2004-01-05'
        trade_end_date = '2004-01-05'
        node_index_3m = 6  # maps to 0.25 year node
        node_index_1y = 9  # maps to 1.0 year node
        node_index_40y = 71  # maps to 40.0 year node
        ric = '0#USDZ=R'

        curve_generator = CurveGenerator.create_curve_generator()
        curve_quotes = curve_generator.get_swap_spread_quotes_v2(
            curve_short_name,
            country_iso,
            currency_code,
            curve_id,
            trade_start_date,
            trade_end_date,
            ric)
        quote_list = curve_quotes[trade_start_date]
        actual_swp_spr_3month = filter_by_tenor(.25, quote_list)
        expected_swp_spr_3month = 0.00213645372926
        self.assertAlmostEqual(actual_swp_spr_3month, expected_swp_spr_3month, delta=1e-8)

        actual_swp_spr_1year = filter_by_tenor(1.0, quote_list)
        expected_swp_spr_1year = 0.001429762603
        self.assertAlmostEqual(actual_swp_spr_1year, expected_swp_spr_1year, delta=1e-8)

        actual_swp_spr_40year = filter_by_tenor(40.0, quote_list)
        expected_swp_spr_40year = 0.00402268961453
        self.assertAlmostEqual(actual_swp_spr_40year, expected_swp_spr_40year, delta=1e-8)

    def test_swap_spread_based_on_irswp_usd(self):

        curve_short_name = 'US.USD.SWP.ZCS'
        country_iso = 'US'
        currency_iso = 'USD'
        curve_id = 200322316 # in DEV
        trade_start_date = '2004-01-05'
        trade_end_date = '2004-01-05'

        curve_generator = CurveGenerator.create_curve_generator()
        curve_quotes = curve_generator.calculate_swap_spreads(
            curve_short_name,
            country_iso,
            currency_iso,
            curve_id,
            trade_start_date,
            trade_end_date)
        quote_list = curve_quotes[trade_start_date]
        actual_swp_spr_10year = filter_by_tenor(10.0, quote_list)
        expected_swp_spr_10year = .00397028
        self.assertAlmostEqual(actual_swp_spr_10year, expected_swp_spr_10year, delta=1e-4)

    def test_get_irswap_quotes_cny(self):

        country_iso = 'CN'
        currency_iso = 'CNY'
        trade_start_date = '2007-01-05'
        trade_end_date = '2007-01-05'

        curve_generator = CurveGenerator.create_curve_generator()
        df_quotes = curve_generator.get_irswap_quotes(
            country_iso,
            currency_iso,
            trade_start_date,
            trade_end_date)

        expected_length = 48
        self.assertTrue(len(df_quotes) == expected_length)


    def test_ois_swap_yield_usd(self):

        curve_short_name = 'US.USD.OISWP.ZC'
        currency_code = 'USD'
        country_iso = 'US'
        curve_id = 207242808
        trade_start_date = '2006-02-03'
        trade_end_date = '2006-02-03'
        ric = 'USD.OIS'

        curve_generator = CurveGenerator.create_curve_generator()
        curve_quotes = curve_generator.get_oiswap_yield_quotes(
            curve_short_name,
            country_iso,
            currency_code,
            curve_id,
            trade_start_date,
            trade_end_date,
            ric)

        quote_list = curve_quotes[trade_start_date]
        actual_swp_16year = filter_by_tenor(16.0, quote_list)
        expected_swp_16year = 0.04856442
        self.assertAlmostEqual(actual_swp_16year, expected_swp_16year, delta=1e-8)


    def test_irswap_yield_usd(self):

        curve_short_name = 'US.USD.IRSWP.ZC'
        currency_code = 'USD'
        country_iso = 'US'
        curve_id = 206815863
        trade_start_date = '2004-01-05'
        trade_end_date = '2004-01-05'
        ric = '0#USDZ=R'

        curve_generator = CurveGenerator.create_curve_generator()
        curve_quotes = curve_generator.get_irswap_yield_quotes(
            curve_short_name,
            country_iso,
            currency_code,
            curve_id,
            trade_start_date,
            trade_end_date,
            ric)

        quote_list = curve_quotes[trade_start_date]
        actual_swp_16year = filter_by_tenor(16.0, quote_list)
        expected_swp_16year = 0.05515474
        self.assertAlmostEqual(actual_swp_16year, expected_swp_16year, delta=1e-8)

        actual_swp_1year = filter_by_tenor(19.0, quote_list)
        expected_swp_1year = 0.05736897
        self.assertAlmostEqual(actual_swp_1year, expected_swp_1year, delta=1e-8)

        actual_swp_40year = filter_by_tenor(40.0, quote_list)
        expected_swp_40year = 0.05715508
        self.assertAlmostEqual(actual_swp_40year, expected_swp_40year, delta=1e-8)

    def test_irswap_yield_cad(self):

        curve_short_name = 'CA.CAD.IRSWP.ZC'
        currency_code = 'CAD'
        country_iso = 'CA'
        curve_id = 206815858
        trade_start_date = '2017-01-10'
        trade_end_date = '2017-01-10'
        ric = '0#CADZ=R'

        curve_generator = CurveGenerator.create_curve_generator()
        curve_quotes = curve_generator.get_irswap_yield_quotes(
            curve_short_name,
            country_iso,
            currency_code,
            curve_id,
            trade_start_date,
            trade_end_date,
            ric)

        quote_list = curve_quotes[trade_start_date]
        actual_swp_5year = filter_by_tenor(5.0, quote_list)
        expected_swp_5year = 0.0142158
        self.assertAlmostEqual(actual_swp_5year, expected_swp_5year, delta=1e-8)


    def test_fwd_swap_yield_usd(self):

        curve_short_name = 'US.USD.FDSWP3M.ZC'
        currency_code = 'USD'
        country_iso = 'US'
        curve_id = 206818834
        trade_start_date = '2006-01-05'
        trade_end_date = '2006-01-05'
        tenor = '3M'

        curve_generator = CurveGenerator.create_curve_generator()
        curve_quotes = curve_generator.get_fdswap_yield_quotes(
            curve_short_name,
            country_iso,
            currency_code,
            curve_id,
            trade_start_date,
            trade_end_date,
            tenor)

        quote_list = curve_quotes[trade_start_date]
        actual_swp_16year = filter_by_tenor(16.0, quote_list)
        expected_swp_16year = 0.04979291
        self.assertAlmostEqual(actual_swp_16year, expected_swp_16year, delta=1e-8)

        actual_swp_19year = filter_by_tenor(19.0, quote_list)
        expected_swp_19year = 0.05017842
        self.assertAlmostEqual(actual_swp_19year, expected_swp_19year, delta=1e-8)

        actual_swp_40year = filter_by_tenor(40.0, quote_list)
        expected_swp_40year = 0.05057048
        self.assertAlmostEqual(actual_swp_40year, expected_swp_40year, delta=1e-8)


    def test_extract_swap_zeros(self):
        country_code = 'EP'
        trade_start_date = '2006-12-19'
        trade_end_date = '2006-12-19'
        tenor = 1.0

        curve_generator = CurveGenerator.create_curve_generator()

        curve_swap_yield_quotes = curve_generator.get_swap_yield_quotes(
            country_code,
            trade_start_date,
            trade_end_date)

        swap_yield_at_tenor = curve_swap_yield_quotes.loc[trade_start_date, tenor][0]

        expected_swap_yield_at_tenor = 0.039438561750918796
        self.assertAlmostEqual(swap_yield_at_tenor, expected_swap_yield_at_tenor, delta=1e-8)

    def test_corporate_rating_usd_aa(self):
        environment='DEV'
        curve_short_name = 'USD.(AA).SPRSWP'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_usd_20170428.csv')
        date_range = EconomyDateRange(start_date='2017-04-28',
                                      end_date='2017-04-28',
                                      currency_code='USD',
                                      country_code='US')
        econ_date = date_range
        curve_spec = CurveSpec(composite_rating='AA')
        corp_quotes_filtered = curve_generator.get_corp_rating_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = 0.006302824
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_corporate_sector_usd_aa_industrial(self):
        environment='DEV'
        curve_short_name = 'USD.RATGICsSEC(AA,Industrials)'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_usd_20170428.csv')
        date_range = EconomyDateRange(start_date='2017-04-28', end_date='2017-04-28',currency_code='USD')
        econ_date = date_range
        curve_spec = CurveSpec(composite_rating='AA', sector='Industrials')
        corp_quotes_filtered = curve_generator.get_corp_sector_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = 0.001937759
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_corporate_sector_cad_sub_ig_utilities(self):
        environment='DEV'
        curve_short_name = 'CAD.RATGICsSEC(SUB-IG,Utilities)'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_cad_20170428.csv')
        date_range = EconomyDateRange(start_date='2017-04-28', end_date='2017-04-28', currency_code='CAD')
        econ_date = date_range
        curve_spec = CurveSpec(composite_rating='SUB-IG', sector='Utilities')
        corp_quotes_filtered = curve_generator.get_corp_sector_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = -0.01287858
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_corporate_sector_chf_aa_consumer_staples(self):
        environment = 'DEV'
        curve_short_name = 'CHF.RATGICsSEC(AA,Consumer Staples)'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_chf_20170428.csv')
        trade_date = '2017-04-28'
        econ_date = EconomyDateRange(start_date=trade_date, end_date=trade_date, currency_code='CHF', country_code='CH')
        curve_spec = CurveSpec(composite_rating='AA', sector='Consumer Staples')
        corp_quotes_filtered = curve_generator.get_corp_sector_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = -0.0009378
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_corporate_sector_eur_aa_industrials(self):
        environment = 'DEV'
        curve_short_name = 'EUR.RATGICsSEC(AA,Industrials)'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        trade_date = '2017-04-28'
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_eur_20170428.csv')
        econ_date = EconomyDateRange(start_date=trade_date, end_date=trade_date, currency_code='EUR', country_code='EU')
        curve_spec = CurveSpec(composite_rating='AA', sector='Industrials')
        corp_quotes_filtered = curve_generator.get_corp_sector_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = 0.00543042178287
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_corporate_issuer_usd_goldman(self):
        environment = 'DEV'
        curve_short_name = 'USD.ISSR(GOLDMAN_SACHS).SPR'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        trade_date = '2017-04-28'
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_usd_20170428.csv')
        econ_date = EconomyDateRange(start_date=trade_date, end_date=trade_date, currency_code='USD', country_code='US')
        curve_spec = CurveSpec(issuer_id=100082014)
        corp_quotes_filtered = curve_generator.get_corp_issuer_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = -0.00313879
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_corporate_sector_cad_goldman(self):
        environment='DEV'
        curve_short_name = 'CAD.ISSR(GOLDMAN_SACHS).SPR'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_cad_20170428.csv')
        date_range = EconomyDateRange(start_date='2017-04-28', end_date='2017-04-28', country_code='CA', currency_code='CAD')
        econ_date = date_range
        curve_spec = CurveSpec(issuer_id=100082014)
        corp_quotes_filtered = curve_generator.get_corp_issuer_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = -0.0043487422
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_corporate_issuer_usd_ameren(self):
        environment='DEV'
        curve_short_name = 'USD.ISSR(AMEREN).SPR'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_corp_usd_20170428.csv')
        date_range = EconomyDateRange(start_date='2017-04-28', end_date='2017-04-28', country_code='US', currency_code='USD')
        econ_date = date_range
        curve_spec = CurveSpec(issuer_id=100061911)
        corp_quotes_filtered = curve_generator.get_corp_issuer_spread_quote(curve_short_name,
                                                                            econ_date,
                                                                            curve_spec,
                                                                            query_exec_func=lambda db, d: df_corp_quotes)
        actual_sector_spread = corp_quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_sector_spread = -0.00331378
        self.assertAlmostEqual(actual_sector_spread, expected_sector_spread, delta=1e-7)

    def test_usd_abs_home(self):
        environment='DEV'
        curve_short_name = 'US.USD.HOME(IG).ABS'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_abs_usd_20170428.csv')
        date_range = EconomyDateRange(start_date='2017-04-28', end_date='2017-04-28', country_code='US', currency_code='USD')
        econ_date = date_range
        quotes_filtered = curve_generator.get_abs_curve_quotes(curve_short_name, econ_date, query_exec_func=lambda db, d: df_corp_quotes)
        actual_spread = quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_spread = 0.0183113
        self.assertAlmostEqual(actual_spread, expected_spread, delta=1e-7)

    def test_usd_supn(self):
        environment='DEV'
        curve_short_name = 'USD.SUPN.(AAA)'
        curve_generator = CurveGenerator.create_curve_generator(environment=environment)
        df_corp_quotes = pd.read_csv(dir_path + '/data/ccy_supn_usd_20170428.csv')
        curve_spec = CurveSpec(composite_rating='AAA')
        date_range = EconomyDateRange(start_date='2017-04-28', end_date='2017-04-28', country_code='US', currency_code='USD')
        econ_date = date_range
        quotes_filtered = curve_generator.get_supranational_curve_quotes(curve_short_name, econ_date, curve_spec, query_exec_func=lambda db, d: df_corp_quotes)
        actual_spread = quotes_filtered[econ_date.start_date]['amount_weighted_corporate_spread']
        expected_spread = 0.0003832
        self.assertAlmostEqual(actual_spread, expected_spread, delta=1e-7)

