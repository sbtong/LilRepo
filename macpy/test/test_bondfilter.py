import unittest
import pandas as pd
from macpy.filters import BondFilter, BondFilterDbWriter
from pandas.util.testing import assert_series_equal


class Test_BondFilter(unittest.TestCase):
    def test_jpy(self):
        country = 'JP'
        currency = 'JPY'
        tradeDate = '2016-01-12'
        bf = BondFilter(currency, country, tradeDate, debug=False)
        df = bf.run_filter()
        # removed_instr_code = 4179070
        # self.assertTrue(df[df['InstrCode'] ==  removed_instr_code].empty)
        self.assertEqual(df.shape[0], 18)

    def test_isk(self):
        country = 'IS'
        currency = 'ISK'
        tradeDate = '2016-01-12'
        database = 'DEV'
        bf = BondFilter(currency, country, tradeDate, debug=False, database=database)
        df = bf.run_filter()
        # removed_instr_code = 4179070
        # self.assertTrue(df[df['InstrCode'] ==  removed_instr_code].empty)
        self.assertEqual(df.shape[0], 18)
        dbconn = BondFilterDbWriter(tradeDate, bf.curveId, df, database=database, production_table=True)
        error_code = dbconn.write_to_dbs()



    def test_jpy2(self):
        country = 'JP'
        currency = 'JPY'
        tradeDate = '2016-02-01'
        bf = BondFilter(currency, country, tradeDate, debug=False)
        df = bf.run_filter()
        # removed_instr_code = 4215478
        # self.assertTrue(df[df['InstrCode'] ==  removed_instr_code].empty)
        self.assertEqual(df.shape[0], 18)

    def test_chf(self):
        country = 'CH'
        currency = 'CHF'
        tradeDate = '2016-02-01'
        bf = BondFilter(currency, country, tradeDate, debug=False)
        df = bf.run_filter()
        self.assertEqual(df.shape[0], 13)

    def test_CAD_0715(self):
        country = 'CA'
        currency = 'CAD'
        tradeDate = '2016-07-15'
        bf = BondFilter(currency, country, tradeDate, debug=False)
        df = bf.run_filter()
        target_instr_code = 3988304
        df_length = len(df.loc[df['InstrCode'].isin([target_instr_code])])
        expected_length = 1
        self.assertAlmostEqual(df_length, expected_length)
        #self.assertEqual(df.shape[0],33)

    def test_SEK_1019(self):
        country = 'SE'
        currency = 'SEK'
        tradeDate = '2016-10-19'
        bf = BondFilter(currency, country, tradeDate, debug=False)
        df = bf.run_filter()
        target_instr_code = 4410468
        df_length = len(df.loc[df['InstrCode'].isin([target_instr_code])])
        expected_length = 1
        self.assertAlmostEqual(df_length, expected_length)

    def test_nok(self):
        country = 'NO'
        currency = 'NOK'
        tradeDate = '2016-02-05'
        bf = BondFilter(currency, country, tradeDate, debug=False)
        df = bf.run_filter()
        self.assertEqual(df.shape[0], 7)

    def test_try(self):
        country = 'TR'
        currency = 'TRY'
        tradeDate = '2016-02-05'
        bf = BondFilter(currency, country, tradeDate, debug=False)
        df = bf.run_filter()
        self.assertEqual(df.shape[0], 5)
        instr_codes = [4080193, 3852793, 2961322, 3906272, 3777031]
        for i in instr_codes:
            self.assertTrue(df[df['InstrCode'] ==  i].empty is False)

    # def test_cad(self):
    #     country = 'CA'
    #     currency = 'CAD'
    #     tradeDate = '2016-02-05'
    #     bf = BondFilter(currency, country, tradeDate, debug=False)
    #     df = bf.run_filter()
    #     self.assertEqual(df.shape[0], 31)

