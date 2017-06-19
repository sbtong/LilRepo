import unittest
import macpy.bond as bond
import macpy.finance as finance
import macpy.utils.database as db
import macpy.bootstrapper as bs
import pandas as pd

class Test_Long_Running(unittest.TestCase):

    def test_bond_stats_write_to_db(self):
        currency = 'GBP'
        yieldcurveName = 'GB.GBP.GVT.ZC'
        spreadcurveName = 'GB.GBP.SWP.ZCS'
        startDate = '2007-01-01'
        endDate = '2007-01-01'
        issuerId = '100082278'

        #bondStatsWriter = bond.BondStatDatabaseWriter(startDate, endDate, currency, yieldcurveName, spreadcurveName, issuerId, None, None, None)
        #bondStatsWriter.write_to_db()

    def test_bootstrapper(self):
        currency='USD'
        startDate='2015-07-01'
        endDate = '2015-07-01'
        bootstrapper = bs.BootStrapper(startDate, endDate, currency)
        bootstrapper.createYieldCurve()


if __name__ == '__main__':
    unittest.main()
