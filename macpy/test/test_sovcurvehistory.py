import os
import unittest
import macpy
import macpy.utils.database as db
import macpy.bond as bond
import plotly.graph_objs as go
import plotly.plotly as py
import macpy.visualization.sovcurvehistory as hist
import pandas as pd
from pandas.util.testing import assert_series_equal
import plotly.tools as tls


class TestCurveFrameView(unittest.TestCase):

    def test_emg_sovereign_history(self):
        curve_list = ['BG.BGN.GVT.ZC', 'BR.BRL.GVT.ZC', 'CL.CLF.GVT.ZC', 'CL.CLP.GVT.ZC', 'CN.CNY.GVT.ZC',
                      'CO.COP.GVT.ZC', 'CZ.CZK.GVT.ZC', 'DK.DKK.GVT.ZC', 'HK.HKD.GVT.ZC', 'HR.HRK.GVT.ZC',
                      'HU.HUF.GVT.ZC', 'ID.IDR.GVT.ZC', 'IL.ILS.GVT.ZC', 'IN.INR.GVT.ZC', 'IS.ISK.GVT.ZC',
                      'KR.KRW.GVT.ZC', 'MX.MXN.GVT.ZC', 'MY.MYR.GVT.ZC', 'NZ.NZD.GVT.ZC', 'PH.PHP.GVT.ZC',
                      'PK.PKR.GVT.ZC', 'PL.PLN.GVT.ZC', 'RO.RON.GVT.ZC', 'RU.RUB.GVT.ZC', 'SG.SGD.GVT.ZC',
                      'TH.THB.GVT.ZC', 'TR.TRY.GVT.ZC', 'TW.TWD.GVT.ZC', 'UA.UAH.GVT.ZC', 'VN.VND.GVT.ZC',
                      'ZA.ZAR.GVT.ZC']

        start_date = '2014-07-01'
        end_date = '2016-08-25'
        curve = curve_list[1]
        environment = 'DEV'
        query = hist.FilteredBondExtraction(start_date, end_date, curve).sql_query
        df_results = db.MSSQL.extract_dataframe(query, environment=environment)

        actual_count = len(df_results)
        expected_count = 10



