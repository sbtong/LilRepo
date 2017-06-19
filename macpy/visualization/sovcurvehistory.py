import os

#  numerical stack
import math
import dateutil.parser as parser
from datetime import date
from datetime import timedelta
from string import Template
import time
import numpy as np
import scipy.interpolate
import pymssql
import pandas as pd
import pandas.io.sql as sql

#  matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns
import colorlover as cl

#  plotly
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import Surface
from plotly.graph_objs import Scatter, Margin, Figure
import plotly.tools as tls
from plotly.offline import plot
from plotly.offline import download_plotlyjs, iplot


class FilteredBondExtraction(object):
    """
    """

    def __init__(self, start_date, end_date, curve_short_name):
        self.sql_template = Template(r"""
DECLARE @StartDate datetime = '$start_date', @EndDate datetime = '$end_date',
@CurveShortName varchar(13) = '$curve_short_name'
SELECT DISTINCT InstrCode= sec.InstrCode, NomTermToMat = DateDiff(d, fb.TradeDate, sec.MatDate)/365.25,
  sec.IssName, p.TradeDate, Price= p.Prc, p.MatStdYld, PriceInclAccrIntFlg = ii.px_incl_accr_int_fl,
  ParValue=ii.par_value, sec.DebtISOCurCode, cv.CurveShortName, sec.CurrCpnRate, cal.DomSettDays, sec.MatDate,
  sec.InitAccDate, sec.DebtIssTypeCode, CompFreqCode = cpn.Value_, c.FrstCpnDate, c.LastCpnDate
FROM [MarketData].[dbo].[DerivCurveFilteredBond] fb
JOIN [MarketData].[dbo].[Curve] cv ON cv.CurveId = fb.CurveId
JOIN [QAI].[dbo].FIEJVPrcDly p ON p.InstrCode = fb.InstrCode and p.TradeDate = fb.TradeDate
JOIN [QAI].[dbo].FIEJVSecInfo sec ON sec.InstrCode = fb.InstrCode
JOIN [QAI].[dbo].FIEJVCpnHdr c ON c.InstrCode = fb.InstrCode
LEFT JOIN [QAI].[dbo].FIEJVCpn cpn  ON cpn.InstrCode = fb.InstrCode
                                                    and cpn.Item = 129 --CouponPaymentFrequencyCode
                                                    and cpn.CpnLegNum = 1
LEFT JOIN [QAI].[dbo].FIEJVSecCalendar cal ON  cal.InstrCode = fb.InstrCode
LEFT JOIN EJV_GovCorp.dbo.orig_iss_info ii ON convert(varbinary(max), sec.EJVAssetId, 1) = ii.asset_id
WHERE fb.TradeDate >= @StartDate and fb.TradeDate <= @EndDate
AND cv.CurveShortName = @CurveShortName
AND ItemID = 1
ORDER BY TradeDate, NomTermToMat""")
        self.sql_query = self.sql_template.substitute(start_date=start_date,
                                                      end_date=end_date,
                                                      curve_short_name=curve_short_name)

