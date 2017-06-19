__author__ = 'hmiao'

import macpy.bond as bond
import macpy.bond_utils as bond_utils
import string
import macpy.utils.database as db
import pandas as pd
import dateutil
import datetime
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import datetime
from optparse import OptionParser
import getpass
from dateutil.relativedelta import relativedelta
import math
from irr import compute_irr, compute_npv
import macpy.bootstrapper as bs
import macpy.bei_curve as BE
from curve_utility import getAnalysisConfigSection, getOracleDatabaseInfo, getMSSQLDatabaseInfo

def get_curve_short_name(currency):
    country = currency[0:2]
    curveShortName = string.Template("$country.$currency.BEI.ZC").substitute({'country':country, 'currency':currency})
    return curveShortName


class BEIBootstrapper:
    def __init__(self, startDate, endDate, currency, mktDBInfo, modelDBInfo, macDBInfo ):
        self.startDate = startDate
        self.endDate = endDate
        self.currency=currency
        self.mktDBInfo = mktDBInfo
        self.modelDBInfo = modelDBInfo
        self.macDBInfo = macDBInfo

    def CreateNominalYield(self):
        curveShortName = get_curve_short_name(self.currency)
        BEI_Benchmark_Query = BE.BEI(curveShortName, self.startDate, self.endDate, self.mktDBInfo, self.modelDBInfo, self.macDBInfo)
        BEI_df = BEI_Benchmark_Query.extract_from_db()
        yieldDatabaseWriter = bs.YieldsDatabaseWriter(self.startDate, self.endDate,self.currency)




