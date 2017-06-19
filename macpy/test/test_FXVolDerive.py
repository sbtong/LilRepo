import pandas as pd
import numpy as np
import macpy.utils.database as db
import macpy.dateflow as df
import unittest
import os
import datetime as datetime
import macpy.FXVolDerive as fv

class Test_FXVolDerive(unittest.TestCase):
    def test_tenor_in_years(self):
        startDate = '2016-03-01'
        endDate = '2016-03-01'
        fxvol = fv.FXVolDerive(startDate, endDate)
        fxvol.derive_FXVol()

    def test_compute_strike(self):
        tenorInYears = 1.0/52
        TradeDate = '2016-03-01'
        SettleCurrency = 'JPY'
        ForeignCurrency = 'USD'
        Vol90 = 16.67
        delta = 0.9
        FXVol = fv.FXVolDerive(TradeDate, TradeDate)
        strike = FXVol.compute_strike(tenorInYears,delta,TradeDate,SettleCurrency,ForeignCurrency,Vol90/100)
        print strike

