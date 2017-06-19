import string
import datetime
import macpy.utils.database as db
from macpy.utils.database import CurveNodeQuoteFinal
import pandas as pd
import time
import macpy.bondPortfolio as bondPortfolio
import numpy as np
from scipy.interpolate import interp1d
from irr import  compute_irr, compute_npv, discount_cashflows_at_irr
from optparse import OptionParser
from irr import compute_irr


class Tranche:

    def __init__(self, cashflow_source, path_key='Path', market_price=None):
        self.path_key = path_key
        self.cashflows = cashflow_source()
        self.market_price = market_price
        self.base_selector = self.cashflows.PathKey.str.contains(self.path_key)
        self.path_count = float(self.cashflows[self.base_selector].groupby(['PathKey']).ngroups)
        self.cashflows['TotalCashflow'] = self.cashflows['Principal'] + self.cashflows['Interest']
        self.cashflows['TotalCashflowPV'] = self.cashflows['TotalCashflow'] * self.cashflows['DiscountFactor']
        self.notional = self.cashflows[self.base_selector].groupby(['TrancheId'])['Balance'].first().sum()
        self.oas = 0 if market_price is None else self.compute_oas()


    @staticmethod
    def add_regex(pattern):
        return '^' + pattern + '*'

    def compute_oas(self):
        path_count = self.path_count
        times_to_cashflow = self.cashflows[self.base_selector].groupby('Period').TimeInYears.first()
        combined_cashflows = self.cashflows[self.base_selector]\
                                 .groupby(['Period'])\
                                 .apply(lambda x: x.TotalCashflowPV.sum())\
                             /(self.notional*path_count)
        oas = compute_irr(times_to_cashflow,
                          combined_cashflows,
                          self.market_price)
        return oas

    def compute_price(self, path_key='', oas=None):
        return self.compute_price_dist(path_key, oas).mean()

    def compute_duration(self, path_key_base='DiagnoseEffectiveDuration-Base',
                         path_key_up='DiagnoseEffectiveDuration-UpCurveShift',
                         path_key_down='DiagnoseEffectiveDuration-DownCurveShift'):
        price_base = self.compute_price(path_key=path_key_base)
        price_up = self.compute_price(path_key=path_key_up)
        price_down = self.compute_price(path_key=path_key_down)
        tsy_up = self.cashflows[
                     self.cashflows.PathKey.str.contains(self.add_regex(path_key_up))]['Tsy10Yr'].values[0]
        tsy_down = self.cashflows[
                       self.cashflows.PathKey.str.contains(self.add_regex(path_key_down))]['Tsy10Yr'].values[0]
        tsy_change = (tsy_up - tsy_down)*.01  # level is in percentage units, we must scale to absolute units
        duration = -(price_up - price_down)/(price_base*tsy_change)
        return duration

    def compute_price_dist(self, path_key='', oas=None):
        oas = self.oas if oas is None else oas
        path_key = self.path_key if path_key == '' else path_key
        path_selector = self.cashflows.PathKey.str.contains(self.add_regex(path_key))
        return self.cashflows[path_selector]\
            .groupby('PathKey')\
            .apply(lambda x : discount_cashflows_at_irr(x['TimeInYears'], x['TotalCashflowPV'], oas)
                   .sum()/self.notional)

    def compute_price_sample_error(self):
        return self.compute_price_dist().std()

    def compute_wal_prin(self):
        return self.compute_wal_prin_dist().mean()

    def compute_wal_prin_dist(self):
        total_principal = self.notional
        return self.get_base_cashflows().groupby('PathKey').apply(lambda x : (x['Principal']*x['TimeInYears']).sum())/ \
               total_principal

    def compute_wal_io(self):
        return self.compute_wal_io_dist().mean()

    def compute_wal_io_dist(self):
        total_interest = self.get_base_cashflows()['Interest'].sum()
        total_interest = total_interest if total_interest > 0 else 1.0
        return self.get_base_cashflows().groupby('PathKey').apply(lambda x : (x['Interest']*x['TimeInYears']).sum())/\
            total_interest

    def compute_loss_pct_dist(self):
        notional = self.notional if self.notional>0.01 else 1.0
        return self.get_base_cashflows().groupby('PathKey').apply(lambda x : (x['Loss']).sum())/notional

    def compute_loss_pct(self):
        return self.compute_loss_pct_dist().mean()

    def get_base_cashflows(self):
        return self.cashflows[self.base_selector]

class TrancheStatistic:

    def __init__(self, tranche):
        self.tranche = tranche
        self.effective_duration = tranche.compute_duration()
        self.price_dist = tranche.compute_price_dist()
        self.price_sample_error = tranche.compute_price_sample_error()
        self.wal_prin = tranche.compute_wal_prin()
        self.wal_io = tranche.compute_wal_io()
        self.wal_prin_dist = tranche.compute_wal_prin_dist()
        self.wal_io_dist = tranche.compute_wal_io_dist()
        self.loss_pct = tranche.compute_loss_pct()
        self.loss_pct_dist = tranche.compute_loss_pct_dist()

