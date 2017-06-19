import time
import finance
import string
import datetime
import macpy.utils.database as db
import pandas as pd
import matplotlib.pyplot as plt
import macpy.bondPortfolio as bondPortfolio
import numpy as np
from optparse import OptionParser
import statsmodels.api as sm

#class NelsonSiegel:
 #   def __init__(self, lambdaValue, b1, b2, b3, tau):
 #       self.lambdaValue = lambdaValue
 #       self.b1 = b1
 #       self.b2 = b2
 #       self.b3 = b3
 #       self.tau = np.array(tau)

 #    def compute_zero_yield(self):
 #    
 #       yld = self.b1 + self.b2 * (1 - np.exp(-self.lambdaValue*self.tau))/(self.lambdaValue*self.tau)\
 #           + self.b3*( (1 - np.exp(-self.lambdaValue*self.tau))/(self.lambdaValue*self.tau)\
 #           - np.exp(-self.lambdaValue*self.tau) );
 #       return yld

def zeroYield(lambdaValue, level, slope, curvature, tau):
    """ Compute zero yield for Nelson-Siegel 
           integrate forward rates to compute zero yields
    """
    b1 = level
    b2 = slope
    b3 = curvature

    yld = b1 + b2 * (1 - np.exp(-lambdaValue*tau))/(lambdaValue*tau)\
        + b3*( (1 - np.exp(-lambdaValue*tau))/(lambdaValue*tau)\
        - np.exp(-lambdaValue*tau) );
    return yld

def discountRate(lambdaValue, level, slope, curvature, tau):
    """ Compute discount rate for Nelson-Siegel
    """

    yld = zeroYield(lambdaValue, level, slope, curvature, tau)
    discount_rate = np.exp(-yld*tau)

    return discount_rate

def forwardRates(lambdaValue, level, slope, curvature, tau):
    """ Compute forward rate for Nelson-Siegel 
    """
    b1 = level
    b2 = slope
    b3 = curvature
    fwdRate  = b1 * b2 * np.exp(-lambdaValue*tau) + b3*lambdaValue*tau * np.exp(-lambdaValue*tau)
    return fwdRate

def calcprice(DiscountRates, CashFlows):
    """
    :param DiscoutnRates: np.array
    :param CashFlows: np.array
    :return: scalar price
    """
    price = (DiscountRates*CashFlows).sum()


def get_NS_bond_price(xdata,b1,b2,b3):
    #xdata is a dataframe with columns:
    #CashFlowTimes = np.array of cashflow time in years, measured from valuation date
    #CashFlows = np.array of cashflows
    L=0.33
    df_prc = pd.DataFrame()
    df_prc['DiscountRates'] = xdata.CashFlowTimes.apply(lambda x: discountRate(L,b1,b2,b3,x))
    df_prc['Prc'] = (df_prc['DiscountRates']*xdata['CashFlows']).apply(lambda x: x.sum())

    return df_prc.Prc.values


class NelsonSiegel(object):
    def __init__(self, curve, shape=1.5):
        """ curve argument specified annualized yields, index is node names 
        """
        self.curve = curve.copy()
        self.shape = 1.5

    def fit(self):
        tmpcurve = self.curve.rename(index=lambda x : CurveLoader.nodeNameKey(x)[1]/365.0)
        X = pd.DataFrame(1.0, index=tmpcurve.index, columns=['level'])
        X['slope'] = pd.Series((1 - np.exp(-self.shape * tmpcurve.index)) / (self.shape * tmpcurve.index), index=tmpcurve.index) 
        X['curvature'] = pd.Series((1 - np.exp(-self.shape * tmpcurve.index)) / (self.shape * tmpcurve.index) - np.exp(-self.shape * tmpcurve.index) , index=tmpcurve.index) 
        self.model = sm.OLS(tmpcurve, X)
        self.result = self.model.fit()

        self.forecast = X.dot(self.result.params)
        self.forecast.index = self.curve.index


