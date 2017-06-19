import pandas as pd
import datetime
import math


def get_BEI_conventions(currency):
    def get_interp_years(currency):
        timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,40,50,60]
        if currency in ['EUR','AUD', 'SEK']:
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,20,25,30,40,50,60]

        if currency == 'GBP':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,20,25,30,35,40,50,60]

        if currency == 'JPY':
            timeInYearsInterp = [0.0, 0.25,0.5,1,2,3,7,10,13,14,15,20,30,35,40,50,60]

        if currency == 'CAD':
            timeInYearsInterp =  [0, 0.25,0.5,1,2,3,7,10,13,14,15,20,30,40,50,60]

        if currency == 'USD':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,30,40,50,60]

        if currency == 'ILS':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,25,30,40,50,60]

        if currency == 'MXN':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,18,26,40,50,60]

        return timeInYearsInterp

    conventions = {}
    conventions['timeInYearsInterp'] = get_interp_years(currency)
    conventions['dirtyPriceFlag'] = []
    conventions['use_clean_price_for_short_end_bond'] = []
    conventions['frontEndAdjustList'] = ['AUD''EUR','JPY','MXN','USD']
    conventions['linearInterpList'] = ['AUD','CAD','EUR','ILS','JPY','MXN','SEK','USD']
    conventions['FlatExtrapList'] = ['USD']
    conventions['EurCubicInterpList'] = []
    conventions['MarketPriceAdjustList']=['MXN']
    conventions['IndexFactorAdjustList']=['THB']

    return conventions



def get_sovereign_conventions(currency):
    def get_interp_years(currency):
        timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,40,50,60]
        if currency in ['EUR','CHF', 'AUD', 'SEK']:
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,20,25,30,40,50,60]

        if currency == 'GBP':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,20,25,30,35,40,50,60]

        if currency == 'JPY':
            timeInYearsInterp = [0.0, 0.25,0.5,1,2,3,7,10,13,14,15,20,30,35,40,50,60]

        if currency == 'CAD':
            timeInYearsInterp =  [0, 0.25,0.5,1,2,3,7,10,13,14,15,20,30,40,50,60]

        if currency == 'USD':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,30,40,50,60]

        if currency == 'ILS':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,25,30,40,50,60]

        if currency == 'SGD':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,14,17,30,40,50,60]

        if currency == 'ZAR':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,19,24,32,40,50,60]

        if currency == 'MXN':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,18,26,40,50,60]

        if currency == 'DKK':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,23,40,50,60]

        if currency == 'RUB':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,15,20,40,50,60]

        if currency == 'CNY':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,7,10,13,14,15,20,30,40,50,60]

        if currency == 'INR':
            timeInYearsInterp = [0, 0.25,0.5,1,2,3,4,5,6,7,9,10,11,12,13,14,18,24,29,40,50,60]

        return timeInYearsInterp
    
    conventions = {}
    conventions['timeInYearsInterp'] = get_interp_years(currency)
    conventions['dirtyPriceFlag'] = ['FR.EUR.GVT.ZC', 'SK.EUR.GVT.ZC']
    conventions['use_clean_price_for_short_end_bond'] = ['LT.EUR.GVT.ZC', 'IS.ISK.GVT.ZC']
    conventions['frontEndAdjustList'] = ['AUD','BGN','BRL','CHF','CLP','CNY','COP','CZK','DKK','EUR','HRK','HUF','IDR','ILS','INR','ISK','KRW','MXN','MYR','NOK','NZD','PKR','PLN','RON','RUB','SGD','TWD','THB','TRY','UAH','VND','ZAR']
    conventions['linearInterpList'] = ['BRL','BGN','CAD','CLP','CNY','COP','CZK','EUR','HKD','HRK','HUF','INR','IDR','ILS','KRW','MYR','MXN','NOK','PKR','PHP','PLN','RON','RUB','SEK','THB','TRY','UAH','VND','ZAR']
    conventions['FlatExtrapList'] = ['USD', 'PKR','HUF']
    conventions['EurCubicInterpList'] = ['EP.EUR.GVT.ZC','DE.EUR.GVT.ZC']
    conventions['MarketPriceAdjustList']=['MXN', 'BRL','IDR','INR','ILS','KRW', 'COP', 'CLP']

    return conventions

def process_reuters_market_data_row(row):
    try:
        issueDate = pd.to_datetime(row['InitAccDate']).strftime('%Y-%m-%d') if not pd.isnull(row['InitAccDate']) else datetime.datetime(row['MatDate'].year-1, row['MatDate'].month, row['MatDate'].day).strftime('%Y-%m-%d')
    except:
        issueDate = pd.to_datetime(row['InitAccDate']).strftime('%Y-%m-%d') if not pd.isnull(row['InitAccDate']) else datetime.datetime(row['MatDate'].year-1, row['MatDate'].month, row['MatDate'].day-1).strftime('%Y-%m-%d')
    maturityDate = pd.to_datetime(row['MatDate']).strftime('%Y-%m-%d')
    valuationDate = pd.to_datetime(row['TradeDate']).strftime('%Y-%m-%d')

    if (row['FrstCpnRate'] == 0.0 and row['Price']<20.0):
        marketPrice= 100/math.pow(1+row['MatStdYld']/100, (pd.to_datetime(row['MatDate']) - pd.to_datetime(row['TradeDate'])).days/365.25)
    elif (row['DebtISOCurCode'] == 'BRL' and valuationDate>'2005-08-03'):
        marketPrice=row['Price']/10.0
    elif row['DebtISOCurCode'] == 'KRW':
        marketPrice=row['Price']/100.0
    else:
        marketPrice = row['Price']

    settlement_adj = float(row['DomSettDays'])

    coupon=row['FrstCpnRate']/100.0

    # Brazilian coupon rates are expressed in terms of annual compounding so before we
    # compute interest payments we must convert the rate following the formula in:
    # http://confluence:8090/display/MAC/Brazilian+Government+Bond+Conventions   - page 7
    if row['DebtISOCurCode'] == 'BRL':
        coupon = (pow(1.0+coupon, 0.5)-1.0) * 2.0

    MarketStandardYield = row['MatStdYld']/100 if row['MatStdYld'] is not None else None

    return issueDate, maturityDate, valuationDate, marketPrice, settlement_adj, coupon, MarketStandardYield