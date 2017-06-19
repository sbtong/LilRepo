import string
from macpy.curve_wrappers import InterpolatedCurve, SwapCurve

# def create_pricer(issueDate, maturityDate, coupon, valuationDate, freq=2.0, settlement_adj=0.0, first_cpn_dt=None, last_cpn_dt=None):
#     return BondPricer(issueDate, maturityDate, coupon, valuationDate, freq, settlement_adj, first_cpn_dt, last_cpn_dt)
    
def create_yield_curve(valuationDate, curveName, database=None):
    yieldCurve = InterpolatedCurve(valuationDate, curveName, database)
    return yieldCurve 

def create_govt_zero_curve(valuationDate, currency, database=None, final=True):
    gvtCurveName = convert_gvt_curve(currency)
    zeroCurve = InterpolatedCurve(valuationDate, gvtCurveName, database, final)
    return zeroCurve

def create_swap_zero_curve(valuationDate, currency, database=None):
    gvtCurveName = convert_gvt_curve(currency)
    swpSpreadCurveName = convert_swp_sprd_curve(currency)
    swapCurve = SwapCurve(valuationDate, gvtCurveName, swpSpreadCurveName, database)
    return swapCurve

def convert_gvt_curve(currency, countryName=None):
    if countryName is None:
        country = currency[0:2]
    else:
        country = countryName
    result = string.Template("$country.$currency.GVT.ZC").substitute({'country':country, 'currency':currency})
    return 'EP.EUR.GVT.ZC' if 'EU' in result else result

def convert_bei_curve(currency, countryName = None):
    if countryName == None:
        country = currency[0:2]
    else:
        country = countryName
    result = string.Template("$country.$currency.BEI.ZC").substitute({'country':country, 'currency':currency})
    return result

def convert_swp_sprd_curve(currency):
    country = currency[0:2]
    result = string.Template("$country.$currency.SWP.ZCS").substitute({'country':country, 'currency':currency})
    return result