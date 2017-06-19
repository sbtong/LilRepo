import numpy as np
import pandas as pd
import macpy.utils.database as db
from scipy.interpolate import interp1d
# import macpy.nelsonsiegel as ns

class CustomYieldCurve:
    
    def __init__(self, forwardRateList, maturityInYearsList):
        self.forwardRateList = forwardRateList
        self.maturityInYearsList = maturityInYearsList
        return

    def __call__(self, timeInYears):        
        return self.getYield(timeInYears)


    def getYield(self, timeInYears):
        
        tempList = [self.maturityInYearsList[i]-self.maturityInYearsList[i-1] for i in xrange(1,len(self.maturityInYearsList))] if len(self.maturityInYearsList)>1 else None
        customTimeInYearsList = [0]
        customTimeInYearsList.extend(tempList if tempList else [])
                
        CustomYield = zip(customTimeInYearsList, self.forwardRateList, self.maturityInYearsList)
        yield_before_Node = [x for x in CustomYield if timeInYears>=x[2]]
        yield_after_Node = [x for x in CustomYield if timeInYears<x[2]]
        yieldSum = sum([x[0]*x[1] for x in yield_before_Node])+(timeInYears-yield_before_Node[-1][2])*(0 if not yield_after_Node else yield_after_Node[0][1])
        yieldOnNode = yieldSum/np.amax([timeInYears, 0.0001])
        return yieldOnNode

    def compute_discount_factor(self, timeInYears):
        return np.exp(-timeInYears*self(timeInYears))


class adjusted_yield_curve:
    """
    Wraps original curve function such that if the independent value (timeInYears) is before the earliest value
    in the curve, it returns the first value in the maturityInYears list
    This class also makes yield curve functions handle two types of inputs: lists of floats and single float values
    """

    def __init__(self, maturityInYearsList, yieldCurve, LastMaturityInYears, extrap = 'False'):

        self.yieldCurve = yieldCurve
        self.maturityInYearsList = maturityInYearsList
        self.LastMaturityInYears = LastMaturityInYears
        self.extrap = extrap

    def __call__(self, timeInYears):

        if isinstance(timeInYears,(tuple, list, np.ndarray)):
            return [self.func_adjust_yield_curve(x).tolist() for x in timeInYears]
        else:
            return self.func_adjust_yield_curve(timeInYears)

    def func_adjust_yield_curve(self, timeInYear):

        if timeInYear < self.maturityInYearsList[1]:
            #print timeInYear, 'rate: ', self.yieldCurve(self.maturityInYearsList[1]), 'rateBefore: ', self.yieldCurve(timeInYear)
            return self.yieldCurve(self.maturityInYearsList[1])

        elif self.extrap == 'True' and timeInYear > self.LastMaturityInYears:
            return self.yieldCurve(self.LastMaturityInYears)

        else:
            return self.yieldCurve(timeInYear)


class InterplatedCurve:

    def __init__(self, bootstrapper):
        self.bootstrapper = bootstrapper
        self.yieldcurve = self.bootstrapper.createYieldCurve()
        return

    def __call__(self, timeInYears):
        return self.yieldcurve(timeInYears)

    def compute_discount_factor(self, timeInYears):
        return np.exp(-timeInYears*self(timeInYears))

class InterpolatedCurve:
    """
    InterpolatedCurve extracts quotes for a supplied curveshortname (e.g. 'US.USD.GVT.ZC.) and creates a scipy interpolator
    """
    LHS_BOUND = 0.0 # curve starts at zero years
    RHS_BOUND = 200.0  #curve ends at 200 years

    #@classmethod
    #def create_curve_history(cls, curveName, startDate, endDate, spreadCurve=None, database='Dev'):

    def __init__(self, valuationDate, curveName, database='Dev', final=True):
        self._valuationDate = valuationDate
        self._curveName = curveName
        self._database = database
        self._curveQuote = db.CurveNodeQuoteFinal(self._curveName, self._valuationDate, database=database, isFinalTable=final)
        self._curveDataFrame = self._curveQuote.extract_from_db()
        self.x = np.append([self.LHS_BOUND], np.append(self._curveDataFrame.InYears.values, [self.RHS_BOUND]))
        self.y = np.append([self._curveDataFrame.Quote.values[0]], np.append(self._curveDataFrame.Quote.values, [self._curveDataFrame.Quote.values[-1]]))
        self.f = interp1d(self.x, self.y)

    def __call__(self, timeInYears):
        return self.f(timeInYears)

    def compute_discount_factor(self,timeInYears):
        return np.exp(-timeInYears*self(timeInYears))

    def get_diagnostics(self):
        diagnostics = pd.DataFrame(data={"TimeInYears": self.x[1:-1]})
        diagnostics["DiscountFactor"] = diagnostics.apply(lambda row: self.compute_discount_factor(row['TimeInYears']), axis=1)
        diagnostics["Yield"] = diagnostics.apply(lambda row: self(row['TimeInYears']), axis=1)
        return diagnostics


class SwapCurve:
    """
    valuationDate = '2014-01-01'
    yieldCurveName = 'US.USD.GVT.ZC'
    spreadCurveName = 'US.USD.SWP.ZCS'
    swapCurve = bond.SwapCurve(valuationDate, yieldCurveName, spreadCurveName)
    timeInYears = 25.0 #25 years
    actualSwapLevel = swapCurve(timeInYears)        
    """

    def __init__(self, valuationDate, govtCurveName, spreadCurveName, database = None):
        self.valuationDate = valuationDate
        self.govtCurveName = govtCurveName
        self.spreadCurveName = spreadCurveName
        self.database = database
        self.govtCurve = InterpolatedCurve(valuationDate, govtCurveName, self.database)
        self.spreadCurve = InterpolatedCurve(valuationDate, spreadCurveName, self.database)
       
    def __call__(self, timeInYears):
        return self.govtCurve(timeInYears) + self.spreadCurve(timeInYears)

    def compute_discount_factor(self,timeInYears):
        return np.exp(-timeInYears*self(timeInYears))

    def get_diagnostics(self):
        diagnostics = pd.DataFrame(data={"TimeInYears": self.govtCurve.x})
        diagnostics["DiscountFactor"] = diagnostics.apply(lambda row: self.compute_discount_factor(row['TimeInYears']), axis=1)
        diagnostics["Yield"] = diagnostics.apply(lambda row: self(row['TimeInYears']), axis=1)
        return diagnostics

class NSCurve:
    """

    """
    def __init__(self,L, b1, b2, b3):
        self.L = L
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        #self.timeInYears = np.array(timeInYears)
        #self.f = interp1d(self.timeInYears, self.curveLevels)

    #def __call__(self, timeInYears):
    #    return self.f(timeInYears)

    def compute_discount_factor(self, timeInYears):
        import macpy.nelsonsiegel as ns
        return ns.zeroYield(self.L, self.b1, self.b2, self.b3, timeInYears)

    @property
    def get_diagnostics(self, timeInYears):
        diagnostics = pd.DataFrame(data={"TimeInYears": timeInYears})
        diagnostics["DiscountFactor"] = diagnostics.apply(lambda row: self.compute_discount_factor(row.TimeInYears), axis=1)
        diagnostics["Yield"] = diagnostics.apply(lambda row: self(row.TimeInYears), axis=1)
        return diagnostics

class CustomCurve:
    """

    """
    def __init__(self, curveFunc, timeInYears):
        self.timeInYears = np.array(timeInYears)
        self.curveFunc = curveFunc
        self.curveLevels = curveFunc(self.timeInYears)
        self.f = interp1d(self.timeInYears, self.curveLevels)

    def __call__(self, timeInYears):
        return self.f(timeInYears)

    def compute_discount_factor(self,timeInYears):
        return [np.exp(-1.0*x*self(x)) for x in timeInYears]

    @property
    def get_diagnostics(self):
        diagnostics = pd.DataFrame(data={"TimeInYears": self.timeInYears})
        diagnostics["DiscountFactor"] = diagnostics.apply(lambda row: self.compute_discount_factor(row.TimeInYears), axis=1)
        diagnostics["Yield"] = diagnostics.apply(lambda row: self(row.TimeInYears), axis=1)
        return diagnostics


