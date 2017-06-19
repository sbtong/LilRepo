
import numpy as np
from statsmodels.stats.stattools import medcouple
from statsmodels.nonparametric.api import KDEUnivariate
from sklearn.neighbors import KernelDensity
#calling R from Python
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import math
import xml.etree.cElementTree as ET
import ngc_queries as q
import pandas as pd
import macpy.irr as irr
import scipy.optimize as so


def mLog_(x, alpha=0.01): #modified log function
    if (x > alpha) :
        y = np.log(x * np.e / alpha)
    else :
        y = x / alpha
    return y
mLog = np.vectorize(mLog_, excluded = ['alpha'], otypes = [np.float])

def mExp_(y, alpha=0.01): #modified exponential function
    if (y > 1.):
        x = np.exp(y)  * alpha / np.e
    else:
        x = y * alpha
    return x
mExp = np.vectorize(mExp_, excluded = ['alpha'], otypes = [np.float])

def gaussian_kernel(x, sigma): #kernel for weighting of data as a function of distance
    #N = 1./np.sqrt(2.*np.pi)/sigma
    #TODO: not using N to normalize weights
    return np.exp(-0.5 * (x/sigma)**2)

def adjBoxplotStats(x, coeff=1.5, a=-4., b=3.): #creates bounds for outlier detection
    x = np.array(x)
    MC = medcouple(x)
    [Q1, Q2, Q3] = np.percentile(x, [25, 50, 75])
    IQR = Q3 - Q1
    if (MC >= 0):
        fence = [Q1 - coeff*np.exp(a * MC)*IQR, Q3 + coeff*np.exp(b * MC)*IQR]
    else:
        fence = [Q1 - coeff*np.exp(-b * MC)*IQR, Q3 + coeff*np.exp(-a * MC)*IQR]
    return {'fence': fence, 'IQR': IQR, 'medcouple':MC}

def shiftedspline(splfun, h): # add constant shift e.g. to a UniversalSpline function
    predict = robjects.r('predict')
    return lambda x: h + np.array(predict(splfun,x)[1])

def logx1Compress_(x, xCompressLong, xCompressShort, xLast, xFirst, compressionLong, compressionShort, widthLong=0.0, widthShort=0.0):
    # = log(1+x)*gamma_short + c_short      for x < xCompressShort
    # = log(1+x)                            for CompressShort <= x<= xCompressLong
    # = log(1+x) * gamma_long + c_long      for x > xCompressLong
    # with c_short and c_long such that level is continuous
    #
    # xCompressLong: point where long end extrapolation begins
    # xCompressShort: point where short end extrapolation begins
    # xLast: maximum x value e.g. max(xout)
    # xShort: minimum x value e.g. min(xout)
    # compressionLong: controls how flat the extrapolation is. If 0 then the extrapolation is flat
    # compressionShort: controls how flat the extrapolation is. If 0 then the extrapolation is flat
    #
    # gamma < 1 compresses log(1+x) such that f(xLast) = (1+compression) * log(1+xCompress)
    x = float(x)
    xCompressShort = float(xCompressShort)
    xCompressLong = float(xCompressLong)
    xFirst = float(xFirst)
    xLast = float(xLast)
    compressionShort = float(compressionShort)
    compressionLong = float(compressionLong)

    #short end
    gammaShort = min(-compressionShort * np.log(1 + xCompressShort) /np.log((1+xFirst)/(1+xCompressShort)) ,  1.) # keep gamma < 1 otherwise starts to inflate not compress
    cShort = np.log(1+xCompressShort) * (1 - gammaShort)

    #long end
    gammaLong = min(compressionLong * np.log(1 + xCompressLong) /np.log((1+xLast)/(1+xCompressLong)) ,  1.) # keep gamma < 1 otherwise starts to inflate not compress
    cLong = np.log(1+xCompressLong) * (1 - gammaLong)

    #method 1: smooth exponential
    # if x < xCompressShort:
    #     ss = smoothingFunction(x,xCompressShort, -1.*width)
    #     y = (1.-ss)*(gammaShort*np.log(1.+x)+cShort) + ss*np.log(1.+x)
    # elif x >= xCompressShort and x <= xCompressLong:
    #     y = np.log(1.+x)
    # elif x > xCompressLong:
    #     ss = smoothingFunction(x,xCompressLong, width)
    #     y = (1.-ss)*(gammaLong*np.log(1.+x)+cLong) + ss*np.log(1.+x)

    #method 2: tanh
    ss = smoothStep(x,xCompressLong,xCompressShort, widthLong, widthShort)
    xMid = (xCompressShort+xCompressLong)/2.
    if x < xMid:
        gamma = gammaShort
        c = cShort
    elif x>=xMid:
        gamma = gammaLong
        c = cLong

    y = (1.-ss)*(gamma*np.log(1.+x)+c) + ss*np.log(1.+x)

    #     y = (1.-ss)*(gammaShort*np.log(1.+x)+cShort) + ss*np.log(1.+x)
    # elif x >= xCompressShort*(1+widthShort) and x <= xCompressLong*(1-widthLong):
    #     y = np.log(1.+x)
    # elif x > xCompressLong*(1-widthLong):
    #     ss = smoothStep(x,xCompressLong, -1.*widthLong)
    #     y = (1.-ss)*(gammaLong*np.log(1.+x)+cLong) + ss*np.log(1.+x)




    return y
logx1Compress = np.vectorize(logx1Compress_, excluded = ['xCompressLong', 'xCompressShort', 'xLast',
                                                         'xFirst', 'compressionLong', 'compressionShort','widthLong','widthShort'], otypes = [np.float])
def smoothingFunction(x,x_0,width):
    x_1 = x_0*(1.+width)
    alpha = np.log(2.0)/(x_1-x_0)

    return np.exp(-alpha*(x-x_0))

def smoothStep_(x,  xLong, xShort, widthLong, widthShort, s_target=0.75):
    # this creates a smooth step function
    #returns an np.array
    #x: input array
    #xShort/xLong: point at which smoothing is applied. Smooth step function = 0.5 at these points.
    #widthShort/widthLong: parameterizes width of smooth step function
    #s_target: specifies value of step function at width

    widthLong = max(widthLong,0.001)   #smooth step function not well defined if width=0
    widthShort = max(widthShort,0.001)

    # x_1 = x_0*(1.+width)
    # a = x_0
    # c = (s_target-0.5)/0.5
    # lc = 0.5*np.log((1+c)/(1-c))
    # b = (x_1 - a)/lc

    c = (s_target-0.5)/0.5
    lc = 0.5*np.log((1+c)/(1-c))

    xLongw = xLong*(1-widthLong)
    aL = xLong
    bL = (xLongw - aL)/lc

    xShortw = xShort*(1+widthShort)
    aS = xShort
    bS = (xShortw - aS)/lc

    xMid = (xShort+xLong)/2.

    #value of left hand sigmoid at xMid:
    sMidS = (0.5 + 0.5*np.tanh((xMid-aS)/bS)) #normalize by this to ensure sigmoids have value 1.0 at xMid
    #value of right hand sigmoid at xMid:
    sMidL = (0.5 + 0.5*np.tanh((xMid-aL)/bL)) #normalize by this to ensure sigmoids have value 1.0 at xMid

    if x < xMid:
        sstep = (0.5 + 0.5*np.tanh((x-aS)/bS))/sMidS
    elif x>= xMid:
        sstep = (0.5 + 0.5*np.tanh((x-aL)/bL))/sMidL

    return sstep
smoothStep = np.vectorize(smoothStep_, excluded = ['xShort', 'xLong', 'widthShort','widthLong','s_target'], otypes = [np.float])

# the kernel density estimation is purely for visualization

def kde_old(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs) # by default uses gaussian kernel
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def kde(x, x_grid, weights=None, bandwidth='normal_reference', kernel='gau', **kwargs):
    """Kernel Density Estimation with KDEUnivariate from statsmodels. **kwargs are the named arguments of KDEUnivariate.fit() """
    x = np.asarray(x)
    density = KDEUnivariate(x)
    if (weights is not None):      # NOTE that KDEUnivariate.fit() cannot perform Fast Fourier Transform with non-zero weights
        weights = np.asarray(weights)
        if (len(x) == 1): # NOTE that KDEUnivariate.fit() cannot cope with one-dimensional weight array
            density.fit(kernel=kernel, weights=None, bw=bandwidth, fft=False, **kwargs)
        else:
            density.fit(weights=weights, fft=False, bw=bandwidth, **kwargs)
    else:
        density.fit(kernel=kernel, bw=bandwidth, **kwargs) #when kernel='gau' fft=true
    return density.evaluate(x_grid)

def NS_zero_yield(xdata,b1,b2,b3):
    L=0.33
    yld = b1 + b2 * (1 - np.exp(-L*xdata))/(L*xdata)\
        + b3*( (1 - np.exp(-L*xdata))/(L*xdata)\
        - np.exp(-L*xdata) );
    return yld

    return yld

def col_normalize(s, N=1.):
    #s is a series, or column of a dataframe
    total = s.sum()/N
    ndf = s.apply(lambda x: x/total)

    return ndf

# def calc_pdf(x,y,bandwidth):
#     x=np.array(x)
#     y=np.array(y)
#     y=y*100.
#     y_raw=np.array([])
#     for i in np.arange(0,len(x)):
#         tmp = np.ones(int(y[i]))*x[i]
#         y_raw = np.append(y_raw,tmp).astype(float)
#     y_raw = np.array(y_raw).flatten()
#     x_grid = np.arange(-1.,1.,0.05)  #this is just to make the charts look smoother
#     pdf = kde(y_raw,x_grid, bandwidth=bandwidth)
#     return pdf, x_grid

def AmtOutstanding_pdf(df, x_grid, tenor_col, weight_col, bandwidth):
    tenors = np.log(1.+ np.array(df[tenor_col]))   #TODO: check this, pdf is normalized with log tenors. Is this OK?
    weights = np.array(df[weight_col])
    x_grid = np.log(np.array(x_grid) + 1.)
    pdf = kde(tenors, x_grid, weights=weights, bandwidth=bandwidth)  # self.TenorWeightDensityBandwidth

    #calculate area under discretized pdf for normalization
    x_grid_spaces=[]
    for i in range(0,len(x_grid)-1):
        x_grid_spaces.append( (x_grid[i+1] - x_grid[i])/2.)
    x_grid_spaces.append( (x_grid[-1:] - x_grid[-2:-1])/2.)
    x_grid_spaces = np.array(x_grid_spaces)
    pdf = np.array(pdf)
    #normalization coefficient

    N=0.
    #first entry
    N+=x_grid_spaces[0]*pdf[0]
    for i in range(1,len(x_grid_spaces)):
        N+= pdf[i]*(x_grid_spaces[i] + x_grid_spaces[i-1])

    #normalize the discrete distribution
    pdf /= N
    return pdf

def industry_level(Industry):
    # this is the length of the industry identifier, for each industry e.g. PHARM has 5 letters
    ind_dict = {5: 'IndustryGroup', 4: 'Sector', 3: 'SectorGroup',2: 'Market', 10:'AllSectors'}
    size = len(Industry)
    industryLevel = ind_dict[size]
    return industryLevel

def ind_level_dist_calc(x,IndustryLevel,c):
    if x==IndustryLevel:
        return c
    else:
        return 0.

def ind_level_dist_calc_DVL(row,IndustryLevel,IssuerIndustryLevel, IssuerRegion,c):
    if (row[IndustryLevel] == IssuerIndustryLevel) & (row['Region'] == IssuerRegion):
        return c
    else:
        return 0.

def ind_level_weight_calc(x,gamma):
    return math.pow(gamma,x)

def currency_support_priority(x, Currency, fallbackCurrency, currencyList, groupWeight, fallbackWeight):
    if x == Currency:
        w = 1.
    elif x in currencyList and x != fallbackCurrency:
        w = groupWeight
    elif x == fallbackCurrency:
        w = fallbackWeight
    return w

def AmtOutstanding_pdf_old(self, df, x_grid):
    tenors = np.log(1.+ np.array(df[self.tenor_col]))
    weights = np.array(df[self.weight_col])*100. #TODO: this is ugly. There must be a better way.
    x_grid = np.array(x_grid)

    weightedTenors=np.array([])
    for i in np.arange(0,len(tenors)): #TODO: this is slow
        tmp = np.ones(int(weights[i]))*tenors[i]
        weightedTenors = np.append(weightedTenors,tmp).astype(float)
    weightedTenors = np.array(weightedTenors).flatten()
    #use ln(1+x) grid
    x_grid = np.log(x_grid + 1.)
    pdf = kde_old(weightedTenors,x_grid, bandwidth=0.2)

    #calculate area under discretized pdf for normalization
    x_grid_spaces=[]
    for i in range(0,len(x_grid)-1):
        x_grid_spaces.append( (x_grid[i+1] - x_grid[i])/2.)
    x_grid_spaces.append( (x_grid[-1:] - x_grid[-2:-1])/2.)
    x_grid_spaces = np.array(x_grid_spaces)
    pdf = np.array(pdf)
    #normalization coefficient

    N=0.
    #first entry
    N+=x_grid_spaces[0]*pdf[0]
    for i in range(1,len(x_grid_spaces)):
        N+= pdf[i]*(x_grid_spaces[i] + x_grid_spaces[i-1])

    #normalize the discrete distribution
    pdf = pdf/N

    return pdf

def weighted_average(df):
    #function to calculate weighted average, including when weights are zero
    # df = dataframe with columns: [eps,w]
    if df.w.sum() == 0:
        av = np.average(df.eps)
    else:
        av = np.average(df.eps,weights=df.w)
    return av

def compress_weights(weights, wrOom):
    # this function performs a compression of weights such that max_weight/min_weight is wrOom orders to magnitude
    lweights = np.log10(weights)
    outl_fence = adjBoxplotStats(lweights)['fence']
    lower = outl_fence[0]
    upper = outl_fence[1]
    lwc = w_interpolate(lweights, upper, lower, wrOom)
    wc = pow(10, lwc)
    return wc

def w_interpolate_(weight, upper, lower, wrOom):
    if weight >= upper:
        wc = 0.
    elif weight <= lower:
        wc = -wrOom
    else:
        wc = -wrOom + ((weight-lower)/(upper-lower)*wrOom)
    return wc
w_interpolate = np.vectorize(w_interpolate_, excluded = ['upper', 'lower', 'wrOom'], otypes = [np.float])

def get_cluster_hier(Industry):
    ind_dict = {5:'IndustryGroup',4:'Sector',3:'SectorGroup',2:'Market',10:'AllSectors'}
    IndustryLevel = ind_dict[len(Industry)]
    df = q.get_cluster_hierarchy(IndustryLevel, Industry)

    df_dict = df.to_dict('records')[0]
    #index = np.array
    #hier = np.array(df.iloc[[0]])[0][5-len(Industry):]

    cdict = {'AllSectors': 'ALLSECTORS'}
    if IndustryLevel != 'AllSectors':
        i=len(Industry)
        #np.sort(ind_dict.keys())[::-1]
        while i>=2:
            cdict.update({str(ind_dict[i]):str(df_dict[ind_dict[i]])})
            i-=1
    return cdict

def calc_rating_weight(rank, InvSumAmtOPdf):
    # InvSumAmtOPdf is an np array of length 21
    # rank is a number in the range [1,21] corresponding to the rating e.g. AA-
    if np.isnan(rank):
        w = 0
    else:
        w = InvSumAmtOPdf[rank - 1]
    return w

def parse_xml(xmlFileName, curveShortName):
    tree = ET.parse(xmlFileName)
    root = tree.getroot()

    param_dict ={}
    tenorGrid=[]
    tenorGridEnum=[]
    #add default params
    for c in root.findall('default'):
        for cc in c:
            if cc.tag == 'logCompressionParams':
                for item in cc:
                    tmp = {item.tag: float(item.text)}
                    param_dict.update(tmp)
            elif cc.tag == 'fitParams':
                for item in cc:
                    tmp = {item.tag: float(item.text)}
                    param_dict.update(tmp)
            elif cc.tag == 'dataParams':
                for item in cc:
                    tmp = {item.tag: item.text}
                    param_dict.update(tmp)
            elif cc.tag == 'tenorGrid':
                for item in cc:
                    tenorGrid.append(float(item[0].text))
                    tenorGridEnum.append(item[1].text)
                tenorGridEnum = np.array(tenorGridEnum)
                tenorGrid = np.array(tenorGrid)
                param_dict.update({'tenorGrid': tenorGrid})
                param_dict.update({'tenorGridEnum': tenorGridEnum})

    for c in root.findall(curveShortName):
        for cc in c:
            if cc.tag == 'logCompressionParams':
                for item in cc:
                    tmp = {item.tag: float(item.text)}
                    param_dict.update(tmp)
            elif cc.tag == 'fitParams':
                for item in cc:
                    tmp = {item.tag: float(item.text)}
                    param_dict.update(tmp)
            elif cc.tag == 'dataParams':
                for item in cc:
                    tmp = {item.tag: item.text}
                    param_dict.update(tmp)
            elif cc.tag == 'tenorGrid':
                for item in cc:
                    tenorGrid.append(float(item[0].text))
                    tenorGridEnum.append(item[1].text)
                tenorGridEnum = np.array(tenorGridEnum)
                tenorGrid = np.array(tenorGrid)
                param_dict.update({'tenorGrid': tenorGrid})
                param_dict.update({'tenorGridEnum': tenorGridEnum})

    return param_dict

def create_xml(xmlFileName):
    tenorGrid_array = np.array([0, 0.083333333333, 0.166666667, 0.25, 0.5, 0.75, 1, 1.25, 1.5,
                                2, 2.5, 3, 3.5, 4, 5, 7, 10, 12, 15, 20, 25, 30, 40])
    tenorGridEnum_array = np.array(['0M', '1M', '2M', '3M', '6M', '9M', '1Y', '15M', '18M',
                                    '2Y', '30M', '3Y', '42M', '4Y', '5Y', '7Y', '10Y', '12Y', '15Y', '20Y', '25Y',
                                    '30Y', '40Y'])

    root = ET.Element("AxCurveParams")
    default = ET.SubElement(root, "default")
    logCompressionParams = ET.SubElement(default, "logCompressionParams")
    dataParams = ET.SubElement(default, "dataParams")
    fitParams = ET.SubElement(default, "fitParams")

    # dataParams
    ET.SubElement(dataParams, "curveCol").text = 'CurveId'
    ET.SubElement(dataParams, "spreadCol").text = 'logOAS'
    ET.SubElement(dataParams, "tenorCol").text = 'Effective_Duration'
    ET.SubElement(dataParams, "weightCol").text = 'AmtOutstanding'
    ET.SubElement(dataParams, "smfitLogxScale").text = 'True'
    ET.SubElement(dataParams, "removeOutliers").text = 'True'
    ET.SubElement(dataParams, "normaliseWeightsByCurve").text = 'False'

    # logCompressionParams:
    ET.SubElement(logCompressionParams, "smfitLongCompressStart").text = str(float(0.1))
    ET.SubElement(logCompressionParams, "smfitShortCompressStart").text = str(float(0.1))
    ET.SubElement(logCompressionParams, "smfitCompressionLong").text = str(float(0.1))
    ET.SubElement(logCompressionParams, "smfitCompressionShort").text = str(float(0.1))
    ET.SubElement(logCompressionParams, "sswidthLong").text = str(float(0.2))
    ET.SubElement(logCompressionParams, "sswidthShort").text = str(float(0.5))

    # fitParams
    ET.SubElement(fitParams, "smfitSpldf").text = str(float(4))
    ET.SubElement(fitParams, "sigmaScaleLevels").text = str(float(0.2))
    ET.SubElement(fitParams, "sigmaScaleChanges").text = str(float(0.1))
    ET.SubElement(fitParams, "outlierPctTolLevels").text = str(float(0.1))
    ET.SubElement(fitParams, "outlierPctTolChanges").text = str(float(0.05))
    ET.SubElement(fitParams, "errTol").text = str(float(1E-3))
    ET.SubElement(fitParams, "maxIter").text = str(float(10))
    ET.SubElement(fitParams, "numOutlierIterations").text = str(float(3))
    ET.SubElement(fitParams, "numIterLevelGuess").text = str(float(2))
    ET.SubElement(fitParams, "smfitMinWeightTol").text = str(float(1E-4))
    ET.SubElement(fitParams, "numBondsLevels").text = str(float(200))
    ET.SubElement(fitParams, "numBondsChanges").text = str(float(100))
    ET.SubElement(fitParams, "numIssuers").text = str(float(1))
    ET.SubElement(fitParams, "tenorWeightDensityBandwidth").text = str(float(0.2))
    ET.SubElement(fitParams, "awDifferential").text = str(float(0.))
    ET.SubElement(fitParams, "gamma").text = str(float(0.5))

    # tenorGrid
    tenorGrid = ET.SubElement(default, "tenorGrid")
    # for i in tenorGrid_array:
    #     ET.SubElement(tenorGrid, "item").text = str(float(i))
    # tenorGridEnum
    # tenorGridEnum = ET.SubElement(default, "tenorGridEnum")
    for i, t in enumerate(tenorGrid_array):
        item = ET.SubElement(tenorGrid, "item")
        ET.SubElement(item, "inYears").text = str(float(t))
        ET.SubElement(item, "enum").text = tenorGridEnum_array[i]

    # Add exception
    EURGGBSEN = ET.SubElement(root, "EUR-GGB-SEN")
    logCompressionParams = ET.SubElement(EURGGBSEN, "logCompressionParams")
    dataParams = ET.SubElement(EURGGBSEN, "dataParams")
    fitParams = ET.SubElement(EURGGBSEN, "fitParams")
    tenorGrid = ET.SubElement(EURGGBSEN, "tenorGrid")

    ET.SubElement(fitParams, "sigmaScaleLevels").text = str(float(0.1))
    ET.SubElement(fitParams, "numBondsLevels").text = str(float(50))

    tree = ET.ElementTree(root)
    tree.write(xmlFileName)

    return 0

def bond_zspread_(price,maturity,coupon,freq,spot_t,spot):
    #calculate z-spread of a bond
    cft, cf = cf_maturities_and_cf(maturity, coupon, freq)
    r = np.interp(cft,spot_t,spot)
    Z = discount_factor(cft,r,0)
    cf_Z = cf*Z
    #print price
    zspread = irr.compute_irr(cft, cf_Z, price)
    return zspread

def discount_factor(t, r, s):
    z = np.exp(-r * t - s * t)
    return z


def discount_factor_s_f(t, r, s_f, s_ft):
    # r is the risk free spot rate
    # s_f is the forward spread array
    s_f = np.array(s_f)
    s_ft = np.array(s_ft)
    z_r = discount_factor(t, r, 0.)

    index = np.where(s_ft < t)
    t = s_ft[index]
    f = s_f[index]

    sum_tf = 0
    for i in range(0, len(t)):
        if i == 0:
            sum_tf += t[i] * f[i]
        else:
            sum_tf += (t[i] - t[i - 1]) * f[i]

    sum_tf += (t - t[i]) / (s_ft[i + 1] - t[i]) * s_f[i + 1]

    # print sum_tf
    z = np.exp(-sum_tf)
    return z


def duration_(maturity, coupon, freq, spot_t, spot, oas):
    # this is the simple discount cashflows using interest rate and spot spread
    cft, cf = cf_maturities_and_cf(maturity, coupon, freq)
    r = np.interp(cft, spot_t, spot)  # use this if we load an array for r
    Z = discount_factor(cft, r, oas)
    cf_discount = cf * Z
    t_cf_discount = cft * cf * Z
    PV = cf_discount.sum()
    D = t_cf_discount.sum() / PV
    return D


duration = np.vectorize(duration_, excluded=['coupon', 'freq', 'oas'], otypes=[np.float])

def bond_price_oas_(maturity, coupon, oas, freq, spot_t, spot):
    cft, cf = cf_maturities_and_cf(maturity, coupon, freq)
    r = np.interp(cft, spot_t, spot)  # use this if we load an array for r
    Z = discount_factor(cft, r, oas)
    cf_discount = cf * Z
    PV = cf_discount.sum()
    return PV
# bond_price_oas = np.vectorize(bond_price_oas_, excluded=['freq','r'], otypes=[np.float])

def cf_maturities_and_cf(maturity, coupon, freq):
    freq = float(freq)
    coupon = float(coupon)
    cft = np.arange(maturity, 0., -1. / freq)
    cft[:] = cft[::-1]  # reverse the order
    cf = np.ones(len(cft)) * coupon / freq
    cf[-1:] += 1.  # add principal cashflow
    return cft, cf


def sum_fwd_time_(t, fwd_t, fwd):
    # fwd is an array of forward
    # fwd_t is an array of maturities
    # t is the target time
    fwd_t = np.array(fwd_t)
    t_diff = []

    for i in range(0, len(fwd_t)):
        if i == 0:
            t_diff.append(fwd_t[i])
        else:
            t_diff.append(fwd_t[i] - fwd_t[i - 1])
    t_diff = np.array(t_diff)

    sum_prod = 0
    il = np.where(fwd_t >= t)[0][0]
    if il == 0:
        sum_prod = t * fwd[0]
    else:
        sum_prod = np.sum(fwd[:il] * t_diff[:il])
        sum_prod += (t - fwd_t[il - 1]) * fwd[il]
    return np.exp(-sum_prod)
sum_fwd_time = np.vectorize(sum_fwd_time_, excluded=['fwd_t', 'fwd'], otypes=[np.float])


def bond_price_solve_fwd(f_solve, fwd_t, fwd, maturity, coupon, freq, target_price):
    cft, cf = cf_maturities_and_cf(maturity, coupon, freq)
    # print cft, cf

    fwd_t = np.append(fwd_t, maturity)
    fwd = np.append(fwd, f_solve)
    # print 'fwd_t', fwd_t
    # print 'fwd', fwd

    cft_discount = []
    for i in range(0, len(cft)):
        # print i
        # print 'sum=', sum_fwd_time_(cft[i], fwd_t, fwd)
        cft_discount.append(sum_fwd_time_(cft[i], fwd_t, fwd))

    cft_discount = np.array(cft_discount)
    price = np.sum(cft_discount * cf)

    return price - target_price


def bond_price_fwd(fwd_t, fwd, maturity, coupon, freq):
    cft, cf = cf_maturities_and_cf(maturity, coupon, freq)
    cft_discount = []
    for i in range(0, len(cft)):
        cft_discount.append(sum_fwd_time_(cft[i], fwd_t, fwd))
    cft_discount = np.array(cft_discount)
    price = np.sum(cft_discount * cf)
    return price


def spot_to_fwd(maturity, spot):
    # maturity = array of maturity points
    # spot = array of spot rates
    ft = maturity
    s = spot
    f = []
    for i, t in enumerate(ft):
        if i == 0:
            f.append(s[i])
        else:
            p_f = np.exp(-spot[i] * ft[i]) / np.exp(-spot[i - 1] * ft[i - 1])
            fwd = -1. / (ft[i] - ft[i - 1]) * np.log(p_f)
            f.append(fwd)
    f = np.array(f)
    return f

def calculate_bond_maturities(duration, oas, coupon, rt, r, freq=2.):
    # for a set of (duration, oas) points this function does the following:
    #     1. for each pair, calculate duration of bonds with integer maturities
    #     2. Interpolate the calculated durations to find the maturity that matches the duration in the pair.
    # rt = risk free rate time points
    # r = risk free rate
    duration = np.array(duration)
    oas = np.array(oas)
    coupon = np.array(coupon)
    results=[]
    mat = np.arange(1. / freq, 100., 1. / freq)
    for i in range(0,len(duration)):  #loop over (d,oas) pairs
        oas_i = oas[i]
        d_i = duration[i]
        coupon_i = coupon[i]
        d=[]
        for m in mat:
            d.append(duration_(m, coupon_i, freq, rt, r, oas_i))
        if d_i <= np.max(d) and d_i > 0.:
            if d_i <= 1./freq:
                m_interp = d_i
            else:
                m_interp = np.round(np.interp(d_i, d, mat),4)  #we round to avoid spurious time increments < 1 day
            tmp_dict = {'maturity':m_interp,'duration':d_i,'oas':oas_i,'coupon':coupon_i}
            results.append(tmp_dict)
    results = pd.DataFrame(results)
    return results

def calculate_bond_prices(results, dfr, freq=2.):
    price = []
    for i, row in results.iterrows():
        p = bond_price_oas_(row['maturity'], row['coupon'], row['oas'], freq, dfr.InYears, dfr.r)
        price.append(p)
    results['price'] = np.array(price)
    return results

def calculate_fwd_rates(results, dfr, freq=2.):
    results = results[results.duration > 0.]  # this should be the case, but double check
    ft = np.array(results[results.duration <= 1. / freq]['duration'])
    spot = np.interp(ft, dfr.InYears, dfr.r)
    s = np.array(results[results.duration <= 1. / freq]['oas']) + spot
    f=[]
    for i in range(0, len(ft)):
        if i == 0:
            f.append(s[i])
        else:
            p_f = np.exp(-s[i] * ft[i]) / np.exp(-s[i - 1] * ft[i - 1])
            fwd = -1. / (ft[i] - ft[i - 1]) * np.log(p_f)
            f.append(fwd)
    f = np.array(f)

    for i, row in results.iterrows():
        if row['maturity'] > 1. / freq:
            f_solve = so.newton(bond_price_solve_fwd, 0.02,
                                args=(ft, f, row['maturity'], row['coupon'], freq, row['price']))
            f = np.append(f, f_solve)
            ft = np.append(ft, row['maturity'])
            # print row['maturity'], ' calc price = ', bond_price_fwd(ft, f, row['maturity'], row['coupon'],
            #                                                           freq), 'initial price = ', row['price']
    results['fwd'] = f

    return results

def calculate_spots_from_fwds(results):
    ft = np.array(results.maturity)
    f = np.array(results.fwd)
    total_spot = []
    for i,row in results.iterrows():
        print row['maturity']
        z = sum_fwd_time_(row['maturity'], ft, f)
        total_spot.append(-np.log(z)/row['maturity'])
    results['total_spot'] = total_spot
    return results

def transform_oas_to_spot(dfoas, dfr, freq=2.):
    # function to transform the (duration, oas) curve into a cont. compounded spot spread curve

    # step 1: calculate the maturities of bonds that match the duration input, for a given coupon
    results = calculate_bond_maturities(dfoas.InYears, dfoas.oas, dfoas.coupon, dfr.InYears, dfr.r, freq=freq)

    # step 2: calculate the prices of the bonds, using the input oas
    results = calculate_bond_prices(results, dfr, freq)

    # step 3: bootstrap the prices to calculate a forward curve
    results = calculate_fwd_rates(results, dfr, freq)

    # step 4: calculate the spot curve from the forward curve
    results = calculate_spots_from_fwds(results)

    spread_spot = results['total_spot'] - np.interp(results.maturity, dfr.InYears, dfr.r)
    results['spread_spot'] = spread_spot

    return results

def calculate_coupon_term_structure(IssuerBonds, tenorGrid, df=3):
    # IssuerBonds is a dataframe with the following columns: [['MatYears', 'Coupon', 'AmtOutstanding']], with outliers removed
    # this function calculates the average coupon at each point in tenorGrid
    # maturity = array of bond maturities
    # coupon = array of coupons
    # weight = array of weights for weighted average calc
    # tenorGrid = set of tenor points
    IssuerBonds = IssuerBonds.sort_values('MatYears')
    x = np.array(IssuerBonds.MatYears)
    y = np.array(IssuerBonds.Coupon)
    w = np.array(IssuerBonds.OriginalAmtOutstanding)

    uniqueX = IssuerBonds.MatYears.unique()

    if len(uniqueX) >= df+1:
        rspline = robjects.r('smooth.spline')
        predict = robjects.r('predict')

        rspl = rspline(x, y=y, w=w, df=3)  # R smooth.spline

        yout = np.array(predict(rspl, tenorGrid)[1])

        min = yout[np.where(tenorGrid > x[0])][0]
        max = yout[np.where(tenorGrid < x[-1:])][-1:][0]

        yout[np.where(tenorGrid < x[0])]=min
        yout[np.where(tenorGrid > x[-1:])] = max
    else:
        level = np.average(y,weights=w)
        yout = np.ones(len(tenorGrid))*level

    return yout

def process_data(df_data, curveCol, spreadCol, weightCol, tenorCol):
    if df_data.shape[0] > 0:
        df_data[curveCol] = df_data.Currency + '-' + df_data.TICKER + '-' + df_data.PricingTier
        df_data = df_data.loc[df_data[curveCol].notnull()]
        df_data = df_data.loc[df_data['MatDate'] < pd.to_datetime('2200-01-01')]
        df_data.loc[:,'MatYears'] = (pd.to_datetime(df_data.loc[:,'MatDate']) - pd.to_datetime(df_data.loc[:,'AnalysisDate'])).dt.days / 365.25  # in case we need it
        #df_data[self.tenorCol] = df_data['Effective_Duration']
        df_data.loc[:,'OriginalAmtOutstanding'] = df_data.loc[:,weightCol]

        #no very negative spreads
        df_data = df_data.loc[df_data.OAS_Swap> -0.2]  #TODO: may want to refine this. Greater than -20% doesn't seem too extreme though!

        df_data[spreadCol] = mLog(df_data.OAS_Swap)  #TODO: generalize the name 'OAS_Swap' should be an input
        #amt outstanding > $1,000,000
        df_data = df_data.loc[df_data[weightCol] > 1.]  #TODO: there is a weight cutoff in the initial universe query, so this is perhaps not needed?
        df_data = df_data.loc[df_data[tenorCol] > 0.5]  #TODO: review, this is probably extreme but may be a good starting point

        df_data = df_data.dropna(axis=0,subset=['RiskEntityId','TICKER'])   #curveId hinges on these being not null
        df_data = df_data.loc[df_data.IndustryGroup.notnull()]  #if no industry group that can't calculate peer weights
        df_data = df_data.drop_duplicates(subset='ISIN')    #if ISIN duplicates are genuine then may need to groupby ISIN and sum amount outstanding

    return df_data

def write_curve_to_db(res_f, res_d, CurveId, date_i, tenorGridEnum):
    level = np.array((res_f['yout']))
    change = np.array((res_d['yout']))
    weight = np.array(res_f['AmtOutstanding_pdf'])
    coupon = np.array(res_f['coupon'])
    IssuerBonds_f = (res_f['IssuerBonds'])
    IssuerBonds_d = (res_d['IssuerBonds'])
    IssuerBonds = IssuerBonds_f.merge(IssuerBonds_d.loc[:, ('ISIN', 'OAS_diff', 'OAS_Model', 'outlier')],
                                      how='outer', on='ISIN', suffixes=('_f', '_d'))

    for i in range(0, level.size):
        q.commit_node(CurveId, date_i, tenorGridEnum[i], level[i], 'll')
        q.commit_node(CurveId, date_i, tenorGridEnum[i], change[i], 'lc')
        q.commit_node(CurveId, date_i, tenorGridEnum[i], weight[i], 'w')
        q.commit_node(CurveId, date_i, tenorGridEnum[i], coupon[i], 'cn')
    for i, r in IssuerBonds.iterrows():
        q.commit_ModelOAS(CurveId, date_i, r['ISIN'], r['OAS_Model_f'],
                          'NULL' if pd.isnull(r['OAS_Model_d']) else r['OAS_Model_d'],
                          r['OAS_diff_f'],
                          'NULL' if pd.isnull(r['OAS_diff_d']) else r['OAS_diff_d'],
                          int(r['outlier_f']),
                          'NULL' if pd.isnull(r['outlier_d']) else int(r['outlier_d']))
    q.commit_SumAmtOutstanding(CurveId, date_i, res_f['SumAmtOutstanding'])

    return


    # maturity = np.array(maturity)
    # coupon = np.array(coupon)
    # weight = np.array(weight)
    # tenorGrid = np.array(tenorGrid)
    #
    # result = []
    # for i,t in enumerate(tenorGrid):
    #     #print 'coupon calc t=', t
    #     m = maturity[np.where(maturity >= t)]
    #     c = coupon[np.where(maturity >= t)]
    #     w = weight[np.where(maturity >= t)]
    #     if w.sum()>0:
    #         avg_c = np.average(c, weights=w)
    #     else:
    #         avg_c = avg_prev
    #     result.append(avg_c)
    #     avg_prev = avg_c

def cluster_interp_and_extrap(xh,yh, eps):
    tenorCurve = []
    xMin = xh[0]  # should be an int
    xMax = xh[-1]  # should be an int
    yShort = yh[0]
    yLong = yh[-1]

    for r in (np.arange(0, 21) + 1):
        if r < xMin:
            dx = xMin - r
            y = yShort - eps * dx
        elif r >= xMin and r <= xMax:
            y = np.interp(r, xh, yh)
        elif r > xMax:
            dx = r - xMax
            y = yLong + eps * dx
        tenorCurve.append(y)

    return tenorCurve






