
# coding: utf-8

# In[ ]:

import numpy as np
import plotly
import macpy
import macpy.utils.database as db
import plotly
import plotly.plotly as py
#print plotly.__version__            # version 1.9.4 required
plotly.offline.init_notebook_mode() # run at the start of every notebook
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import Surface
import cufflinks
import plotly_layout as pl
from plotly import tools

import macpy.next_gen_curve as ngc
import macpy.utils.ngc_utils as u
import pandas as pd # isn't pandas already imported??

#stuff for R2py library: calling R-functions through python
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

plotly.offline.init_notebook_mode()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:70% !important; }</style>"))


# In[ ]:


def calc_curve(df_both, alpha, tenor):
    #alpha = level/change weighting coefficient
    #tenor = choice of tenor
    #df_both = merged dataframe of levels and changes for each tenor by TradeDate
    tenor=str(tenor)
    curve=[]
    first_iteration=True
    for i,r in df_both[[tenor + '_l', tenor + '_c']].iterrows(): 
        if first_iteration:
            c=r[tenor + '_l']
        else:
            c=(1.-alpha)*r[tenor + '_l'] + alpha*(c_prev+ r[tenor + '_c'])  
        #store results
        tmp={}
        tmp.update({'TradeDate':i, 'Curve':c})
        curve.append(tmp)
        #get ready for next iteration
        c_prev = c
        first_iteration=False
    df_curve = pd.DataFrame(curve)
    return df_curve


# In[ ]:

sql="""
use marketdata
select distinct CurveName
from researchcurves
where Category like 'loglevel'
and Lud > '2017-03-08'
and Lud < '2017-03-10'
"""
df_names = db.MSSQL.extract_dataframe(sql, environment='DEV')
names = np.array(df_names.CurveName)


# ## Available Curves:

# In[ ]:

print df_names
print ''
print 'NUMBER of CURVES=' , len(names)


# ## Choose a curve

# In[ ]:

curve_index=4
specific_curve = str(names[curve_index])
print specific_curve


# In[ ]:

sql="""
use marketdata
select distinct TradeDate
from researchcurves
where CurveName = '%s'
order by TradeDate
""" % str(specific_curve)
df_dates = db.MSSQL.extract_dataframe(sql, environment='DEV')
dates = np.array(df_dates.TradeDate.values)


# ## Available Dates

# In[ ]:

print 'Start Date', dates[0]
print 'End Date', dates[len(dates)-1]
print 'NUMBER of DATES', len(dates)


# ## Get Curve History and Chart it

# In[ ]:


sql="""
use marketdata
select CurveName, Category, TimeInYears, [Level], TradeDate from researchcurves
where CurveName = '%s'
and Category = 'loglevel'
order by TradeDate, TimeInYears
""" % (specific_curve)
df_lvl = db.MSSQL.extract_dataframe(sql, environment='DEV')
df_lvl.head()


# In[ ]:

sql="""
use marketdata
select CurveName, Category, TimeInYears, [Level], TradeDate from researchcurves
where CurveName = '%s'
and Category = 'logchange'
order by TradeDate, TimeInYears
""" % (specific_curve)
df_chng = db.MSSQL.extract_dataframe(sql, environment='DEV')
df_chng.head()


# In[ ]:

df_lvlp = df_lvl.pivot('TradeDate','TimeInYears','Level')
df_chngp = df_chng.pivot('TradeDate','TimeInYears','Level')
cols = df_lvl.columns
df_both = df_lvlp.merge(df_chngp, how='outer',left_index=True,right_index=True, suffixes=('_l', '_c')) ## outer join so we see which fail
df_both.head()


# In[ ]:

# see if there are any dates where only levels or changes were calculated
tmp=df_both.isnull().any(axis=1)
tmp.loc[tmp.values == True]


# In[ ]:

alpha=0.9
tenor=20.0
df_curve_lvl = calc_curve(df_both, 0.,tenor)
df_curve_chng = calc_curve(df_both, 1.,tenor)
df_curve = calc_curve(df_both, alpha,tenor)

layout=pl.PlotLayout1('Date','Spread',width=900,title='%s: curve History Tenor=%sy, alpha=%s' % (specific_curve, tenor,alpha))
trace2 = go.Scatter(x=df_curve.TradeDate,y=u.mExp(np.array(df_curve.Curve)),name='alpha = %s' % alpha, mode='lines')
trace3 = go.Scatter(x=df_curve_lvl.TradeDate,y=u.mExp(np.array(df_curve_lvl.Curve)),name='Levels',mode='lines',line={'dash':'dot'})
trace4 = go.Scatter(x=df_curve_chng.TradeDate,y=u.mExp(np.array(df_curve_chng.Curve)),name='Changes', mode='lines', line={'dash':'dash'})

data=[trace2,trace3,trace4]

plotly.offline.iplot({'data':data,'layout':layout})
py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG')
filename = 'C:\\Users\\dantonio\\Documents\\Projects\\NextGenFit\\USD_TEN_SEN_20.pdf'
fig = go.Figure(data=data,layout=layout)
py.image.save_as(fig, filename=filename,format='pdf')


# In[ ]:

alpha=0.9
tenor=30.0
df_curve_1 = calc_curve(df_both, alpha,1.0)
df_curve_2 = calc_curve(df_both, alpha,2.0)
df_curve_5 = calc_curve(df_both, alpha,5.0)
df_curve_10 = calc_curve(df_both, alpha,10.0)
df_curve_20 = calc_curve(df_both, alpha,20.0)
df_curve_30 = calc_curve(df_both, alpha,30.0)

layout=pl.PlotLayout1('Date','Spread',width=900,title='Curve History, alpha=%s' % (alpha))
t1 = go.Scatter(x=df_curve_1.TradeDate,y=u.mExp(np.array(df_curve_1.Curve)),name='1y', mode='lines')
t2 = go.Scatter(x=df_curve_2.TradeDate,y=u.mExp(np.array(df_curve_2.Curve)),name='2y', mode='lines')
t3 = go.Scatter(x=df_curve_5.TradeDate,y=u.mExp(np.array(df_curve_5.Curve)),name='5y', mode='lines')
t4 = go.Scatter(x=df_curve_10.TradeDate,y=u.mExp(np.array(df_curve_10.Curve)),name='10y', mode='lines')
t5 = go.Scatter(x=df_curve_20.TradeDate,y=u.mExp(np.array(df_curve_20.Curve)),name='20y', mode='lines')
t6 = go.Scatter(x=df_curve_30.TradeDate,y=u.mExp(np.array(df_curve_30.Curve)),name='30y', mode='lines')

data=[t1, t2, t3, t4, t5, t6]

plotly.offline.iplot({'data':data,'layout':layout})
py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG')


# ## Run fit on particular date

# In[ ]:

date = '2016-03-31'
date_index = df_dates.loc[df_dates.TradeDate == date].index[0]
start_date = dates[date_index-1]
end_date = dates[date_index]
Ticker = ''
Currency = 'USD'
category=''
#specific_curve = Currency + '-' + Ticker + '-SEN'


print start_date, ',', end_date, ', ', specific_curve


# In[ ]:

specific_curve = 'JPY-TERUMO-SEN'

xmlFileName = 'C:\ContentDev-MAC\macpy\utils\\xml_test.xml'
xml_params = u.parse_xml(xmlFileName,specific_curve)

#xml params very useful for running all curves, but for ad hoc testing still useful to have direct input here:
input_params = {'specific_curve': specific_curve,
               'start_date': '2017-02-16',
               'end_date': '2017-02-17',
               'Currency': 'JPY',
               'alpha':0.5,
               'gamma':0.5,
               'smfitCompressionShort':0.1,
               'smfitCompressionLong':0.1,
               'sigmaScaleLevels':0.2,
               'sigmaScaleChanges':0.2,
               'smfitSpldf':4,
               'write_data':False,
               'debugplot':True,
               'debugplotSmoothFit':False,
               'debugplot_ixlist':[-1],
              'plot_f':False,
              'sswidthLong':.05,
              'sswidthShort':0.8,
               'maxIter':10,
               'numOutlierIterations': 3,
               'numIterLevelGuess': 2,
               'numBondsLevels':200,
               'numBondsChanges': 200,
               'overwriteStart':False}

params={}
params.update(xml_params)
params.update(input_params)

ssc_i = ngc.SmoothSplineCurve(**params)
ssc_i.run_single()


# 
# ## Misc testing

# In[ ]:

print a


# In[ ]:

from statsmodels.nonparametric.api import KDEUnivariate
def kde(x, x_grid, weights=None, bandwidth='normal_reference', kernel='gau', **kwargs):
    """Kernel Density Estimation with KDEUnivariate from statsmodels. **kwargs are the named arguments of KDEUnivariate.fit() """
    x = np.asarray(x)
    density = KDEUnivariate(x)
    if (weights is not None):      # NOTE that KDEUnivariate.fit() cannot perform Fast Fourier Transform with non-zero weights
        weights = np.asarray(weights)
        weights / np.sum(weights)
        neff = 1.0 / np.sum(weights ** 2)
        print neff
        d = 1.
        
        bw = np.power(neff, -1./(d+4.))  #scott
        print 'bw', bw
        
        print 'sum', np.sum(weights)
        if (len(x) == 1): # NOTE that KDEUnivariate.fit() cannot cope with one-dimensional weight array
            density.fit(kernel=kernel, weights=None, fft=False, **kwargs)
        else:
            print 'here'
            print x
            print weights
            density.fit(kernel='gau', weights=weights, bw=bw, fft=False, **kwargs)
    else:
        density.fit(kernel=kernel, **kwargs) #when kernel='gau' fft=true
    return density.evaluate(x_grid), weights


# In[ ]:

x = df.x
w = df.w
x_grid = np.arange(-1.,1.,0.05)
test, weights = kde(x, x_grid, weights=w) #bandwidth automatically selected


# In[ ]:

pd.DataFrame(test).to_clipboard()


# In[ ]:



