
# coding: utf-8

# In[1]:


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

import macpy.visualization.curve_slider as cs

import macpy.next_gen_curve as ngc
import pandas as pd # isn't pandas already imported??

#stuff for R2py library: calling R-functions through python
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

plotly.offline.init_notebook_mode()


# In[2]:

#dataset = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_fit.csv')
dataset = pd.read_clipboard()
dataset = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_time_VEGV_1yRate.csv')
dataset.TradeDate = dataset.TradeDate.apply(lambda x: pd.to_datetime(x))


# In[3]:

cs.curve_slider_time(dataset,'VEGV')


# In[4]:

dataset = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_fit.csv')
curve = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_fit_curve.csv')


# In[5]:

cs.curve_slider_fit(dataset,curve,'MXGV')


# In[6]:


dataset = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_1.csv')
dfc = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_curves_1.csv')


# In[7]:

cs.curve_slider_sigma(dataset,dfc,'PAGV')


# In[8]:

dataset = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_gamma_CHD.csv')
dfc = pd.read_csv('C:/ContentDev-MAC/macpy/visualization/cs_data_curves_gamma_CHD.csv')


# In[9]:

cs.curve_slider_gamma(dataset,dfc,'CHD')

