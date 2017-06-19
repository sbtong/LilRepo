import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import weighted as wq # wquantiles, weighted quantiles

#calling R from Python
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import macpy.utils.ngc_utils as u
import macpy.utils.ngc_queries as q
import visualization.ngc_plot as p
import macpy.utils.database as db
import macpy
import scipy.optimize as so

dfoas = q.get_example_curve(207241456)
sql="""
	  select t.InYears, cn.TenorEnum, r = Quote from marketdata..curve c
	  join MarketData..curvenodes cn on c.curveid = cn.curveid
		join MarketData..curvenodequote cq on cq.curvenodeid = cn.curvenodeid
		join MarketData..TenorEnum t on t.TenorEnum = cn.TenorEnum
	  where curveshortname = 'US.USD.GVT.ZC'
	  and TradeDate = '2017-02-17'
	  order by t.InYears

"""
dfr = db.MSSQL.extract_dataframe(sql, environment='PROD')
dfoas['oas'] = u.mExp(dfoas['Quote'])
dfoas['coupon'] = np.ones(dfoas.shape[0])*0.025

results = u.transform_oas_to_spot(dfoas, dfr, freq=2.)

print results







