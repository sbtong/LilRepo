import os
import unittest
import macpy
import macpy.utils.database as db
import macpy.bond as bond
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import Scatter, Margin, Figure, XAxis, YAxis, ZAxis, Scene, Layout, Surface
from macpy.CurveFrameView import MeshView, Mesh3D, Tenor

import pandas as pd
from pandas.util.testing import assert_series_equal

import plotly.tools as tls


class TestCurveFrameView(unittest.TestCase):

    def test_swap_spread_graph_generation(self):
        curves = ('US.USD.SWP.ZCS.v2', 'GB.GBP.SWP.ZCS.v2', 'EU.EUR.SWP.ZCS.v2', 'JP.JPY.SWP.ZCS.v2')
        #curves = ( 'GB.GBP.SWP.ZCS.v2',)
        #curves = ('GB.GBP.SWP.ZCS','EU.EUR.SWP.ZCS')
        curve_list = """
        ('{}')
        """.format("','".join(curves))

        date_range = '2006-01-01:2016-07-12'
        trade_date_begin, trade_date_end = date_range.split(':')

        sql_swpspr = """
        select CurveShortName=SUBSTRING(CurveShortName,1,14), TradeDate, Quote, te.TenorEnum
        from {0}[MarketData].[dbo].[Curve] cv
        join {0}[MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
        join {0}[MarketData].[dbo].TenorEnum te on cn.TenorEnum = te.TenorEnum
        join {0}[MarketData].[dbo].CurveNodeQuote{1} cq on cq.CurveNodeId = cn.CurveNodeId
        where
        cq.TradeDate >= '{2}' and
        cq.TradeDate <= '{3}' and
        cv.CurveShortName in {4}
        """
        #tls.embed('https://plot.ly/~cufflinks/8')  # very slow remote call ... not sure what it does
        #plotly_user = 'wgajate1'
        #plotly_password = 'tnf7krpd0c'
        #py.sign_in(plotly_user, plotly_password)  # Replace the username, and API key with your credentials.
        py.sign_in('davidjantonio', '73b4emn642') # Replace the username, and API key with your credentials.

        sql_swpspr_staging = sql_swpspr.format('', '', trade_date_begin, trade_date_end, curve_list)
        dfspr_staging = db.MSSQL.extract_dataframe(sql_swpspr_staging)
        dfspr_staging['Source'] = 'Staging'

        curves = ('US.USD.SWP.ZCS', 'GB.GBP.SWP.ZCS', 'EU.EUR.SWP.ZCS', 'JP.JPY.SWP.ZCS')
        #curves = ('GB.GBP.SWP.ZCS',)
        curve_list = """
        ('{}')
        """.format("','".join(curves))
        sql_swpspr_final = sql_swpspr.format('[prod_mac_mkt_db].', "Final", trade_date_begin, trade_date_end, curve_list)
        dfspr_final = db.MSSQL.extract_dataframe(sql_swpspr_final)
        dfspr_final['Source'] = 'Prod'
        dfspr_final = dfspr_final.append(dfspr_staging)

        dfspr_final["TenorInYears"] = dfspr_final["TenorEnum"].apply(
            lambda x: round(Tenor.convert_tenor_to_years(x), 3))

        columns = ['CurveShortName', 'TradeDate', 'Source', 'TenorInYears', 'Quote']

        dfts = dfspr_final[columns].groupby(columns[:-1]).sum().unstack(level=0).unstack(level=-2)

        directory = 'E:/Phoenix-Research/Risk/DataDerivation/latex/'
        tenors = [.5]  #  , 2.0, 5.0, 10.0, 30.0]
        tenors = [.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        import time
        for curve in curves:
            time.sleep(2)
            #  plot_swap_spread(curve, dfts, directory, date_range=date_range)
            mesh = Mesh3D(dfts, curve, date_range=date_range)
            mesh_view = MeshView(mesh, tenors)
            mesh_view.plot_tenors(tenors, directory, plot_vol=False, use_html=False)


    def test_swap_spread_3dgraph_generation(self):
        #curves = ('US.USD.SWP.ZCS', 'GB.GBP.SWP.ZCS', 'EU.EUR.SWP.ZCS')
        curves = ('US.USD.SWP.ZCS',)
        curve_list = """
        ('{}')
        """.format("','".join(curves))

        date_range = '2001-03-01:2016-06-13'
        trade_date_begin, trade_date_end = date_range.split(':')

        sql_swpspr = """
        select CurveShortName, TradeDate, Quote, te.TenorEnum
        from  [Elara].[MarketData].[dbo].[Curve] cv
        join [Elara].[MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
        join [Elara].MarketData.dbo.TenorEnum te on cn.TenorEnum = te.TenorEnum
        join [Elara].[MarketData].[dbo].CurveNodeQuote{} cq on cq.CurveNodeId = cn.CurveNodeId
        where
        cq.TradeDate >= '{}' and
        cq.TradeDate <= '{}' and
        cv.CurveShortName in {}
        """
        tls.embed('https://plot.ly/~cufflinks/8')
        plotly_user = 'davidjantonio'
        plotly_password = '73b4emn642'
        py.sign_in(plotly_user, plotly_password)  # Replace the username, and API key with your credentials.

        sql_swpspr_staging = sql_swpspr.format('_Research', trade_date_begin, trade_date_end, curve_list)
        dfspr_staging = db.MSSQL.extract_dataframe(sql_swpspr_staging)
        dfspr_staging['Source'] = 'Staging'

        sql_swpspr_final = sql_swpspr.format("Final", trade_date_begin, trade_date_end, curve_list)
        dfspr_final = db.MSSQL.extract_dataframe(sql_swpspr_final)
        dfspr_final['Source'] = 'Prod'
        dfspr_final = dfspr_final.append(dfspr_staging)

        dfspr_final["TenorInYears"] = dfspr_final["TenorEnum"].apply(
            lambda x: round(Tenor.convert_tenor_to_years(x), 3))

        columns = ['CurveShortName', 'TradeDate', 'Source', 'TenorInYears', 'Quote']

        dfts = dfspr_final[columns].groupby(columns[:-1]).sum().unstack(level=0).unstack(level=-2)

        directory = 'E:/Phoenix-Research/Risk/DataDerivation/latex/figs'
        tenors = [.5]  #  , 2.0, 5.0, 10.0, 30.0]
        #tenors = [.5, 2.0, 5.0, 10.0, 30.0]

        #  plot_swap_spread(curve, dfts, directory, date_range=date_range)
        mesh = Mesh3D(dfts, 'US.USD.SWP.ZCS', date_range=date_range)
        # Generate 3D plot

        camera = dict(
            up=dict(x=1.0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.0, y=-2.0, z=2.5)
        )

        scene = Scene(
            xaxis=XAxis(title='Trade Date'),
            yaxis=YAxis(title='Tenor'),
            zaxis=ZAxis(title='Swap Yield (%)'),
            camera=camera
                )

        layout = Layout(
            title='Swap Yield Curve: ' + 'US',
            autosize=False,
            width=1200,
            height=800,
            margin=dict(
                l=80,
                r=80,
                b=80,
                t=60
            ),
            scene=scene
        )



        #fig = tools.make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])
        trace = Surface(x=dfswp_pivot.index, y=zip(*dfswp_pivot.columns)[1],z=dfswp_pivot.values.T, colorscale='Viridis',showscale=True)
        #fig.append_trace(go.Scatter3d(x=dfswp_pivot.index, y=zip(*dfswp_pivot.columns)[1],z=dfswp_pivot.values.T, marker=go.Marker(color='rgb(31,119,180)'), name='Fit', scene='scene1'), 1, 1)

        data = [trace]

        fig = dict(data=data, layout=layout)

        #fig['layout']['scene1'].update(scene)
        #fig['layout'].update(layout)

        filename='e:/plotly3d-' + country_code + '.SwapCurve.html'

        plot(fig, filename=filename)
        fig

        mesh_view = MeshView(mesh, tenors)
        mesh_view.plot_tenors(tenors, directory, plot_vol=False, use_html=True)
