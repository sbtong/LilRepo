import os

#  numerical stack
import math
import dateutil.parser as parser
from datetime import date
from datetime import timedelta
import time
import numpy as np
import scipy.interpolate
import pymssql
import pandas as pd
import pandas.io.sql as sql

#  matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns
import colorlover as cl

#  plotly
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import Surface
from plotly.graph_objs import Scatter, Margin, Figure
import plotly.tools as tls
from plotly.offline import plot
from plotly.offline import download_plotlyjs, iplot



import cufflinks as cf

#  macpy package
import macpy
import macpy.utils.database as db


class Tenor:

    YearInYears = 1.0
    MonthInYears = 1.0/12.0
    WeekInYears = 1.0/52.0
    DayInYears = 1.0/252.0

    @classmethod
    def convert_tenor_to_years(cls, tenor_enum):
        val_str = tenor_enum[0:-1]
        term = tenor_enum[-1]
        val = float(val_str)
        if term == 'W':
            return val*cls.WeekInYears
        elif term == 'Y':
            return val*cls.YearInYears
        elif term == 'D':
            return val*cls.DayInYears
        elif term == 'M':
            return val*cls.MonthInYears
        return 0.0

    def __init__(self):
        pass


class Mesh3D:

    def __init__(self, dfts, curve_name, date_range=None, halflife=60):

        self.curve_name = curve_name
        self.dfts = dfts
        self.date_range = date_range
        self.df_prod = self.dfts['Quote'][curve_name]['Prod'].unstack().dropna()
        self.df_prod_vol = pd.ewmstd(self.df_prod.diff(), halflife=halflife, min_periods=40)*math.sqrt(250)
        self.x_prod = self.df_prod.columns
        self.y_prod = self.df_prod.index
        self.z_prod = self.df_prod.values

        self.df_staging = self.dfts['Quote'][curve_name]['Staging'].unstack().dropna()
        self.df_staging_vol = pd.ewmstd(self.df_staging.diff(), halflife=halflife, min_periods=40)*math.sqrt(250)
        self.x_staging = self.df_staging.columns
        self.y_staging = self.df_staging.index
        self.z_staging = self.df_staging.values

        self.df_diff = (self.df_prod - self.df_staging).dropna()
        self.df_diff_vol = pd.ewmstd(self.df_diff.diff(), halflife=halflife, min_periods=40)*math.sqrt(250)
        self.x_diff = self.df_diff.columns
        self.y_diff = self.df_diff.index
        self.z_diff = self.df_diff.values


class PlotType(object):

    def __init__(self, mesh, directory, tenor):
        self.name = None
        self.mesh = mesh
        self.tenor = tenor
        self.showlegend = True

        self.filename = None
        self.directory = directory
        self.trace_staging = None
        self.trace_prod = None
        self.x_prod = None
        self.y_prod = None
        self.x_staging = None
        self.y_staging = None

    def create_trace_production(self):
        color = cl.scales.values()[0]['qual']['Paired'][-2]
        return Scatter(x=self.x_prod,
                       y=self.y_prod,
                       name='Production Swap ' + self.name,
                       marker=dict(
                            size=1,
                            color=color,
                            line=dict(
                                width=1,
                                color=color
                                )
                            ),
                       showlegend=self.showlegend,
                       mode='lines')

    def create_trace_staging(self):
        color = cl.scales.values()[0]['qual']['Paired'][3]
        return Scatter(x=self.x_staging,
                       y=self.y_staging,
                       name='V2 (Cardano) Swap ' + self.name,
                       marker=dict(
                            size=1,
                            color=color,
                            line=dict(
                                width=1,
                                color=color
                                )
                            ),
                       showlegend=self.showlegend,
                       mode='lines')

    def create_plot_title(self):
        currency = str(self.mesh.curve_name.split('.')[1])
        title = currency + " {0:.0f}".format(self.tenor) + ' Year'
        if self.tenor < 1.0:
            title = currency + ' 6 Month'
        title += ' Swap Spread Comparison'
        return title

    def create_trace(self):
        trace_production = self.create_trace_production()
        trace_staging = self.create_trace_staging()
        data = [trace_production, trace_staging]
        return data


class SpreadPlotType(PlotType):

    def __init__(self, mesh, directory, tenor):
        super(SpreadPlotType, self).__init__(mesh, directory, tenor)
        self.name = 'Spread Level'
        self.x_prod = self.mesh.df_prod.index
        self.y_prod = self.mesh.df_prod[tenor].values*10000  # convert to basis points
        self.x_staging = self.mesh.df_staging.index
        self.y_staging = self.mesh.df_staging[tenor].values*10000  # convert to basis points


class VolPlotType(PlotType):

    def __init__(self, mesh, directory, tenor):
        super(VolPlotType, self).__init__(mesh, directory, tenor)
        self.showlengend = False
        self.name = 'Spread Volatility'
        self.x_prod = self.mesh.df_prod_vol.index
        self.y_prod = self.mesh.df_prod_vol[tenor].values*10000  # convert to basis points
        self.x_staging = self.mesh.df_staging_vol.index
        self.y_staging = self.mesh.df_staging_vol[tenor].values*10000  # convert to basis points


class PlotTypeCreator(object):

    @classmethod
    def create_plot_type(cls, mesh, directory, tenor, plot_vol=False):
        plot_type = VolPlotType(mesh, directory, tenor) if plot_vol else SpreadPlotType(mesh, directory, tenor)
        return plot_type


class MeshView:

    def __init__(self, mesh, tenors):
        self.mesh = mesh
        self.date_range = mesh.date_range
        self.tenors = tenors
        self.autosize = False
        self.width = 2000
        self.height = 2000
        self.fontname = 'Arial, sans-serif'
        self.titlefont = dict(family=self.fontname, size=22, color='black')
        self.xaxis = dict(
                        title='Year',
                        titlefont=dict(
                            family=self.fontname,
                            size=22,
                            color='black'
                        ),
                        tickfont=dict(
                            family=self.fontname,
                            size=20,
                            color='black'
                        ),
                        showgrid=True,
                        gridwidth=3
        )
        self.yaxis = dict(
                        title='Spread (basis points)',
                        titlefont=dict(
                            family=self.fontname,
                            size=22,
                            color='black'
                        ),
                        tickfont=dict(
                            family=self.fontname,
                            size=20,
                            color='black'
                        ),
                        showgrid=True

        )
        self.legend = dict(
                    x=0.5,
                    y=1,
                    traceorder='normal',
                    font=dict(
                        family=self.fontname,
                        size=18,
                        color='black'
                    ),
                    #  bgcolor='white',
                    bordercolor='#FFFFFF',
                    borderwidth=2,
                    orientation='h'
        )

        self.margin = Margin(l=100, r=50, b=100, t=100, pad=5)

    def _create_layout(self, title):
        return dict(
            title=title,
            titlefont=self.titlefont,
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            legend=self.legend,
            margin=self.margin,
            autosize=self.autosize,
            width=self.width,
            height=self.height
        )

    def plot_tenors(self, tenors, directory, use_html=False, plot_vol=False):
        subplot_titles = ['Spread Level, basis points', 'Annualized Spread Volatility, basis points']

        for tenor in tenors:
            time.sleep(2)
            subplot_titles_tenors = [x for x in subplot_titles]

            if tenor < 1.0:
                subplot_titles_tenors = [x for x in subplot_titles]

            shared_xaxes = True if use_html else False
            fig = tls.make_subplots(rows=2, cols=1, subplot_titles=subplot_titles_tenors, shared_xaxes=shared_xaxes)
            spread_plotter = PlotTypeCreator.create_plot_type(self.mesh, directory, tenor, plot_vol=False)
            spread_trace = spread_plotter.create_trace()
            for idx, trace in enumerate(spread_trace):
                fig.append_trace(trace, 1, 1)

            vol_plotter = PlotTypeCreator.create_plot_type(self.mesh, directory, tenor, plot_vol=True)
            vol_trace = vol_plotter.create_trace()
            for idx, trace in enumerate(vol_trace):
                fig.append_trace(trace, 2, 1)

            height, width = (600, 1000)
            if use_html:
                height, width = (height*1.3, width*1.3)
                fig['layout']['xaxis1'].update(title='Year',
                                               titlefont=dict(family=self.fontname, size=15, color='black'),
                                               tickfont=dict(family=self.fontname, size=15, color='black'))
                fig['layout']['yaxis1'].update(title=None,
                                               titlefont=dict(family=self.fontname, size=15, color='black'),
                                               tickfont=dict(family=self.fontname, size=15, color='black'))
                fig['layout']['yaxis2'].update(title=None,
                                               titlefont=dict(family=self.fontname, size=15, color='black'),
                                               tickfont=dict(family=self.fontname, size=15, color='black'))
            else:
                fig['layout']['xaxis1'].update(title='Year',
                                               titlefont=dict(family=self.fontname, size=15, color='black'),
                                               tickfont=dict(family=self.fontname, size=15, color='black'))
                fig['layout']['xaxis2'].update(title='Year',
                                               titlefont=dict(family=self.fontname, size=15, color='black'),
                                               tickfont=dict(family=self.fontname, size=15, color='black'))
                fig['layout']['yaxis1'].update(title=None,
                                               titlefont=dict(family=self.fontname, size=15, color='black'),
                                               tickfont=dict(family=self.fontname, size=15, color='black'))
                fig['layout']['yaxis2'].update(title=None,
                                               titlefont=dict(family=self.fontname, size=15, color='black'),
                                               tickfont=dict(family=self.fontname, size=15, color='black'))
            title = spread_plotter.create_plot_title()
            fig['layout'].update(height=height,
                                 width=width,
                                 title=title,
                                 titlefont=self.titlefont,
                                 margin=self.margin,
                                 orientation='h',
                                 autosize=False)

            fig['layout']['legend'].update(x=0, y=-200)

            filename = (self.mesh.curve_name+'_'+str(tenor)).replace('.', '_')
            if use_html is False:
                filename = os.path.join(directory, filename)
                py.image.save_as(fig, filename, format='png')
            else:
                filename = os.path.join(directory, filename + '.html')
                plot(fig, filename=filename, auto_open=False)


def main():
    pass

if __name__ == '__main__':
    main()


