# import relevant packages including macpy
import os

import unittest
import math
from StringIO import StringIO
import string
import time
import dateutil.parser as parser # helps parse datetime strings
import numpy as np
import pandas as pd
from string import Template
import base64
from itertools import cycle
from collections import namedtuple
import pprint

#### Plotting packages ######
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML # displays dataframes as html tables
from IPython.display import Image

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import Surface
import plotly.tools as tls
import plotly.tools as tools
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.graph_objs import Surface
import cufflinks as cf
import plotly.tools as tls
#tls.embed('https://plot.ly/~cufflinks/8')
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import Scatter
#import seaborn as sns # Changes default grapical settings

### macpy package - available at ->  http://git.axiomainc.com:8080/tfs/axiomadev/_git/ContentDev-MAC
import macpy
import macpy.utils.database as db
import macpy.visualization.plotly_helper as pl



ExtractConfig = namedtuple('ExtractConfig',
                           'curve_name start_date end_date country_name')
ExtractSwapConfig = namedtuple('ExtractConfigSwap',
                               'curve_name curve_name_gvt curve_name_swpspr start_date end_date country_name')


def extract_curve_from_db(extract_config):
    df = extract_curve_dataframe(extract_config.curve_name, extract_config.start_date, extract_config.end_date)
    return df


def create_stat_dataframe(extract_config_functor,
                          extract_config,
                          combine_nodes=True,
                          compute_vol=False,
                          no_decimals=False,
                          two_decimals=False,
                          scale_by=100):

    """Returns dataframe based statistic report grouped by tenor and year (rows:tenors, columns:year)
    """

    df = extract_config_functor(extract_config)

    #format_str = '{0:.1f}|{1:=.1f}|{2:=.1f}'
    format_str = '{0:.5f}'

    if compute_vol:
        df = compute_daily_vol(df)

    if no_decimals:
        #format_str = '{0:.0f}|{1:=.0f}|{2:=.0f}'
        format_str = '{0:.5f}'

    if two_decimals:
        #format_str = '{0:.2f}|{1:=.2f}|{2:=.2f}'
        format_str = '{0:=.5f}'


    def map_tenor(tenor):
        if tenor < 2.0:
            return '(a) Short end'
        if tenor >= 2.0 and tenor < 10.0:
            return '(b) Mid'
        if tenor >= 10.0 and tenor <= 40.0:
            return '(c) Long end'
        return None

    def map_tenor_node(tenor):
        if tenor == 0.002739726:
            return 0.002739726
        if tenor == 0.083333333333:
            return 0.083333333333
        if tenor == .25:
            return .25
        if tenor == .5:
            return .5
        if tenor == 1.0:
            return 1.0
        if tenor == 2.0:
            return 2.0
        if tenor == 5.0:
            return 5.0
        if tenor == 10.0:
            return 10.0
        if tenor == 30.0:
            return 30.0
        return None

    map_nodes = map_tenor if combine_nodes else map_tenor_node

    def compute_stats(column):
        #return pd.DataFrame({'min':column.min()*scale_by, 'max':column.max()*scale_by, 'mean':column.mean()*scale_by})
        return pd.DataFrame({'mean':column.mean()*scale_by})

    df['Year'] = df.index.map(lambda x: x.year).T
    df_rotate = df.groupby('Year').apply(compute_stats)
    df_rotate = df_rotate.reset_index()
    df_rotate['TenorGrp'] = df_rotate['TenorInYears'].map(map_nodes)
    #fields = ['Year','TenorGrp','min','mean','max']
    fields = ['Year','TenorGrp','mean']
    df_rotate = df_rotate[fields]
    #stats_func = lambda x: format_str.format(x['min'].min(), x['mean'].mean(), x['max'].max())
    stats_func = lambda x: format_str.format(x['mean'].mean())
    df_rotate = df_rotate.groupby(['Year','TenorGrp']).apply(stats_func).unstack()
    df_rotate = df_rotate.T

    curve_name_cycle = cycle([extract_config.curve_name])
    df_rotate.set_index(pd.MultiIndex.from_tuples(zip(curve_name_cycle, df_rotate.index)), inplace=True)
    return df_rotate


class Tenor:
    """
    """

    YearInYears = 1.0
    MonthInYears = 1.0/12.0
    WeekInYears = 1.0/52.0
    DayInYears = 1.0/252.0

    MonthsPerYear = 12
    MonthsPerMonth = 1
    MonthsPerDay = 1.0/30.0
    MonthPerWeek = 1.0/4.0

    @classmethod
    def _check_if_null(cls, tenor_enum):
        if tenor_enum is None:
            return (True, None)
        if pd.isnull(tenor_enum):
            return (True, 0.0)
        return (False, None)

    @classmethod
    def _parse_tenor_enum(cls, tenor_enum):
        val_str, term = tenor_enum[0:-1], tenor_enum[-1]
        is_zero, result = (True, 0.0) if val_str == '0' else (False, None)
        return (val_str, term, is_zero, result)

    @classmethod
    def convert_tenor_to_months(cls, tenor_enum):
        is_null, result = cls._check_if_null(tenor_enum)
        if is_null:
            return result

        val_str, term, is_zero, result = cls._parse_tenor_enum(tenor_enum)
        if is_zero:
            return result

        val = float(val_str)
        if term == 'W':
            return val*cls.MonthPerWeek
        elif term == 'Y':
            return int(val*cls.MonthsPerYear)
        elif term == 'D':
            return val*cls.MonthsPerDay
        elif term == 'M':
            return int(val*cls.MonthsPerMonth)
        return 0

    @classmethod
    def convert_tenor_to_years(cls, tenor_enum):
        is_null, result = cls._check_if_null(tenor_enum)
        if is_null:
            return result

        val_str, term, is_zero, result = cls._parse_tenor_enum(tenor_enum)
        if is_zero:
            return result

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

from datetime import datetime as dt
from datetime import date
#extract derived curves

def extract_curve_dataframe(curve_name, start_date, end_date, database_server=None, date_format='%Y-%m-%d',
                            use_final_table=False, use_preview=False, reindex=True):
    """Returns dataframe of yield curve for one country according
    args:
        curve_name:str  = 'US.USD.GVT.ZC'
        start_date:str  = '2016-01-27'
        end_date:str    = '2017-01-27'
        database_server:str = optional; 'Dev'|'Prod' specified in database.config (CurvesDB.DEV or CurvesDB.PROD)
        date_format:str  =  optional; passed into datetime.strptime
    returns:
        pandas dataframe of the form shown below
    TenorInYears	0.002739726	0.01369863	0.038461538	0.057692308	0.083333333333	0.166666667	0.25
    2000-01-04	0.05517	0.05517	0.05517	0.05517	0.05517	0.05517	0.05517	0.057427	0.062298	0.07
    2000-01-05	0.05462	0.05462	0.05462	0.05462	0.05462	0.05462	0.05462	0.057675	0.062818	0.07
    """

    sql_curve_template = Template("""
    select CurveShortName, TradeDate, Quote, te.TenorEnum, TenorInYears = te.InYears
    from [MarketData$use_preview].[dbo].[Curve] cv
    join [MarketData$use_preview].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
    join [MarketData$use_preview].[dbo].TenorEnum te on cn.TenorEnum = te.TenorEnum
    join [MarketData$use_preview].dbo.CurveNodeQuote$use_final_table cq on cq.CurveNodeId = cn.CurveNodeId
    where
    cq.TradeDate >= '$begin_date' and
    cq.TradeDate <= '$end_date' and
    cv.CurveShortName = '$curve_name'
    """)
    #[MarketData].[dbo].CurveNodeQuote
    #MarketDataPreview.dbo.CurveNodeQuoteFinal
    #print 'use_final_table', use_final_table

    def convert_date(date_arg):
        date_arg = dt.strptime(date_arg, date_format).date() if isinstance(date_arg, str) else date_arg
        date_arg = date_arg if isinstance(date_arg, date) else date_arg
        date_arg = date_arg.date() if isinstance(date_arg, dt) else date_arg
        return date_arg

    start_date_dt =  convert_date(start_date)
    end_date_dt =  convert_date(end_date)


    sql_curve = sql_curve_template.substitute(begin_date=start_date_dt,
                                              end_date=end_date_dt,
                                              curve_name=curve_name,
                                              use_final_table='Final' if use_final_table else '',
                                              use_preview='Preview' if use_preview else '')
    #print sql_curve
    df_curve_history = db.MSSQL.extract_dataframe(sql_curve, environment=database_server)
    if len(df_curve_history) == 0:
        raise Exception('No data found for sql:' + sql_curve)
    df_curve_history['TradeDate'] = pd.to_datetime(df_curve_history['TradeDate'])
    #print df_curve_history
    #df_curve_history["TenorInYears"] = df_curve_history["TenorInYears"].apply(lambda x:
    #                                                                          round(Tenor.convert_tenor_to_years(x), 3))
    fields = ['CurveShortName', 'TradeDate', 'TenorInYears', 'Quote']

    df_pivot = df_curve_history[fields].set_index(fields[:-1]).unstack()
    #print start_date_dt, end_date_dt

    if not reindex:
        return df_pivot['Quote'].loc[curve_name,:]
    date_range = pd.date_range(start_date_dt, end_date_dt, freq='B')
    df_pivot_reindexed = df_pivot['Quote'].loc[curve_name,:].reindex(date_range)
    return df_pivot_reindexed


def extract_combined_curve_dataframe(curve_name_1, curve_name_2, start_date, end_date, database_server=None,
                                     use_final_table=False):
    """Returns dataframe of yield curve for one country according
    args:
        curve_name_1 = 'US.USD.GVT.ZC'
        curve_name_2 = 'US.USD.SWP.ZC'
        start_date = Timestamp('2016-01-27')
        end_date = Timestamp('2017-01-27')
    returns:
        dataframe shown below
    TenorInYears	0.002739726	0.01369863	0.038461538	0.057692308	0.083333333333	0.166666667	0.25
    2000-01-04	0.05517	0.05517	0.05517	0.05517	0.05517	0.05517	0.05517	0.057427	0.062298	0.07
    2000-01-05	0.05462	0.05462	0.05462	0.05462	0.05462	0.05462	0.05462	0.057675	0.062818	0.07
    """
    #print use_final_table
    df_curve_1 = extract_curve_dataframe(curve_name_1, start_date, end_date, database_server,
                                         use_final_table=use_final_table)
    #print len(df_curve_1.dropna())
    df_curve_2 = extract_curve_dataframe(curve_name_2, start_date, end_date, database_server,
                                         use_final_table=use_final_table)
    #print len(df_curve_2.dropna())
    df_combined = df_curve_1 + df_curve_2
    #print len(df_combined.dropna())
    return df_combined


def extract_curve_history_from_dev(curve_name, start_date, end_date):
    sql_curve_str = """
    select CurveShortName, TradeDate, te.InYears as Tenor, Quote
    from [MarketData].[dbo].[Curve] cv
    join [MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
    join [MarketData].[dbo].[TenorEnum] te on cn.TenorEnum = te.TenorEnum
    join [MarketData].[dbo].[CurveNodeQuote] cq on cq.CurveNodeId = cn.CurveNodeId
    where
    cq.TradeDate >= '{start_date}' and
    cq.TradeDate <= '{end_date}' and
    cv.CurveShortName = '{curve_name}'
    """.format(curve_name=curve_name, start_date=start_date, end_date=end_date)

    df = db.MSSQL.extract_dataframe(sql_curve_str, environment='DEV')
    df.TradeDate = pd.to_datetime(df.TradeDate)
    return df


def convert_html_to_pdf(source_html, output_filename):
    """
    args:
        source_html
        output_filename

    returns pisa_status.error_code
    """
    from xhtml2pdf import pisa

    # open output file for writing (truncated binary)
    result_file = open(output_filename, "w+b")

    # convert HTML to PDF
    pisa_status = pisa.CreatePDF(
            source_html,                # the HTML to convert
            dest=result_file)           # file handle to recieve result

    # close output file
    result_file.close()                 # close output file

    # return True on success and False on errors
    return pisa_status.err


class PlotConfig:
    """
    """

    def __init__(self,
                 curve_name='None',
                 show_legend=False,
                 filename_prefix='Plot_Name',
                 plot_title='Plot_Title',
                 x_axis_title='',
                 y_axis_title='',
                 y_axis_min=None,
                 y_axis_max=None,
                 img_format='jpeg',
                 source_plot_file_path='.',
                 destination_directory='.',
                 layout_width=900,
                 layout_height=600,
                 font_size=15,
                 plot_sleep=1,
                 scatter_vectors=Scatter(x=[0,1],y=[0,1])):

        self.curve_name = curve_name
        self.show_legend = show_legend
        self.filename_prefix = filename_prefix
        self.plot_title = plot_title
        self.x_axis_title = x_axis_title
        self.y_axis_title = y_axis_title
        self.y_axis_min = y_axis_min
        self.y_axis_max = y_axis_max
        self.img_format = img_format
        self.layout_width = layout_width
        self.layout_height = layout_height,
        self.font_size = 15
        self.plot_sleep = plot_sleep # sleep is required because plotly's 'save' method returns before the image file is available on disk
        self.destination_directory=destination_directory
        self.scatter_vectors = scatter_vectors


def create_html_figures(figures, width=600, height=600, caption=''):

    template = (''
        '<hr>'
        '<img style="width: {width}; height: {height}" src="data:image/png;base64,{image}">'
    '')

    #Generate their images using `py.image.get`
    images = [base64.b64encode(py.image.get(figure, width=width, height=height, scale=3)).decode('utf-8')
              for figure in figures]

    report_html = '{caption}'.format(caption=caption)
    for index, image in enumerate(images):
        _ = template
        _ = _.format(image=image, caption=caption, width=width, height=height)
        report_html += _
    report_html += '<hr>'

    return report_html


def create_figure(plot_config):
    """
    function to plot scatter vectors
    creates figure

    """
    scatter_vectors = plot_config.scatter_vectors
    plot_title = plot_config.plot_title
    layout = pl.create_basic_layout(x_axis_title=plot_config.x_axis_title,
                                    y_axis_title=plot_config.y_axis_title,
                                    title=plot_title,
                                    font_size=plot_config.font_size,
                                    show_legend=plot_config.show_legend,
                                    y_axis_min = plot_config.y_axis_min if plot_config.y_axis_min is not None else None,
                                    y_axis_max = plot_config.y_axis_max if plot_config.y_axis_max is not None else None,
                                    x_axis_min = plot_config.scatter_vectors[0]['x'][0],
                                    x_axis_max = plot_config.scatter_vectors[0]['x'][-1])
    figure = pl.create_figure(scatter_vectors, layout)
    return figure


def create_zero_curve_scatter_vectors(df_zero_curves,
                                      transform = lambda x: x,
                                      label_transform=lambda y: r'$y(t,' + str(float(y)) + r')$'):
    """Returns list of plotly scatter vectors
    df_zero_curves
    transform
    label_transform
    """

    derived_curve_data = []
    for i, node_label in enumerate(df_zero_curves):  # iterate through curve nodes
        df_nodes = df_zero_curves[node_label]
        name = node_label
        nodes_subset = [0.002739726, 0.083333333333,.25, .5, 1.0, 2.0, 5.0, 10.0, 30.0]
        if (name not in nodes_subset):
            #print name
            continue
        name = df_nodes.name
        label = label_transform(name)
        curve_node_x = df_nodes.index
        curve_node_y = transform(df_nodes)
        scatter = Scatter(x=curve_node_x,
                          y=curve_node_y,
                          connectgaps=False,
                          name=label,
                          mode='lines',
                          line=dict(shape='linear',width=1))
        derived_curve_data.append(scatter)
    return derived_curve_data


def create_plot_config_curve_yield(curve_name, start_date, end_date):
    """
    generate and save derived curve yield plot
    """
    plot_config = PlotConfig()
    plot_config.plot_title= r'$y(t,T)$' + curve_name
    plot_config.show_legend=True
    plot_config.x_axis_title=''
    plot_config.y_axis_title='Percent'
    plot_config.font_size = 14
    transform_functor = lambda y: np.round(y*100,2)
    df_zero_curves=extract_curve_dataframe(curve_name, start_date, end_date)
#    plot_config.y_axis_max = df_source_bonds.describe().T[['max','min']].describe().loc['max','max']+1.0
#    plot_config.y_axis_min = df_source_bonds.describe().T[['max','min']].describe().loc['min','min']
    plot_config.scatter_vectors = create_zero_curve_scatter_vectors(df_zero_curves, transform=transform_functor)
    return plot_config


def create_plot_config_curve_vol(curve_name, start_date, end_date):
    """
    # generate and save derived curve volatility plot
    :param curve_name:
    :param start_date:
    :param end_date:
    :return:
    """
    plot_config = PlotConfig()
    plot_config.curve_name = curve_name
    plot_config.plot_title= r'EMWA Volatility (60 day half-life)): $\sigma(t,T)$'
    plot_config.show_legend=True
    plot_config.x_axis_title=''
    plot_config.y_axis_title='Basis Points'
    plot_config.font_size = 14
    transform_functor=lambda y: np.round(pd.ewmstd(y.diff()*10000*math.sqrt(250),
                                                   halflife=60,
                                                   ignore_na=True,
                                                   min_periods=60),0)
    label_transform=lambda y: r'$\sigma(t,' + str(float(y)) + r')$'
    df_zero_curves=extract_curve_dataframe(curve_name, start_date, end_date)
    plot_config.scatter_vectors=create_zero_curve_scatter_vectors(df_zero_curves,
                                                                  transform=transform_functor,
                                                                  label_transform=label_transform)
    return plot_config


def create_plot_config_curve_diff(curve_name, start_date, end_date):
    """
    # generate and save derived curve volatility plot
    :param curve_name:
    :param start_date:
    :param end_date:
    :return:
    """
    plot_config = PlotConfig()
    plot_config.curve_name = curve_name
    plot_config.plot_title= r'Daily Arithmetic Return: $\delta(t,T)$'
    plot_config.show_legend=True
    plot_config.x_axis_title=''
    plot_config.y_axis_title='Percent'
    plot_config.font_size = 14
    transform_functor=lambda y: y.diff()*100
    label_transform=lambda y: r'$\delta(t,' + str(float(y)) + r')$'
    df_zero_curves=extract_curve_dataframe(curve_name, start_date, end_date)
    plot_config.scatter_vectors=create_zero_curve_scatter_vectors(df_zero_curves,
                                                                  transform=transform_functor,
                                                                  label_transform=label_transform)
    return plot_config


def extract_curve_difference(extract_config, comb_database_server='PROD', comb_use_final_table=True):
    """ Returns dataframe of time-series of differences between old and new swap curves
    """

    df_curve = extract_curve_dataframe(extract_config.curve_name, extract_config.start_date, extract_config.end_date)

    df_curve_prod = extract_curve_dataframe(extract_config.curve_name, extract_config.start_date, extract_config.end_date,
                                       database_server=comb_database_server, use_final_table=comb_use_final_table)

    df_diff = df_curve - df_curve_prod
    return df_diff


def extract_swap_curve_difference(extract_swap_config, comb_database_server='PROD', comb_use_final_table=True):
    """ Returns dataframe of time-series of differences between old and new swap curves
    """

    df_combined = extract_combined_curve_dataframe(extract_swap_config.curve_name_gvt,
                                                      extract_swap_config.curve_name_swpspr,
                                                      extract_swap_config.start_date,
                                                      extract_swap_config.end_date,
                                                      database_server=comb_database_server,
                                                      use_final_table=comb_use_final_table)

    df_curve = extract_curve_dataframe(extract_swap_config.curve_name,
                                          extract_swap_config.start_date,
                                          extract_swap_config.end_date)

    df_diff = df_curve - df_combined
    return df_diff


def extract_curve_difference_prod_dev(extract_config):
    """ Returns dataframe of time-series of differences between old and new swap curves
    """

    df_curve_prod = extract_curve_dataframe(extract_config.curve_name,
                                       extract_config.start_date,
                                       extract_config.end_date,
                                       database_server='PROD',
                                       use_final_table=True)

    df_curve_dev = extract_curve_dataframe(extract_config.curve_name,
                                          extract_config.start_date,
                                          extract_config.end_date)

    df_diff = df_curve_dev - df_curve_prod
    return df_diff


def extract_curve_vol_of_difference(extract_config):
    """ Returns dataframe of time-series of differences between new and legacy swap curves
    """

    df_combined = extract_curve_difference_prod_dev(extract_config)

    df_combined_vol = compute_daily_vol(df_combined)
    #print display(df_combined_vol.describe())
    df_curve = extract_curve_dataframe(extract_config.curve_name,
                                          extract_config.start_date,
                                          extract_config.end_date)
    df_curve_vol = compute_daily_vol(df_curve)
    #print display(df_curve_vol.describe())
    df_diff = df_curve_vol - df_combined_vol
    return df_diff


def extract_swap_curve_combined(extract_swap_config, comb_database_server='PROD', use_final_table=False):
    """ Returns dataframe of time-series of differences between old and new swap curves
    """
    #print use_final_table
    df_combined = extract_combined_curve_dataframe(extract_swap_config.curve_name_gvt,
                                                      extract_swap_config.curve_name_swpspr,
                                                      extract_swap_config.start_date,
                                                      extract_swap_config.end_date,
                                                      database_server=comb_database_server,
                                                      use_final_table=use_final_table)
    return df_combined



def create_pdfs(extract_config, directory, filename=None, file_suffix='', width=800, height=325):

    extract_args = (extract_config.curve_name, extract_config.start_date, extract_config.end_date)
    try:
        figures = [create_figure(create_plot_config_curve_yield(*extract_args)),
                   create_figure(create_plot_config_curve_diff(*extract_args)),
                   create_figure(create_plot_config_curve_vol(*extract_args))]

        caption = '<center><font size="7">' + extract_config.country_name \
                  + ' (' + extract_config.curve_name + ')' + '</font></center>'
        report_html = create_html_figures(figures, caption=caption, width=width, height=height)
        filename = extract_config.curve_name.replace('.','_') + file_suffix + '.pdf' \
            if filename is None else filename + file_suffix + '.pdf'
        full_filename = directory + filename
        convert_html_to_pdf(report_html, full_filename)
        return filename
    except Exception as e:
        return repr(e)


def compute_daily_vol(df):
    """ Returns dataframe with time-series of vol estimates in basis points
    """
    df_vol = pd.ewmstd(df.diff(), halflife=60, ignore_na=True, min_periods=60)
    return df_vol


def extract_swap_curve_vol_diff(extract_swap_config):
    """ Returns dataframe of time-series of differences between new and legacy swap curves
    """

    df_combined = extract_combined_curve_dataframe(extract_swap_config.curve_name_gvt,
                                                      extract_swap_config.curve_name_swpspr,
                                                      extract_swap_config.start_date,
                                                      extract_swap_config.end_date,
                                                      database_server='PROD')
    df_combined_vol = compute_daily_vol(df_combined)
    #print display(df_combined_vol.describe())
    df_curve = extract_curve_dataframe(extract_swap_config.curve_name,
                                          extract_swap_config.start_date,
                                          extract_swap_config.end_date)
    df_curve_vol = compute_daily_vol(df_curve)
    #print display(df_curve_vol.describe())
    df_diff = df_curve_vol - df_combined_vol
    return df_diff


def create_extract_swap_config(df_config):

    def replace_eu_gvt_with_ep(curve_name):
        if 'EU.EUR.GVT.ZC' in curve_name:
            return 'EP.EUR.GVT.ZC'  #Handles inconsistency with country name for European union

        return curve_name

    extract_config = ExtractSwapConfig(curve_name=df_config.CurveShortName,
                                          curve_name_gvt=
                                          replace_eu_gvt_with_ep(df_config.CurveShortName.replace('IRSWP','GVT')),
                                          curve_name_swpspr=
                                          df_config.CurveShortName.replace('IRSWP.ZC','SWP.ZCS'),
                                          start_date=df_config.StartDate,
                                          end_date=df_config.EndDate,
                                          country_name=df_config.CountryName)
    return extract_config


def create_extract_swap_configs(extract_config, end_date='2017-01-27'):

    extract_swap_configs = [create_extract_swap_config(df_config)
                            for index, df_config in define_swap_attributes_sov(end_date=end_date).iterrows()
                            if 'IRSWP' in df_config.CurveShortName]

    return extract_swap_configs


def create_extract_swap_configs(end_date='2017-01-27'):

    extract_swap_configs = [create_extract_swap_config(df_config)
                            for index, df_config in define_swap_attributes_sov(end_date=end_date).iterrows()
                            if 'IRSWP' in df_config.CurveShortName]

    return extract_swap_configs

def create_extract_swap_configs_from_attributes(end_date):

    extract_swap_configs = [create_extract_swap_config(df_config)
                            for index, df_config in define_swap_attributes_sov(end_date=end_date).iterrows()]

    return extract_swap_configs


def define_swap_attributes_sov(end_date):

    daterangeccy = StringIO(\
"""CurveTypeEnum	CountryEnum	CurrencyEnum	CurveLongName	CurveShortName	ActiveToDate	ActiveFromDate	UnderlyingTypeEnum	CurveDataId	InterpolationMethod	RIC	CountryName
SwapZC	AU	AUD	AUD Interest Rate Swap Curve	AU.AUD.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|AU.AUD.IRSWP.ZC	PieceWiseLinear	0#AUDZ=R	Australia
SwapZC	CA	CAD	CAD Interest Rate Swap Curve	CA.CAD.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|CA.CAD.IRSWP.ZC	PieceWiseLinear	0#CADZ=R	Canada
SwapZC	CH	CHF	CHF Interest Rate Swap Curve	CH.CHF.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|CH.CHF.IRSWP.ZC	PieceWiseLinear	0#CHFZ=R	Switzerland
SwapZC	EU	EUR	EUR Interest Rate Swap Curve	EU.EUR.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|EU.EUR.IRSWP.ZC	PieceWiseLinear	0#EURZ=R	Europe
SwapZC	GB	GBP	GBP Interest Rate Swap Curve	GB.GBP.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|GB.GBP.IRSWP.ZC	PieceWiseLinear	0#GBPZ=R	Great Britain
SwapZC	JP	JPY	JPY Interest Rate Swap Curve	JP.JPY.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|JP.JPY.IRSWP.ZC	PieceWiseLinear	0#JPYZ=R	Japan
SwapZC	US	USD	USD Interest Rate Swap Curve	US.USD.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|US.USD.IRSWP.ZC	PieceWiseLinear	0#USDZ=R	United States
SwapZC	BG	BGN	BGN Interest Rate Swap Curve	BG.BGN.IRSWP.ZC	12/31/9999	12/15/2011	IRSWP	RefType|Name=SwapZeroCurve|BG.BGN.IRSWP.ZC	PieceWiseLinear	0#BGNZ=R	Bulgaria
SwapZC	BR	BRL	BRL Interest Rate Swap Curve	BR.BRL.IRSWP.ZC	12/31/9999	8/1/2002	IRSWP	RefType|Name=SwapZeroCurve|BR.BRL.IRSWP.ZC	PieceWiseLinear	0#BRLPREZ=R	Brazil
SwapZC	CN	CNY	CNY Interest Rate Swap Curve	CN.CNY.IRSWP.ZC	12/31/9999	1/4/2007	IRSWP	RefType|Name=SwapZeroCurve|CN.CNY.IRSWP.ZC	PieceWiseLinear	0#CNYZ=R	China
SwapZC	CN	CNH	CNH Interest Rate Swap Curve	CN.CNH.IRSWP.ZC	12/31/9999	1/4/2007	IRSWP	RefType|Name=SwapZeroCurve|CN.CNH.IRSWP.ZC	PieceWiseLinear	0#CNHQMQCHZ=R	China
SwapZC	CZ	CZK	CZK Interest Rate Swap Curve	CZ.CZK.IRSWP.ZC	12/31/9999	3/10/2000	IRSWP	RefType|Name=SwapZeroCurve|CZ.CZK.IRSWP.ZC	PieceWiseLinear	0#CZKZ=R	Czech Republic
SwapZC	DK	DKK	DKK Interest Rate Swap Curve	DK.DKK.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|DK.DKK.IRSWP.ZC	PieceWiseLinear	0#DKKZ=R	Denmark
SwapZC	HK	HKD	HKD Interest Rate Swap Curve	HK.HKD.IRSWP.ZC	12/31/9999	3/10/2000	IRSWP	RefType|Name=SwapZeroCurve|HK.HKD.IRSWP.ZC	PieceWiseLinear	0#HKDZ=R	Hong Kong
SwapZC	HU	HUF	HUF Interest Rate Swap Curve	HU.HUF.IRSWP.ZC	12/31/9999	3/7/2002	IRSWP	RefType|Name=SwapZeroCurve|HU.HUF.IRSWP.ZC	PieceWiseLinear	0#HUFZ=R	Hungary
SwapZC	ID	IDR	IDR Interest Rate Swap Curve	ID.IDR.IRSWP.ZC	12/31/9999	12/18/2003	IRSWP	RefType|Name=SwapZeroCurve|ID.IDR.IRSWP.ZC	PieceWiseLinear	0#IDRZ=R	Indonesia
SwapZC	IL	ILS	ILS Interest Rate Swap Curve	IL.ILS.IRSWP.ZC	12/31/9999	7/2/2008	IRSWP	RefType|Name=SwapZeroCurve|IL.ILS.IRSWP.ZC	PieceWiseLinear	0#ILSZ=R	Israel
SwapZC	IN	INR	INR Interest Rate Swap Curve	IN.INR.IRSWP.ZC	12/31/9999	3/8/2002	IRSWP	RefType|Name=SwapZeroCurve|IN.INR.IRSWP.ZC	PieceWiseLinear	0#INRZ=R	India
SwapZC	IS	ISK	ISK Interest Rate Swap Curve	IS.ISK.IRSWP.ZC	12/31/9999	12/3/2008	IRSWP	RefType|Name=SwapZeroCurve|IS.ISK.IRSWP.ZC	PieceWiseLinear	0#ISKZ=R	Iceland
SwapZC	KR	KRW	KRW Interest Rate Swap Curve	KR.KRW.IRSWP.ZC	12/31/9999	7/12/2002	IRSWP	RefType|Name=SwapZeroCurve|KR.KRW.IRSWP.ZC	PieceWiseLinear	0#KRWZ=R	South Korea
SwapZC	MX	MXN	MXN Interest Rate Swap Curve	MX.MXN.IRSWP.ZC	12/31/9999	7/9/2002	IRSWP	RefType|Name=SwapZeroCurve|MX.MXN.IRSWP.ZC	PieceWiseLinear	0#MXNZ=R	Mexico
SwapZC	MY	MYR	MYR Interest Rate Swap Curve	MY.MYR.IRSWP.ZC	12/31/9999	4/30/2003	IRSWP	RefType|Name=SwapZeroCurve|MY.MYR.IRSWP.ZC	PieceWiseLinear	0#MYRZ=R	Malaysia
SwapZC	NG	NGN	NGN Interest Rate Swap Curve	NG.NGN.IRSWP.ZC	12/31/9999	3/28/2016	IRSWP	RefType|Name=SwapZeroCurve|NG.NGN.IRSWP.ZC	PieceWiseLinear	0#NGNZ=R	Nigeria
SwapZC	NO	NOK	NOK Interest Rate Swap Curve	NO.NOK.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|NO.NOK.IRSWP.ZC	PieceWiseLinear	0#NOKZ=R	Norway
SwapZC	NZ	NZD	NZD Interest Rate Swap Curve	NZ.NZD.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|NZ.NZD.IRSWP.ZC	PieceWiseLinear	0#NZDZ=R	New Zealand
SwapZC	PH	PHP	PHP Interest Rate Swap Curve	PH.PHP.IRSWP.ZC	12/31/9999	5/28/2004	IRSWP	RefType|Name=SwapZeroCurve|PH.PHP.IRSWP.ZC	PieceWiseLinear	0#PHPZ=R	Philippines
SwapZC	PL	PLN	PLN Interest Rate Swap Curve	PL.PLN.IRSWP.ZC	12/31/9999	10/12/2000	IRSWP	RefType|Name=SwapZeroCurve|PL.PLN.IRSWP.ZC	PieceWiseLinear	0#PLNZ=R	Poland
SwapZC	RO	RON	RON Interest Rate Swap Curve	RO.RON.IRSWP.ZC	12/31/9999	10/20/2010	IRSWP	RefType|Name=SwapZeroCurve|RO.RON.IRSWP.ZC	PieceWiseLinear	0#RONCCSZ=R	Romania
SwapZC	RU	RUB	RUB Interest Rate Swap Curve	RU.RUB.IRSWP.ZC	12/31/9999	12/12/2007	IRSWP	RefType|Name=SwapZeroCurve|RU.RUB.IRSWP.ZC	PieceWiseLinear	0#RUBZ=R	Russian Federation
SwapZC	SE	SEK	SEK Interest Rate Swap Curve	SE.SEK.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|SE.SEK.IRSWP.ZC	PieceWiseLinear	0#SEKZ=R	Sweden
SwapZC	SG	SGD	SGD Interest Rate Swap Curve	SG.SGD.IRSWP.ZC	12/31/9999	3/13/2000	IRSWP	RefType|Name=SwapZeroCurve|SG.SGD.IRSWP.ZC	PieceWiseLinear	0#SGDZ=R	Singapore
SwapZC	TH	THB	THB Interest Rate Swap Curve	TH.THB.IRSWP.ZC	12/31/9999	1/1/2000	IRSWP	RefType|Name=SwapZeroCurve|TH.THB.IRSWP.ZC	PieceWiseLinear	0#THBZ=R	Thailand
SwapZC	TR	TRY	TRY Interest Rate Swap Curve	TR.TRY.IRSWP.ZC	12/31/9999	7/3/2006	IRSWP	RefType|Name=SwapZeroCurve|TR.TRY.IRSWP.ZC	PieceWiseLinear	0#TRYZ=R	Turkey
SwapZC	TW	TWD	TWD Interest Rate Swap Curve	TW.TWD.IRSWP.ZC	12/31/9999	7/12/2002	IRSWP	RefType|Name=SwapZeroCurve|TW.TWD.IRSWP.ZC	PieceWiseLinear	0#TWDZ=R	Taiwan
SwapZC	ZA	ZAR	ZAR Interest Rate Swap Curve	ZA.ZAR.IRSWP.ZC	12/31/9999	10/24/2000	IRSWP	RefType|Name=SwapZeroCurve|ZA.ZAR.IRSWP.ZC	PieceWiseLinear	0#ZARZ=R	South Africa
SwapZrSpr	AU	AUD	Australia Swap Zero Spread Curve (AUD)	AU.AUD.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|AU.AUD.SWP.ZCS			Australia
SwapZrSpr	CA	CAD	Canada Swap Zero Spread Curve (CAD)	CA.CAD.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|CA.CAD.SWP.ZCS			Canada
SwapZrSpr	CH	CHF	Switzerland Swap Zero Spread Curve (CHF)	CH.CHF.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|CH.CHF.SWP.ZCS			Switzerland
SwapZrSpr	EP	EUR	Euro Swap Zero Spread Curve (EUR)	EU.EUR.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|EU.EUR.SWP.ZCS			European Union
SwapZrSpr	GB	GBP	United Kingdom Swap Zero Spread Curve (GBP)	GB.GBP.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|GB.GBP.SWP.ZCS			United Kingdom
SwapZrSpr	JP	JPY	Japan Swap Zero Spread Curve (JPY)	JP.JPY.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|JP.JPY.SWP.ZCS			Japan
SwapZrSpr	US	USD	United States Swap Zero Spread Curve (USD)	US.USD.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|US.USD.SWP.ZCS			United States
SwapZrSpr	BG	BGN	Bulgaria Swap Zero Spread Curve (BGN)	BG.BGN.SWP.ZCS	12/31/9999	12/15/2011		RefType|Name=BondSpreadCurve|BG.BGN.SWP.ZCS			Bulgaria
SwapZrSpr	BR	BRL	Brazil Swap Zero Spread Curve (BRL)	BR.BRL.SWP.ZCS	12/31/9999	8/1/2002		RefType|Name=BondSpreadCurve|BR.BRL.SWP.ZCS			Brazil
SwapZrSpr	BW	BWP	Botswana Swap Zero Spread Curve (BWP)	BW.BWP.SWP.ZCS	12/31/9999	3/28/2016		RefType|Name=BondSpreadCurve|BW.BWP.SWP.ZCS			Botswana
SwapZrSpr	CN	CNY	China Swap Zero Spread Curve (CNY)	CN.CNY.SWP.ZCS	12/31/9999	1/4/2007		RefType|Name=BondSpreadCurve|CN.CNY.SWP.ZCS			China
SwapZrSpr	CZ	CZK	Czech Republic Swap Zero Spread Curve (CZK)	CZ.CZK.SWP.ZCS	12/31/9999	3/10/2000		RefType|Name=BondSpreadCurve|CZ.CZK.SWP.ZCS			Czech Republic
SwapZrSpr	DK	DKK	Denmark Swap Zero Spread Curve (DKK)	DK.DKK.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|DK.DKK.SWP.ZCS			Denmark
SwapZrSpr	HK	HKD	Hong Kong Swap Zero Spread Curve (HKD)	HK.HKD.SWP.ZCS	12/31/9999	3/10/2000		RefType|Name=BondSpreadCurve|HK.HKD.SWP.ZCS			Hong Kong
SwapZrSpr	HR	HRK	Croatia Swap Zero Spread Curve (HRK)	HR.HRK.SWP.ZCS	12/31/9999	12/15/2011		RefType|Name=BondSpreadCurve|HR.HRK.SWP.ZCS			Croatia
SwapZrSpr	HU	HUF	Hungary Swap Zero Spread Curve (HUF)	HU.HUF.SWP.ZCS	12/31/9999	3/7/2002		RefType|Name=BondSpreadCurve|HU.HUF.SWP.ZCS			Hungary
SwapZrSpr	ID	IDR	Indonesia Swap Zero Spread Curve (IDR)	ID.IDR.SWP.ZCS	12/31/9999	12/18/2003		RefType|Name=BondSpreadCurve|ID.IDR.SWP.ZCS			Indonesia
SwapZrSpr	IL	ILS	Israel Swap Zero Spread Curve (ILS)	IL.ILS.SWP.ZCS	12/31/9999	7/2/2008		RefType|Name=BondSpreadCurve|IL.ILS.SWP.ZCS			Israel
SwapZrSpr	IN	INR	India Swap Zero Spread Curve (INR)	IN.INR.SWP.ZCS	12/31/9999	3/8/2002		RefType|Name=BondSpreadCurve|IN.INR.SWP.ZCS			India
SwapZrSpr	IS	ISK	Iceland Swap Zero Spread Curve (ISK)	IS.ISK.SWP.ZCS	12/31/9999	12/3/2008		RefType|Name=BondSpreadCurve|IS.ISK.SWP.ZCS			Iceland
SwapZrSpr	KE	KES	Kenya Swap Zero Spread Curve (KES)	KE.KES.SWP.ZCS	12/31/9999	10/21/2010		RefType|Name=BondSpreadCurve|KE.KES.SWP.ZCS			Kenya
SwapZrSpr	KR	KRW	Korea Swap Zero Spread Curve (KRW)	KR.KRW.SWP.ZCS	12/31/9999	7/12/2002		RefType|Name=BondSpreadCurve|KR.KRW.SWP.ZCS			Korea, Republic of
SwapZrSpr	MX	MXN	Mexico Swap Zero Spread Curve (MXN)	MX.MXN.SWP.ZCS	12/31/9999	7/9/2002		RefType|Name=BondSpreadCurve|MX.MXN.SWP.ZCS			Mexico
SwapZrSpr	MY	MYR	Malaysia Swap Zero Spread Curve (MYR)	MY.MYR.SWP.ZCS	12/31/9999	4/30/2003		RefType|Name=BondSpreadCurve|MY.MYR.SWP.ZCS			Malaysia
SwapZrSpr	NO	NOK	Norway Swap Zero Spread Curve (NOK)	NO.NOK.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|NO.NOK.SWP.ZCS			Norway
SwapZrSpr	NZ	NZD	New Zealand Swap Zero Spread Curve (NZD)	NZ.NZD.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|NZ.NZD.SWP.ZCS			New Zealand
SwapZrSpr	PE	PEN	Peru Swap Zero Spread Curve (PEN)	PE.PEN.SWP.ZCS	12/31/9999	8/1/2006		RefType|Name=BondSpreadCurve|PE.PEN.SWP.ZCS			Peru
SwapZrSpr	PH	PHP	Philippines Swap Zero Spread Curve (PHP)	PH.PHP.SWP.ZCS	12/31/9999	5/28/2004		RefType|Name=BondSpreadCurve|PH.PHP.SWP.ZCS			Philippines
SwapZrSpr	PK	PKR	Pakistan Swap Zero Spread Curve (PKR)	PK.PKR.SWP.ZCS	12/31/9999	4/18/2006		RefType|Name=BondSpreadCurve|PK.PKR.SWP.ZCS			Pakistan
SwapZrSpr	PL	PLN	Poland Swap Zero Spread Curve (PLN)	PL.PLN.SWP.ZCS	12/31/9999	10/12/2000		RefType|Name=BondSpreadCurve|PL.PLN.SWP.ZCS			Poland
SwapZrSpr	RO	RON	Romania Swap Zero Spread Curve (RON)	RO.RON.SWP.ZCS	12/31/9999	10/20/2010		RefType|Name=BondSpreadCurve|RO.RON.SWP.ZCS			Romania
SwapZrSpr	RU	RUB	Russian Federation Swap Zero Spread Curve (RUB)	RU.RUB.SWP.ZCS	12/31/9999	12/12/2007		RefType|Name=BondSpreadCurve|RU.RUB.SWP.ZCS			Russian Federation
SwapZrSpr	SE	SEK	Sweden Swap Zero Spread Curve (SEK)	SE.SEK.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|SE.SEK.SWP.ZCS			Sweden
SwapZrSpr	SG	SGD	Singapore Swap Zero Spread Curve (SGD)	SG.SGD.SWP.ZCS	12/31/9999	3/13/2000		RefType|Name=BondSpreadCurve|SG.SGD.SWP.ZCS			Singapore
SwapZrSpr	TH	THB	Thailand Swap Zero Spread Curve (THB)	TH.THB.SWP.ZCS	12/31/9999	1/1/2000		RefType|Name=BondSpreadCurve|TH.THB.SWP.ZCS			Thailand
SwapZrSpr	TR	TRY	Turkey Swap Zero Spread Curve (TRY)	TR.TRY.SWP.ZCS	12/31/9999	7/3/2006		RefType|Name=BondSpreadCurve|TR.TRY.SWP.ZCS			Turkey
SwapZrSpr	TW	TWD	Taiwan Swap Zero Spread Curve (TWD)	TW.TWD.SWP.ZCS	12/31/9999	7/12/2002		RefType|Name=BondSpreadCurve|TW.TWD.SWP.ZCS			Taiwan, Province of China
SwapZrSpr	VN	VND	Viet Nam Swap Zero Spread Curve (VND)	VN.VND.SWP.ZCS	12/31/9999	6/8/2006		RefType|Name=BondSpreadCurve|VN.VND.SWP.ZCS			Viet Nam
SwapZrSpr	ZA	ZAR	South Africa Swap Zero Spread Curve (ZAR)	ZA.ZAR.SWP.ZCS	12/31/9999	10/24/2000		RefType|Name=BondSpreadCurve|ZA.ZAR.SWP.ZCS			South Africa
CvRatSpr	DE	EUR	AAA German Covered Bond spread over EUR sovereign	DE.EUR.CVBND.(AAA).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.(AAA).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	AA German Covered Bond spread over EUR sovereign	DE.EUR.CVBND.(AA).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.(AA).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	A German Covered Bond spread over EUR sovereign	DE.EUR.CVBND.(A).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.(A).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	BBB German Covered Bond spread over EUR sovereign	DE.EUR.CVBND.(BBB).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.(BBB).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	IG German Covered Bond spread over EUR sovereign	DE.EUR.CVBND.(IG).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.(IG).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	AAA German Jumbo Covered Bond spread over EUR sovereign	DE.EUR.CVBND.JUMBO.(AAA).RTGSPR	9999-12-31	2004-05-24	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.JUMBO.(AAA).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	AA German Jumbo Covered Bond spread over EUR sovereign	DE.EUR.CVBND.JUMBO.(AA).RTGSPR	9999-12-31	2004-05-24	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.JUMBO.(AA).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	A German Jumbo Covered Bond spread over EUR sovereign	DE.EUR.CVBND.JUMBO.(A).RTGSPR	9999-12-31	2004-05-24	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.JUMBO.(A).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	BBB German Jumbo Covered Bond spread over EUR sovereign	DE.EUR.CVBND.JUMBO.(BBB).RTGSPR	9999-12-31	2004-05-24	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.JUMBO.(BBB).RTGSPR	NULL	NULL	Germany
CvRatSpr	DE	EUR	IG German Jumbo Covered Bond spread over EUR sovereign	DE.EUR.CVBND.JUMBO.(IG).RTGSPR	9999-12-31	2004-05-24	NULL	RefType|Name=BondSpreadCurve|DE.EUR.CVBND.JUMBO.(IG).RTGSPR	NULL	NULL	Germany
CvRatSpr	DK	EUR	AAA Danish Covered Bond spread over EUR sovereign	DK.EUR.CVBND.(AAA).RTGSPR	9999-12-31	2004-12-28	NULL	RefType|Name=BondSpreadCurve|DK.EUR.CVBND.(AAA).RTGSPR	NULL	NULL	Denmark
CvRatSpr	DK	EUR	AA Danish Covered Bond spread over EUR sovereign	DK.EUR.CVBND.(AA).RTGSPR	9999-12-31	2005-04-11	NULL	RefType|Name=BondSpreadCurve|DK.EUR.CVBND.(AA).RTGSPR	NULL	NULL	Denmark
CvRatSpr	DK	EUR	A Danish Covered Bond spread over EUR sovereign	DK.EUR.CVBND.(A).RTGSPR	9999-12-31	2006-10-03	NULL	RefType|Name=BondSpreadCurve|DK.EUR.CVBND.(A).RTGSPR	NULL	NULL	Denmark
CvRatSpr	DK	EUR	IG Danish Covered Bond spread over EUR sovereign	DK.EUR.CVBND.(IG).RTGSPR	9999-12-31	2004-12-28	NULL	RefType|Name=BondSpreadCurve|DK.EUR.CVBND.(IG).RTGSPR	NULL	NULL	Denmark
CvRatSpr	ES	EUR	AAA Spanish Covered Bond spread over EUR sovereign	ES.EUR.CVBND.(AAA).RTGSPR	9999-12-31	2005-03-08	NULL	RefType|Name=BondSpreadCurve|ES.EUR.CVBND.(AAA).RTGSPR	NULL	NULL	Spain
CvRatSpr	ES	EUR	AA Spanish Covered Bond spread over EUR sovereign	ES.EUR.CVBND.(AA).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|ES.EUR.CVBND.(AA).RTGSPR	NULL	NULL	Spain
CvRatSpr	ES	EUR	A Spanish Covered Bond spread over EUR sovereign	ES.EUR.CVBND.(A).RTGSPR	9999-12-31	2004-09-16	NULL	RefType|Name=BondSpreadCurve|ES.EUR.CVBND.(A).RTGSPR	NULL	NULL	Spain
CvRatSpr	ES	EUR	BBB Spanish Covered Bond spread over EUR sovereign	ES.EUR.CVBND.(BBB).RTGSPR	9999-12-31	2011-04-18	NULL	RefType|Name=BondSpreadCurve|ES.EUR.CVBND.(BBB).RTGSPR	NULL	NULL	Spain
CvRatSpr	ES	EUR	IG Spanish Covered Bond spread over EUR sovereign	ES.EUR.CVBND.(IG).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|ES.EUR.CVBND.(IG).RTGSPR	NULL	NULL	Spain
CvRatSpr	FR	EUR	AAA French Covered Bond spread over EUR sovereign	FR.EUR.CVBND.(AAA).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|FR.EUR.CVBND.(AAA).RTGSPR	NULL	NULL	France
CvRatSpr	FR	EUR	AA French Covered Bond spread over EUR sovereign	FR.EUR.CVBND.(AA).RTGSPR	9999-12-31	2005-05-05	NULL	RefType|Name=BondSpreadCurve|FR.EUR.CVBND.(AA).RTGSPR	NULL	NULL	France
CvRatSpr	FR	EUR	A French Covered Bond spread over EUR sovereign	FR.EUR.CVBND.(A).RTGSPR	9999-12-31	2010-04-05	NULL	RefType|Name=BondSpreadCurve|FR.EUR.CVBND.(A).RTGSPR	NULL	NULL	France
CvRatSpr	FR	EUR	IG French Covered Bond spread over EUR sovereign	FR.EUR.CVBND.(IG).RTGSPR	9999-12-31	2004-06-21	NULL	RefType|Name=BondSpreadCurve|FR.EUR.CVBND.(IG).RTGSPR	NULL	NULL	France
CvRatSpr	SE	EUR	AAA Swedish Covered Bond spread over EUR sovereign	SE.EUR.CVBND.(AAA).RTGSPR	9999-12-31	2006-11-23	NULL	RefType|Name=BondSpreadCurve|SE.EUR.CVBND.(AAA).RTGSPR	NULL	NULL	Sweden
CvRatSpr	SE	EUR	AA Swedish Covered Bond spread over EUR sovereign	SE.EUR.CVBND.(AA).RTGSPR	9999-12-31	2006-10-25	NULL	RefType|Name=BondSpreadCurve|SE.EUR.CVBND.(AA).RTGSPR	NULL	NULL	Sweden
CvRatSpr	SE	EUR	IG Swedish Covered Bond spread over EUR sovereign	SE.EUR.CVBND.(IG).RTGSPR	9999-12-31	2006-10-25	NULL	RefType|Name=BondSpreadCurve|SE.EUR.CVBND.(IG).RTGSPR	NULL	NULL	Sweden
CvRatSpr	SE	SEK	AAA Swedish Covered Bond spread over sovereign	SE.SEK.CVBND.(AAA).RTGSPR	9999-12-31	2006-07-20	NULL	RefType|Name=BondSpreadCurve|SE.SEK.CVBND.(AAA).RTGSPR	NULL	NULL	Sweden
CvRatSpr	SE	SEK	AA Swedish Covered Bond spread over sovereign	SE.SEK.CVBND.(AA).RTGSPR	9999-12-31	2006-07-13	NULL	RefType|Name=BondSpreadCurve|SE.SEK.CVBND.(AA).RTGSPR	NULL	NULL	Sweden
CvRatSpr	SE	SEK	BBB Swedish Covered Bond spread over sovereign	SE.SEK.CVBND.(BBB).RTGSPR	9999-12-31	2007-09-20	NULL	RefType|Name=BondSpreadCurve|SE.SEK.CVBND.(BBB).RTGSPR	NULL	NULL	Sweden
CvRatSpr	SE	SEK	IG Swedish Covered Bond spread over sovereign	SE.SEK.CVBND.(IG).RTGSPR	9999-12-31	2006-07-13	NULL	RefType|Name=BondSpreadCurve|SE.SEK.CVBND.(IG).RTGSPR	NULL	NULL	Sweden
""")

    df = pd.read_csv(daterangeccy, sep='\t')
    df['EndDate'] = end_date
    df['EndDate'] = pd.to_datetime(df['EndDate']).dt.date
    df['StartDate'] = pd.to_datetime(df['ActiveFromDate']).dt.date
    return df


def identify_outliers(extract_config, threshold, database_server='Dev', use_final_table=False,use_preview=False):
    """ Returns dataframe of dates that exceed threshold for arithmetic return (e.g. y(t) - y(t-1))
    args: instance of ExtractConfig  - namedtuple
    """
    df = extract_curve_dataframe(extract_config.curve_name, extract_config.start_date, extract_config.end_date,
                                 database_server=database_server,
                                 use_preview=use_preview,
                                 use_final_table=use_final_table).diff()
    outlier_days = df[(np.abs(df) > threshold).max(axis=1)].index.map(lambda x: x.strftime("%Y-%m-%d"))
    return outlier_days


def sql_delete_outliers(curve_name, days):
    """ Returns sql to eliminate trade dates in days argument
    """

    if len(days) == 0:
        return '-- length of days is zero. No sql generated'

    sql =  r"""DELETE CurveNodeQuote
    WHERE CurveNodeId in
        (SELECT CurveNodeId
         FROM CurveNodes cn
         JOIN Curve cv on cv.CurveId = cn.CurveId
         AND cv.CurveId=(SELECT CurveId FROM Curve WHERE CurveShortName='{curve_name}'))
    AND TradeDate IN
    """.format(curve_name=curve_name)
    sql += '(' + ',\n'.join(["'" + day + "'" for day in days]) + ')'
    return sql


def extract_curve_history_from_prod(curve_name, start_date, end_date):
    sql_curve_str = """
    select CurveShortName, TradeDate, te.InYears as Tenor, Quote
    from [MarketData].[dbo].[Curve] cv
    join [MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
    join [MarketData].[dbo].[TenorEnum] te on cn.TenorEnum = te.TenorEnum
    join [MarketData].[dbo].[CurveNodeQuoteFinal] cq on cq.CurveNodeId = cn.CurveNodeId
    where
    cq.TradeDate >= '{start_date}' and
    cq.TradeDate <= '{end_date}' and
    cv.CurveShortName = '{curve_name}'
    """.format(curve_name=curve_name, start_date=start_date, end_date=end_date)

    df = db.MSSQL.extract_dataframe(sql_curve_str, environment='PROD')
    df.TradeDate = pd.to_datetime(df.TradeDate)
    df_pivot = df #df.set_index(columns[0:-1]).unstack()['Quote'].loc[curve_name,:]
    return df_pivot


def extract_curve_history_from_dev_basic(curve_name, start_date, end_date):
    sql_curve_str = """
    select CurveShortName, TradeDate, cn.CurveNodeId as Tenor, Quote
    from [MarketData].[dbo].[Curve] cv
    join [MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
    join [MarketData].[dbo].[TenorEnum] te on cn.TenorEnum = te.TenorEnum
    join [MarketData].[dbo].[CurveNodeQuote] cq on cq.CurveNodeId = cn.CurveNodeId
    where
    cq.TradeDate >= '{start_date}' and
    cq.TradeDate <= '{end_date}' and
    cv.CurveShortName = '{curve_name}'
    """.format(curve_name=curve_name, start_date=start_date, end_date=end_date)

    df = db.MSSQL.extract_dataframe(sql_curve_str, environment='DEV')
    df.TradeDate = pd.to_datetime(df.TradeDate)
    columns = ['CurveShortName','TradeDate','Tenor','Quote']
    df_pivot = df.set_index(columns[0:-1]).unstack()['Quote'].loc[curve_name,:]

    return df_pivot

def infer_new_covered_bond(cvdbond_curve_name, start_date, end_date):
    swpspr_curve_name = '{ctry}.{ccy}.SWP.ZCS'.format(ctry=cvdbond_curve_name[3:5],ccy=cvdbond_curve_name[3:6])

    df_curve_history_cvdbnd = extract_curve_history_from_prod(cvdbond_curve_name, start_date, end_date)
    df_curve_history_cvdbnd['CurveShortName'] = cvdbond_curve_name

    df_curve_history_swpspr_prod = extract_curve_history_from_prod(swpspr_curve_name, start_date, end_date)
    swpspr_curve_name_prod = swpspr_curve_name + '.Prod'
    df_curve_history_swpspr_prod['CurveShortName'] = swpspr_curve_name_prod

    df_curve_history_swpspr_dev = extract_curve_history_from_dev(swpspr_curve_name, start_date, end_date)
    swpspr_curve_name_dev = swpspr_curve_name + '.Dev'
    df_curve_history_swpspr_dev['CurveShortName'] = swpspr_curve_name_dev

    df_combined = df_curve_history_cvdbnd.append(df_curve_history_swpspr_prod).append(df_curve_history_swpspr_dev)
   # return df_curve_history_cvdbnd, df_curve_history_swpspr_prod, df_curve_history_swpspr_dev
    df_pivot = df_combined.set_index(['TradeDate','Tenor','CurveShortName']).unstack()['Quote']
    df_pivot['TotalSpread'] = df_pivot[cvdbond_curve_name] + df_pivot[swpspr_curve_name_prod]
    df_pivot['CvdBondNew'] = df_pivot['TotalSpread'] - df_pivot[swpspr_curve_name_dev]
    df_pivot.dropna(inplace=True)
    return df_pivot


def extract_tenor_nodeids(curve_name):
    sql_nodeids_str = """
SELECT cv.CurveShortName, cn.TenorEnum, te.InYears, cn.CurveNodeId
FROM MarketData.dbo.Curve cv
JOIN MarketData.dbo.CurveNodes cn on cn.CurveId = cv.CurveId
JOIN MarketData.dbo.TenorEnum te on te.TenorEnum = cn.TenorEnum
WHERE cv.CurveShortName = '{curve_name}'
""".format(curve_name=curve_name)

    df_nodeids = db.MSSQL.extract_dataframe(sql_nodeids_str)
    return df_nodeids


def create_insert_sql_infer_cvd_bond(cvdbond_curve_name, start_date, end_date):
    sql_template = """INSERT INTO CurveNodeQuote (CurveNodeId, TradeDate, Quote, Lud, Lub) VALUES ({node_id}, '{trade_date}', {quote}, GETDATE(), 'wgajate');"""
    df_nodeids = extract_tenor_nodeids(cvdbond_curve_name)
    df_cvdbnd = infer_new_covered_bond(cvdbond_curve_name, start_date, end_date)
    sql_insert_list = []
    for (trade_date, tenor), series in df_cvdbnd.iterrows():
        cvdbond_spread_new = series['CvdBondNew']
        node_id = df_nodeids[df_nodeids.InYears==tenor].CurveNodeId.values[0]
        sql_insert = sql_template.format(node_id=node_id, trade_date=trade_date, quote=cvdbond_spread_new)
        sql_insert_list.append(sql_insert)
    return sql_insert_list


def forward_fill_business_days(curve_name, start_date, end_date):
    df_curve_history = extract_curve_history_from_dev_basic(curve_name, start_date, end_date)
    date_range = pd.date_range(start_date, end_date, freq='B')
    df_curve_history_reindexed = df_curve_history.reindex(date_range, method='ffill')
    df_curve_history['Step'] = 'UnFilled'
    df_curve_history_reindexed['Step'] = 'Filled'
    df_combined = df_curve_history.append(df_curve_history_reindexed).reset_index()
    df_combined.rename(columns={'index':'TradeDate'}, inplace=True)
    df_combined = df_combined.set_index(['TradeDate','Step']).unstack()

    return df_combined


def filter_filled_days(curve_name, start_date, end_date):
    df_combined = forward_fill_business_days(curve_name, start_date, end_date)
    df_nan_index = np.isnan(df_combined.select(lambda x: x[1] in ['UnFilled'], axis=1))
    df_filtered = df_combined.loc[df_nan_index[df_nan_index.apply(lambda x: reduce(lambda v,a: v and a, x), axis=1)].index]
    return df_filtered


def create_sql_insert_forward_fill(curve_name, start_date, end_date):
    sql_template = """INSERT INTO CurveNodeQuote (CurveNodeId, TradeDate, Quote, Lud, Lub) VALUES ({node_id}, '{trade_date}', {quote}, GETDATE(), 'wgajate');"""
    df_result = forward_fill_business_days(curve_name, start_date, end_date)

    # filter out unfilled values
    df_unfilled = np.isnan(df_result.loc[:, (slice(None), 'UnFilled')])
    filled_trade_date = df_unfilled[df_unfilled[df_unfilled.columns] == True].dropna().index
    df_result = df_result.loc[filled_trade_date,:]

    sql_insert_list = []
    for trade_date, series in df_result.iterrows():
        sql_insert_list_temp = []
        for (node_id, stage), quote in series.iterkv():
            if stage=='Filled':
                sql_insert=sql_template.format(node_id=node_id, trade_date=trade_date, quote=quote)
                sql_insert_list.append(sql_insert)
    return sql_insert_list

fwd_curves = """CurveTypeEnum	CountryEnum	CurrencyEnum	CurveLongName	CurveShortName	ActiveToDate	ActiveFromDate	UnderlyingTypeEnum	CurveDataId	InterpolationMethod	RIC	CountryName
SwapZCFwd	CH	CHF	Forwarding Swap Yield Curve CHF	CH.CHF.FDSWP1M.ZC	12/31/9999	9/7/2009	FDSWP1M	RefType|Name=SwapZeroCurve|CH.CHF.FDSWP1M.ZC	PieceWiseLinear	FDSWP1M	Switzerland
SwapZCFwd	CH	CHF	Forwarding Swap Yield Curve CHF	CH.CHF.FDSWP3M.ZC	12/31/9999	9/1/2009	FDSWP3M	RefType|Name=SwapZeroCurve|CH.CHF.FDSWP3M.ZC	PieceWiseLinear	FDSWP3M	Switzerland
SwapZCFwd	CH	CHF	Forwarding Swap Yield Curve CHF	CH.CHF.FDSWP6M.ZC	12/31/9999	9/1/2009	FDSWP6M	RefType|Name=SwapZeroCurve|CH.CHF.FDSWP6M.ZC	PieceWiseLinear	FDSWP6M	Switzerland
SwapZCFwd	EU	EUR	Forwarding Swap Yield Curve EUR	EU.EUR.FDSWP1M.ZC	12/31/9999	7/22/2009	FDSWP1M	RefType|Name=SwapZeroCurve|EU.EUR.FDSWP1M.ZC	PieceWiseLinear	FDSWP1M	Europe
SwapZCFwd	EU	EUR	Forwarding Swap Yield Curve EUR	EU.EUR.FDSWP1Y.ZC	12/31/9999	7/22/2009	FDSWP1Y	RefType|Name=SwapZeroCurve|EU.EUR.FDSWP1Y.ZC	PieceWiseLinear	FDSWP1Y	Europe
SwapZCFwd	EU	EUR	Forwarding Swap Yield Curve EUR	EU.EUR.FDSWP3M.ZC	12/31/9999	3/31/2005	FDSWP3M	RefType|Name=SwapZeroCurve|EU.EUR.FDSWP3M.ZC	PieceWiseLinear	FDSWP3M	Europe
SwapZCFwd	EU	EUR	Forwarding Swap Yield Curve EUR	EU.EUR.FDSWP6M.ZC	12/31/9999	3/31/2005	FDSWP6M	RefType|Name=SwapZeroCurve|EU.EUR.FDSWP6M.ZC	PieceWiseLinear	FDSWP6M	Europe
SwapZCFwd	GB	GBP	Forwarding Swap Yield Curve GBP	GB.GBP.FDSWP1M.ZC	12/31/9999	9/7/2009	FDSWP1M	RefType|Name=SwapZeroCurve|GB.GBP.FDSWP1M.ZC	PieceWiseLinear	FDSWP1M	Great Britain
SwapZCFwd	GB	GBP	Forwarding Swap Yield Curve GBP	GB.GBP.FDSWP3M.ZC	12/31/9999	1/1/2000	FDSWP3M	RefType|Name=SwapZeroCurve|GB.GBP.FDSWP3M.ZC	PieceWiseLinear	FDSWP3M	Great Britain
SwapZCFwd	GB	GBP	Forwarding Swap Yield Curve GBP	GB.GBP.FDSWP6M.ZC	12/31/9999	5/3/2007	FDSWP6M	RefType|Name=SwapZeroCurve|GB.GBP.FDSWP6M.ZC	PieceWiseLinear	FDSWP6M	Great Britain
SwapZCFwd	JP	JPY	Forwarding Swap Yield Curve JPY	JP.JPY.FDSWP1M.ZC	12/31/9999	9/22/2009	FDSWP1M	RefType|Name=SwapZeroCurve|JP.JPY.FDSWP1M.ZC	PieceWiseLinear	FDSWP1M	Japan
SwapZCFwd	JP	JPY	Forwarding Swap Yield Curve JPY	JP.JPY.FDSWP3M.ZC	12/31/9999	9/7/2009	FDSWP3M	RefType|Name=SwapZeroCurve|JP.JPY.FDSWP3M.ZC	PieceWiseLinear	FDSWP3M	Japan
SwapZCFwd	JP	JPY	Forwarding Swap Yield Curve JPY	JP.JPY.FDSWP6M.ZC	12/31/9999	8/10/2009	FDSWP6M	RefType|Name=SwapZeroCurve|JP.JPY.FDSWP6M.ZC	PieceWiseLinear	FDSWP6M	Japan
SwapZCFwd	US	USD	Forwarding Swap Yield Curve USD	US.USD.FDSWP1M.ZC	12/31/9999	7/7/2008	FDSWP1M	RefType|Name=SwapZeroCurve|US.USD.FDSWP1M.ZC	PieceWiseLinear	FDSWP1M	United States
SwapZCFwd	US	USD	Forwarding Swap Yield Curve USD	US.USD.FDSWP1Y.ZC	12/31/9999	3/10/2015	FDSWP1Y	RefType|Name=SwapZeroCurve|US.USD.FDSWP1Y.ZC	PieceWiseLinear	FDSWP1Y	United States
SwapZCFwd	US	USD	Forwarding Swap Yield Curve USD	US.USD.FDSWP3M.ZC	12/31/9999	1/4/2006	FDSWP3M	RefType|Name=SwapZeroCurve|US.USD.FDSWP3M.ZC	PieceWiseLinear	FDSWP3M	United States
SwapZCFwd	US	USD	Forwarding Swap Yield Curve USD	US.USD.FDSWP6M.ZC	12/31/9999	9/7/2009	FDSWP6M	RefType|Name=SwapZeroCurve|US.USD.FDSWP6M.ZC	PieceWiseLinear	FDSWP6M	United States
"""

covered_bond_curve_names_list = sorted(list(set(\
"""DE.EUR.CVBND.(A).RTGSPR
DE.EUR.CVBND.(A).RTGSPR
DE.EUR.CVBND.(A).RTGSPR
DE.EUR.CVBND.(A).RTGSPR
DE.EUR.CVBND.(A).RTGSPR
DE.EUR.CVBND.(AA).RTGSPR
DE.EUR.CVBND.(AA).RTGSPR
DE.EUR.CVBND.(AA).RTGSPR
DE.EUR.CVBND.(AA).RTGSPR
DE.EUR.CVBND.(AA).RTGSPR
DE.EUR.CVBND.(AAA).RTGSPR
DE.EUR.CVBND.(AAA).RTGSPR
DE.EUR.CVBND.(AAA).RTGSPR
DE.EUR.CVBND.(AAA).RTGSPR
DE.EUR.CVBND.(AAA).RTGSPR
DE.EUR.CVBND.(BBB).RTGSPR
DE.EUR.CVBND.(BBB).RTGSPR
DE.EUR.CVBND.(BBB).RTGSPR
DE.EUR.CVBND.(BBB).RTGSPR
DE.EUR.CVBND.(BBB).RTGSPR
DE.EUR.CVBND.(IG).RTGSPR
DE.EUR.CVBND.(IG).RTGSPR
DE.EUR.CVBND.(IG).RTGSPR
DE.EUR.CVBND.(IG).RTGSPR
DE.EUR.CVBND.(IG).RTGSPR
DE.EUR.CVBND.JUMBO.(A).RTGSPR
DE.EUR.CVBND.JUMBO.(A).RTGSPR
DE.EUR.CVBND.JUMBO.(A).RTGSPR
DE.EUR.CVBND.JUMBO.(A).RTGSPR
DE.EUR.CVBND.JUMBO.(A).RTGSPR
DE.EUR.CVBND.JUMBO.(AA).RTGSPR
DE.EUR.CVBND.JUMBO.(AA).RTGSPR
DE.EUR.CVBND.JUMBO.(AA).RTGSPR
DE.EUR.CVBND.JUMBO.(AA).RTGSPR
DE.EUR.CVBND.JUMBO.(AA).RTGSPR
DE.EUR.CVBND.JUMBO.(AAA).RTGSPR
DE.EUR.CVBND.JUMBO.(AAA).RTGSPR
DE.EUR.CVBND.JUMBO.(AAA).RTGSPR
DE.EUR.CVBND.JUMBO.(AAA).RTGSPR
DE.EUR.CVBND.JUMBO.(AAA).RTGSPR
DE.EUR.CVBND.JUMBO.(IG).RTGSPR
DE.EUR.CVBND.JUMBO.(IG).RTGSPR
DE.EUR.CVBND.JUMBO.(IG).RTGSPR
DE.EUR.CVBND.JUMBO.(IG).RTGSPR
DE.EUR.CVBND.JUMBO.(IG).RTGSPR
DK.EUR.CVBND.(A).RTGSPR
DK.EUR.CVBND.(A).RTGSPR
DK.EUR.CVBND.(A).RTGSPR
DK.EUR.CVBND.(A).RTGSPR
DK.EUR.CVBND.(A).RTGSPR
DK.EUR.CVBND.(AA).RTGSPR
DK.EUR.CVBND.(AA).RTGSPR
DK.EUR.CVBND.(AA).RTGSPR
DK.EUR.CVBND.(AA).RTGSPR
DK.EUR.CVBND.(AA).RTGSPR
DK.EUR.CVBND.(AAA).RTGSPR
DK.EUR.CVBND.(AAA).RTGSPR
DK.EUR.CVBND.(AAA).RTGSPR
DK.EUR.CVBND.(AAA).RTGSPR
DK.EUR.CVBND.(AAA).RTGSPR
DK.EUR.CVBND.(IG).RTGSPR
DK.EUR.CVBND.(IG).RTGSPR
DK.EUR.CVBND.(IG).RTGSPR
DK.EUR.CVBND.(IG).RTGSPR
DK.EUR.CVBND.(IG).RTGSPR
ES.EUR.CVBND.(A).RTGSPR
ES.EUR.CVBND.(A).RTGSPR
ES.EUR.CVBND.(A).RTGSPR
ES.EUR.CVBND.(A).RTGSPR
ES.EUR.CVBND.(A).RTGSPR
ES.EUR.CVBND.(AA).RTGSPR
ES.EUR.CVBND.(AA).RTGSPR
ES.EUR.CVBND.(AA).RTGSPR
ES.EUR.CVBND.(AA).RTGSPR
ES.EUR.CVBND.(AA).RTGSPR
ES.EUR.CVBND.(BBB).RTGSPR
ES.EUR.CVBND.(BBB).RTGSPR
ES.EUR.CVBND.(BBB).RTGSPR
ES.EUR.CVBND.(BBB).RTGSPR
ES.EUR.CVBND.(BBB).RTGSPR
ES.EUR.CVBND.(IG).RTGSPR
ES.EUR.CVBND.(IG).RTGSPR
ES.EUR.CVBND.(IG).RTGSPR
ES.EUR.CVBND.(IG).RTGSPR
ES.EUR.CVBND.(IG).RTGSPR
FR.EUR.CVBND.(AA).RTGSPR
FR.EUR.CVBND.(AA).RTGSPR
FR.EUR.CVBND.(AA).RTGSPR
FR.EUR.CVBND.(AA).RTGSPR
FR.EUR.CVBND.(AA).RTGSPR
FR.EUR.CVBND.(AAA).RTGSPR
FR.EUR.CVBND.(AAA).RTGSPR
FR.EUR.CVBND.(AAA).RTGSPR
FR.EUR.CVBND.(AAA).RTGSPR
FR.EUR.CVBND.(AAA).RTGSPR
FR.EUR.CVBND.(IG).RTGSPR
FR.EUR.CVBND.(IG).RTGSPR
FR.EUR.CVBND.(IG).RTGSPR
FR.EUR.CVBND.(IG).RTGSPR
FR.EUR.CVBND.(IG).RTGSPR
SE.EUR.CVBND.(AA).RTGSPR
SE.EUR.CVBND.(AA).RTGSPR
SE.EUR.CVBND.(AA).RTGSPR
SE.EUR.CVBND.(AA).RTGSPR
SE.EUR.CVBND.(AA).RTGSPR
SE.EUR.CVBND.(AAA).RTGSPR
SE.EUR.CVBND.(AAA).RTGSPR
SE.EUR.CVBND.(AAA).RTGSPR
SE.EUR.CVBND.(AAA).RTGSPR
SE.EUR.CVBND.(AAA).RTGSPR
SE.EUR.CVBND.(IG).RTGSPR
SE.EUR.CVBND.(IG).RTGSPR
SE.EUR.CVBND.(IG).RTGSPR
SE.EUR.CVBND.(IG).RTGSPR
SE.EUR.CVBND.(IG).RTGSPR
SE.SEK.CVBND.(AA).RTGSPR
SE.SEK.CVBND.(AA).RTGSPR
SE.SEK.CVBND.(AA).RTGSPR
SE.SEK.CVBND.(AA).RTGSPR
SE.SEK.CVBND.(AA).RTGSPR
SE.SEK.CVBND.(AAA).RTGSPR
SE.SEK.CVBND.(AAA).RTGSPR
SE.SEK.CVBND.(AAA).RTGSPR
SE.SEK.CVBND.(AAA).RTGSPR
SE.SEK.CVBND.(AAA).RTGSPR
SE.SEK.CVBND.(IG).RTGSPR
SE.SEK.CVBND.(IG).RTGSPR
SE.SEK.CVBND.(IG).RTGSPR
SE.SEK.CVBND.(IG).RTGSPR
SE.SEK.CVBND.(IG).RTGSPR""".split('\n'))))