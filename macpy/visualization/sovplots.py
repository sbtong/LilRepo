import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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
import macpy.visualization.plotly_helper as pl
from plotly.graph_objs import Scatter
import seaborn as sns # Changes default grapical settings

### macpy package - available at ->  http://git.axiomainc.com:8080/tfs/axiomadev/_git/ContentDev-MAC
import macpy
import macpy.utils.database as db
import shutil
from StringIO import StringIO
from PIL import Image

py.sign_in('Axioma01', 'rbgwcvrp8i')

class EMCountries(object):
    """
    Contains list and attributes of emerging markets
    """
    def __init__(self, end_date):
        self.end_date = end_date
        self.date_range_ccy = StringIO("""
CurveName	Currency	CountryName	ImpliedRating	StartDate
BG.BGN.GVT.ZC	BGN	Bulgaria	BBB	8/20/2002
BR.BRL.GVT.ZC	BRL	Brazil	B	1/1/2008
CL.CLP.GVT.ZC	CLP	Chile	A	11/3/2005
CN.CNY.GVT.ZC	CNY	China	BBB	3/31/2004
CO.COP.GVT.ZC	COP	Colombia	BB	8/21/2002
CZ.CZK.GVT.ZC	CZK	Czech Republic	AAA	1/1/2000
DK.DKK.GVT.ZC	DKK	Denmark	AAA	1/1/2000
HK.HKD.GVT.ZC	HKD	Hong Kong	AA	1/1/2000
HR.HRK.GVT.ZC	HRK	Croatia	BB	1/30/2008
HU.HUF.GVT.ZC	HUF	Hungary	BBB	1/4/2000
ID.IDR.GVT.ZC	IDR	Indonesia	BB	2/1/2010
IL.ILS.GVT.ZC	ILS	Israel	A	6/1/2012
IN.INR.GVT.ZC	INR	India	BBB	1/1/2006
IS.ISK.GVT.ZC	ISK	Iceland	A	9/8/2004
KR.KRW.GVT.ZC	KRW	Korea (Republic of)	AA	1/1/2006
MX.MXN.GVT.ZC	MXN	Mexico	BBB	2/21/2002
MY.MYR.GVT.ZC	MYR	Malaysia	BBB	1/1/2006
NZ.NZD.GVT.ZC	NZD	New Zealand	AAA	1/1/2000
PH.PHP.GVT.ZC	PHP	Philippines	A	1/1/2009
PK.PKR.GVT.ZC	PKR	Pakistan	CCC	1/1/2012
PL.PLN.GVT.ZC	PLN	Poland	A	1/1/2000
RO.RON.GVT.ZC	RON	Romania	A	11/1/2005
RU.RUB.GVT.ZC	RUB	Russian Federation	BB	1/1/2000
SG.SGD.GVT.ZC	SGD	Singapore	AAA	1/1/2006
TH.THB.GVT.ZC	THB	Thailand	BBB	1/1/2006
TR.TRY.GVT.ZC	TRY	Turkey	BB	1/1/2003
TW.TWD.GVT.ZC	TWD	Taiwan, Province of China	AA	1/1/2003
VN.VND.GVT.ZC	VND	Viet Nam	BB	1/1/2012
UA.UAH.GVT.ZC   UAH Ukraine CCC 3/31/2011
ZA.ZAR.GVT.ZC	ZAR	South Africa	BB	1/1/2000
""")
        self.df_extract_config = pd.read_csv(self.date_range_ccy, sep='\t')
        self.df_extract_config['EndDate'] = end_date

        trade_date_begin, trade_date_end, curve_name = self.df_extract_config.ix[0].StartDate,\
                                                       self.df_extract_config.ix[0].EndDate, \
                                                       self.df_extract_config.ix[0].CurveName

class Tenor:
    """
    """

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
    
help(Tenor)


# In[10]:

class PlotConfig:
    """
    """
    
    def __init__(self, 
                 curve_name='None',
                 show_legend=False, 
                 filename_prefix='Plot_Name',
                 plot_title='Plot_Title',
                 x_axis_title='x axis title',
                 y_axis_title='y axis title',
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
        self.img_format = img_format  
        self.layout_width = layout_width
        self.layout_height = layout_height,                         
        self.font_size = 15
        self.plot_sleep = plot_sleep # sleep is required because plotly's 'save' method returns before the image file is available on disk
        self.destination_directory=destination_directory
        self.scatter_vectors = scatter_vectors

help(PlotConfig)


# In[11]:

def extract_source_bonds_dataframe(df_config, messages=[]):
    """
    
    """
    
    sql_bonds_template = Template("""
    SELECT cv.CurveShortName, prc.TradeDate, prc.InstrCode, Quote=prc.MatStdYld
    FROM DerivCurveFilteredBond fb
    JOIN [QAI]..FIEJVPrcDly prc on prc.InstrCode = fb.InstrCode and prc.TradeDate = fb.TradeDate
    JOIN Curve cv on fb.CurveId = cv.CurveId
    WHERE prc.TradeDate <= '$trade_date_end'
    AND prc.TradeDate >= '$trade_date_begin'
    AND cv.CurveShortName in ('$curve_name')
    AND itemid = 2
    """)   
    
    trade_date_begin, trade_date_end, curve_name = df_config.StartDate, df_config.EndDate, df_config.CurveName
    source_bond_data = [] 
    sql_bonds = sql_bonds_template.substitute(trade_date_begin=trade_date_begin, trade_date_end=trade_date_end, curve_name=curve_name)
    messages.append(sql_bonds)
    df_source_bonds = db.MSSQL.extract_dataframe(sql_bonds)
    df_source_bonds['TradeDate'] = pd.to_datetime(df_source_bonds['TradeDate'])
    fields = ['CurveShortName','TradeDate','InstrCode','Quote']
    df_pivot = df_source_bonds[fields].groupby(fields[:-1]).sum().unstack()    
    date_range = pd.date_range(trade_date_begin, trade_date_end, freq='B')    
    df_subset = df_pivot.loc[curve_name,:]['Quote'].reindex(date_range)
    return df_subset

messages=[]
df_zero_curves = extract_source_bonds_dataframe(df_extract_config.ix[0],messages)

print messages[0]
df_zero_curves.describe()


# In[15]:

# create sovereign bond yield scatter vectors
def create_bond_scatter_vectors(df_source_bonds):
    """
    
    """
    source_bond_scatter_vectors = []
    #iterate through curve nodes    
    for node_label in df_source_bonds.loc[:, df_source_bonds.columns]:  
        df_nodes = df_source_bonds.loc[:, node_label]
        name = node_label
        name = df_nodes.name        
        curve_node_x = df_nodes.index
        curve_node_y = df_nodes 
        scatter = Scatter(x=curve_node_x, 
                          y=curve_node_y, 
                          connectgaps=False,
                          name=str(name), 
                          mode='lines', 
                          #marker = dict(size=3),
                          line=dict(shape='linear',width=1))
        source_bond_scatter_vectors.append(scatter)   
    return source_bond_scatter_vectors

df_source_bonds = extract_source_bonds_dataframe(df_extract_config.ix[0])
create_bond_scatter_vectors(df_source_bonds)[0]['x']
#plot_history(curves[3], create_plot_config_source_bond())


# In[18]:

#extract derived curves
def extract_curve_dataframe(df_config):
    """
    
    """
    sql_curve_template = Template("""
    select CurveShortName=SUBSTRING(CurveShortName,1,14), TradeDate, Quote, te.TenorEnum
    from [MarketData].[dbo].[Curve] cv
    join [MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
    join [MarketData].[dbo].TenorEnum te on cn.TenorEnum = te.TenorEnum
    join [MarketData].[dbo].CurveNodeQuote cq on cq.CurveNodeId = cn.CurveNodeId
    where
    cq.TradeDate >= '$begin_date' and
    cq.TradeDate <= '$end_date' and
    cv.CurveShortName = '$curve_name'
    """)
    
    trade_date_begin, trade_date_end, curve_name = df_config.StartDate, df_config.EndDate, df_config.CurveName    
    sql_curve = sql_curve_template.substitute(begin_date=trade_date_begin, end_date=trade_date_end, curve_name=curve_name)
    #print sql_curve
    df_curve_history = db.MSSQL.extract_dataframe(sql_curve)
    df_curve_history['TradeDate'] = pd.to_datetime(df_curve_history['TradeDate'])
    df_curve_history["TenorInYears"] = df_curve_history["TenorEnum"].apply(lambda x: round(Tenor.convert_tenor_to_years(x), 3))
    fields = ['CurveShortName','TradeDate','TenorInYears','Quote']
    df_pivot = df_curve_history[fields].groupby(fields[:-1]).sum().unstack()
    date_range = pd.date_range(trade_date_begin, trade_date_end, freq='B')    
    df_pivot_reindexed = df_pivot['Quote'].loc[curve_name,:].reindex(date_range)
    return df_pivot_reindexed


print help(extract_curve_dataframe)

extract_curve_dataframe(df_extract_config.ix[0]).head(2)


# In[19]:

## function to plot scatter vectors
def plot_history(plot_config):
    """
    TESTER
    
    """
    try:
        curve_name = plot_config.curve_name
        scatter_vectors = plot_config.scatter_vectors
        # save derived curve
        filename = plot_config.filename_prefix + curve_name
        plot_title = plot_config.plot_title + curve_name

        # Plot derived yield curves
        layout = pl.create_basic_layout(x_axis_title=plot_config.x_axis_title,
                                        y_axis_title=plot_config.y_axis_title, 
                                        title=plot_title, 
                                        width=plot_config.layout_width,
                                        height=plot_config.layout_height,
                                        font_size=plot_config.font_size,
                                        show_legend=plot_config.show_legend)

        figure = pl.create_figure(scatter_vectors, layout)
        img_format = plot_config.img_format


        full_file_name = filename + '.' + img_format
        iplot(figure, filename=filename, image=img_format, show_link=False)  
        #py.image.save_as(figure, full_file_name, format='pdf')        
        source_plot_file_path = plot_config.source_plot_file_path + full_file_name
        destination_plot_file_path = plot_config.destination_directory + full_file_name
        #print source_plot_file_path, destination_plot_file_path
        time.sleep(plot_config.plot_sleep)
        shutil.move(source_plot_file_path, destination_plot_file_path)
        #im = Image.open(destination_plot_file_path, mode='r')
        #im.save(destination_plot_file_path[0:-4]+'eps')
        #os.remove(source_plot_file_path)
        return ",".join([source_plot_file_path, destination_plot_file_path])
    except Exception as e:
        return repr(e)
    
def create_test_plot_config(df_extract_config):
    plot_config = PlotConfig()
    plot_config.curve_name = df_extract_config.CurveName
    plot_config.filename_prefix='TEST'
    plot_config.plot_title='Test PLOT_'
    plot_config.source_plot_file_path= 'C:/Users/wgajate/Downloads/'
    plot_config.destination_directory = 'E:/EM_Cleanup_Dec16/plots/'
    plot_config.layout_width = 300
    plot_config.layout_height = 200    
    plot_config.scatter_vectors = [Scatter(x=range(1,10), y=range(1,10))]
    return plot_config

print help(plot_history)

plot_history(create_test_plot_config(df_extract_config.ix[0]))


# In[20]:

def create_zero_curve_scatter_vectors(df_zero_curves, transform = lambda x: x):
    """
    
    """
    
    derived_curve_data = []
    for node_label in df_zero_curves:  #iterate through curve nodes
        df_nodes = df_zero_curves[node_label]
        name = node_label
        if name not in [.083, .5, 1.0, 2.0, 5.0, 10.0, 30.0]:
            continue
        name = df_nodes.name        
        curve_node_x = df_nodes.index
        curve_node_y = transform(df_nodes)
        scatter = Scatter(x=curve_node_x, 
                          y=curve_node_y, 
                          connectgaps=False,
                          name=str(name)+'Y', 
                          mode='lines', 
                          #marker = dict(size=3),
                          line=dict(shape='linear',width=1))
        derived_curve_data.append(scatter)   
    return derived_curve_data

print help(create_zero_curve_scatter_vectors)
df_zero_curves = extract_curve_dataframe(df_extract_config.ix[0])
create_zero_curve_scatter_vectors(df_zero_curves)[0]['x']


# In[ ]:

# generate and save source bond plot
def create_plot_config_source_bond(df_config):
    """
    
    """
    plot_config = PlotConfig()
    plot_config.curve_name=df_config.CountryName  
    plot_config.filename_prefix='SovereignBonds_'
    plot_config.plot_title='Source Bond Yields: '
    plot_config.show_legend=False
    plot_config.x_axis_title='Analysis Date'
    plot_config.y_axis_title='Yield (%)'
    plot_config.source_plot_file_path= 'C:/Users/wgajate/Downloads/'
    plot_config.destination_directory = 'E:/EM_Cleanup_Dec16/plots/'
    plot_config.layout_width = 1000
    plot_config.font_size = 15
    plot_config.sleep = 5
    df_source_bonds = extract_source_bonds_dataframe(df_config).tail(250*4)
    plot_config.scatter_vectors = create_bond_scatter_vectors(df_source_bonds)
    return plot_config

[plot_history(create_plot_config_source_bond(extract_config)) for index, extract_config in df_extract_config.iterrows()]


# In[565]:

# generate and save derived curve yield plot
def create_plot_config_curve_yield(df_extract_config):
    """
    
    """
    plot_config = PlotConfig()
    plot_config.curve_name = df_extract_config.CountryName
    plot_config.filename_prefix='DerivedZeroYield_'
    plot_config.plot_title='Sovereign Zero Yield (%): '
    plot_config.show_legend=True
    plot_config.x_axis_title='Analysis Date'
    plot_config.y_axis_title='Yield (%)'
    plot_config.source_plot_file_path= 'C:/Users/wgajate/Downloads/'
    plot_config.destination_directory = 'E:/EM_Cleanup_Dec16/plots/'
    plot_config.layout_width = 1000
    plot_config.layout_height = 600
    plot_config.font_size = 15
    plot_config.sleep = 5
    transform_functor = lambda y: np.round(y*100,2)
    df_zero_curves = extract_curve_dataframe(df_extract_config)
    plot_config.scatter_vectors = create_zero_curve_scatter_vectors(df_zero_curves, transform=transform_functor)
    return plot_config

extract_configs = [df_config for index, df_config in df_extract_config.iterrows()]

[plot_history(create_plot_config_curve_yield(extract_config)) for extract_config in extract_configs]


# In[499]:

# generate and save derived curve volitility plot
def create_plot_config_curve_vol(df_extract_config):
    plot_config = PlotConfig()
    plot_config.filename_prefix='DerivedZeroVol_'
    plot_config.plot_title='Sovereign Zero Yield Volatility (60 day rolling): '
    plot_config.show_legend=True
    plot_config.x_axis_title='Analysis Date'
    plot_config.y_axis_title='Annualized Vol (bps)'
    plot_config.source_plot_file_path= 'C:/Users/wgajate/Downloads/'
    plot_config.destination_directory = 'E:/EM_Cleanup_Dec16/plots/'
    plot_config.layout_width = 1000
    plot_config.layout_height = 600
    plot_config.font_size = 15
    plot_config.sleep = 5
    transform_functor = lambda y: np.round(pd.rolling_std(y.diff()*10000*math.sqrt(250), window=60),0)
    df_zero_curves = extract_curve_dataframe(df_extract_config)
    plot_config.scatter_vectors = create_zero_curve_scatter_vectors(df_zero_curves, transform=transform_functor)
    return plot_config

plot_config = create_plot_config_curve_vol(df_extract_config.ix[0])

[plot_history(plot_config=plot_config) for curve_name in curves[1:2]]


# In[321]:

curves


# In[ ]:



