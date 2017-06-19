#$Id: CurveControl.py 184397 2014-07-31 14:08:40Z vsmani $
import matplotlib
matplotlib.use('Agg')
import os
import pymssql
import pandas.io.sql as sql
import pandas as pd
import datetime
from prototype import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from datetime import date
from datetime import timedelta
from optparse import OptionParser
import scipy.interpolate
import logging
from string import Template
from matplotlib.font_manager import FontProperties

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
from matplotlib.backends.backend_pdf import PdfPages


@constructor
def DataFrameCreator(self, sqltemplate):
    """
    Converts templated SQL into SQL statement and runs SQL against database returning a pandas dataframe
    """
    self.sqltemplate = sqltemplate;
    
    def createSQL(self, **kwargs):
        template = Template(self.sqltemplate)
        subst = template.safe_substitute(kwargs)
        return subst
    
    self.createSQL = createSQL

    def createDataFrame(self, dbconn, **kwargs):
        sqlValue = createSQL(kwargs)
        df = sql.read_sql(sqlValue, dbconn)
        return df
        
    self.createDataFrame = createDataFrame


class UsageHelp:
    @staticmethod
    def commandLineExamples():
        r"""
        Offers common use usage examples for CurveControl module as command line script        
        """

        usage = r"""
        #generates a curve comparison graph and writes it to the file usgraph.jbg
        python CurveControl.py -f usgraph.jpg -d 2013-09-20 -c us.usd.gvt.zc -s prod_mac_mkt_db
        """

    @staticmethod
    def examples():
        r"""
        Offers common use usage examples for CurveControl module
        """
        usage = r"""
        import CurveControl as cc
        from CurveControl import CurveShortName as csn

        #Plot G7 for given date
        csn.plot_g7_asOf('2013-08-27','prod_mac_mkt_db')

        #Plot Elara and Saturn histories from 2013-2011 Country US
        csn.comp_hist('2013|2012|2011','US','prod_mac_mkt_db')

        #Plot specific date range and curve
        ax = csn.quick_hist('NO.NOK.GVT.ZC','2008-09-01:2008-12-15',server='saturn')

        #Plot detailed comparison across servers and sources for specified date
        results = csn.quick_comp('NO.NOK.GVT.ZC','2008-11-05','saturn','prod_mac_mkt_db')
        """
        print usage

class CurvePlot:
    r"""
    Base class containing core plotting logic.
    """
    __ypos=0.0
    __xpos=.5

    @classmethod
    def __y_pos(cls):
        dy = .005
        while True:
            cls.__ypos = cls.__ypos + dy
            if cls.__ypos > .06:
                cls.__ypos = 0.0
            yield cls.__ypos

    @classmethod
    def __plot_nodata(cls, axes, name):
        ypos = cls.__y_pos().next()
        xpos = cls.__xpos
        axes.text(xpos, ypos, name + ' has no data', style='italic')
        
    def read_frame(self):
        self.sqlQuery = self.create_query()
        self.df = self.db.read_frame(self.sqlQuery)
        return self.df

    def read_frameDev(self):
        self.sqlQuery = self.create_query()
        self.df = self.db.read_frame(self.sqlQuery)
        return self.df

    def plot_2d(self, ax2, axes, dateRange, server, maxYears=50, dirName='~\\'):
        
        self.db = DBServer(server=server)
        self.dateRange = DateRange(dateRange)
        self.df = self.read_frame()
        
        if (self.name == 'Swap'): #extrapolate to term 0 years and save the interpolation function
            if(len(self.df) > 0):
                df0 = self.df[:1]
                df0['InYears'] = 0
                self.df = pd.concat([df0, self.df])
                self.linfit = scipy.interpolate.interp1d(self.df['InYears'], self.df['Quote'], bounds_error=False)
        elif (self.name == 'SwapT-1'): #extrapolate to term 0 years and save the interpolation function
            if(len(self.df) > 0):
                df0 = self.df[:1]
                df0['InYears'] = 0
                self.df = pd.concat([df0, self.df])
                self.linfitTm1 = scipy.interpolate.interp1d(self.df['InYears'], self.df['Quote'], bounds_error=False)
                
        self.df = self.df.ix[self.df.InYears <= maxYears] 
        #print self.df
        if(len(self.df) > 0):
            self.maxterm = self.df.InYears.max()
            x, y = self.df['InYears'], self.df['Quote']
            self.plotImpl(axes, x, y)
        else:
            self.maxterm = 1
            CurvePlot.__plot_nodata(axes, self.name)
        return axes
    
    def plot_3DHist(self, df, ax, server, alpha=0.8, textRot=80):

        curveHist = df['Quote'].groupby([df['TradeDate'],df['InYears']]).sum().unstack()
        axsDate = pd.Series(curveHist.index)
        [lenDates, _ ] = curveHist.shape
        
        xind = np.arange(0, lenDates)
        yind = np.asarray(curveHist.columns)
        [Y, X] = np.meshgrid(yind, xind)
        Z = np.asarray(curveHist)
        #cmap = matplotlib.cm.cool
        ax.plot_surface(X, Y, Z, rstride=1, cstride=2, alpha=alpha, linewidth=0, antialiased=False) #cmap=cmap
        #cset = ax.contour(X, Y, Z, zdir='x', offset=-15, cmap=cmap)
        xTickLabel = np.asarray(axsDate.T)
        #print 'tick labels', xTickLabel
        plt.xticks(xind, xTickLabel,rotation=textRot,fontsize=7,horizontalalignment='center')
        ax.set_xlim(-10, 50)
        ax.set_ylabel('Term in Years')
        #ax.set_ylim(0, 50)
        ax.set_zlabel('Yield (Absolute)')
        #ax.set_zlim(-0.01, 0.05)
        title = self.name + ' on server: ' + server 
        ax.set_title(title)

    def get_history_df(self, dateRange, server='Saturn'):
        self.db = DBServer(server=server)
        self.dateRange = DateRange(dateRange)
        self.df = self.read_frame()
        return self.df

    def plot_history(self, dateRange, sumDfList, server='Saturn', dim='3d'):
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
        self.db = DBServer(server=server)
        self.db2 = DBServer(server='Saturn')
        self.dateRange = DateRange(dateRange)
        
        def getTopScores(dOutParam, numLimit=10):
            if (len(dOutParam) < 1):
                return pd.DataFrame()
            dfpTop10 = None
            for k, v in dOutParam.iteritems():
                dfv = v['data']
                dfv['Term'] = v['term']
                if (dfpTop10 is None):
                    dfpTop10 = dfv
                else:
                    dfpTop10 = dfpTop10.append(dfv)
            dfpTop10 = dfpTop10.sort_index(by=['score'], ascending=[False])
            dfpTop10['DateTime'] = dfpTop10.index
            dfpTop10.drop_duplicates('DateTime', inplace=True)
            dfpTop10 = dfpTop10[:numLimit]
            dfpTop10['Score'] = np.round(dfpTop10['score'], decimals=1)
            dfpTop10['Term'] = np.round(dfpTop10['Term'], decimals=3)
            dfpTop10['Yield'] = np.round(dfpTop10['level']*100, decimals=1)
            dfpTop10['Date']= dfpTop10['DateTime'].apply(lambda x: x.date())
            #print 'top10', dfpTop10
            dfpTop10 = pd.DataFrame(dfpTop10, columns=['Score','Date','Term','Yield'])
            return dfpTop10

        if (dim == '2dcomp'):
            self.finalTable = False;
            self.df = self.read_frame()
            curveId = 0
            OutlierBound = 12
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)#, sharey=True)
            self.dfTestRes = None
                           
            if (len(self.df)>0):
                self.df.TradeDate = pd.to_datetime(self.df.TradeDate) #, format='%Y-%m-%d')
            
                #self.df.TradeDate = self.df.TradeDate.apply(lambda x: x.date())
                self.curveHist = self.df['Quote'].groupby([self.df['TradeDate'],self.df['InYears']]).sum().unstack()
                curveId = self.df.ix[0].CurveId
                self.dpOutParam = generateOutlierParams(formateRawDataFrame(self.df), bound = OutlierBound)
                self.dfpOutParSel = pd.DataFrame(self.dpOutParam, index=['term','count','score']).T
                #print self.dpOutParam, self.dfpOutParSel
                self.dfpTop10 = getTopScores(self.dpOutParam)

                #self.dfpOutTotalScore = self.dfpOutParSel['score'].sum()
                #self.dfpOutTotalCount = self.dfpOutParSel['count'].sum()
                self.dfTestRes = pd.DataFrame({'Score':[self.dfpOutParSel['score'].sum()],
                                               'Count':[self.dfpOutParSel['count'].sum()]}, index=['Proposed'])
            else:
                self.curveHist = None
            self.finalTable = True;
            self.dfFinal = self.read_frame()

            if (len(self.dfFinal)>0):            
                self.dfFinal.TradeDate = pd.to_datetime(self.dfFinal.TradeDate) #, format='%Y-%m-%d')
                self.curveHistFinal = self.dfFinal['Quote'].groupby([self.dfFinal['TradeDate'],self.dfFinal['InYears']]).sum().unstack()
                self.dffOutParam = generateOutlierParams(formateRawDataFrame(self.dfFinal), bound = OutlierBound)
                self.dffOutParSel = pd.DataFrame(self.dffOutParam, index=['term','count','score']).T
                self.dffTop10 = getTopScores(self.dffOutParam)

                self.dfTestRes = self.dfTestRes.append(pd.DataFrame({'Score':[self.dffOutParSel['score'].sum()],'Count':[self.dffOutParSel['count'].sum()]}, index=['Baseline']))
                self.dfTestRes = self.dfTestRes[['Score','Count']]
                #self.dffOutTotalScore = self.dffOutParSel['score'].sum()
                #self.dffOutTotalCount = self.dffOutParSel['count'].sum()
                #print 'dff Total Outlier count ',self.dffOutTotalCount,' Total Score ',self.dffOutTotalScore
            else:
                self.curveHistFinal = None
            if (self.curveHist is not None and self.curveHistFinal is not None):
                #i2=self.curveHist.index.join(self.curveHistFinal.index, how='outer')
                #i2=pd.bdate_range(self.dateRange.startDate, self.dateRange.endDate, freq='B')
                i2=self.curveHist.index.join(pd.bdate_range(self.curveHist.index[0], self.curveHist.index[-1], freq='B'), how='outer')
                self.dfp=pd.DataFrame(self.curveHist, index=i2)
                self.dff=pd.DataFrame(self.curveHistFinal, index=i2)
            else:
                self.dfp=pd.DataFrame(self.curveHist)
                self.dff=pd.DataFrame(self.curveHistFinal)
            
            #print 'dfp', self.dfp, self.dfp.index

            title = self.name + ' (CurveId: ' + str(curveId) + ')\n' + 'Proposed (Table: CurveNodeQuote)'
            axes[0].set_title(title)
            def to_percent(y, position):
                s = str(100 * y)
                return s + r'$\%$'
                
            axepText = 'Server: ' + server + ' Date range: ' + dateRange + ' TimeStamp: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            txtp1 = axes[0].text(.25, 1.15, axepText, va = 'top', ha = 'left', transform=axes[0].transAxes, fontsize=5)
            totalScorePositionX = .83
            totalScorePositionY = 1.08
            self.tblBbox=[1.005,0,.2,1]
            #axepText.set_alpha(0.5)
            self.tblFontSize = 6
            self.tblColWidths = [0.15, 0.25, 0.13, 0.13]
            self.tblColLabels = ['Score','Date','Term\n(years)','Yield\n(%)']
            font0 = FontProperties()
            
            fontBold = font0.copy()
            fontBold.set_weight('bold')
            for i in (0, 1):
                txtpi = axes[i].text(1.02, 1.05, 'Top 10 Outliers by Score', fontproperties=fontBold, 
                                     va = 'top', ha = 'left', transform=axes[i].transAxes, fontsize=5)
            self.outlierScoreText = 'Total Outlier Score: %7.0f \nTotal Outlier Count: %7.0f'
            self.outlierAlpha = 0.6
            if (len(self.dfp)>0):
                self.dfp.plot(ax=axes[0], zorder=100)#np.round(self.dfpTop10, 2), linewidth=0, marker='.',markersize=2)#, sharex=True, sharey=True);
                #print  self.curveHist
                axes[0].legend().set_visible(False)
                axes[0].set_ylabel('Yield')
                axes[0].set_xlabel('')
                axes[0].yaxis.set_major_formatter(FuncFormatter(to_percent))
                axeptText = self.outlierScoreText % (self.dfTestRes.loc['Proposed','Score'], self.dfTestRes.loc['Proposed','Count'])#(self.dfpOutTotalScore, self.dfpOutTotalCount)
                txtp2 = axes[0].text(totalScorePositionX, totalScorePositionY, axeptText, va = 'top', ha = 'left', transform=axes[0].transAxes, fontsize=5)
                if (len(self.dfpTop10)>0):
                    table0 = axes[0].table(cellText=np.array(self.dfpTop10),
                      #rowLabels=np.array(self.dfpTop10.index),
                      #rowColours=colors,
                      colWidths = self.tblColWidths,
                      colLabels = self.tblColLabels,
                      loc='right', bbox=self.tblBbox)
                    table0.set_fontsize(self.tblFontSize)
                #table0.set_bordersize(0)
                    table0.scale(.4,.5)
                    axes[0].scatter(self.dfpTop10.index, self.dfpTop10['Yield']/100, facecolors='#A60628', edgecolors='black', lw=1, s=100, alpha=self.outlierAlpha, zorder=1)
                for label in axes[0].get_xticklabels(minor = True):
                    label.set_visible(False)

                
            if (len(self.dff)>0):
                #if (len(self.dfp)>0):
                #    self.dff.plot(ax=axes[1])#, xlim=axes[0].get_xlim(), xticks=axes[0].get_xticks()) #sharex=True, sharey=True);
                #else:
                self.dff.plot(ax=axes[1], ylim=axes[0].get_ylim(), zorder=100)#0, marker='.',markersize=3) 
                axes[1].legend().set_visible(False)
                title = 'Baseline (Table: CurveNodeQuoteFinal)'
                axes[1].set_title(title)
                axes[1].set_ylabel('Yield')
                axes[1].set_xlabel('')
                axes[1].yaxis.set_major_formatter(FuncFormatter(to_percent))
                axebtText = self.outlierScoreText % (self.dfTestRes.loc['Baseline','Score'], self.dfTestRes.loc['Baseline','Count'])
                txtf2 = axes[1].text(totalScorePositionX, totalScorePositionY, axebtText, va = 'top', ha = 'left', transform=axes[1].transAxes, fontsize=5)
                if (len(self.dffTop10)>0):
                    table1 = axes[1].table(cellText=np.array(self.dffTop10),
                      #rowLabels=np.array(self.dfpTop10.index),
                      #rowColours=colors,
                      colWidths = self.tblColWidths,
                      colLabels = self.tblColLabels,
                      loc='right', bbox=self.tblBbox)#, bbox=[.8,.2,.3,.4])
                    table1.set_fontsize(self.tblFontSize)
                    table1.scale(.4,.5)
                    axes[1].scatter(self.dffTop10.index, self.dffTop10['Yield']/100, facecolors='#A60628', edgecolors='black', lw=1, s=100, alpha=self.outlierAlpha, zorder=1)
            #print self.dfTestRes
            if (self.dfTestRes is None):
                self.dfTestRes = pd.DataFrame({'Score':[0, 0],
                                               'Count':[0, 0]}, index=['Proposed', 'Baseline'])
            dfChange = self.dfTestRes.T
            dfChange['Change']=np.round(dfChange['Proposed']-dfChange['Baseline'], decimals=1)
            print dfChange
            sumDfList.append(dfChange)
            #print 'Summary', sumDfList
            tableChange = axes[0].table(cellText=np.array(pd.DataFrame(dfChange['Change'])),
              rowLabels=['Outlier Score Chg','Outlier Count Chg'],
              #rowColours=colors,
              colWidths = [.5,.3],
              #colLabels = ['',''],
              loc='top left', bbox=[0.11,1.01,.1,.08])
            tableChange.scale(.2,.08)
            tableChange.set_fontsize(5)
            
            txtpc = axes[0].text(0.05, 1.12, 'Test Results', fontproperties=fontBold, 
                                     va = 'top', ha = 'left', transform=axes[0].transAxes, fontsize=5)
            #print self.dff
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.1, right=0.8)
                        
            return axes[0]

        else:
            self.df = self.read_frame()
            fig = plt.figure(figsize=(18,8))
            axe = fig.add_subplot(1,2,1,projection='3d')
            if(len(self.df) > 0):
                self.plot_3DHist(self.df, axe, server, alpha=0.7, textRot=80)
                axe.set_xlabel('TradeDate')
                
            self.curveShortName.curveShortName = self.curveShortName.curveShortName.replace("GVT.ZC", "SWP.ZCS", 1);
            if (self.curveShortName.curveShortName == "EP.EUR.SWP.ZCS"):
                self.curveShortName.curveShortName = "EU.EUR.SWP.ZCS"
            self.name = self.curveShortName.curveShortName
    
            self.swapdf = self.read_frame()
            #print self.name, self.swapdf
            if (len(self.swapdf) > 0):
                swapax = fig.add_subplot(1,2,2,projection='3d')
                swapax.view_init(20,20)
                #self.df['TradeDate']=self.df['TradeDate'].apply(lambda x: x.date())
                self.plot_3DHist(self.swapdf, swapax, server, alpha=0.5, textRot=-50)
                swapax.set_xlabel('')
                plt.tight_layout()
            return axe

    
    def create_query(self, serverOverride = None):
        curveNames = self.curveShortName.get_curve_names()
        if (self.finalTable):
            self.sqlQuery = self.sqlTemplate.format(self.dateRange.startDate, self.dateRange.endDate, curveNames, "final")
        else:
            self.sqlQuery = self.sqlTemplate.format(self.dateRange.startDate, self.dateRange.endDate, curveNames, "")
        return self.sqlQuery    

class SQLQuery:
    """ SQLQuery's sole responsible is to generate SQL from the underlying template. 
        Generated SQL is not required to be valid against a database
    """

    def __init__(self, curveShortName='', dateRange='dateRangeNotSet', serverLocation='location:NotSet'):
        self.curveShortNames = CurveShortName(curveShortName)
        self.dateRange = DateRange(dateRange)
        self.serverLocation = serverLocation
        self.template = Template('default template: $startDate, $endDate, $curveShortNames, $serverLocation')

    def create_query(self):
        dictForTemplate = self._generate_dic_for_template_sub()
        self.sqlQuery = self.template.substitute(dictForTemplate)
        return self.sqlQuery

    def _generate_dic_for_template_sub(self):
        dictForTemplate = { 
            'startDate' : self.dateRange.startDate,
            'endDate' : self.dateRange.endDate,              
            'curveShortNames' : str(self.curveShortNames),
            'serverLocation' : self.serverLocation }
        return dictForTemplate

    def get_server_location(self):
        return self.serverLocation

class DerivedCurve(CurvePlot):
    """ DerivedCurve knows how to generate queries and dataframes of underling timeseries for derived curves
    Examples:
        dc = DerivedCurve('SE.SEK.GVT.ZC')
        fig, ax = dc.plot_2d('2013-07-01', gca())
    """

    def __init__(self, curveShortName, name=None, dirName='.', finalTable=False):
        self.curveShortName = CurveShortName(curveShortName, finalTable=finalTable)
        self.finalTable = finalTable
        self.name = name if (name is not None) else str(self.curveShortName)
        self.cntry = self.curveShortName.get_country()
        self.fileRoot = os.path.join(dirName, 'swap-data-'+self.cntry) if (dirName is not None) else None
        self.dataDir = dirName
        self.plotImpl = lambda axes, x, y: axes.plot(x, y, label=self.name)
        self.sqlTemplate = r"""
        Declare @StartDate datetime = '{0}', @EndDate datetime = '{1}'
        select cv.CurveShortName, cv.CurveId, cq.TradeDate, te.InYears, cq.Quote
        from MarketData.dbo.curve cv
        join MarketData.dbo.curvenodes cn on cv.curveid = cn.curveid
        join MarketData.dbo.curvenodequote{3} cq on cn.curvenodeid = cq.curvenodeid
        join MarketData.dbo.TenorEnum te on te.TenorEnum = cn.TenorEnum
        where cv.curveshortname in ({2}) and cq.tradedate >= @StartDate and cq.tradedate <= @EndDate
        order by cq.tradedate, InYears asc"""        
        self.sqlTemplateTm1 = r"""
        Declare @StartDate datetime = '{0}', @EndDate datetime = '{1}'
        select cv.CurveShortName, cv.CurveId, cq.TradeDate, te.InYears, cq.Quote
        from MarketData.dbo.curve cv
        join MarketData.dbo.curvenodes cn on cv.curveid = cn.curveid
        join MarketData.dbo.curvenodequote{3} cq on cn.curvenodeid = cq.curvenodeid
        join MarketData.dbo.TenorEnum te on te.TenorEnum = cn.TenorEnum
        where cv.curveshortname in ({2}) and cq.tradedate = (SELECT Max(TradeDate)
                    FROM MarketData.dbo.curvenodequote{3}
                    where TradeDate < @EndDate and CurveShortName in ({2}))
        order by cq.tradedate, InYears asc"""  
        self.sqlTempSwap = r"""
        Declare @StartDate datetime = '{0}', @EndDate datetime = '{1}'
        Declare @countryCode varchar(2) = '{2}'
        SELECT hist.date AS 'TradeDate', hist.term AS 'InYears', hist.spot/100 AS 'Quote'
        from ejv_rigs..curves c
        join ejv_ejv_common..analytic_yield_curve_hist hist on hist.curve_id = c.curve_id 
        where c.[curve_cd] = 'SW' 
        and c.[curve_sub_cd] = 'INRS'
        and c.cntry_cd = @countryCode
        and c.ayc_fl = 'y'
        and c.status_cd = 'AOK'
        and hist.date >= @StartDate and hist.date <= @EndDate
        order by hist.date, hist.term asc
        """
        self.sqlTempSwapTm1 = r"""
        Declare @StartDate datetime = '{0}', @EndDate datetime = '{1}'
        Declare @countryCode varchar(2) = '{2}'
        SELECT hist.term AS 'InYears', hist.spot/100 AS 'Quote'
        from ejv_rigs..curves c
        join ejv_ejv_common..analytic_yield_curve_hist hist on hist.curve_id = c.curve_id 
        where c.[curve_cd] = 'SW' 
        and c.[curve_sub_cd] = 'INRS'
        and c.cntry_cd = @countryCode
        and c.ayc_fl = 'y'
        and c.status_cd = 'AOK'
        and hist.date = (select Max(date) from ejv_ejv_common..analytic_yield_curve_hist
            where date < @EndDate and curve_id = c.curve_id)
        order by hist.term
        """
    def plot_2d(self, axespread, axes, dateRange, server, maxYears=50):
        #self.finalTable = finalTable
        self.plotImpl = lambda axes, x, y: axes.plot(x, y, label=self.name, linestyle='solid', color='black')
        CurvePlot.plot_2d(self, axespread, axes, dateRange, server, maxYears)
        self.name += '_T-1'
        self.nameSave = self.name
        self.curveId = 0
        self.linfitTm1 = None
        self.linfit = None
        
        if (len(self.df) > 0):
            self.curveId = self.df['CurveId'][0]
        self.gvtzcDf = self.df
        
        self.name = self.nameSave
        self.plotImpl = lambda axes, x, y: axes.plot(x, y, label=self.name, linestyle='dashed')
        self.sqlTemplate = self.sqlTemplateTm1
        CurvePlot.plot_2d(self, axespread, axes, dateRange, server, maxYears)
        self.gvtzcDfTm1 = self.df
        if (len(self.df) > 0):
            self.curveId = self.df['CurveId'][0]
        
        self.name = 'Swap'
        self.plotImpl = lambda axes, x, y: axes.plot(x, y, label=self.name, linestyle='dashdot')
        self.sqlTemplate = self.sqlTempSwap
        CurvePlot.plot_2d(self, axespread, axes, dateRange, server, maxYears)

        if (len(self.gvtzcDf) > 0 and (self.linfit is not None)):
            self.gvtzcDf['Swap']=self.linfit(self.gvtzcDf['InYears']) #fit data
            self.gvtzcDf['Spread']=self.gvtzcDf['Swap'] - self.gvtzcDf['Quote']
            axespread.plot(self.gvtzcDf['InYears'], self.gvtzcDf['Spread'], label='Spread', linestyle='solid')
            axespread.set_ylabel('Spread')
            #for tl in axespread.get_yticklabels():
                #tl.set_color('r')
        #print self.gvtzcDf
        
        self.name = 'SwapT-1'
        self.plotImpl = lambda axes, x, y: axes.plot(x, y, label=self.name, linestyle='dotted')
        #self.sqlTemplate = self.sqlTempSwapTm1
        CurvePlot.plot_2d(self, axespread, axes, dateRange, server, maxYears)
        if (len(self.gvtzcDfTm1) > 0 and (self.linfitTm1 is not None)):
            self.gvtzcDfTm1['Swap']=self.linfitTm1(self.gvtzcDfTm1['InYears']) #fit data
            self.gvtzcDfTm1['Spread']=self.gvtzcDfTm1['Swap'] - self.gvtzcDfTm1['Quote']
            axespread.plot(self.gvtzcDfTm1['InYears'], self.gvtzcDfTm1['Spread'], label='SpreadT-1', linestyle='dashed', color='r')
            #print 'gvtzcDfTm1',self.gvtzcDfTm1
            
        if ((self.linfitTm1 is not None) and (self.linfit is not None) and (len(self.gvtzcDf) > 0) 
            and (len(self.gvtzcDfTm1) > 0) and (self.fileRoot is not None)):
            outputFile = self.dataDir + os.sep + 'swap-data-%s.%s'%(self.cntry, 'csv')
            self.gvtzcDf.pop('CurveShortName')
            self.gvtzcDf.pop('CurveId')
            self.gvtzcDf.pop('TradeDate')
            self.gvtzcDf.rename(columns={'InYears':'TimeToMaturity', 'Quote':'GVT.ZC'}, inplace=True)
            self.gvtzcDf['GVT.ZC_T-1'] = self.gvtzcDfTm1['Quote']
            self.gvtzcDf['Swap_T-1'] = self.gvtzcDfTm1['Swap']
            self.gvtzcDf['Spread_T-1'] = self.gvtzcDfTm1['Spread']
            #print self.gvtzcDf
            self.gvtzcDf.to_csv(outputFile, float_format='%.6f')
            #df.reindex_axis(sorted(df.columns), axis=1)
        
    def create_query(self):
        if (self.name == 'Swap'):
            self.sqlQuery = self.sqlTempSwap.format(self.dateRange.startDate, self.dateRange.endDate, self.cntry)
        elif (self.name == 'SwapT-1'):
            self.sqlQuery = self.sqlTempSwapTm1.format(self.dateRange.startDate, self.dateRange.endDate, self.cntry)
            #print self.sqlQuery
        else:
            curveNames = self.curveShortName.get_curve_names()
            if (self.finalTable):
                self.sqlQuery = self.sqlTemplate.format(self.dateRange.startDate, self.dateRange.endDate, curveNames, "final")
            else:
                self
                self.sqlQuery = self.sqlTemplate.format(self.dateRange.startDate, self.dateRange.endDate, curveNames, "")
            #print 'DerivedCurve', self.sqlQuery
        return self.sqlQuery    

class USValidationCurve(CurvePlot):
    """ 
    Examples:
        sb = USValidationCurve('Fed Yield Curve')
        ax = dc.plot_2d('2013-07-01')
    """
    
    def __init__(self, name=None,server="carme"):
        self.db = DBServer(server=server)        
        self.name = name if name else "Fed Yield"   
        self.curveShortName = CurveShortName('US.USD.GVT.ZC')
        self.name = name if (name is not None) else str(self.curveShortName)
        self.plotImpl = lambda axes, x, y: axes.scatter(x, y, label=self.name, s=50, marker='d')
        self.sqlTemplate = r"""
        Declare @StartDate datetime = '{0}', @EndDate datetime = '{1}'
        SELECT 'US.USD.GVT.ZC' AS CurveShortName, TradeDate, te.TenorEnum, te.InYears, Quote
        FROM
        (        SELECT TradeDate, TenorEnum, Quote
                FROM
        (
                SELECT    [Date] as TradeDate, [1 Mo] as [1M], [3 Mo] as [3M], [6 Mo] as [6M], [1 Yr] as [1Y], [2 Yr] as [2Y], [3 Yr] as [3Y],
                          [5 Yr] as [5Y], [7 Yr] as [7Y], [10 Yr] as [10Y], [20 Yr] as [20Y], [30 Yr] as [30Y]            
                FROM [VendorDataMaint].[dbo].[USTreasuryYieldCurveRates] yc
                WHERE [Date] >= @StartDate and [Date] <= @EndDate                
        ) AS cq
                UNPIVOT(Quote For TenorEnum 
                IN([1M], [3M], [6M], [1Y], [2Y], [3Y], [5Y], [7Y], [10Y], [20Y], [30Y]) ) AS up ) AS source
                join MarketData..TenorEnum te on source.TenorEnum = te.TenorEnum
                """            

class SourceBondCurve(CurvePlot):
    """ SourceBondCurve knows how to generate queries and dataframes of underling timeseries for sec info 
    Examples:
        sb = SourceBondCurve('US.USD.GVT.ZC',name='US-Filtered')
        ax = dc.plot_2d(gca(),,'2013-07-01')
    """
    def __init__(self, curveShortName, name=None, dirName='.'):
        self.curveShortName = CurveShortName(curveShortName)
        self.country = self.curveShortName.get_country()
        self.name = name if name else "Source Bond " + self.country #graph lable name
        self.fileRoot = os.path.join(dirName, 'source-bond-data-'+self.country) if (dirName is not None) else None
        self.dataDir = dirName
        self.sqlTemplate = r"""
                Declare @StartDate datetime = '{0}', @EndDate datetime = '{1}'
                select CurveShortname, TradeDate, fb.InstrCode, InYears = DateDiff(d,TradeDate,MatDate)/365.25, ItemValue/100.0 AS [Quote]
                from marketdata.dbo.DerivCurveFilteredBond fb
                join MarketData.dbo.Curve cv on cv.CurveId = fb.CurveId
                join QAI.dbo.FIEJVSecInfo sec on fb.InstrCode = sec.InstrCode
                where cv.CurveShortName in ({2})
                and fb.ItemId = 2
                and fb.TradeDate >= @StartDate 
                and fb.TradeDate <= @EndDate
                order by MatDate
                """        
        self.sqlTemplateFull = r"""
                Declare @StartDate datetime = '{0}', @EndDate datetime = '{1}'
               select fb.CurveShortname, fb.TradeDate, isnull(fb.InstrCode, fby.InstrCode) AS [InstrCode], isnull(fb.InYears, fby.InYears) as [InYears], isNull(fb.Price, fby.Price) as [Price], fb.Quote, fby.Quote AS [Yield_T-1], 
                    isnull(fb.ISIN, fby.ISIN) AS [ISIN], isnull(fb.Description, fby.Description) AS [Description], isnull(fb.MatDate, fby.MatDate) AS [MatDate], fb.CurveId
               from (select fbo.*, CurveShortname, InYears = DateDiff(d,fbo.TradeDate,MatDate)/365.25, fbPrice.ItemValue AS [Price], fbo.ItemValue/100.0 AS [Quote], id.Value_ AS [ISIN], sec.issName AS [Description], sec.MatDate
                    from [MarketData].[dbo].[DerivCurveFilteredBond] fbo
                    join marketdata.dbo.DerivCurveFilteredBond fbPrice 
                        on fbo.CurveId = fbPrice.CurveId and fbo.InstrCode = fbPrice.InstrCode and fbo.TradeDate = fbPrice.TradeDate
                    join MarketData.dbo.Curve cv on cv.CurveId = fbo.CurveId
                    join QAI.dbo.FIEJVSecInfo sec on fbo.InstrCode = sec.InstrCode
                    left join 
                        (select * from QAI.dbo.FIEJVSecIdent 
                        where EndDate   is null 
                        and Item       = 35  
                        and SeqNum     = 1
                        ) id on id.InstrCode = fbo.InstrCode
                    where cv.CurveShortName in ({2})
                    and fbo.ItemId    = 2
                    and fbo.TradeDate = @EndDate    
                    and fbPrice.ItemId = 1
                       
               ) fb   
               full join 
                (select fbo.*, CurveShortname, InYears = DateDiff(d,fbo.TradeDate,MatDate)/365.25, fbPrice.ItemValue AS [Price], fbo.ItemValue/100.0 AS [Quote], id.Value_ AS [ISIN], sec.issName AS [Description], sec.MatDate
                    from [MarketData].[dbo].[DerivCurveFilteredBond] fbo
                    join marketdata.dbo.DerivCurveFilteredBond fbPrice 
                        on fbo.CurveId = fbPrice.CurveId and fbo.InstrCode = fbPrice.InstrCode and fbo.TradeDate = fbPrice.TradeDate
                    join MarketData.dbo.Curve cv on cv.CurveId = fbo.CurveId
                    join QAI.dbo.FIEJVSecInfo sec on fbo.InstrCode = sec.InstrCode
                    left join (select * from QAI.dbo.FIEJVSecIdent 
                        where EndDate   is null 
                        and Item       = 35  
                        and SeqNum     = 1
                        ) id on id.InstrCode = fbo.InstrCode
                    where cv.CurveShortName in ({2})
                    and fbo.ItemId    = 2
                    and fbo.TradeDate = (select max(TradeDate) from [MarketData].[dbo].[DerivCurveFilteredBond]
                                        where CurveId = fbo.CurveId
                                        and ItemId    = 2
                                        and TradeDate < @EndDate
                                        )      
                    and fbPrice.ItemId = 1  
               ) fby 
                on  fby.CurveId    = fb.CurveId
                and fby.InstrCode  = fb.InstrCode
                and fby.ItemId     = fb.ItemId
                order by MatDate
                """        
    def create_query(self):
        curveNames = self.curveShortName.get_curve_names()
        if (self.fileRoot is None):
            self.sqlQuery = self.sqlTemplate.format(self.dateRange.startDate, self.dateRange.endDate, curveNames)
        else:
            self.sqlQuery = self.sqlTemplateFull.format(self.dateRange.startDate, self.dateRange.endDate, curveNames)
        #print self.sqlQuery
        return self.sqlQuery    
    def plotImpl(self, axes, x, y):
        try:
            axes.scatter(x, y, label=self.name, marker='x', s=200, color='black')
        except:
            print "exception while scatter source bond"
        logging.info( "Saving source bond table : {0}.csv".format(self.fileRoot))
        if (self.fileRoot is not None):
            self.df.pop('CurveShortname')
            self.df.pop('TradeDate')
            self.df.rename(columns={'InYears':'TimeToMaturity', 'Quote':'Yield'}, inplace=True)
            #print self.df.columns
            self.df['MatDate']=self.df['MatDate'].apply(lambda x: x.date())
            #self.df.pop('MatDate')
            outputFile = self.dataDir + os.sep + 'source-bond-data-%s.%s'%(self.country, 'csv')
            #Add swap and spread columns
            #print self.df
            
            self.df.to_csv(outputFile, float_format='%.6f')
            #self.df.to_html(self.fileRoot+'.html')
class RigsAnalyticGovtCurve(CurvePlot):
    """ RigsAnalyticGovtCurve knows how to generate queries and dataframes of underling Rigs tables 
    Examples:
        sb = RigsAnalyticGovtCurve('TR-Filtered','US.USD.GVT.ZC')
        ax = dc.plot(gca(),'2013-07-01')
    """
    def __init__(self, curveShortName, name=None):
        self.curveShortName = CurveShortName(curveShortName)
        self.name = name if name else "Rigs Analytic " + self.curveShortName.get_country()
        self.plotImpl = lambda axes, x, y: axes.plot(x, y , label=self.name, c='r', linewidth=2,linestyle='-.')
        self.sqlTemplate = r"""
        DECLARE @StartDate datetime = '{0}', @EndDate datetime = '{1}', @CountryCode varchar(2) = '{2}'
        SELECT date AS TradeDate, term AS InYears, spot/100.0 AS Quote
        from ejv_rigs..curves c
        join ejv_ejv_common..analytic_yield_curve_hist hist on hist.curve_id = c.curve_id
        WHERE 
        c.[curve_sub_cd] = 'GVBM' and
        c.cntry_cd = @CountryCode and
        date >= @StartDate and
        date <= @EndDate
        """ 
        self.sqlTemplateMX = r"""
        DECLARE @StartDate datetime = '{0}', @EndDate datetime = '{1}', @CountryCode varchar(2) = '{2}'
        SELECT date AS TradeDate, term AS InYears, spot/100.0 AS Quote
        from ejv_rigs..curves c
        join ejv_ejv_common..analytic_yield_curve_hist hist on hist.curve_id = c.curve_id
        WHERE 
        c.[curve_sub_cd] = 'GVBM' and
        c.cntry_primary_fl = 'y' and
        c.cntry_cd = @CountryCode and
        date >= @StartDate and
        date <= @EndDate
        """ 

    def create_query(self):
        cntry = self.curveShortName.get_country()
        if (cntry == "MX" or cntry == "CN" or cntry == "CL") :
            self.sqlQuery = self.sqlTemplateMX.format(self.dateRange.startDate, self.dateRange.endDate, cntry)
        else :
            self.sqlQuery = self.sqlTemplate.format(self.dateRange.startDate, self.dateRange.endDate, cntry)
        return self.sqlQuery          


class RigsAnalyticSwapCurve(CurvePlot):
    """ RigsAnalyticSwapCurve knows how to generate queries and dataframes of underling Rigs tables 
    Examples:
        sb = RigsAnalyticSwapCurve('TR-Filtered','US.USD.SWP.ZC')
        ax = dc.plot(gca(),'2013-07-01')
    """
    def __init__(self, curveShortName, name=None):
        self.curveShortName = CurveShortName(curveShortName)
        self.name = name if name else "Rigs Analytic " + self.curveShortName.get_country()
        self.plotImpl = lambda axes, x, y: axes.plot(x, y , label=self.name, c='r', linewidth=2,linestyle='-.')
        self.sqlTemplate = r"""
        DECLARE @StartDate datetime = '{0}', @EndDate datetime = '{1}', @CountryCode varchar(2) = '{2}'
        SELECT date AS TradeDate, term AS InYears, spot/100.0 AS Quote
        from ejv_rigs..curves c
        join ejv_ejv_common..analytic_yield_curve_hist hist on hist.curve_id = c.curve_id
        WHERE 
        c.[curve_sub_cd] = 'INRS' and
        c.cntry_cd = @CountryCode and
        date >= @StartDate and
        date <= @EndDate
        """ 

    def create_query(self):
        cntry = self.curveShortName.get_country()
        self.sqlQuery = self.sqlTemplate.format(self.dateRange.startDate, self.dateRange.endDate, cntry)
        return self.sqlQuery         

class DBServer:
    """Contains server specific configuraiton e.g. server name and connection and interacts with SQLServer instance"""
    def __init__(self, server, dbName='MarketData', auth='Trusted_Connection=yes', user='MarketDataLoader', password='mdl1234'):
        self.server = server
        self.dbName = dbName
        self.user = user
        self.password = password

    def read_frame(self, sqlQuery):
        self.cnxn = pymssql.connect(user=self.user, password=self.password, database=self.dbName, host=self.server)
        sqlQuery = sqlQuery
        df = sql.read_sql(sqlQuery, self.cnxn)
        self.cnxn.close()
        return df

class DateRange:
    """ Makes date range manipulation easier
        dr = DateRange('2013-07-01:2013-07-03')
        startDate = dr.startDate
        """

    def __init__(self,dateRepr):
        self.dateRepr=dateRepr
        self.startDate, self.endDate = dateRepr.split(':') if ':' in dateRepr else (dateRepr,dateRepr)

    yearSeqStr =r"""
        2013-01-01:2013-12-31
        2012-01-01:2012-12-31
        2011-01-01:2011-12-31
        2010-01-01:2010-12-31
        2009-01-01:2009-12-31
        2008-01-01:2008-12-31        
        2007-01-01:2007-12-31                
        2006-01-01:2006-12-31                        
        2005-01-01:2005-12-31    
        2004-01-01:2004-12-31            
        """

    @staticmethod
    def range_years(fltr=':'):
        for y in [yr for yr in DateRange.yearSeqStr.split() for f in fltr.split('|') if f in yr]:
            yield y

    @staticmethod
    def range_history():
        return '2001-01-01:2013-12-31'

class Server:
    @staticmethod
    def servers():
        servers = ['Saturn','Elara']
        for v in servers:
            yield v
            


class CurveShortName:
    """Helps interact with CurveShortNames such as US.USD.GVT.ZC
       csn = CurveShortName('US.USD.GVT.ZC')
       fig, ax = csn.plot_2D('2013-07-01','saturn')
    
    """
    @staticmethod
    def create_govt_short_name(country,currency):
        return '{country}.{currency}.GVT.ZC'.format(country=country,currency=currency)

    def __init__(self,curveShortName, dirName='', finalTable=False):
        self.curveShortName = curveShortName
        self.dirName = dirName
        self.finalTable = finalTable

    def __str__(self):
        return self.curveShortName

    def __repr__(self):
        return self.curveShortName
   
    def get_curve_names(self):
        self.curveNames = ','.join(["'"+ x +"'" for x in self.curveShortName.split('|')])
        return self.curveNames

    def get_country(self):
        return self.curveShortName.split('.')[0]

    def create_derived_curve(self):    
        return DerivedCurve(self.curveShortName, dirName=self.dirName, finalTable=self.finalTable)
        
    def create_sourcebond_curve(self):    
        sb = SourceBondCurve(self.curveShortName, dirName=self.dirName)
        return sb

    def create_US_validation_curve(self):    
        vc = USValidationCurve(self.curveShortName)
        return vc

    def create_rigs_govt_curve(self):    
        c = RigsAnalyticGovtCurve(self.curveShortName)
        return c    

    def create_curves(self, server, plotSource=True):
        self.curves = []
        self.curves.append(self.create_derived_curve())
        if plotSource : 
            self.curves.append(self.create_sourcebond_curve())
        self.curves.append(self.create_rigs_govt_curve())
        #if (self.get_country().lower()=='us') : self.curves.append(USValidationCurve())
        return self.curves

    @staticmethod
    def quick_plot_2d(curveShortName, dateRange, server, plotSource=True, maxYears=50, dirName='.\\', finalTable=False):
        csn = CurveShortName(curveShortName, dirName, finalTable)
        fig, ax, cc = csn.plot_2d(dateRange, server, plotSource, maxYears = maxYears)
        return fig, ax, cc

    @classmethod
    def plot_g7_2013(cls, dateRange, server, plotSource=True):
        return [cls.plot_2d(csn,dateRange,server, plotSource) for csn in cls.g7()]

    @staticmethod
    def quick_hist(curveShortName, dateRange, server, finalTable=False):
        dc = DerivedCurve(curveShortName, finalTable=finalTable)
        ax = dc.plot_history(dateRange, server)

        return ax.figure, ax, dc
    
    @staticmethod
    def hist_comp(curveShortName, dateRange, sumDf, server):
        dc = DerivedCurve(curveShortName, finalTable=False)
        ax = dc.plot_history(dateRange, sumDf, server, dim='2dcomp')
        #ax.figure.show()
        return ax.figure, ax, dc
    
    @classmethod
    def quick_comp(cls, curveShortName, dateRange, serverBase, serverDev, plotSource=True, maxYears=50):
        fig, ax, cc = cls.quick_plot(curveShortName, dateRange, serverDev, plotSource,maxYears=maxYears)
        dc = DerivedCurve(curveShortName,name=serverBase+':'+curveShortName, finalTable=cls.finalTable)
        dc.plot_2d(ax,dateRange,server=serverBase)
        ax.legend(ncol=4,loc='lower center')       
        return fig, ax, cc        

    @classmethod
    def quick_plot(cls, curveShortName, dateRange, plot, server, plotSource=True, maxYears=50, dirName='.', finalTable=False):
        cls.finalTable=finalTable
        if(plot.lower()=='day'):
            fig, ax, cc = cls.quick_plot_2d(curveShortName, dateRange, server, plotSource,maxYears=maxYears, dirName=dirName, finalTable=finalTable)
            ax.legend(ncol=4,loc='lower center')           
            return fig, ax, cc
        elif(plot.lower()=='short'):
            fig, ax, cc = cls.quick_plot_2d(curveShortName, dateRange, server, plotSource,maxYears=maxYears, dirName=None, finalTable=finalTable)
            ax.legend(ncol=4,loc='lower center')           
            return fig, ax, cc
        elif(plot.lower()=='histcomp'):
            fig, ax, dc = cls.hist_comp(curveShortName, dateRange, server)
            return fig, ax, dc
        else: 
            year, month, day = dateRange.split('-')
            endDate = date(int(year), int(month), int(day))
            startDate = endDate - timedelta(days=60) 
            dateRange = "{0}:{1}".format(startDate.isoformat(), endDate.isoformat())
            fig, ax, dc = cls.quick_hist(curveShortName, dateRange, server, finalTable=finalTable)

            return fig, ax, dc
    
    def plot_2d(self, dateRange, server, plotSource=True, maxYears=60):

        dr = DateRange(dateRange)
        fig = plt.figure(figsize=(9,10))            
        curves = self.create_curves(plotSource)
        xlims = [1]
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        for cv in curves:
            cv.plot_2d(ax2, ax, dateRange, server, maxYears=maxYears)
            xlims.append(cv.maxterm)
        ax.set_title('Server: '+ server + '\n TradeDate: ' +  dr.startDate +'    Curve: ' + str(self.curveShortName) + ' ' + str(curves[0].curveId))
        ax.set_xlabel('Term In Years')
        ax.set_ylabel('Level (abs)')
        #print xlims
        xLowerBound = 0
        xUpperBound = min(max(xlims), maxYears)
        #print 'xbound',ax.get_xbound(),' xUpperBound ',xUpperBound
        yBound = ax.get_ybound()
        yUpperBound = yBound[1]
        if(yUpperBound==1.0): yUpperBound = .1
        #print yBound
        if (yBound[0] < 0):
            yLowerBound = yBound[0]
        else:
            yRange = yBound[1] - yBound[0]
            yLowerBound = ax.get_ybound()[0] -.08 * yRange
        
        ax.set_xlim([xLowerBound,xUpperBound])
        ax.set_ylim([yLowerBound,yUpperBound])        
        
        ax.axhline(c='black')
        ax.grid(True)
        ax2.set_xlim([xLowerBound,xUpperBound])
        ax2.grid(True)
        ax2.legend(ncol=1,loc='best')
        plt.tight_layout()        

        return fig, ax, curves

    g7p1Str = r"""
    US.USD.GVT.ZC
    GB.GBP.GVT.ZC
    EP.EUR.GVT.ZC
    CA.CAD.GVT.ZC
    AU.AUD.GVT.ZC
    JP.JPY.GVT.ZC
    CH.CHF.GVT.ZC
    SE.SEK.GVT.ZC
    NO.NOK.GVT.ZC
            """                                

    @classmethod
    def comp_hist(cls, yearFilter=':', curveFilter='.'):
        for c in [curve for curve in cls.g7() for cf in curveFilter.split('|') if cf in curve]:        
            for y in [ry for ry in DateRange.range_years() for yf in yearFilter.split('|') if yf in ry]:
                for sv in Server.servers():
                    cls.quick_hist(c, y, server=sv)
    @classmethod
    def g7(cls):
        return [g7 for g7 in cls.g7p1Str.split()]

    @classmethod
    def plot_history(cls, curveShortName, server):
        return cls.quick_hist(curveShortName,'2000-01-01:2013-12-31',server)
    
    @classmethod
    def plot_g7_history(cls, server):
        return [cls.plot_history(g, server) for g in cls.g7()]

    @classmethod
    def plot_g7_asOf(cls, dateRange, server, maxYears=50):
        return [cls.quick_comp(g, dateRange, server,maxYears=maxYears) for g in cls.g7()]    

def plotiqrbounds(dataframe):
    from matplotlib.ticker import FuncFormatter
    fig = plt.figure(figsize=(16,10), )
    axret = fig.add_subplot(312)
    axret.plot(dataframe.index, dataframe.upper, color="#e31a1c")
    axret.plot(dataframe.index, dataframe.upperMad, color="#ffa500")
    axret.scatter(dataframe.index, dataframe.returns, color="#377eb8", s=5)
    axret.plot(dataframe.index, dataframe.lower, color="#e31a1c")
    axret.plot(dataframe.index, dataframe.lowerMad, color="#ffa500")
    axret.set_title('Returns')
    def to_percent(y, position):
        s = str(100 * y)
        if matplotlib.rcParams['text.usetex'] == True:
            return s + r'$\%$'
        else:
            return s + ''
    axret.yaxis.set_major_formatter(FuncFormatter(to_percent))
    outliers = dataframe[dataframe.outliers.notnull()]
    if(len(outliers) > 0):
        axret.scatter(outliers.index, outliers.returns, facecolors='#A60628', edgecolors='#7A68A6', lw=1, s=100, alpha=0.2)
    axlevel = fig.add_subplot(311,sharex=axret)
    axlevel.set_title('Levels')
    axlevel.scatter(dataframe.index, dataframe.level, color="#377eb8", s=5)
    axlevel.plot(dataframe.index, dataframe.level, color="#377eb8")
    axlevel.yaxis.set_major_formatter(FuncFormatter(to_percent))
    if(len(outliers) > 0):
        axlevel.scatter(outliers.index, outliers.level, facecolors='#A60628', edgecolors='#7A68A6', lw=1, s=100, alpha=0.2)
    outlierCount = 0
    axdates = fig.add_subplot(313,sharex=axret, sharey=axret)
    axdates.set_title('Date Detail')
    axdates.yaxis.set_major_formatter(FuncFormatter(to_percent))
    if(len(outliers) > 0):
        axdates.scatter(outliers.index, outliers.returns, facecolors='#A60628', edgecolors='#7A68A6', lw=1, s=100, alpha=0.2)
        outlierCount = 0
        for x in outliers.ix[np.where(outliers['outliers'])[0]].index:
            outlierCount = outlierCount + 1
            label = x.strftime('%m/%d/%y')
            axdates.annotate(label, xy=(x, outliers['returns'].ix[x]), xytext=(x, outliers['returns'].ix[x]), horizontalalignment='left', verticalalignment='top')
    #plt.show()
    return axdates, dataframe[dataframe.outliers.notnull()==True].ix[:,0]

def generateBondReturn(shortName, dateRange, server='Saturn'):
    #dataclient =  
    #df = dataclient.read_frame(SourceBondCurve(shortName, dateRange, server))
    #df.index = pd.MultiIndex.from_arrays([df.ix[:,1],df.ix[:,2]])
    #return df
    return None

def generateHist(shortName, dateRange, server='Saturn', finalTable=False):
    dc = DerivedCurve(shortName, finalTable=finalTable)
    df = dc.get_history_df(dateRange, server)
    return formateRawDataFrame(df)
    
def formateRawDataFrame(df):
    df.index = pd.MultiIndex.from_arrays([df.CurveShortName.values, df.TradeDate.values, df.InYears.values])
    df = df.Quote
    df = df.unstack()
    df.index = pd.DatetimeIndex(zip(*df.index.values)[1])
    return df

def generateOutlierParams(df, bound=12):
    endTermScorePenalty = 3
    #score = distance / mad, if first or last line, score *= endTermScorePenalty
    indexMap = { df.columns.tolist().index(x) : x for x in df.columns }
    indexMapNodes = { x : indexMap[x] for x in indexMap.iterkeys() if indexMap.has_key(x) }
    outlierMap = { x : pd.DataFrame(df.ix[:, indexMapNodes[x]], index=df.index) for x in indexMapNodes.keys() }
    firstIdx = indexMapNodes.keys()[0]
    lastIdx = indexMapNodes.keys()[-1]
    outlierMapBds = { k : addlevelboundsSingle(v) for k, v in outlierMap.iteritems() }
    outlierMapBds[firstIdx].score *= endTermScorePenalty
    outlierMapBds[lastIdx].score *= endTermScorePenalty
    #outlierDates = { k : v.outliers[v.outliers==True] for k, v in outlierMapBds.iteritems() }
    outlierParams = { k : {'term': indexMapNodes[k], 'data':v[v.score>bound], 'count':v[v.score>bound].score.count(), 'score':v[v.score>bound].score.sum()} for k, v in outlierMapBds.iteritems() }
    
    return outlierParams

def plotOutlierDetail(shortName, dateRange, server):
    dfh = generateHist(shortName, dateRange, server)
    outlierParams = generateOutlierParams(dfh)
    df=pd.DataFrame(outlierParams, index=['term','count','score']).T

    print 'Total Outlier count ',df['count'].sum(),' Total Score ',df['score'].sum()
    
def mad(a, axis=None):
    """Compute *Median Absolute Deviation* of an array along given axis."""
    med = np.median(a, axis=axis)                # Median along given axis
    if axis is None:
        umed = med                              # med is a scalar
    else:
        umed = np.expand_dims(med, axis)         # Bring back the vanished axis
    mad = np.median(np.absolute(a - umed), axis=axis) # MAD along given axis

    return mad
def addlevelboundsSingle(df, window=60):
    minMad = 0.00005
    df.columns=['level']
    df['returns'] = df.diff()
    df['median'] = pd.rolling_median(df['returns'], window=window, min_periods=1)
    df['mad'] = pd.rolling_median(np.absolute(df['returns']-df['median']), window=window, min_periods=1)
    #df['upperMad'] = bound*df['mad'] + df['median']
    #df['lowerMad'] = -bound*df['mad'] + df['median']
    #df['outliers'] = df[np.logical_or(df['returns'] > df['upperMad'], df['returns'] < df['lowerMad'])].returns.notnull()
    df.ix[df['mad']<minMad, 'mad'] = minMad
    df['score'] = np.absolute(df['returns'] - df['median']) / df['mad']
    df.pop('returns')
    df.pop('median')
    df.pop('mad')
    return df

def addlevelbounds(dataframe, column=7, bound=10, window=60):
    
    df = pd.DataFrame(dataframe.ix[:, column], index=dataframe.index)#, columns=['level'])
    df.columns=['level']
    print dataframe, df
    df['returns'] = df.diff()
    df['median'] = pd.rolling_median(df['returns'], window=window, min_periods=1)
    df['iqr'] = pd.rolling_quantile(df['returns'], window=window, min_periods=1, quantile=.75) - pd.rolling_quantile(df['returns'], window=window, min_periods=1, quantile=.25)
    df['mad'] = pd.rolling_median(np.absolute(df['returns']-df['median']), window=window, min_periods=1)
    df['upper'] = bound*df['iqr'] + df['median']
    df['lower'] = -bound*df['iqr'] + df['median']
    df['upperMad'] = bound*1.5*df['mad'] + df['median']
    df['lowerMad'] = -bound*1.5*df['mad'] + df['median']
    #df['outliers'] = df[np.logical_or(df['returns'] > df['upper'], df['returns'] < df['lower'])].returns.notnull()
    df['outliers'] = df[np.logical_or(df['returns'] > df['upperMad'], df['returns'] < df['lowerMad'])].returns.notnull()
    df['score'] = np.absolute(df['returns'] - df['median']) / df['mad']
    return df

def G9hist(dateRange, inputfile, server='prod_mac_mkt_db', reportFile='G9History.pdf'):
    G9df = pd.read_csv(inputfile, sep='|')
    outputFileNames = ['G9hist.pdf', 'G9Summary.pdf', reportFile]
    pp = PdfPages(outputFileNames[0])
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['axes.titlesize'] = 'medium'
    matplotlib.rcParams['axes.labelsize'] = 'xx-small'
    matplotlib.rcParams['xtick.labelsize'] = 'xx-small'
    matplotlib.rcParams['ytick.labelsize'] = 'xx-small'
    matplotlib.rcParams['lines.linewidth'] = 0.1
    matplotlib.rcParams['patch.linewidth'] = 0.1
    sumDfList = []
    finalNames = []
    for idx, shortName in G9df.iterrows():
        if (shortName[0] == 'CurveShortName' or shortName[0].find("#") == 0 ):
            continue
        print idx, shortName[0]
        finalNames.append(shortName[0])
        #fig, ax, dc = CurveShortName.hist_comp(shortName[0],'2004-12-01:2005-02-01', server='prod_mac_mkt_db')
        fig, _,_ = CurveShortName.hist_comp(shortName[0], dateRange, sumDfList, server)
        fig.savefig(pp, format='pdf')
        plt.close(fig)
    pp.close()
    if (len(sumDfList) > 0):
        tblDataList = []
        titleColumn = ['CurveShortName','Proposed\nScore','Baseline\nScore','Change\nScore',
                       'Proposed\nCount','Baseline\nCount','Change\nCount']
        tblDataList.append(titleColumn)
        sumDf = None
        colorMap = []
        rowOffset = 2
        idx = 0
        for shortName in finalNames:
            item = np.round(sumDfList[idx], decimals=1)
            if (sumDf is None):
                sumDf = item
            else:
                sumDf += item
            outList = [shortName]
            for x1 in list(item.stack()):
                outList.append( "{:,}".format(x1) )
            #print item
            changedScore = item.loc['Score','Change']
            if (changedScore > 0):
                colorMap.append(('TEXTCOLOR', (3, rowOffset+idx), (3, rowOffset+idx), colors.red))
            if (changedScore < 0):
                colorMap.append(('TEXTCOLOR', (3, rowOffset+idx), (3, rowOffset+idx), colors.blue))
            
            tblDataList.append(outList)
            idx +=1
        totalList = ['TOTAL']
        for x1 in list(sumDf.stack()):
            totalList.append( "{:,}".format(x1) )
        tblDataList.insert(1, totalList)
        
        doc = SimpleDocTemplate(outputFileNames[1], pagesize=letter)
        # container for the 'Flowable' objects
        elements = []
        styles = getSampleStyleSheet()
        styleH = styles['Heading1']
        styleN = styles['Code']
        
        tsText = 'Server: ' + server + ' Date range: ' + dateRange + ' TimeStamp: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        titleText = 'Derived Curve Quality Report'
        elements.append(Paragraph(tsText, styleN))
        elements.append(Paragraph(titleText, styleH))
        #colWidths=5*[0.4*reportlab.lib.units.inch]
        tbl=Table(tblDataList, repeatRows = 1)
        tblStyleList = [('ROWBACKGROUNDS', (0,0),(-1,-1), [colors.lightgrey, colors.white]),
                                 ('ALIGNMENT', (1,1),(-1,-1), 'DECIMAL'), 
                                 ('LINEABOVE',(0,2),(-1,2),1,colors.blue),
                                 ('FONTSIZE', (0,0),(-1,0), 11),
                                 ('FONTSIZE', (0,1),(-1,1), 10),
                                 ('FONTSIZE', (0,2),(-1,-1), 9)]
        tblStyleList.extend(colorMap)
        tbl.setStyle(TableStyle(tblStyleList))
        elements.append(tbl)
        
        # write the document to disk
        doc.build(elements)
        
        from PyPDF2 import PdfFileMerger, PdfFileReader
        merger = PdfFileMerger()
        for filename in reversed(outputFileNames[:-1]):
            merger.append(PdfFileReader(file(filename, 'rb')))

        merger.write(outputFileNames[-1])
            #rowList = [item.iloc[]]
        
        '''sumFig, sumAxe = plt.subplots(1, 1)
        sumDfList[0]['ChangePercent'] = np.round(sumDfList[0].Change/sumDfList[0].Baseline*100, decimals=2)
        print 'G9 Summary', sumDfList[0]
        sumDfList[0]['ChangePercent'].plot(ax=sumAxe, kind='bar')
        rw=0.15
        tableChange = sumAxe.table(cellText=np.round(np.array(sumDfList[0]), decimals=1),
                                   rowLabels=['Score','Count'],
                                   #rowColours=colors,
                                   colWidths = [rw, rw, rw, rw],
                                   colLabels = np.array(sumDfList[0].columns),
                                   loc='top', bbox=[0.11,1.01,.8,.2])
        #tableChange.scale(.2,1)
        tableChange.set_fontsize(5)
        txtt1 = sumAxe.text(.5, 1.3, 'Outlier Summary', va = 'top', ha = 'center', transform=sumAxe.transAxes, fontsize=8)

        #sumAxe.set_title('Outlier Summary')
        plt.subplots_adjust(hspace=0.1, bottom = 0.2, top=0.7)
        
        pp.savefig(sumFig)
        #plt.show()'''

def main():
    print "Running: " + os.path.abspath(os.curdir) + "\\CurveControl.py"
    parser = OptionParser()
    parser.add_option("-f", "--outputFile", dest="filename", help="write graph to file. Example: -f graph.jpg")
    parser.add_option("-i", "--inputFile", dest="inputFileName", help="write graph to file. Example: -i .txt")    
    parser.add_option("-d", "--date", dest="endDate", help="As of date. Example: -d 2013-09-20", metavar="DATE")
    parser.add_option("-c", "--curveShortName", dest="curveshortname", help="Example: -c US.USD.GVT.ZC")
    parser.add_option("-s", "--server", dest="server", help="Example: -s prod_mac_mkt_db")   
    parser.add_option("-p", "--plot", dest="plot", help="Example: -p [day|hist|outlier]", default='day')    
    parser.add_option("-r", "--dateRangeFocus", dest="dateRange2", help="Example: -r 2001-01-01:2014-05-01", default='')   
    parser.add_option("-t", "--test", dest="test", help="Runs unit tests and exits. Example:-t On", default="Off")
    parser.add_option("-y", "--historyComp", action="store_true", dest="histComp", help="History report between CurveNodeQuote and CurveNodeQuote Final. Example:-t On", default=False)

    (options, _ ) = parser.parse_args()

    print "outputFile: {0} \ndate: {1}\ncurveShortName: {2}\nserver: {3}\nplot: {4}\ndateRange: {5}".format(options.filename, options.endDate, options.curveshortname, options.server, options.plot, options.dateRange2)
    if options.server is None or options.filename is None:
        raise AssertionError()
    #plotiqrbounds(addlevelbounds(generateHist('SE.SEK.GVT.ZC','2005-02-13:2006-04-19', server='saturn'),column=7, bound=6).ix['2006-02-13':'2006-04-19',])
    #plotiqrbounds(addlevelbounds(generateHist('SE.SEK.GVT.ZC','2006-04-13:2006-04-18', server='saturn'),column=7, bound=6).ix['2005-01-01':'2007-05-01',])
    #plotiqrbounds(addlevelbounds(generateHist('CH.CHF.GVT.ZC','2001-01-01:2014-05-01', server='saturn'),column=5, bound=6).ix['2001-02-05':'2014-05-01',])
    #plotOutlierDetail('CH.CHF.GVT.ZC','2001-01-01:2014-05-01', server='saturn')
    #-r '2000-01-01:2014-07-17' -s prod_mac_mkt_db -f histCompE.pdf
    if (options.histComp):
        G9hist(options.dateRange2, inputfile=options.inputFileName, server=options.server, reportFile=options.filename)
    #fig, ax, dc = CurveShortName.hist_comp('CH.CHF.GVT.ZC','2000-01-01:2014-07-01', server='prod_mac_mkt_db')
    '''
    from CurveControl import CurveShortName as csn
    results = csn.quick_plot(options.curveshortname, options.endDate, options.plot, options.server)
    fig = results[0]
    fig.savefig(options.filename)
    print "Saved {0} plot at {1}".format(options.plot, options.filename)
    '''
    
if __name__ == '__main__':
    main()
