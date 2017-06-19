import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import weighted as wq # this is from the package 'wquantiles', for calculating a weighted median.
import datetime
import CurveQueries as cq
import utils.Utilities as U
import curve_utility as cu
import os
import concurrent.futures
import argparse
import time

def get_current_time():
    return str(datetime.datetime.now())[0:-3]

#calling R from Python
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import macpy.utils.ngc_utils as u
import macpy.utils.ngc_queries as q
import visualization.ngc_plot as p

import logging

class SmoothSplineCurve(object):
    def __init__(self, *args, **kwargs):
        self.specific_curve = kwargs.pop('specific_curve')
        self.start_date = kwargs.pop('start_date')
        self.end_date = kwargs.pop('end_date')
        self.Currency = kwargs.pop('Currency')
        self.RiskEntityId = None
        self.IndustryGroup = kwargs.get('IndustryGroup', None)
        self.Market = kwargs.get('Market', None)
        self.SectorGroup = kwargs.get('SectorGroup', None)
        self.Sector = kwargs.get('Sector', None)
        self.Region = kwargs.get('Region', None)
        self.RegionGroup = kwargs.get('RegionGroup', None)
        self.PricingTier = kwargs.get('PricingTier', None)
        self.alpha = kwargs.get('alpha', 0.5)
        self.gamma = kwargs.get('gamma', 0.5)
        self.tenorGrid = kwargs.get('tenorGrid', np.array([0, 0.083333333333, 0.166666667, 0.25, 0.5, 0.75, 1, 1.25, 1.5,
                                    2, 2.5, 3, 3.5, 4, 5, 7, 10, 12, 15, 20, 25, 30, 40]))
        self.tenorGridEnum  = kwargs.get('tenorGridEnum',np.array(['0M', '1M', '2M', '3M', '6M', '9M', '1Y', '15M', '18M',
                                        '2Y', '30M', '3Y', '42M', '4Y', '5Y', '7Y', '10Y', '12Y', '15Y', '20Y', '25Y',
                                        '30Y', '40Y']))
        self.curveCol = kwargs.get('curveCol','CurveId')
        self.spreadCol = kwargs.get('spreadCol', 'logOAS')
        self.tenorCol = kwargs.get('tenorCol', 'Effective_Duration')
        self.weightCol = kwargs.get('weightCol', 'AmtOutstanding')
        self.smfitMinWeightTol = kwargs.get('smfitMinWeightTol', 0.0001)
        self.smfitLogxScale = kwargs.get('smfitLogxScale', True)
        self.smfitSpldf = kwargs.get('smfitSpldf', 4.) #degrees of freedom of smooth spline
        self.smfitLongCompressStart= kwargs.get('smfitLongCompressStart',0.1)
        self.smfitShortCompressStart=  kwargs.get('smfitShortCompressStart',0.1)
        self.smfitCompressionLong= kwargs.get('smfitCompressionLong',0.1)
        self.smfitCompressionShort= kwargs.get('smfitCompressionShort',0.1)
        self.normaliseWeightsByCurve = kwargs.get('normaliseWeightsByCurve',False)
        self.awDifferential = kwargs.get('awDifferential', 0.0) # compress asset weights to differential factor aw_differential = max(aw)/min(aw). Choose 1 for equalising all weights, np.inf or 0 for preserving current differential, e.g. 100 for max(aw)/min(aw) = 100
                                      # a compression factor afact will be calculated and applied to weights across *all* curves : aw = afact + (1 - afact) * aw/sum(aw)
        self.tenorWeightDensityBandwidth = kwargs.get('tenorWeightDensityBandwidth',0.2)
        self.sigmaScaleLevels = kwargs.get('sigmaScaleLevels', None)
        self.sigmaScaleChanges = kwargs.get('sigmaScaleChanges', None)
        self.numBondsLevels = kwargs.get('numBondsLevels', None)
        self.numBondsChanges = kwargs.get('numBondsChanges', None)
        self.numIssuers = kwargs.get('numIssuers', 1)
        self.levelsFit = kwargs.get('levelsFit', None)
        self.removeOutliers = kwargs.get('removeOutliers', True)
        self.outlierPctTolLevels = kwargs.get('outlierPctTolLevels', None) # absolute threshold below which an outlier is accepted (not marked as outlier). In mLog(alpha=0.01) space this is equivalent to a threshold in percent, e.g. 0.1 means outlier 10% off curve is accepted.
        self.outlierPctTolChanges = kwargs.get('outlierPctTolChanges', None)
        self.errTol = kwargs.get('errTol', None)
        self.maxIter = kwargs.get('maxIter', None)
        self.numOutlierIterations = kwargs.get('numOutlierIterations', None)
        self.numIterLevelGuess = kwargs.get('numIterLevelGuess', None)
        self.sswidthLong = kwargs.get('sswidthLong', None)
        self.sswidthShort = kwargs.get('sswidthShort', None)
        # plot instructions:
        self.debugplot = kwargs.get('debugplot',False)
        self.debugplot_ixlist = kwargs.get('debugplot_ixlist', [-1]) # plot at selected iteration indices eg. [0,2,5] or *** [-1] for final iteration ***
        self.debugplot_curve = kwargs.get('debugplot_curve', False) # None #'USD-UYGV-SEN' # None for default (specific_curve)
        self.debugplotSmoothFit = kwargs.get('debugplotSmoothFit', False)
        self.debugplotSmoothFit_ixlist = kwargs.get('debugplotSmoothFit_ixlist', None) # None sets it to = = debugplot_ixlist, or else provide list, e.g. [0, 5]
        self.plot_f = kwargs.get('plot_f', False)
        self.write_data = kwargs.get('write_data', True)
        self.plIndustries = None
        self.fitSingle = kwargs.get('fitSingle', False)

    def run(self):
        self.fit_history()
        return

    def SmoothFit(self,df, iteration=None, xout=None):
        #df is a dataframe with x, y, w, outlier
        #outliers have weight zero so should not impact the fit. They are included for charting: TODO: make sure low weights don't skew optimization
        xout = np.array(xout) if (xout is not None) else np.array(df.x)
        df = df.sort_values(by=['x'])

        x=np.array(df.x)
        y=np.array(df.y)
        w=np.array(df.w)
        indList=np.array(df.index)
        #print 'smoothfit len(x)', len(x)

        xCompressLong = (1-self.smfitLongCompressStart)*max(x)# np.ceil(max(x))
        xCompressShort = (1+self.smfitShortCompressStart)*min(x)
    #     lxinp = np.log(1+xinp) if (logx1) else xinp
        lxinp = u.logx1Compress(x, xCompressLong, xCompressShort, max(xout), min(xout), self.smfitCompressionLong, self.smfitCompressionShort, widthLong=self.sswidthLong, widthShort=self.sswidthShort) if bool(self.smfitLogxScale) else x
        lxout = u.logx1Compress(xout, xCompressLong, xCompressShort, max(xout), min(xout), self.smfitCompressionLong, self.smfitCompressionShort, widthLong=self.sswidthLong, widthShort=self.sswidthShort) if bool(self.smfitLogxScale) else xout
        #lx0 = logx1Compress(x0, xCompressLong, xCompressShort, max(xout), min(xout), self.smfitCompressionLong, self.smfitCompressionShort, widthLong=self.sswidthLong, widthShort=self.sswidthShort) if (self.smfitLogxScale) else x

        #we round the inputs to the Spline because it can be unstable otherwise.
        lxout = np.round(lxout,decimals=4)
        lxinp = np.round(lxinp,decimals=4)
        y=np.round(y,decimals=4)
        #w=np.round(w,decimals=6)

        rspline = robjects.r('smooth.spline')
        predict = robjects.r('predict')

        df.w = u.col_normalize(df.w)
        wfiltered_size = df.loc[df.w > self.smfitMinWeightTol].shape[0]  #this is the number of bonds that have a weight larger than (self.smfitMinWeightTol)%

        if wfiltered_size >= 10:  #TODO (DA): don't want to overfit, so have a minimum number of points. Also, if len(x) < 4 then rspline returns an error.
            rspl = rspline(lxinp,y=y,w=w, df=self.smfitSpldf)     # R smooth.spline
        else:
            #print 'smoothfit: <10 bonds to fit, going flat'
            med_y = np.median(y)
            rspl = rspline(np.array([1.,10.,20.,30.]),y=np.array([med_y,med_y,med_y,med_y]),w=np.array([1.,1.,1.,1.]), df=self.smfitSpldf)

        youtr = np.array(predict(rspl,lxout)[1])
        yinp = np.array(predict(rspl,lxinp)[1])

        xs = np.linspace(0, np.ceil(xout.max()), 200)
        lxs = u.logx1Compress(xs, xCompressLong, xCompressShort, max(xout), min(xout), self.smfitCompressionLong, self.smfitCompressionShort, widthLong=self.sswidthLong, widthShort=self.sswidthShort) if bool(self.smfitLogxScale) else xs
        y_test=np.array(predict(rspl,lxs)[1])
        eps_y = np.array(y - np.array(predict(rspl,lxinp)[1]))
        plot_y = np.array(predict(rspl,lxs)[1])
        ys = np.array(predict(rspl,lxinp)[1])


        if (self.debugplotSmoothFit):
            subtitle = iteration if (iteration is not None) else ''
            xs = np.linspace(0, np.ceil(xout.max()), 200)
    #         yxs = np.log(1+xs) if (logx1) else xs
            lxs = u.logx1Compress(xs, xCompressLong, xCompressShort, max(xout), min(xout), self.smfitCompressionLong, self.smfitCompressionShort, widthLong=self.sswidthLong, widthShort=self.sswidthShort) if bool(self.smfitLogxScale) else xs
            plotly.offline.iplot(go.Figure(data=[
                                    go.Scatter(x=x, y=y, mode='markers', marker=dict(opacity=w/max(w)+0.3)),
                                    go.Scatter(x=df.loc[df.outlier == True]['x'], y=df.loc[df.outlier == True]['y'], mode='markers', name='OUTLIERS'),
                                    go.Scatter(x=xs, y=plot_y)]
                                , layout=go.Layout(title='SmoothFit, %s Instruments, Iteration %s' % (len(x),iteration) , xaxis=dict(title = 't'), yaxis=dict(title = 'log(s)'))))

        #eps_yU = np.array(y - spl(lxinp))
        #eps_y0U = np.array(y0 - spl(lx0))
        #eps_y0 = np.array(y0 - np.array(predict(rspl,lx0)[1]))

        return {'yout':youtr,
                'err': sum(w*(eps_y)**2),
                'ys':ys,
                'xs':x,
                'eps_arr': [x, eps_y, w],
                'indices_list':indList,
                'splfun':rspl,
                'logx1Compress':{'xCompressLong':xCompressLong,'xCompressShort':xCompressShort,
                                 'xLast':max(xout),'xFirst': min(xout),
                                 'compressionLong':self.smfitCompressionLong,'compressionShort':self.smfitCompressionShort,
                                 'widthLong':self.sswidthLong,
                                 'widthShort':self.sswidthShort}}
                #'eps0_arr': [x0, eps_y0, w0]}

    def smoothcurve(self, data_i, spread_col):  #data_i is a slice of spread data on one date
        self.debugplotSmoothFit_ixlist = self.debugplot_ixlist if (self.debugplotSmoothFit_ixlist is None) else self.debugplotSmoothFit_ixlist
        # asset weights:
        if bool(self.normaliseWeightsByCurve):
            data_i['normweight'] = data_i.groupby(self.curveCol)[self.weightCol].transform(lambda x: x/x.sum()) # weights are normalised by issuer, test: # print data.groupby(curve_col)['normweight'].sum()
            self.weightCol = 'normweight'
        aw = data_i[self.weightCol]
        afact =  0.0 if ((self.awDifferential==np.inf) | (self.awDifferential==0)) else (self.awDifferential * min(aw)/sum(aw) - max(aw)/sum(aw)) / (1-max(aw) / sum(aw) - self.awDifferential *(1-min(aw) / sum(aw)))
        #normalize weights
        aoTotal = data_i[self.weightCol].sum()/1000.
        data_i[self.weightCol] = data_i[self.weightCol].apply(lambda x: afact + (1. - afact) * x/aoTotal) #TODO: this should go in self.process_data

        curvelist, curvelevels = self.curve_guess(data_i, spread_col)         # first guess of curve levels

        data_i['outlier'] = False
        data_i['suppressed'] = False
        it_ix = 0
        while ((it_ix < int(self.numIterLevelGuess)) & (data_i.loc[(data_i[self.curveCol] == self.specific_curve)].shape[0] >1) &
                   (data_i.loc[(data_i.outlier == False) & (data_i[self.curveCol] == self.specific_curve)].shape[0] >1)):
            d = pd.DataFrame({self.curveCol: curvelist,'distance': -curvelevels})  # join by curve to individual assets in data => assign 'curvelevel' to each asset
            data_i = pd.merge(data_i, d, how='inner',on=self.curveCol)  # assign bucket to individual assets
            data_i.distance.fillna(0.,inplace=True)  # suppressed assets (have NaN curvelevel) receive 0 curvelevel
            data_i['w'] = data_i[self.weightCol].where(~data_i.outlier.values, 0.0)
            data_i['w'] = data_i.w.where(~data_i.suppressed.values, 0.0)

            if (it_ix == 0):  # TODO: should sigma be updated in a subsequent iteration?
                # sigma = np.std(data_i.distance) * self.sigma_scale # if (fittingobjfun.it == 0 )  else fittingobjfun.sigma TODO: sigma evaluated before data_i is limited to max_num_bonds. Problem?
                # sigma = np.std(data_i.loc[(data_i['IL_dist']==0)].distance) * self.sigma_scale
                sigma = self.calc_sigma(data_i)

            data_i = self.calc_weights(data_i, sigma)
            data_i['y'] = data_i[spread_col].add(data_i.distance)

            data_i_topN = self.select_top_bonds(data_i)

            df_fit = data_i_topN.loc[:, [self.tenorCol, 'y', 'wkil', 'outlier']]
            df_fit = df_fit.rename(columns={self.tenorCol: 'x', 'wkil': 'w'})  # rename x, y, w
            smfit = self.SmoothFit(df_fit, iteration=it_ix, xout=self.tenorGrid)  # fit curve

            indList = smfit['indices_list']
            epsdf = pd.DataFrame(
                {self.curveCol: np.array(data_i.loc[indList, self.curveCol]), 'eps': smfit['eps_arr'][1],
                 'w': smfit['eps_arr'][2], 'ys': smfit['ys'], 'xs': smfit['xs']},
                index=indList)  # new dataframe because SmoothFit might have filtered out some curves (too low weight)

            if self.levelsFit:
                outlierPctTol = self.outlierPctTolLevels
            else:
                outlierPctTol = self.outlierPctTolChanges

            NumIssuerBonds = data_i_topN.loc[(data_i_topN[self.curveCol] == self.specific_curve)].shape[0]
            IssuerBond_outlier = np.array(data_i_topN.loc[(data_i_topN[self.curveCol] == self.specific_curve)]['outlier'])
            OutlierInd_prev = np.array(data_i_topN.loc[(data_i_topN[self.curveCol] == self.specific_curve)&(data_i_topN['outlier'] == True)].index)
            spc_eps = np.array(epsdf.eps)
            if np.size(spc_eps) > 1:  # mdcouple needs a minimum of 2 data points
                outl_fence = u.adjBoxplotStats(spc_eps)['fence']
                spc_outlepsInd = np.array(epsdf.index)[
                    ((spc_eps < outl_fence[0]) | (spc_eps > outl_fence[1])) & (np.abs(spc_eps) > outlierPctTol)]
                epsdf.loc[spc_outlepsInd, 'outlier'] = True
                data_i_topN.loc[spc_outlepsInd, 'outlier'] = True
                data_i.loc[spc_outlepsInd, 'outlier'] = True

            #get new outliers
            epsdf = epsdf.drop(OutlierInd_prev)
            errs = epsdf.loc[(epsdf.CurveId == self.specific_curve) & (epsdf.outlier == True)]['eps'].apply(lambda x: np.absolute(x)).sort_values()
            OutlierInd_new = np.array(errs.index)
            #only allow one outlier
            epsdf.loc[OutlierInd_new[:-1], 'outlier'] = False
            data_i_topN.loc[OutlierInd_new[:-1], 'outlier'] = False
            data_i.loc[OutlierInd_new[:-1], 'outlier'] = False

            data_i.loc[(data_i[self.curveCol]!=self.specific_curve), 'outlier'] = False #for guesses just removing issuer outliers
            adjustment = epsdf.loc[(epsdf.outlier != True)].groupby(self.curveCol).apply(lambda r: u.weighted_average(r))
            specificshift = adjustment[self.specific_curve]
            #specificshift_test = epsdf.loc[(epsdf.outlier != True) & (epsdf[self.curveCol]==self.specific_curve)].groupby(self.curveCol).apply(lambda r: u.weighted_average(r))
            curvelevels_df = pd.DataFrame(data=curvelevels, index=curvelist, columns=['curvelevel'])
            curvelevels_df.set_value(self.specific_curve, 'curvelevel', specificshift)
            curvelevels_df = curvelevels_df - specificshift
            curvelevels = np.array(curvelevels_df.values).flatten()

            data_i.pop('distance')
            it_ix+=1

    ###################################################################################################################
        ##print 'Beginning second optimization'
        data_i['outlier'] = False
        data_i['suppressed'] = False
        stderr = 1.
        it_ix = 0
        while ((it_ix < int(self.maxIter)) & (stderr > self.errTol)):
            d = pd.DataFrame({self.curveCol:curvelist, 'distance':-curvelevels})             # join by curve to individual assets in data => assign 'curvelevel' to each asset
            data_i = pd.merge(data_i, d, how='inner', on=self.curveCol) # assign bucket to individual assets, (suppressed) curves not in right-hand d will have NaN curvelevel if how='outer', or get removed if join='inner'
            data_i.distance.fillna(0., inplace=True) # suppressed assets (have NaN curvelevel) receive infinite curvelevel => infinite distance DA: changed this to have high curvelelve, but not infintite
            data_i['w'] = data_i[self.weightCol].where(~data_i.outlier.values,0.0)
            data_i['w'] = data_i.w.where(~data_i.suppressed.values,0.0)

            if (it_ix == 0):   # TODO: should sigma be updated in a subsequent iteration?
                #sigma = np.std(data_i.distance) * self.sigma_scale # if (fittingobjfun.it == 0 )  else fittingobjfun.sigma TODO: sigma evaluated before data_i is limited to max_num_bonds. Problem?
                #sigma = np.std(data_i.loc[(data_i['IL_dist']==0)].distance) * self.sigma_scale
                sigma = self.calc_sigma(data_i)

            data_i = self.calc_weights(data_i, sigma)  #Calculate weights
            data_i['y'] = data_i[spread_col].add(data_i.distance)   #adjust curve levels by distance

            if it_ix == 0:
                data_i = self.select_top_bonds(data_i)   #pick top N bonds again. Having improved the initial curve guess this should be a more stable set of bonds
                curvelist, curvelevels = self.curve_guess(data_i, spread_col)

            df_fit = data_i.loc[:,[self.tenorCol,'y','wkil','outlier']]
            df_fit = df_fit.rename(columns={self.tenorCol:'x','wkil':'w'})              #rename x, y, w
            smfit = self.SmoothFit(df_fit, iteration=it_ix, xout=self.tenorGrid)        #fit curve

            indList = smfit['indices_list']
            epsdf = pd.DataFrame({self.curveCol:np.array(data_i.loc[indList, self.curveCol]), 'eps': smfit['eps_arr'][1],
                                  'w': smfit['eps_arr'][2],'ys': smfit['ys'],'xs': smfit['xs']}, index=indList) # new dataframe because SmoothFit might have filtered out some curves (too low weight)

            if ((it_ix <= self.numOutlierIterations ) & bool(self.removeOutliers)):  # outlier detection for fist few iterations!
                outlepsMask = pd.DataFrame(np.array(epsdf.eps != epsdf.eps),index=epsdf.index)  # initialise boolean mask, all elements set to False
                if self.levelsFit:
                    outlierPctTol = self.outlierPctTolLevels
                else:
                    outlierPctTol = self.outlierPctTolChanges

                data_i.outlier = False # reset outlier assessment at every iteration
                spc_eps = np.array(epsdf.eps)
                if np.size(spc_eps) > 1:  #mdcouple needs a minimum of 2 data points
                    outl_fence = u.adjBoxplotStats(spc_eps)['fence']
                    spc_outlepsInd = np.array(epsdf.index)[((spc_eps < outl_fence[0]) | (spc_eps > outl_fence[1])) & (np.abs(spc_eps) > outlierPctTol)]
                    epsdf.loc[spc_outlepsInd, 'outlier'] = True
                    data_i.loc[spc_outlepsInd, 'outlier'] = True

                    NumIssuerOutliers = data_i.loc[(data_i[self.curveCol] == self.specific_curve) & (data_i.outlier == True)].shape[0]
                    NumIssuerBonds = data_i.loc[(data_i[self.curveCol] == self.specific_curve)].shape[0]
                    if NumIssuerOutliers == NumIssuerBonds:
                        if NumIssuerBonds == 1: #1 bond can't be an outlier
                            ind = epsdf.loc[epsdf.CurveId == self.specific_curve].index[0]
                            epsdf.loc[ind, 'outlier'] = False
                            data_i.loc[ind, 'outlier'] = False
                        else:  # if more than one bond, keep one
                            errs = epsdf.loc[epsdf.CurveId == self.specific_curve]['eps'].apply(lambda x: np.absolute(x)).sort_values()
                            ind = np.array(errs.index)[0]
                            epsdf.loc[ind, 'outlier'] = False
                            data_i.loc[ind, 'outlier'] = False
            outlepsMask = data_i.loc[indList]['outlier']
            outlepsMask = np.array(outlepsMask).flatten()

            adjustment = epsdf[~outlepsMask].groupby(self.curveCol).apply(lambda r: u.weighted_average(r))  # curve-wise weighted average fitting error
            specificshift = adjustment[self.specific_curve] # the line fitting of ALL (weighted) points does not necessarily "go through" the specific issuer curve => adjustment is constant and remains (see below: it has to be added to )
            adjustment = adjustment - specificshift # for specific curve this is 0, otherwise there is a drift at each iteration
            curvelevels_prev = curvelevels
            curvelist_prev = curvelist
            curvelist = epsdf[~outlepsMask][self.curveCol].unique()
            curvelevels = np.array([curvelevels_prev[np.where(curvelist_prev == curve)] + adjustment[curve] for curve in curvelist]).flatten()

            ## objective: the curve levels are determined (convergence); not the fitting error (fitting error takes *sufficiently* care of that)
            dclvl = np.array([curvelevels[np.where(curvelist == curve)] - curvelevels_prev[np.where(curvelist_prev == curve)]  for curve in curvelist]).flatten()
            stderr = np.sqrt(sum(dclvl * dclvl) / len(dclvl))
            splfun = u.shiftedspline(smfit['splfun'], specificshift)

            lastIteration = True if (((it_ix+1) == self.maxIter) | (stderr <= self.errTol)) else False # only for debug plot
            #if ((self.debugplot == True) & ((it_ix in self.debugplot_ixlist) | ((-1 in self.debugplot_ixlist) & lastIteration))):
            if ((self.debugplot == True) & lastIteration):
                p.plot_curve_context(self.specific_curve, data_i, self.IndustryGroup, self.Sector, self.SectorGroup, self.Market,
                                  self.tenorGrid, smfit, splfun, self.plIndustries, self.tenorWeightDensityBandwidth, self.RiskEntityId,
                                     curve_col=self.curveCol, spread_col=spread_col, tenor_col=self.tenorCol,
                                    distance_col='distance', weight_col='wkil',
                                  smfitLogxScale=bool(self.smfitLogxScale), levels_fit=self.levelsFit, RegionGroup=self.RegionGroup)

            if not lastIteration:
                data_i.pop('distance')
            it_ix += 1

        x_grid = self.tenorGrid
        ao_pdf = u.AmtOutstanding_pdf(data_i.loc[(data_i[self.curveCol] == self.specific_curve) & (data_i.outlier !=True),[self.weightCol,self.tenorCol]], x_grid,
                                      self.tenorCol,self.weightCol,self.tenorWeightDensityBandwidth)

        IssuerBonds = data_i.loc[(data_i[self.curveCol] == self.specific_curve)][[self.tenorCol, spread_col,'OriginalAmtOutstanding',
                                                                                  'outlier','ISIN','MatYears','Coupon','OAS_Swap']]

        SumAmtOutstanding = IssuerBonds.loc[~IssuerBonds.outlier].OriginalAmtOutstanding.sum()  #this is the total amout outstanding of bonds from the issuer, in currenty units
        ao_pdf = ao_pdf*SumAmtOutstanding #ao_pdf is normalized so this step ensure the weight pdf sums to TotalAmountOutstanding

        #calculate coupon term structure
        coupon = u.calculate_coupon_term_structure(IssuerBonds.loc[~IssuerBonds.outlier]
                                                   [['MatYears', 'Coupon', 'OriginalAmtOutstanding']], self.tenorGrid)  #NOTE: tenorgrid is treated as maturity here, not duration

        lxout = u.logx1Compress(self.tenorGrid, smfit['logx1Compress']['xCompressLong'],
                                          smfit['logx1Compress']['xCompressShort'],
                                          smfit['logx1Compress']['xLast'],
                                          smfit['logx1Compress']['xFirst'],
                                          smfit['logx1Compress']['compressionLong'],
                                          smfit['logx1Compress']['compressionShort'],
                                          widthLong=smfit['logx1Compress']['widthLong'],
                                          widthShort=smfit['logx1Compress']['widthShort']) if bool(self.smfitLogxScale) else self.tenorGrid

        x = np.array(IssuerBonds[self.tenorCol])
        xc = u.logx1Compress(x, smfit['logx1Compress']['xCompressLong'],
                                smfit['logx1Compress']['xCompressShort'],
                                smfit['logx1Compress']['xLast'],
                                smfit['logx1Compress']['xFirst'],
                                smfit['logx1Compress']['compressionLong'],
                                smfit['logx1Compress']['compressionShort'],
                                widthLong=smfit['logx1Compress']['widthLong'],
                                widthShort=smfit['logx1Compress']['widthShort']) if bool(
        self.smfitLogxScale) else self.x
        IssuerBonds['OAS_Model'] = u.mExp(splfun(xc)) if self.levelsFit else splfun(xc)
        IssuerBonds['OAS_diff'] = IssuerBonds['OAS_Model'] - (u.mExp(np.array(IssuerBonds[spread_col])) if self.levelsFit else np.array(IssuerBonds[spread_col]))

        res = {'stderr':stderr, 'eyout':u.mExp(smfit['yout']+specificshift), 'yout':smfit['yout']+specificshift, 'xout':self.tenorGrid, 'splfun':splfun, 'iter_index':it_ix-1,
               'AmtOutstanding_pdf':ao_pdf, 'IssuerBonds':IssuerBonds, 'SumAmtOutstanding': SumAmtOutstanding, 'coupon':coupon}

        if self.debugplot_curve:
            p.plot_curve_multi(self.specific_curve, data_i, self.IndustryGroup, self.Sector, self.SectorGroup, self.Market,
                               self.tenorGrid, smfit, splfun, self.plIndustries, self.Currency, coupon, IssuerBonds,
                               self.tenorWeightDensityBandwidth, self.RiskEntityId,
                               curve_col=self.curveCol, spread_col=spread_col, tenor_col=self.tenorCol,
                               distance_col='distance', weight_col='wkil',
                               smfitLogxScale=bool(self.smfitLogxScale), levels_fit=self.levelsFit,
                               RegionGroup=self.RegionGroup, Region=self.Region)

        return res

    def fit_date_pair(self, date_i, s_i, s_prev):
        s_i, self.plIndustries = self.Industry_level_distance(s_i)  # get industry level distance,
        self.levelsFit = True
        res_f = self.smoothcurve(s_i, self.spreadCol)
        logging.info('done LEVELS')

        # fit changes for pairs of dates
        dspread_col = 'd' + self.spreadCol
        s_d = s_i.merge(s_prev.loc[:, ('ISIN', self.spreadCol)], how='inner', on='ISIN', suffixes=('_f', '_i'))
        s_d[dspread_col] = s_d[self.spreadCol + '_f'] - s_d[self.spreadCol + '_i']

        res_d = {}
        if (s_d[s_d[self.curveCol] == self.specific_curve].shape[0] > 0):  # intersection is non-empty
            self.levelsFit = False
            res_d = self.smoothcurve(s_d, dspread_col)
            logging.info('done CHANGES')

        return res_f, res_d

    def fit_history(self):
        udates = pd.date_range(self.start_date,self.end_date, freq='B')  #TODO: list of dates could be obtained through SQL query also.
        dates_ts = pd.Series(udates, index=range(len(udates)))
        s_prev = pd.DataFrame()
        date_prev = 0
        first_iteration = True
        for i_dt in range(len(dates_ts)):  # iterate over dates
            logging.info('RUNNING: date: {}'.format(dates_ts[i_dt]))
            date_i = dates_ts.ix[i_dt]
            s_i = q.get_OAS_data(date_i,date_i,self.Currency)             #get data on date i_dt
            if s_i.shape[0] > 0:
                s_i = u.process_data(s_i, self.curveCol, self.spreadCol, self.weightCol, self.tenorCol)  # process data
                if not first_iteration:  # can't do changes on first date
                    # curve_list = pd.DataFrame(
                    #     s_i.loc[(s_i.Sector == 'HLTH')][[self.curveCol, 'IndustryGroup', 'RiskEntityId', 'PricingTier', 'Region', 'RegionGroup',
                    #          'Sector', 'SectorGroup', 'Market']].groupby(
                    #         self.curveCol).first())  # list of all curves
                    if self.fitSingle:
                        curve_list = pd.DataFrame(
                            s_i.loc[(s_i[self.curveCol] == self.specific_curve)][
                                [self.curveCol, 'IndustryGroup', 'RiskEntityId', 'PricingTier', 'Region', 'RegionGroup',
                                 'Sector',
                                 'SectorGroup', 'Market']].groupby(
                                self.curveCol).first())
                    else:
                        curve_list = pd.DataFrame(
                            s_i[[self.curveCol, 'IndustryGroup', 'RiskEntityId', 'PricingTier', 'Region',
                                 'RegionGroup', 'Sector', 'SectorGroup', 'Market']].groupby(
                                self.curveCol).first())  # list of all curves
                    count = 0
                    for i, row in curve_list.iterrows():  #loop over issuer curves
                        self.specific_curve = i
                        self.IndustryGroup = row['IndustryGroup']
                        self.RiskEntityId = int(row['RiskEntityId'])
                        self.PricingTier = row['PricingTier']
                        self.Region = row['Region']
                        self.RegionGroup = row['RegionGroup']
                        self.Sector = row['Sector']
                        self.SectorGroup = row['SectorGroup']
                        self.Market = row['Market']
                        CurveId = q.get_curveId(CurveShortName=self.specific_curve, RiskEntityId=self.RiskEntityId,
                                                Currency=self.Currency, PricingTier=self.PricingTier)
                        try:
                            logging.info(
                                'RUNNING: date: {} curve: {}, IndustryGroup: {}'.format(date_i, i, self.IndustryGroup))
                            print 'Curve #', count  # , 'Curve ', c, dates_ts.ix[i_dt], self.IndustryGroup
                            res_f, res_d = self.fit_date_pair(date_i, s_i, s_prev)

                            if self.write_data:
                                logging.info('Writing data to DB started')
                                u.write_curve_to_db(res_f, res_d, CurveId, date_i, self.tenorGridEnum)
                                logging.info('Writing to DB complete')

                            count+=1
                        except Exception, e:
                            print 'ERROR curve', i
                            logging.info('FAIL: date: {} curve: {}'.format(date_i, i))
                            logging.error(e)
                            logging.exception("message")

                s_prev = s_i
                #date_prev = date_i
                first_iteration = False
        return

    def select_top_bonds(self, data_i):
        if self.levelsFit:
            numBonds = int(self.numBondsLevels)
        else:
            numBonds = int(self.numBondsChanges)

        data_i['w_sort'] = data_i['kernel'].multiply(data_i['IL_weight'])
        data_i = data_i.sort_values('w_sort', ascending=False)
        data_i['TickerPrcTier'] = data_i.TICKER + data_i.PricingTier

        # group by issuer (TICKER), sort by decreasing weight and calcualte the cumulative sum of number of bonds
        issuers = data_i[['TickerPrcTier', 'ISIN']].groupby('TickerPrcTier', sort=False).count().cumsum()
        issuers = issuers.rename(columns={'ISIN': 'NumBondsCumulative'})
        issuers = pd.DataFrame(issuers)

        if issuers.shape[0] <= self.numIssuers:
            self.numIssuers = issuers.shape[0]
        if issuers.iloc[-1:].NumBondsCumulative.values[0] <= numBonds:
            numBonds = issuers.iloc[-1:].NumBondsCumulative.values[0]

        if issuers.iloc[int(self.numIssuers) - 1].NumBondsCumulative >= numBonds:
            issuers = issuers.iloc[0:int(self.numIssuers)]
        else:
            iss_cutoff = issuers.loc[issuers.NumBondsCumulative >= numBonds].index[0]
            issuers = issuers.loc[:iss_cutoff]
        bond_mask = data_i.TickerPrcTier.isin(issuers.index)
        bond_mask = bond_mask[bond_mask]
        #data_i = data_i[bond_mask]
        #return bond_mask
        data_i.pop('TickerPrcTier')
        data_i_topN = data_i.loc[bond_mask.index]
        return data_i_topN

    def Industry_level_distance(self, df_data):
        # logic to calculate Industry_Level_distance for each bond
        #To start with all bond get a distance of 5.: Industry=AllSectors, Region=GLOBAL
        df_data['IL_dist'] = 5.
        plIndustries = {5: 'Global-ALLSECTORS'}  # need this for charting

        if self.RegionGroup == 'EMG':
            # EMG: we remain in Region and move up the industry hierarchy, only moving to GlobalRegion at last step
            # for EMG:
            #     5: Global, All
            #     4: Region, All
            #     3: Region, Market
            #     2: Region, SectorGroup
            #     1: Region, Sector
            #     0: Region, IndustryGroup
            df_data['IL_dist'] = df_data['IL_dist'].mask(df_data.Region == self.Region, 4.)
            plIndustries.update({4:self.Region + '-AllSectors'})
            #if q.get_cluster_curves(self.Currency,self.Market,self.Region).shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.Market == self.Market) & (df_data.Region == self.Region)),3.)
            plIndustries.update({3: self.Region + '-' + self.Market})
            #if q.get_cluster_curves(self.Currency,self.SectorGroup,self.Region).shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.SectorGroup == self.SectorGroup) & (df_data.Region == self.Region)), 2.)
            plIndustries.update({2: self.Region + '-' + self.SectorGroup})
            #if q.get_cluster_curves(self.Currency,self.Sector,self.Region).shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.Sector == self.Sector) & (df_data.Region == self.Region)), 1.)
            plIndustries.update({1: self.Region + '-' + self.Sector})
            #if q.get_cluster_curves(self.Currency,self.IndustryGroup,self.Region).shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.IndustryGroup == self.IndustryGroup) & (df_data.Region == self.Region)), 0.)
            plIndustries.update({0: self.Region + '-' + self.IndustryGroup})
        elif self.RegionGroup == 'DVL':
            # DVL: we remain in sector, go global and move up the industry hierarchy
            # for DVL:
            #     5: Global, All
            #     4: Global, Market
            #     3: Global, SectorGroup
            #     2: Global, Sector
            #     1: Global, IndustryGroup
            #     0: Region, IndustryGroup

            #LstClusterIndustry = []  #this identifies the lowest point in the hierarchy where we have a cluster
                                    # we need this to ensure we switch from Global -> Region at the right point
            #if q.get_cluster_curves(self.Currency, self.Market, 'GLOBAL').shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.Market == self.Market)), 4.)
            #LstClusterIndustry = ['Market',self.Market,4.]
            plIndustries.update({4:'GLOBAL-' + self.Market})
            #if q.get_cluster_curves(self.Currency, self.SectorGroup, 'GLOBAL').shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.SectorGroup == self.SectorGroup) ), 3.)
            #LstClusterIndustry = ['SectorGroup',self.SectorGroup,3.]
            plIndustries.update({3: 'GLOBAL-' + self.SectorGroup})
            #if q.get_cluster_curves(self.Currency, self.Sector, 'GLOBAL').shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.Sector == self.Sector)), 2.)
            #LstClusterIndustry = ['Sector',self.Sector,2.]
            plIndustries.update({2: 'GLOBAL-' + self.Sector})
            #if q.get_cluster_curves(self.Currency, self.IndustryGroup, 'GLOBAL').shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.IndustryGroup == self.IndustryGroup)), 1.)
            #LstClusterIndustry = ['IndustryGroup',self.IndustryGroup,1.]
            plIndustries.update({1: 'GLOBAL-' + self.IndustryGroup})
            #finally, jump to Region cluster if it exists
            #if q.get_cluster_curves(self.Currency, LstClusterIndustry[1], self.Region).shape[0] > 0:
            df_data['IL_dist'] = df_data['IL_dist'].mask(((df_data.IndustryGroup == self.IndustryGroup) & (df_data.Region == self.Region)), 0.)
            plIndustries.update({0.: self.Region + '-' + self.IndustryGroup})

        #ensure that the lowest level of the hierarchy (the most granular) has IndustryLevel distance of 0
        #d_min =  df_data['IL_dist'].min()    #d_min identifies the most granular cluster curve available.
        #df_data['IL_dist'] = df_data['IL_dist'] - d_min  #TODO: d_min here because we used to use existence of cluster curves.
                                                        # d_min should always be 0 now, as we always have at least one bond at IndustryGroup level by definition
        plIndustries.update({'d_min':0})
        #print 'd_min = ', d_min
        return df_data, plIndustries

    def calc_sigma(self,data_i):
        if self.levelsFit:
            sigma = np.std(data_i.loc[data_i.outlier == False]['distance']) * self.sigmaScaleLevels
        else:
            sigma = np.std(data_i.loc[data_i.outlier == False]['distance']) * self.sigmaScaleChanges

        return sigma

    def calc_weights(self,data_i, sigma):
        if sigma > 0:
            data_i['kernel'] = data_i.distance.apply(lambda x: u.gaussian_kernel(x, sigma))
        else:
            data_i['kernel'] = data_i.distance.apply(lambda x: 1. if x == 0. else 0.)
        data_i['IL_weight'] = data_i['IL_dist'].apply(lambda x: u.ind_level_weight_calc(x, self.gamma))
        data_i['wkil'] = data_i['kernel'].multiply(data_i['w']).multiply(data_i['IL_weight'])

        return data_i

    def curve_guess(self, data_i, spread_col):
        curvelist = data_i[self.curveCol].unique()  # new curve list
        curvemedians = data_i.groupby(data_i[self.curveCol]).apply(
            lambda r: wq.median(r[spread_col], r[self.weightCol]))  # curve-wise median
        curvelevels = np.array([curvemedians[curvelist[i]] for i in range(len(curvelist))])
        curvelevels = curvelevels - curvelevels[np.where(
            curvelist == self.specific_curve)]  # normalise so that curve level of specific curve = 0, but keep levels different from distances (opposite sign) so more transparent for plotting

        return curvelist, curvelevels

def write_curves_to_db(macDBInfo, delete, level, weight, change, coupon, bonds, properties, chunkSize=1000):
    connection = U.createMSSQLConnection(macDBInfo)
    curs = connection.cursor()
    xcIssuerCurveDelete = 'delete from [MarketData].[dbo].[xcIssuerCurve] where AxiomaDataId = %s and TradeDate = %s'
    xcIssuerBondDelete = 'delete from [MarketData].[dbo].[xcIssuerModelOAS] where AxiomaDataId = %s and TradeDate = %s'
    xcIssuerPropertyDelete = 'delete from [MarketData].[dbo].[xcIssuerProperty] where AxiomaDataId = %s and TradeDate = %s'
    for chunk in U.grouper(delete, chunkSize):
        curs.executemany(xcIssuerCurveDelete, chunk)
        curs.executemany(xcIssuerBondDelete, chunk)
        curs.executemany(xcIssuerPropertyDelete, chunk)
        connection.commit()

    xcIssuerCurveQuery = 'insert into [MarketData].[dbo].[xcIssuerCurve] values (%s,%s,%s,%d,%s,%s, GETDATE())'
    logging.info('writing curve level to DB')
    for chunk in U.grouper(level, chunkSize):
        curs.executemany(xcIssuerCurveQuery, chunk)
        connection.commit()

    logging.info('writing curve weight to DB')
    for chunk in U.grouper(weight, chunkSize):
        curs.executemany(xcIssuerCurveQuery, chunk)
        connection.commit()

    logging.info('writing curve change to DB')
    for chunk in U.grouper(change, chunkSize):
        curs.executemany(xcIssuerCurveQuery, chunk)
        connection.commit()

    logging.info('writing curve coupon to DB')
    for chunk in U.grouper(coupon, chunkSize):
        curs.executemany(xcIssuerCurveQuery, chunk)
        connection.commit()

    xcIssuerOASQuery = 'insert into [MarketData].[dbo].[xcIssuerModelOAS] values (%s,%s,%s,%d,%d,%s, %s, %s, GETDATE(), %d, %d)'
    for chunk in U.grouper(bonds, chunkSize):
        curs.executemany(xcIssuerOASQuery, chunk)
        connection.commit()

    xcIssuerPropertyQuery = 'insert into [MarketData].[dbo].[xcIssuerProperty] values (%s,%s,%d,%s,GETDATE())'
    for chunk in U.grouper(properties, chunkSize):
        curs.executemany(xcIssuerPropertyQuery, chunk)
        connection.commit()
    connection.close()
    return 0


def run_single_curve(args):
    params = args[0]
    i = args[1]
    row=args[2]
    s_i = args[3]
    s_prev = args[4]
    date_i = args[5].strftime('%Y-%m-%d')

    ssc = SmoothSplineCurve(**params)

    ssc.specific_curve = i
    ssc.IndustryGroup = row['IndustryGroup']
    ssc.RiskEntityId = int(row['RiskEntityId'])
    ssc.PricingTier = row['PricingTier']
    ssc.Region = row['Region']
    ssc.RegionGroup = row['RegionGroup']
    ssc.Sector = row['Sector']
    ssc.SectorGroup = row['SectorGroup']
    ssc.Market = row['Market']
    CurveId = q.get_curveId(CurveShortName=ssc.specific_curve, RiskEntityId=ssc.RiskEntityId,
                            Currency=ssc.Currency, PricingTier=ssc.PricingTier)

    results_level = []
    results_weight = []
    results_change = []
    results_coupon = []
    delete_rows = []
    results_bonds = []
    results_properties = []

    try:
        print 'Curve ', ssc.specific_curve, 'Date: ', date_i
        res_f, res_d = ssc.fit_date_pair(ssc.end_date, s_i, s_prev)

        #write lists of tuples to write to db:
        level = np.array((res_f['yout']))
        change = np.array((res_d['yout']))  if bool(res_d) else np.ones(len(ssc.tenorGridEnum))*(-10)
        weight = np.array(res_f['AmtOutstanding_pdf'])
        coupon = np.array(res_f['coupon'])
        sumAmtOutstanding = res_f['SumAmtOutstanding']
        IssuerBonds_f = (res_f['IssuerBonds'])
        IssuerBonds_d = (res_d['IssuerBonds']) if bool(res_d) else pd.DataFrame(columns = IssuerBonds_f.columns)
        IssuerBonds = IssuerBonds_f.merge(IssuerBonds_d.loc[:, ('ISIN', 'OAS_diff', 'OAS_Model', 'outlier')],
                                              how='outer', on='ISIN', suffixes=('_f', '_d'))

        for i in range(0,level.size):
            results_level.append((int(CurveId), date_i, ssc.tenorGridEnum[i], level[i], 'll','dantonio'))
            results_weight.append((int(CurveId), date_i, ssc.tenorGridEnum[i], weight[i], 'w', 'dantonio'))
            results_change.append((int(CurveId), date_i, ssc.tenorGridEnum[i], change[i], 'lc', 'dantonio'))
            results_coupon.append((int(CurveId), date_i, ssc.tenorGridEnum[i], coupon[i], 'cn', 'dantonio'))
        delete_rows.append((int(CurveId), date_i))

        for i, r in IssuerBonds.iterrows():
            results_bonds.append((int(CurveId), date_i, r['ISIN'], r['OAS_Model_f'],
                              -10 if pd.isnull(r['OAS_Model_d']) else r['OAS_Model_d'],   #TODO: execute complained about 'NULL' so -10 used as workaround.
                              int(r['outlier_f']),
                              1 if pd.isnull(r['outlier_d']) else int(r['outlier_d']),    #TODO: if bond not in changes fit marked as outlier
                                  'dantonio',
                              r['OAS_diff_f'],
                              -10 if pd.isnull(r['OAS_diff_d']) else r['OAS_diff_d']
                              ))
        results_properties.append((int(CurveId), date_i, sumAmtOutstanding,'dantonio'))

    except Exception, e:
        print 'ERROR curve', i
        logging.info('FAIL: date: {} curve: {}'.format(date_i, i))
        logging.error(e)
        logging.exception("message")

    results = {'results_level':results_level,
               'results_change':results_change,
               'results_weight':results_weight,
               'results_coupon':results_coupon,
               'delete_rows':delete_rows,
               'results_bonds':results_bonds,
               'results_properties':results_properties} #dictionary of lists of tuples
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--delete_existing_quotes',
                        help="Execute delete on all quotes within date range curves",
                        action="store_true",
                        default=False)
    parser.add_argument('-d', '--environment',
                        help="Environment to use. Available options: PROD, UAT, DEVTEST. Default is DEV",
                        action="store",
                        default='DEV')
    parser.add_argument("--currency",
                        help="Curves currency",
                        action="store",
                        default=None)
    parser.add_argument("-s", "--startDate", help="Start date", action="store", default=None)
    parser.add_argument("-e", "--endDate", help="End date", action="store", default=None)
    parser.add_argument("-p", "--parallelProcessing", help="Split tasks over multiple processors", action="store_true", default=True)
    args = parser.parse_args()

    startDate = args.startDate
    endDate = args.endDate
    Currency=args.currency
    parallelProcessing = args.parallelProcessing
    environment = args.environment

    curveCol = 'CurveId'
    spreadCol = 'logOAS'
    weightCol = 'AmtOutstanding'
    tenorCol =  'Effective_Duration'

    pwd = os.path.dirname(os.path.realpath(__file__)) + os.sep
    max_workers = 2
    configFile = open(pwd + "/production.config", 'r')
    configuration = U.loadConfigFile(configFile)

    sectionID = cu.getAnalysisConfigSection(environment)
    envInfoMap = U.getConfigSectionAsMap(configuration, sectionID)
    # mktDBInfo = cu.getOracleDatabaseInfo(U.getConfigSectionAsMap(
    #     configuration, envInfoMap.get('equitymarketdb', None)))
    # modelDBInfo = cu.getOracleDatabaseInfo(U.getConfigSectionAsMap(
    #     configuration, envInfoMap.get('equitymodeldb', None)))
    macDBInfo = cu.getMSSQLDatabaseInfo(U.getConfigSectionAsMap(
        configuration, envInfoMap.get('macdb', None)))

    logging.basicConfig(filename='C:\\Users\\lyang\\Desktop\\ngc.log', level=logging.INFO, filemode='w',
                        format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    params = {'specific_curve': '',
              'start_date': startDate,
              'end_date': endDate,
              'Currency': Currency,
              'alpha': 0.5,
              'gamma': 0.5,
              'smfitCompressionShort': 0.1,
              'smfitCompressionLong': 0.1,
              'sigmaScaleLevels': 0.2,
              'sigmaScaleChanges': 0.2,
              'smfitSpldf': 4,
              'debugplot': True,
              'debugplot_ixlist': [-1],
              'plot_f': False,
              'sswidthLong': .05,
              'sswidthShort': 0.8,
              'maxIter': 10,
              'numOutlierIterations': 3,
              'numIterLevelGuess': 2,
              'numBondsLevels': 200,
              'numBondsChanges': 200,
              'fitSingle': False}

    udates = pd.date_range(startDate, endDate,freq='B')   #TODO: holidays
    dates_ts = pd.Series(udates, index=range(len(udates)))
    s_prev = pd.DataFrame()
    first_iteration = True
    startTime = get_current_time()
    print "startTime: ", startTime
    for i_dt in range(len(dates_ts)):  # iterate over dates
        logging.info('RUNNING: date: {}'.format(dates_ts[i_dt]))
        date_i = dates_ts.ix[i_dt]
        s_i = q.get_OAS_data(date_i, date_i, Currency)  # get data on date i_dt
        if s_i.shape[0] > 0:
            s_i = u.process_data(s_i, curveCol, spreadCol, weightCol, tenorCol)  # process data
            if not first_iteration:  # can't do changes on first date
                # curve_list = pd.DataFrame(
                #     s_i.loc[(s_i['CurveId'] == 'USD-BWA-SEN')][
                #         [curveCol, 'IndustryGroup', 'RiskEntityId', 'PricingTier', 'Region',
                #          'RegionGroup', 'Sector', 'SectorGroup', 'Market']].groupby(
                #         curveCol).first())  # list of all curves
                # curve_list = pd.DataFrame(
                #     s_i.loc[(s_i['IndustryGroup'] == 'AUTOS')][[curveCol, 'IndustryGroup', 'RiskEntityId', 'PricingTier', 'Region',
                #          'RegionGroup', 'Sector', 'SectorGroup', 'Market']].groupby(
                #         curveCol).first())  # list of all curves
                curve_list = pd.DataFrame(
                    s_i[
                        [curveCol, 'IndustryGroup', 'RiskEntityId', 'PricingTier', 'Region',
                         'RegionGroup', 'Sector', 'SectorGroup', 'Market']].groupby(
                        curveCol).first())  # list of all curves
                count = 0

                arg_list = [(params,
                             i,
                             row,
                             s_i,
                             s_prev,
                             date_i
                             ) for i,row in curve_list.iterrows()]

                results_level = []
                results_weight = []
                results_change = []
                results_coupon = []
                delete_rows=[]
                results_bonds=[]
                results_properties = []
                # run in parallel
                if parallelProcessing:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                    #with concurrent.futures.ProcessPoolExecutor() as executor:
                        r =(executor.map(run_single_curve, arg_list))
                    for x in r:
                        results_level.append(x['results_level'])
                        results_weight.append(x['results_weight'])
                        results_change.append(x['results_change'])
                        results_coupon.append(x['results_coupon'])
                        delete_rows.append(x['delete_rows'])
                        results_bonds.append(x['results_bonds'])
                        results_properties.append(x['results_properties'])
                    results_level = [val for sublist in results_level for val in sublist]  #this is apparently the fatest way https://stackoverflow.com/questions/11264684/flatten-list-of-lists
                    results_weight = [val for sublist in results_weight for val in sublist]
                    results_change = [val for sublist in results_change for val in sublist]
                    results_coupon = [val for sublist in results_coupon for val in sublist]
                    delete_rows = [val for sublist in delete_rows for val in sublist]
                    results_bonds = [val for sublist in results_bonds for val in sublist]
                    results_properties = [val for sublist in results_properties for val in sublist]
                else:
                    for i, row in curve_list.iterrows():
                        arg_list = [params,
                                     i,
                                     row,
                                     s_i,
                                     s_prev,
                                     date_i
                                     ]
                        x = run_single_curve(arg_list)
                        results_level.append(x['results_level'])
                        results_weight.append(x['results_weight'])
                        results_change.append(x['results_change'])
                        results_coupon.append(x['results_coupon'])
                        delete_rows.append(x['delete_rows'])
                        results_bonds.append(x['results_bonds'])
                        results_properties.append(x['results_properties'])

                write_data = True
                if write_data:
                    write_curves_to_db(macDBInfo, delete_rows, results_level, results_weight,
                                       results_change, results_coupon, results_bonds,results_properties)

            s_prev = s_i
            first_iteration = False

    endTime = get_current_time()
    print "startTime: ", startTime, "endTime: ", endTime

if __name__ == '__main__':
    t1 = time.clock()
    main()
    t2 = time.clock()
    print t2-t1







