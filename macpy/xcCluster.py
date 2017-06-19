import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import weighted as wq # wquantiles, weighted quantiles
import scikits.datasmooth as ds
from sklearn import datasets, linear_model
import argparse

import macpy.utils.ngc_utils as u
import macpy.utils.ngc_queries as q
import visualization.ngc_plot as p
import time

class ClusterSurface(object):
    def __init__(self, *args, **kwargs):
        self.clusterShortName = None
        self.Currency = kwargs.get('Currency', None)
        self.currencyList = []
        self.fallbackCurrency = None
        self.Industry = kwargs.get('Industry', None)
        self.IndustryLevel = None
        self.Region = kwargs.get('Region', None)
        self.RegionGroup = ''
        self.tradeDate = kwargs.get('tradeDate', None)
        self.df = 6
        self.wrOom = 2.  # order of relative magnitude between smallest and highest weight **** SEEMS CRUCIAL for stability of fit !! ****
        self.clambda = 1.0
        self.Ns = 21  # number of support pillars defining intervals for monotonicity, i.e between these pillars we have yh(i+1) - yh(i) >= eps
        self.Np = 50  # number of support points at which the function is evaluated: ensure that dimension of xhat (necessary when specifying constraints!) ... defaults to x if not specified) is larger than d !
        self.Nc = None  # number of constraints
        self.eps = 1e-2
        self.xmonoFrom = 1.0
        self.xmonoMin = 10.0  # rank 10 (investment grade BBB- / Baa3) is lowest rating up to which the rating fit is monotonic: monotonicity is enforced AT LEAST up to this rating (and beyond for tenors below tRelaxedConstr)
        self.tRelaxedConstr = 10.0  # from this tenor onwards we only enforce monotonicity up to investment grade
        self.tRelaxedConstrStart = 5.
        self.tenorGrid = np.array([0, 0.083333333333, 0.166666667, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7, 10, 12, 15, 20, 25,30, 40])  # 50, 60, 80, 100])
        self.tenorGridEnum = np.array(['0M', '1M', '2M', '3M', '6M', '9M', '1Y', '15M', '18M', '2Y', '30M', '3Y', '42M', '4Y', '5Y', '7Y', '10Y','12Y', '15Y', '20Y', '25Y', '30Y', '40Y'])
        self.weightCutoff = 1E-06
        self.notebookMode = False
        self.gamma = 0.5  #for IndustryLevel Weighting
        self.numIssuers = 200
        self.needXccySupport = False
        self.currencyGroupWeight = 0.5
        self.currencyFallbackWeight = 0.25

    def run_single(self):
        self.fitSurface()

    def fitSurface(self):
        #TODO: will we have a 'needXccySupport' flag?
        #self.needXccySupport = q.NeedXccySupport(self.Currency, self.Industry, self.Region)
        self.needXccySupport = False
        if self.needXccySupport:
            df_currSupport = q.get_supporting_currencies(self.Currency)
            self.fallbackCurrency = df_currSupport[df_currSupport.FallbackOnly == 'y'].Currency.astype(str).tolist()
            self.currencyList = df_currSupport.Currency.astype(str).tolist()
        else:
            self.fallbackCurrency = self.Currency
            self.currencyList = [self.Currency]

        dfIssuerProperty = q.get_cluster_issuer_property(self.tradeDate)
        dfIssuerProperty = dfIssuerProperty.loc[(dfIssuerProperty.Currency.isin(self.currencyList))]

        dfCurves = q.get_cluster_issuer_curves(self.tradeDate)
        badCurves = np.array(dfCurves.loc[dfCurves.Quote < -1.].AxiomaDataId)
        dfCurves = dfCurves[~dfCurves.isin(badCurves)]# there are some crazy curves, based on crazy data
        HasRating = dfCurves.AxiomaDataId.isin(dfIssuerProperty.AxiomaDataId)  # filter for curves which have a rating attached
        dfCurves = dfCurves[HasRating]
        print dfCurves.shape, dfIssuerProperty.PricingTier.unique()
        ratingscale = q.get_cluster_rating_scale()

        self.IndustryLevel = u.industry_level(self.Industry)

        eps = self.eps * self.Ns / self.Np
        self.Nc = self.Np -1

        ind_dict = {5:'IndustryGroup',4:'Sector',3:'SectorGroup',2:'Market', 10:'AllSectors'}  # this is the length of the industry identifier, for each industry e.g. PHARM has 5 letters
        dfIssuerProperty = self.industry_level_distance(dfIssuerProperty)

        dfIssuerProperty['IL_weight'] = dfIssuerProperty.IL_dist.apply(lambda x: u.ind_level_weight_calc(x,self.gamma))
        dfIssuerProperty['Currency_weight'] = \
            dfIssuerProperty.Currency.apply(lambda x: u.currency_support_priority(x,
                                                                                  self.Currency,
                                                                                  self.fallbackCurrency,
                                                                                  self.currencyList,
                                                                                  self.currencyGroupWeight,
                                                                                  self.currencyFallbackWeight))

        dfIssuerProperty = dfIssuerProperty[dfIssuerProperty.SumAmtOutstanding.notnull()]

        #count the number of issuers in the specific cluster
        if self.Region == 'GLOBAL':
            NumIssuersInCluster = dfIssuerProperty[(dfIssuerProperty[ind_dict[len(self.Industry)]] == self.Industry) & (dfIssuerProperty.Currency == self.Currency)].shape[0]
        else:
            NumIssuersInCluster = dfIssuerProperty[(dfIssuerProperty[ind_dict[len(self.Industry)]] == self.Industry) & (
            dfIssuerProperty.Region == self.Region) & (dfIssuerProperty.Currency == self.Currency)].shape[0]
        NumIssuersFit = np.maximum(NumIssuersInCluster, self.numIssuers)

        # obtain the pdf of SumAmtOustanding for cluster
        if self.Region == 'GLOBAL':
            dfIssuerPropertyCluster = dfIssuerProperty[(dfIssuerProperty[ind_dict[len(self.Industry)]] == self.Industry) & (dfIssuerProperty.Currency == self.Currency)]
        else:
            dfIssuerPropertyCluster = dfIssuerProperty[
                (dfIssuerProperty[ind_dict[len(self.Industry)]] == self.Industry) & (dfIssuerProperty.Region == self.Region)
                & (dfIssuerProperty.Currency == self.Currency)]
        dfICluster = pd.concat([dfIssuerPropertyCluster[~dfIssuerPropertyCluster.MDRank.isnull()].assign(RatingRank=dfIssuerPropertyCluster.MDRank[~dfIssuerPropertyCluster.MDRank.isnull()], Agency='MD'),
                                dfIssuerPropertyCluster[~dfIssuerPropertyCluster.SPRank.isnull()].assign(RatingRank=dfIssuerPropertyCluster.SPRank[~dfIssuerPropertyCluster.SPRank.isnull()], Agency='SP')])
        x = np.array(dfICluster.RatingRank)
        x_grid = np.array(ratingscale.RatingRank)
        weights = np.array(dfICluster.SumAmtOutstanding)
        SumAmtOPdf = u.kde(x, x_grid, weights, bandwidth=2.)
        InvSumAmtOPdf = np.max(SumAmtOPdf)  - SumAmtOPdf

        #TODO: AmtOutstanding needs to be in a single currency, otherwise ordering is nonsense
        dfIssuerProperty.loc[:,'RatingWeight'] =0.5*(dfIssuerProperty.MDRank.apply(lambda x: u.calc_rating_weight(x, InvSumAmtOPdf))\
                                                     +dfIssuerProperty.SPRank.apply(lambda x: u.calc_rating_weight(x, InvSumAmtOPdf)))

        #rank issuers and select top N
        dfIssuerProperty = dfIssuerProperty.sort_values(['Currency_weight','IL_weight','RatingWeight','SumAmtOutstanding'], ascending=[False, False, False, False])
        IdInRegInd = dfIssuerProperty.iloc[:int(NumIssuersFit)].AxiomaDataId
        dfI = dfIssuerProperty.loc[dfIssuerProperty.AxiomaDataId.isin(IdInRegInd)]
        # put union of non-null agency rankings in column RatingRanking:
        dfI = pd.concat([dfI[~dfI.MDRank.isnull()].assign(RatingRank=dfI.MDRank[~dfI.MDRank.isnull()], Agency='MD'),
                         dfI[~dfI.SPRank.isnull()].assign(RatingRank=dfI.SPRank[~dfI.SPRank.isnull()], Agency='SP')])

        dfCurves_copy = pd.merge(dfCurves, dfIssuerProperty[['AxiomaDataId', 'IL_weight']],
                       on='AxiomaDataId').sort_values('IL_weight', ascending=False)

        wCl = np.array(dfCurves_copy.Quote[(dfCurves_copy.AxiomaDataId.isin(IdInRegInd)) & (dfCurves_copy.category == 'w')]*
                       dfCurves_copy.IL_weight[(dfCurves_copy.AxiomaDataId.isin(IdInRegInd)) & (dfCurves_copy.category == 'w')])
        #TODO: add the currency weight to wCl? At the moment it's just being used to order.
        # scale weights (to stay within wrOoM) across surface (not just tenor by tenor, otherwise the relative weight proportion across tenors gets disrupted)
        wc = u.compress_weights(wCl, self.wrOom)
        dfCurves_copy.loc[(dfCurves_copy.AxiomaDataId.isin(IdInRegInd)) & (dfCurves_copy.category == 'w'), 'Quote'] = wc
        # surfc = np.zeros((Ns, len(TenorGridEnum)), dtype=float)
        surfc = np.zeros((len(self.tenorGridEnum), self.Ns), dtype=float)
        dfC = dfCurves_copy[(dfCurves_copy.AxiomaDataId.isin(IdInRegInd))]
        dfS = pd.merge(dfI[['AxiomaDataId', 'RatingRank', 'Agency', 'IndustryGroup', 'Sector', 'SectorGroup', 'Market','IL_dist']], dfC[['AxiomaDataId', 'Quote', 'category','TenorEnum']],
                       on='AxiomaDataId').sort_values('RatingRank', ascending=True)

        for te in self.tenorGridEnum:
            dfSte = dfS.loc[dfS.TenorEnum == te]
            x = dfSte.RatingRank[dfSte.category == 'll']
            y = dfSte.Quote[dfSte.category == 'll']
            ws = dfSte.Quote[dfSte.category == 'w']

            xMin = x.min()
            xMax = x.max()

            if x.shape[0] != ws.shape[0]:
                print 'ERROR: x and ws not the same'

            #     w = dfS.Quote[dfS.category=='w']
            #     ws = [pow(10, -wrOoM + wrOoM/(len(w)-1) * (np.max(np.where(np.sort(w)==w_i)))) for w_i in w] # simple way of compressing weights to scale between e.g. 0.01 and 1.
            #     # plotly.offline.iplot(go.Figure(data=[go.Scatter(y=np.log10(np.sort(w)), mode= 'markers'), go.Scatter(y=np.log10(np.sort(ws)), mode= 'markers'),
            #     #                                      go.Scatter(y=regr.predict(np.arange(len(np.log10(np.sort(w)))).reshape(-1, 1)))]))

            #xh = np.linspace(1, self.Ns, self.Np)
            xh = np.linspace(xMin, xMax, self.Np)
            ## ** ALTERNATIVE :
            #     xh = np.linspace(min(x), max(x),Np)
            ##    xh = np.linspace(1, np.max(x), Np)

            bconstr = np.repeat(0.0, self.Nc)
            Aconstr = np.zeros((self.Nc, self.Np), dtype=float)
            # enforce monotonicity in a specific interval:
            teYr = self.tenorGrid[np.where(self.tenorGridEnum == te)][0]

            if teYr <= self.tRelaxedConstrStart:
                xmono_to = max(xh)
            else:
                xmono_to = max(xh) - min(teYr - 0., self.tRelaxedConstr) / self.tRelaxedConstr * (max(xh) - self.xmonoMin)

            #     build the constraint matrices
            constr_msk = np.in1d(xh, xh[xh >= self.xmonoFrom]) & np.in1d(xh, xh[xh <= xmono_to])
            for i in np.arange(0, self.Nc)[constr_msk[1:self.Np]]:
                Aconstr[i, i], Aconstr[i, i + 1] = (1, -1)  # monotonicity constraint: increasing
                bconstr[i] = -eps # *strictly* monotonic

            print 'tenor= ', te
            yh = ds.smooth_data_constr(x, y, xhat=xh, weights=ws, d=self.df, lmbd=self.clambda, inequality=(Aconstr, bconstr),
                                       relative=False, midpointrule=False)
            i = np.where(self.tenorGridEnum == te)[0][0]
            #     surfc[:, i] = np.interp(np.arange(0,21)+1 , xh, yh)
            #surfc[i, :] = np.interp(np.arange(0, 21) + 1, xh, yh)
            surfc[i,:] = u.cluster_interp_and_extrap(xh,yh, self.eps)

            #     yhuc,lmbd = ds.smooth_data (x, y, xhat=xh, weights=ws, d=df) # unconstrained smooth fit, to compare with constrained fit
            #     plotly.offline.iplot(go.Figure(data=[go.Scatter(x=x, y=y, mode= 'markers')
            #                                         ,go.Scatter(x=xh, y= yh),go.Scatter(x=xh, y= yhuc)]
            #                                    , layout=go.Layout(title=te, xaxis=dict(title = 'rating rank'), yaxis=dict(title = 'log spread'))))

        p.plot_3D_cluster_surface(ratingscale, self.tenorGrid,surfc,self.Currency, self.Region, self.Industry, notebookMode=self.notebookMode)

        if self.Industry == 'ALLSECTORS':
            allSectorFlag = '--'
        else:
            allSectorFlag = ''
        if self.Region == 'GLOBAL':
            dfOAS = q.get_cluster_raw_oas_global(self.tradeDate, self.Currency, self.Industry, self.IndustryLevel, allSectorFlag)
        else:
            dfOAS = q.get_cluster_raw_oas_region(self.tradeDate, self.Currency, self.Industry, self.IndustryLevel, self.Region, allSectorFlag)
        dfOAS['logOAS'] = u.mLog(dfOAS.OAS_Swap)
        dfOAS = dfOAS[dfOAS.AmtOutstanding.notnull()]
        surfcT = surfc.transpose()
        p.plot_3D_cluster_scatter(ratingscale, self.tenorGrid, surfcT, dfOAS, self.Currency, self.Region, self.Industry, notebookMode=self.notebookMode)

        return dfS, surfc, dfOAS, ratingscale

    def industry_level_distance(self, dfIssuerProperty):
        dfIssuerProperty['IL_dist'] = 5.  # start with a distance of 5 for Global, AllSectors
        cdict = u.get_cluster_hier(self.Industry)
        Industries = ['AllSectors', 'Market', 'SectorGroup', 'Sector', 'IndustryGroup']  # layers of the Industry hierarchy
        ind_dict = {5: 'IndustryGroup', 4: 'Sector', 3: 'SectorGroup',
                    2: 'Market', 10: 'AllSectors'}  # this is the length of the industry identifier, for each industry e.g. PHARM has 5 letters

        if self.Region == 'GLOBAL':
            # 5: Global, AllSectors
            # 4: Global, Market
            # 3: Global, SectorGroup
            # 2: Global, Sector
            # 1: Global, IndustryGroup
            i=5
            for key in Industries:  # important that we move down the hierarchy in this loop
                if key in cdict:
                    dfIssuerProperty['IL_dist'] = dfIssuerProperty['IL_dist'].mask(
                        (dfIssuerProperty[key] == cdict[key]), i)
                    i-=1
        else:
            # region not GLOBAL
            self.RegionGroup = dfIssuerProperty[(dfIssuerProperty[ind_dict[len(self.Industry)]] == self.Industry) & (
                dfIssuerProperty.Region == self.Region)].RegionGroup.values[0]
            if self.RegionGroup == 'EMG':
                # for EMG:
                #     5: Global, AllSectors
                #     4: Region, AllSectors
                #     3: Region, Market
                #     2: Region, SectorGroup
                #     1: Region, Sector
                #     0: Region, IndustryGroup
                dfIssuerProperty['IL_dist'] = 5.  # start with a distance of 5 for Global, AllSectors
                # dfIssuerProperty['IL_dist'] = dfIssuerProperty['IL_dist'].mask(
                #     (dfIssuerProperty.Region == self.Region), 4.)
                i=4
                for key in Industries:  # important that we move down the hierarchy in this loop
                    if key in cdict:
                        dfIssuerProperty['IL_dist'] = dfIssuerProperty['IL_dist'].mask(
                            ((dfIssuerProperty[key] == cdict[key]) & (dfIssuerProperty.Region == self.Region)),
                            i)
                        i-=1
            elif self.RegionGroup == 'DVL':
                # for DVL:
                #     5: Global, AllSectors
                #     4: Global, Market
                #     3: Global, SectorGroup
                #     2: Global, Sector
                #     1: Global, IndustryGroup
                #     0: Region, IndustryGroup
                i=5
                for key in Industries:  # important that we move down the hierarchy in this loop
                    if key in cdict:
                        dfIssuerProperty['IL_dist'] = dfIssuerProperty['IL_dist'].mask(
                            (dfIssuerProperty[key] == cdict[key]), i)
                        LstClusterIndustry = key
                        i-=1
                dfIssuerProperty['IL_dist'] = dfIssuerProperty['IL_dist'].mask(
                    ((dfIssuerProperty[LstClusterIndustry] == cdict[LstClusterIndustry]) & (dfIssuerProperty.Region == self.Region)),
                    i)

                # for key in Industries:
                #     if key in cdict:
                #         dfIssuerProperty['IL_dist'] = dfIssuerProperty['IL_dist'].mask(
                #             ((dfIssuerProperty[key] == cdict[key]) & (dfIssuerProperty.Region == self.Region)), 5.-len(cdict[key])+1)
        # ensure that the lowest level of the hierarchy (the most granular) has IndustryLevel distance of 0
        d_min = dfIssuerProperty['IL_dist'].min()  # d_min identifies the most granular cluster curve available.
        dfIssuerProperty['IL_dist'] = dfIssuerProperty['IL_dist'] - d_min

        return dfIssuerProperty

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
    parser.add_argument("--industry",
                        help="The industry level e.g. PHARM, CR, NFN, ALLSECTORS",
                        action="store",
                        default=None)
    parser.add_argument("--region",
                        help="The surface region e.g. LATM, NAMR, GLOBAL",
                        action="store",
                        default=None)
    parser.add_argument("-s", "--startDate", help="Start date", action="store", default=None)
    parser.add_argument("-e", "--endDate", help="End date", action="store", default=None)
    parser.add_argument("-p", "--parallelProcessing", help="Split tasks over multiple processors", action="store_true",
                        default=True)
    args = parser.parse_args()
    Currency = args.currency
    Industry = args.industry
    Region = args.region
    startDate = args.startDate

    params = {'Currency': Currency,
              'Industry': Industry,
              'Region':Region,
              'tradeDate':startDate}

    xcs = ClusterSurface(**params)
    xcs.run_single()


if __name__ == '__main__':
    t1 = time.clock()
    main()
    t2 = time.clock()
    print t2-t1
