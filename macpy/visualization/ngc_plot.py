import numpy as np
import macpy.utils.database as db
import plotly
import plotly.graph_objs as go
import colorlover as cl
import macpy.utils.ngc_utils as u
import macpy.utils.ngc_queries as q
import pandas as pd
import colorlover as cl
import plotly.plotly as py
from plotly import tools
import os
import plotly_layout as pl


def plot_curve_context(specific_curve, data_i, IndustryGroup, Sector, SectorGroup, Market,
                                  TenorGrid, smfit, splfun, plIndustries, TenorWeightDensityBandwidth, RiskEntityId, debugplot_curve=None,
                                curve_col='CurveId', spread_col='logOAS', tenor_col='EffectiveTenor',
                                    distance_col='distance', weight_col='weights',
                                  smfitLogxScale=True, levels_fit=True, RegionGroup='DVL'):

    ChartFolder = 'C:\\Users\\dantonio\\Documents\\Projects\\NextGenFit\\Figs\\'
    writeCharts = True

    IndustryGroup = q.get_industry_map_description(IndustryGroup)
    Sector = q.get_industry_map_description(Sector)
    SectorGroup = q.get_industry_map_description(SectorGroup)
    Market = q.get_industry_map_description(Market)

    print IndustryGroup, Sector, SectorGroup, Market

    x = data_i[tenor_col][data_i[curve_col]==specific_curve]#MatYears or EffectiveTenor
    x_curve = TenorGrid[TenorGrid <= x.max()]
    pldata = []
    N = len(data_i.groupby([curve_col]))
    ### just for color palette
#         col= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
    #from IPython.display import HTML
    #cols = cl.scales['9']['seq']['Greys']
    cols = cl.scales['9']['qual']['Set1']
    col = cl.interp( cols, np.int_(np.ceil(N/10.)*10)) # multiple of 10s for some reason
    kmeds = np.array(data_i['wkil'].groupby(data_i[curve_col]).apply(np.median)) # curve-wise median
    kmorder = kmeds.argsort() # to reverse: [::-1]
#         print kmorder
#         print data[curve_col].unique()[kmorder]
    ###

    if levels_fit:
        subtitle = 'LEVELS'
    else:
        subtitle = 'CHANGES'

    dotsizenorm = 30. / max(data_i[weight_col][data_i[curve_col] == specific_curve])
    data_i['plw'] = 10. * np.log(1. + data_i[weight_col] * dotsizenorm)
    data_i['plw'] = np.maximum(data_i['plw'], 1.)
    data_i['plw_text'] = np.round(1000. * data_i[weight_col], 1)
    data_i['plw_text'] = data_i['plw_text'].astype(str)
    opacitynorm = 1. / np.median(data_i['wkil'][data_i[curve_col]==specific_curve])

    indHier = ['IndustryGroup','Sector','SectorGroup','Market']
    #ildist_dict = {4:'ALL',3:Market,2:SectorGroup,1:Sector,0:IndustryGroup}
    if RegionGroup == 'DVL':
        ildist_dict = {5:'Global AllS',4:'Global ' + Market,3:'Global ' + SectorGroup,2:'Global ' + Sector,1:'Global ' + IndustryGroup,0:'Region ' + IndustryGroup}
    elif RegionGroup == 'EMG':
        ildist_dict = {5:'Global AllS',4:'Region AllS ',3: 'Region ' + Market,2:'Region ' + SectorGroup,1:'Region ' + Sector,0:'Region ' + IndustryGroup}

    data_i['IL_dist_shift'] = data_i['IL_dist'].apply(lambda x: x + plIndustries['d_min'])

    indHier = {'IndustryGroup':IndustryGroup,'Sector':Sector,'SectorGroup':SectorGroup,'Market':Market}
    tmp = data_i.loc[(data_i[curve_col]==specific_curve) & (data_i['outlier']!=True)]
    trace0= go.Scatter(
                    x= tmp[tenor_col],
                    y= u.mExp(tmp['y']) if levels_fit else tmp['y'],
                    mode= 'markers',
                    marker= dict(size= tmp['plw'],
                                line= dict(width=5,color=cols[0]),
                                 color=cols[0]
                               ),
                    name= specific_curve,
                    text= specific_curve + ', w=' + tmp['plw_text']) # The hover text goes here...

    counter = 1
    for ild in np.sort(data_i['IL_dist_shift'].unique()):
        tmp = data_i.loc[(data_i['IL_dist_shift']==ild) & (data_i[curve_col]!=specific_curve) & (data_i['outlier']!=True)]
        trace1= go.Scatter(
                    x= tmp[tenor_col],
                    y= u.mExp(tmp['y']) if levels_fit else tmp['y'],
                    mode= 'markers',
                    marker= dict(size= tmp['plw'],
                                line= dict(width=5, color=cols[int(ild)+1]),
                                 color=cols[int(ild)+1]
                               ),
                    name= plIndustries[ild],
                    text= tmp['TICKER']+ ', w=' + tmp['plw_text'],
                    opacity=0.5)
        pldata.append(trace1)
        counter+=1
    pldata.append(trace0)

    tmp = data_i.loc[(data_i[curve_col]==specific_curve) & (data_i['outlier']==True)]
    if subtitle == 'LEVELS':
        trace0= go.Scatter(
                        x= tmp[tenor_col],
                        y= u.mExp(tmp['y']) if levels_fit else tmp['y'],
                        mode= 'markers',
                        marker= dict(size= 10.,
                                    line= dict(width=5,color=cols[6]),
                                     color=cols[6]
                                   ),
                        name= specific_curve + ' OUTLIERS',
                        text= tmp['ISIN']) # The hover text goes here...
        pldata.append(trace0)
    tmp = data_i.loc[(data_i[curve_col]!=specific_curve) & (data_i['outlier']==True)]
    trace1= go.Scatter(
                x= tmp[tenor_col],
                y= u.mExp(tmp['y']) if levels_fit else tmp['y'],
                mode= 'markers',
                marker= dict(size= 10.,
                            line= dict(width=5, color=cols[7]),
                             color=cols[7]
                           ),
                name= 'Other OUTLIERS',
                text= tmp['TICKER'] + ', ' +  tmp['ISIN']) #The hover text goes here...
    pldata.append(trace1)

    pl_curve = specific_curve if not debugplot_curve else debugplot_curve # can also plot any other 'helper' curve # 'USD-PEGV-SEN'
    plx = data_i.loc[data_i[curve_col]==pl_curve, tenor_col]
    plxs = np.linspace(0, np.ceil(data_i[tenor_col].max()), 200)
    plxsc = u.logx1Compress(plxs, smfit['logx1Compress']['xCompressLong'],
                                      smfit['logx1Compress']['xCompressShort'],
                                      smfit['logx1Compress']['xLast'],
                                      smfit['logx1Compress']['xFirst'],
                                      smfit['logx1Compress']['compressionLong'],
                                      smfit['logx1Compress']['compressionShort'],
                                      widthLong=smfit['logx1Compress']['widthLong'],
                                      widthShort=smfit['logx1Compress']['widthShort']) if (smfitLogxScale) else plxs
    plys = u.mExp(splfun(plxsc)) if levels_fit else splfun(plxsc)

    #fitted curve
    trace1 = go.Scatter(x=plxs, y=plys, name='Fitted curve',mode='lines', line=dict(color=cols[0]))
    pldata.append(trace1)
    # layout = go.Layout(title=specific_curve + ', NBonds=' + str(data_i.shape[0]) + ', NIss=' + str(data_i['TICKER'].nunique()),
    #                    hovermode= 'closest', xaxis=dict(title = 'Effective Tenor'), yaxis=dict(title = 'spread'))
    AnalysisDate = data_i.AnalysisDate.unique()[0]
    IssuerName = data_i.loc[(data_i[curve_col] == specific_curve)].IssuerName.unique()[0]
    NumOutliers = str(data_i.loc[(data_i.outlier == True)].shape[0])
    title = IssuerName + ', ' + str(AnalysisDate)
    print 'NumIssuer', str(data_i['TICKER'].nunique())
    # title =str(AnalysisDate) + ': ' + IssuerName + ', REId=' + str(int(RiskEntityId)) + '<br>' + subtitle + ' ' + specific_curve + ', NBonds=' + str(
    #     data_i.shape[0]) + ', NOut=' + NumOutliers + ', NIss=' + str(data_i['TICKER'].nunique())

    layout = pl.PlotLayout1('Effective Tenor', 'Spread', width=1000, height=600, title=title)
    # layout = go.Layout(title=str(AnalysisDate) + ': ' + IssuerName + ', REId=' + str(int(RiskEntityId)) + '<br>' + subtitle + ' ' + specific_curve + ', NBonds=' + str(
    #     data_i.shape[0]) + ', NOut=' + NumOutliers + ', NIss=' + str(data_i['TICKER'].nunique()),
    #                    hovermode='closest', xaxis=dict(title='Effective Tenor'), yaxis=dict(title='spread'))
    plotly.offline.iplot(go.Figure(data=pldata, layout = layout))

    if writeCharts:
        py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG')
        filename = ChartFolder + '%s_%s_fit.png' % (specific_curve, subtitle)
        fig = go.Figure(data=pldata, layout=layout)
        py.image.save_as(fig, filename=filename, format='png', scale=6.)

    ####################################################################################################3
    #plot by issuer ###################################################################################
    plys = splfun(plxsc) if levels_fit else splfun(plxsc)  #
    idata=[]
    tmp = data_i.loc[(data_i[curve_col]==specific_curve) & (data_i['outlier']!=True)]

    N = data_i['TICKER'].nunique() -1 # number of issuers excluding specific curve
    colsi = cl.scales['9']['seq']['Greys'][3:]
    if N>1:
        col = cl.interp( colsi, np.int_(np.ceil(N/10.)*10)) # multiple of 10s for some reason
    else:
        col = colsi

    trace0= go.Scatter(
                    x= tmp[tenor_col],
                    y= tmp[spread_col] if levels_fit else tmp[spread_col],  #u.mExp(tmp[spread_col]) if levels_fit else tmp[spread_col],
                    mode= 'markers',
                    marker= dict(size= tmp['plw'],
                                line= dict(width=5,color=cols[0]),
                                 color=cols[0]
                               ),
                    name= specific_curve,
                    text= specific_curve) # The hover text goes here...

    counter=0
    data_i['mod_dist'] = data_i[distance_col].abs()
    data_i = data_i.sort_values('mod_dist', ascending=False)
    legFlag = True
    for ticker in data_i.loc[(data_i[curve_col]!=specific_curve)]['TICKER'].unique():
        tmp = data_i.loc[(data_i['TICKER']==ticker) & (data_i[curve_col]!=specific_curve) & (data_i['outlier']!=True)]
        dist = tmp[distance_col].values[0]
        trace1= go.Scatter(
                    x= tmp[tenor_col],
                    y= tmp[spread_col] if levels_fit else tmp[spread_col],  #u.mExp(tmp[spread_col]) if levels_fit else tmp[spread_col],
                    mode= 'markers',
                    marker= dict(size= tmp['plw'],
                                line= dict(width=5, color=col[counter]),
                                 color=col[counter]
                               ),
                    name= 'Peer support',
                    text=  'dist=' + str(dist) + ', ' + tmp['ISIN'],
                    showlegend=legFlag,
                    opacity=0.5)
        idata.append(trace1)
        trace1 = go.Scatter(x=plxs, y=splfun(plxsc) - dist if levels_fit else splfun(plxsc) - dist,
                            name='Peer curves', mode='lines', line=dict(color=col[counter]),showlegend=legFlag)
        #trace1 = go.Scatter(x=plxs, y=u.mExp(splfun(plxsc)-dist) if levels_fit else splfun(plxsc)-dist, name=ticker + ' curve',mode='lines',line=dict(color=col[counter]))
        idata.append(trace1)
        counter+=1
        legFlag = False
    idata.append(trace0)
    trace1 = go.Scatter(x=plxs, y=plys, name='Fitted curve', mode='lines', line=dict(color=cols[0]))
    idata.append(trace1)
    # title = str(AnalysisDate) + ': ' + IssuerName + ', REId=' + str(
    #     int(RiskEntityId)) + '<br>' + subtitle + ' ' + specific_curve + ', NBonds=' + str(
    #     data_i.shape[0]) + ', NOut=' + NumOutliers + ', NIss=' + str(data_i['TICKER'].nunique())
    title = IssuerName + ', ' + str(AnalysisDate)
    layout = pl.PlotLayout1('Effective Tenor', 'Log Spread', width=1000, height=800, title=title)
    # layout = go.Layout(
    #     title=str(AnalysisDate) + ': ' + IssuerName + ', REId=' + str(int(RiskEntityId)) + '<br>' + subtitle + ' ' + specific_curve + ', NBonds=' + str(
    #         data_i.shape[0]) + ', NOut=' + NumOutliers + ', NIss=' + str(data_i['TICKER'].nunique()),
    #     hovermode='closest', xaxis=dict(title='Effective Tenor'), yaxis=dict(title='spread'))
    plotly.offline.iplot(go.Figure(data=idata, layout = layout))

    if writeCharts:
        py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG')
        filename = ChartFolder + '%s_%s_byIssuer.png' % (specific_curve, subtitle)
        fig = go.Figure(data=idata, layout=layout)
        py.image.save_as(fig, filename=filename, format='png', scale=6.)


    ####################################################################################################
    ######## weight histogram
    ####################################################################################################
    wdata=[]
    tmp = data_i.loc[data_i[curve_col]==specific_curve]
    x_grid = np.arange(0.,40.,0.5)
    ao_pdf = u.AmtOutstanding_pdf(tmp, x_grid, tenor_col, weight_col,TenorWeightDensityBandwidth)
    trace0= go.Scatter(
                    x= tmp[tenor_col],
                    y= u.mExp(tmp[spread_col]) if levels_fit else tmp[spread_col],
                    mode= 'markers',
                    marker= dict(size= tmp['plw'],
                                line= dict(width=2,color=cols[0]),
                                 color=cols[0]
                               ),
                    name= specific_curve,
                    text= specific_curve)
    wdata.append(trace0)
    trace1 = go.Scatter(
                    x= x_grid,
                    y= ao_pdf,
                    mode= 'lines',
                    line=dict(color=cols[1],shape='spline'),
                    name= 'Amt PDF',
                    yaxis='y2')
    wdata.append(trace1)

    layout=go.Layout(title='', xaxis=go.XAxis(title = 'Tenor'),
                                                        yaxis=go.YAxis(title = 'spread',anchor='x',showgrid=False),yaxis2=go.YAxis(title='density',overlaying='y',side='right',
                                                                                                                    anchor='x',showgrid=False))

    layout = pl.PlotLayoutDualAxis('Tenor', 'Spread', y2Title='Amt Density',width=1000, height=600, title=IssuerName + ': Amt Outstanding PDF')
    plotly.offline.iplot(go.Figure(data=wdata, layout = layout))

    if writeCharts:
        py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG')
        filename = ChartFolder + '%s_%s_weights.png' % (specific_curve, subtitle)
        fig = go.Figure(data=wdata, layout=layout)
        py.image.save_as(fig, filename=filename, format='png', scale=6.)

    #########################################################################################################
    ####   weight distribution chart
    #########################################################################################################
    data=[]
    wSum=data_i[weight_col].sum()
    x = np.array(data_i[distance_col])
    w = np.array(data_i[weight_col])
    x_grid = np.arange(-1.,1.,0.05)
    pdf = u.kde(x, x_grid, weights=w, bandwidth=0.05) #bandwidth automatically selected

    #pdf, x_grid = u.calc_pdf(x,y,0.1)
    trace = go.Scatter(x=x_grid, y=pdf, mode='lines', name='weight pdf')
    data.append(trace)
    tmp = data_i.loc[data_i[curve_col]==specific_curve]
    x=tmp[distance_col]
    y=tmp[weight_col]
    trace = go.Scatter(x=x, y=y, mode='markers', name=specific_curve, yaxis='y2',marker=dict(color=cols[0]))
    data.append(trace)
    counter=1
    for ild in data_i['IL_dist_shift'].unique():
        x=data_i.loc[(data_i[curve_col]!=specific_curve) & (data_i['IL_dist_shift']==ild),distance_col]
        y=data_i.loc[(data_i[curve_col]!=specific_curve) & (data_i['IL_dist_shift']==ild),weight_col]
        trace = go.Scatter(x=x, y=y, mode='markers', name=plIndustries[ild], yaxis='y2', marker=dict(color=cols[counter]))
        data.append(trace)
        counter+=1
    layout=go.Layout(title='', xaxis=go.XAxis(title = 'distance'),
                                                        yaxis=go.YAxis(title = 'wieght density',anchor='x',showgrid=False,rangemode='tozero'),yaxis2=go.YAxis(title='weight',overlaying='y',side='right',
                                                                                                                    anchor='x',showgrid=False,rangemode='tozero'))
    plotly.offline.iplot(go.Figure(data=data, layout = layout))

    #################################################################################################
    #### Weight pie chart
    #################################################################################################

    labels=[]
    values=[]
    colors=[]

    value_specific = data_i.loc[data_i[curve_col] == specific_curve][weight_col].sum()
    labels.append(specific_curve)
    values.append(value_specific)
    colors.append(cols[0]) #issuer color

    #dg = data_i.groupby('IL_dist_shift')[weight_col].sum()
    #print dg
    if value_specific/data_i[weight_col].sum() < 0.999:  #if values are too small the pie chart crashes
        counter=1
        for ild in data_i['IL_dist_shift'].unique():
            labels.append(plIndustries[ild])
            weightSum_i = data_i.loc[(data_i['IL_dist_shift'] == ild) & (data_i[curve_col] != specific_curve)][weight_col].sum()
            values.append(weightSum_i)
            colors.append(cols[counter])
            counter+=1

    trace = go.Pie(labels=labels, values=values, marker = dict(colors = colors))
    layout = go.Layout(title='Industry Sector Weighting %s' % subtitle)

    plotly.offline.iplot(go.Figure(data=[trace],layout=layout))

    data_i.pop('IL_dist_shift')  #this column was just used for charting

    # plotly.offline.iplot(go.Figure(data=data,
    #                         layout=go.Layout(title='Weight pdf', xaxis=dict(title = 'Distance'), yaxis=dict(title = 'Density'))))

    # #distance vs. weights
    # data=[]
    # tmp = data_i.loc[data_i[curve_col]==specific_curve]
    # x=tmp[distance_col]
    # y=tmp[weight_col]
    # trace = go.Scatter(x=x, y=y, mode='markers', name=specific_curve)
    # data.append(trace)
    # for i in data_i['IL_dist'].unique():
    #     x=data_i.loc[(data_i[curve_col]!=specific_curve) & (data_i['IL_dist']==i),distance_col]
    #     y=data_i.loc[(data_i[curve_col]!=specific_curve) & (data_i['IL_dist']==i),weight_col]
    #     trace = go.Scatter(x=x, y=y, mode='markers', name=ildist_dict[i] )
    #     data.append(trace)
    # plotly.offline.iplot(go.Figure(data=data,
    #                         layout=go.Layout(title='Weight vs. Distance', xaxis=dict(title = 'Distance'), yaxis=dict(title = 'Weight'))))



    return 0


# def plot_final_curve():
#     ##



def plot_curve(specific_curve, data, curve_params, all_params, TenorGrid, plotweights=True, plotcurve=True, plotcontext=True, plotadjlevel = True, curve_col='CurveId', spread_col='logOAS', tenor_col='EffectiveTenor', distance_col='distance', weights_col='weights', kernel_weights_col = 'kernel_weights', MatYears_col='MatYears'):
    if (plotweights) :
        x = np.linspace(data[distance_col].min(), data[distance_col].max(), 100)
        y = u.kde(data[distance_col], x, bandwidth=0.2)
        w = u.kde(data[weights_col], x, bandwidth=0.2)
        plotly.offline.iplot(go.Figure(data=[go.Scatter(x=x, y=y, name='density of approx logOAS level'),
                                             go.Scatter(x=x, y=w, name='Gaussian weighting kernel')
                                            ], layout=go.Layout(title='', xaxis=dict(title = 'distance'), yaxis=dict(title = 'density'))))

    if (plotcurve):
        x = data[MatYears_col][data[curve_col]==specific_curve]#MatYears or EffectiveTenor
        xt = data[tenor_col][data[curve_col]==specific_curve]#MatYears or EffectiveTenor
        y = data[spread_col][data[curve_col]==specific_curve]
        dotsize = data[weights_col][data[curve_col]==specific_curve]
        dotsize = dotsize / max(dotsize) * 50
        x_curve = TenorGrid[TenorGrid <= x.max()]
        y_curve = u.paramfun(curve_params[0], curve_params[1], curve_params[2], TenorGrid)
        y_curve = y_curve[TenorGrid <= x.max()]
        # curve is plot in spread space (not logSpread)
        plotly.offline.iplot(go.Figure(data=[go.Scatter(x=xt, y=u.mExp(y), marker=dict(size=dotsize), mode= 'markers', name='quotes (EffTenor)'),
                                             go.Scatter(x=x_curve, y=u.mExp(y_curve), name=specific_curve),
                                             go.Scatter(x=x, y=u.mExp(y), mode= 'markers', name='quotes (Mat)')
                                            ], layout=go.Layout(title=specific_curve, xaxis=dict(title = 't'), yaxis=dict(title = 'spread'))))

    if (plotcontext):
        x = data[tenor_col][data[curve_col]==specific_curve]#MatYears or EffectiveTenor
        x_curve = TenorGrid[TenorGrid <= x.max()]
        pldata = []
        N = len(data.groupby([curve_col]))

        ### just for color palette
#         col= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

        #from IPython.display import HTML
        cols = cl.scales['9']['seq']['Greys']
        col = cl.interp( cols, np.int_(np.ceil(N/10.)*10)) # multiple of 10s for some reason
        kmeds = np.array(data[kernel_weights_col].groupby(data[curve_col]).apply(np.median)) # curve-wise median
        kmorder = kmeds.argsort() # to reverse: [::-1]
#         print kmorder
#         print data[curve_col].unique()[kmorder]
        ###

        dotsizenorm = 30. / max(data[weights_col][data[curve_col]==specific_curve])
        opacitynorm = 1. / np.median(data[kernel_weights_col][data[curve_col]==specific_curve])
        i = 0
        for curveid, grp in data.groupby([curve_col]):
        #     print curveid
        #     print grp.EffectiveTenor.max()
            col_i = 'steelblue' if (curveid == specific_curve) else col[kmorder[i]]
            trace0= go.Scatter(
                        x= grp[tenor_col],
                        y= grp[spread_col],
                        mode= 'markers',
                        marker= dict(size= data[weights_col][data[curve_col]==curveid] * dotsizenorm,
                                    line= dict(width=1),
                                    color= col_i,
                                    opacity = np.median(data[kernel_weights_col][data[curve_col]==curveid]) * opacitynorm
                                   ),name= curveid,
                        text= curveid) # The hover text goes here...
            pldata.append(trace0)

            coeff1 = curve_params[0] if plotadjlevel else all_params[curve_col+'['+curveid+']']
            y_curve = u.paramfun(coeff1, curve_params[1], curve_params[2], TenorGrid)
            linewidth = np.median(data[kernel_weights_col][data[curve_col]==curveid]) * opacitynorm * 5#np.mean(weights[dspread.CurveId==curveid]) * dotsizenorm/10
            trace1 = go.Scatter(x=x_curve, y=y_curve,
                                mode = 'lines',
                                #marker = dict(
                                line = dict(width=linewidth,
                                     color= col_i),
                                showlegend=False, opacity = np.mean(data[weights_col][data[curve_col]==curveid]) * opacitynorm
                                )
            if plotadjlevel :
                if (curveid == specific_curve) :
                    pldata.append(trace1)
            else :
                pldata.append(trace1)
            i += 1 ;
        layout = go.Layout(title=specific_curve,  hovermode= 'closest', xaxis=dict(title = 't'), yaxis=dict(title = 'log(spread)'))
        plotly.offline.iplot(go.Figure(data=pldata, layout = layout))

    return 0

def plot_level_comparison(s_i,s_f,start_date,end_date,specific_curve,curve_col,tenor_col,weight_col,
                            res_i,res_f,TenorGrid):
    cols = cl.scales['9']['qual']['Set1']
    s_f = s_f[s_f[curve_col]==specific_curve]
    s_i = s_i[s_i[curve_col]==specific_curve]
    plx_f = s_f[tenor_col]
    ply_f = s_f['OAS_Swap']
    plx_i = s_i[tenor_col]
    ply_i = s_i['OAS_Swap']
    dotsizenorm = 30. / max(s_f[weight_col][s_f[curve_col]==specific_curve])
    plotly.offline.iplot(go.Figure(data=[
                            go.Scatter(x=plx_i, y=ply_i, mode='markers', name=specific_curve + ' ' + start_date, marker= dict(size= s_i[weight_col] * dotsizenorm,
                                                                                                                              color=cols[1],
                                                                                                                              line= dict(width=5,color=cols[1])),
                                       text=s_i['ISIN']),
                            go.Scatter(x=plx_f, y=ply_f, mode='markers', name=specific_curve + ' ' + end_date, marker= dict(size= s_f[weight_col] * dotsizenorm,
                                                                                                                            color=cols[0],
                                                                                                                            line= dict(width=5,color=cols[0])),
                                       text=s_i['ISIN']),
                            go.Scatter(x=TenorGrid, y=u.mExp(np.array(res_i['yout'])),name='fit %s' % start_date, mode='lines',line=dict(color=cols[1])),
                            go.Scatter(x=TenorGrid, y=u.mExp(np.array(res_f['yout'])),name='fit %s' % end_date, mode='lines',line=dict(color=cols[0]))],
                            layout = go.Layout(title=specific_curve + ' - levels change',
                            hovermode= 'closest', xaxis=dict(title = 'Effective Tenor'), yaxis=dict(title = 'spread'))))

    return 0

def plot_curve_multi(specific_curve, data_i, IndustryGroup, Sector, SectorGroup, Market,
                                  TenorGrid, smfit, splfun, plIndustries, Currency, coupon, IssuerBonds, TenorWeightDensityBandwidth, RiskEntityId, debugplot_curve=None,
                                curve_col='CurveId', spread_col='logOAS', tenor_col='EffectiveTenor',
                                    distance_col='distance', weight_col='weights',
                                  smfitLogxScale=True, levels_fit=True, RegionGroup='DVL', Region=''):

    showOutliers = False  #set to False to make the display charts more attractive
    IndustryGroup = q.get_industry_map_description(IndustryGroup)
    Sector = q.get_industry_map_description(Sector)
    SectorGroup = q.get_industry_map_description(SectorGroup)
    Market = q.get_industry_map_description(Market)

    x = data_i[tenor_col][data_i[curve_col]==specific_curve]#MatYears or EffectiveTenor
    x_curve = TenorGrid[TenorGrid <= x.max()]
    pldata = []
    N = len(data_i.groupby([curve_col]))

    cols = cl.scales['9']['qual']['Set1']
    col = cl.interp( cols, np.int_(np.ceil(N/10.)*10)) # multiple of 10s for some reason
    kmeds = np.array(data_i['wkil'].groupby(data_i[curve_col]).apply(np.median)) # curve-wise median
    #kmorder = kmeds.argsort() # to reverse: [::-1]

    dotsizenorm = 30. / max(data_i[weight_col][data_i[curve_col]==specific_curve])
    data_i['plw'] = 10.*np.log(1.+ data_i[weight_col] * dotsizenorm)
    data_i['plw'] = np.maximum(data_i['plw'],1.)
    data_i['plw_text'] = np.round(1000.*data_i[weight_col],1)
    data_i['plw_text'] = data_i['plw_text'].astype(str)
    #data_i['plw'] = data_i[weight_col] * dotsizenorm
    opacitynorm = 1. / np.median(data_i['wkil'][data_i[curve_col]==specific_curve])

    if RegionGroup == 'DVL':
        ildist_dict = {5:'Global AllS',4:'Global-' + Market,3:'Global-' + SectorGroup,2:'Global-' + Sector,1:'Global-' + IndustryGroup,0: Region + '-' + IndustryGroup}
    elif RegionGroup == 'EMG':
        ildist_dict = {5:'Global AllS',4:Region + '-AllS ',3: Region + '-' + Market,2: Region + '-' + SectorGroup,1: Region + '-' + Sector,0: Region + '-' + IndustryGroup}

    data_i['IL_dist_shift'] = data_i['IL_dist'].apply(lambda x: x + plIndustries['d_min'])
    #fig = tools.make_subplots(rows=1, cols=2)  #multi plot

    tmp = data_i.loc[(data_i[curve_col]==specific_curve) & (data_i['outlier']!=True)]
    #data_i.to_clipboard()
    #tmp[weight_col] = np.minimum(np.array(tmp[weight_col]),100)

    trace0= go.Scatter(
                    x= tmp[tenor_col],
                    y= u.mExp(tmp['y']) if levels_fit else tmp['y'],
                    mode= 'markers',
                    marker= dict(size= tmp['plw'],
                                line= dict(width=5,color=cols[0]),
                                 color=cols[0]
                               ),
                    name= specific_curve,
                    text= tmp['TICKER'] + ', w = ' + tmp['plw_text'] + '<br>' + 'ISIN=' + tmp['ISIN']) # The hover text goes here...

    counter = 1
    for ild in np.sort(data_i['IL_dist_shift'].unique()):
        tmp = data_i.loc[(data_i['IL_dist_shift']==ild) & (data_i[curve_col]!=specific_curve) & (data_i['outlier']!=True)]
        trace1= go.Scatter(
                    x= tmp[tenor_col],
                    y=u.mExp(tmp['y']) if levels_fit else tmp['y'],
                    mode= 'markers',
                    marker= dict(size= tmp['plw'],
                                line= dict(width=5, color=cols[int(ild)+1]),
                                 color=cols[int(ild)+1]
                               ),
                    name= plIndustries[ild],
                    text= tmp['TICKER'] + ', w = ' + tmp['plw_text'] + '<br>' + 'ISIN=' + tmp['ISIN'],
                    opacity=0.5)
        pldata.append(trace1)
        #fig.append_trace(trace1, 1, 1)
        counter+=1
    pldata.append(trace0)
    #fig.append_trace(trace0, 1, 1)

    tmp = data_i.loc[(data_i[curve_col]==specific_curve) & (data_i['outlier']==True)]
    trace0= go.Scatter(
                    x= tmp[tenor_col],
                    y=u.mExp(tmp['y']) if levels_fit else tmp['y'],
                    mode= 'markers',
                    marker= dict(size= 10.,
                                line= dict(width=5,color=cols[6]),
                                 color=cols[6]
                               ),
                    name= specific_curve + ' OUTLIERS',
                    text= tmp['TICKER'] + '<br>' + 'ISIN=' + tmp['ISIN']) # The hover text goes here...
    pldata.append(trace0)

    #fig.append_trace(trace0, 1, 1)

    if showOutliers:
        tmp = data_i.loc[(data_i[curve_col]!=specific_curve) & (data_i['outlier']==True)]
        trace1= go.Scatter(
                    x= tmp[tenor_col],
                    y=u.mExp(tmp['y']) if levels_fit else tmp['y'],
                    mode= 'markers',
                    marker= dict(size= 10.,
                                line= dict(width=5, color=cols[7]),
                                 color=cols[7]
                               ),
                    name= 'Other OUTLIERS',
                    text= tmp['TICKER'] + ', ' + '<br>' + 'ISIN=' + tmp['ISIN']) #The hover text goes here...
        pldata.append(trace1)
    #fig.append_trace(trace1, 1, 1)

    pl_curve = specific_curve if not debugplot_curve else debugplot_curve # can also plot any other 'helper' curve # 'USD-PEGV-SEN'
    plx = data_i.loc[data_i[curve_col]==pl_curve, tenor_col]
    plxs = np.linspace(0, np.ceil(data_i[tenor_col].max()), 200)
    plxsc = u.logx1Compress(plxs, smfit['logx1Compress']['xCompressLong'],
                                      smfit['logx1Compress']['xCompressShort'],
                                      smfit['logx1Compress']['xLast'],
                                      smfit['logx1Compress']['xFirst'],
                                      smfit['logx1Compress']['compressionLong'],
                                      smfit['logx1Compress']['compressionShort'],
                                      widthLong=smfit['logx1Compress']['widthLong'],
                                      widthShort=smfit['logx1Compress']['widthShort']) if (smfitLogxScale) else plxs
    plys = u.mExp(splfun(plxsc)) if levels_fit else splfun(plxsc)

    #fitted curve
    trace1 = go.Scatter(x=plxs, y=plys, name='Fitted curve',mode='lines', line=dict(color=cols[0],shape='spline'))
    pldata.append(trace1)
    trace1 = go.Scatter(x=TenorGrid, y=coupon/100., name='coupon', mode='lines', line=dict(color=cols[8]))
    pldata.append(trace1)
    IssuerBonds = IssuerBonds[IssuerBonds.outlier ==False]
    trace1 = go.Scatter(x=IssuerBonds.Effective_Duration, y=IssuerBonds.OAS_Model, name='OAS Model', mode='markers', marker=dict(color='rgb(0,0,0)',symbol='cross'))
    pldata.append(trace1)
    #fig.append_trace(trace1, 1, 1)
    AnalysisDate = data_i.AnalysisDate.unique()[0]
    IssuerName = data_i.loc[(data_i[curve_col]==specific_curve)].IssuerName.unique()[0]
    NumOutliers = str(data_i.loc[(data_i.outlier == True)].shape[0])
    layout = go.Layout(
        title=str(AnalysisDate) + ': ' + IssuerName + ', REId=' + str(int(RiskEntityId)) + '<br>' + 'LEVELS' + ' ' + specific_curve + ', NBonds=' + str(
            data_i.shape[0]) + ', NOut=' + NumOutliers + ', NIss=' + str(data_i['TICKER'].nunique()),
        hovermode='closest', xaxis=dict(title='Effective Tenor'), yaxis=dict(title='spread'))

    py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG') # Replace the username, and API key with your credentials.
    fig = go.Figure(data=pldata,layout=layout)
    if levels_fit:
        ChartFolder = 'C:\Users\dantonio\Documents\Projects\NextGenFit\Charts1\%s\%s\LEVELS\\' % (Currency, IndustryGroup)
    else:
        ChartFolder = 'C:\Users\dantonio\Documents\Projects\NextGenFit\Charts1\%s\%s\CHANGES\\' % (Currency, IndustryGroup)
    directory = os.path.dirname(ChartFolder)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    if levels_fit:
        filename=ChartFolder + '%s_levels.html' % specific_curve
    else:
        filename = ChartFolder + '%s_changes.html' % specific_curve
    #py.image.save_as(fig, filename=filename,format='pdf')
    plotly.offline.plot({'data':pldata,'layout':layout},filename=filename,show_link=False,auto_open=False)
    #plotly.offline.iplot(go.Figure(data=pldata, layout = layout))

    data_i.pop('IL_dist_shift')

    return 0

def plot_3D_cluster_surface(ratingscale, tenorGrid, surfc, Currency, region, industryGroup,notebookMode=False):
    data = [go.Surface(x=ratingscale.Rating, y=tenorGrid, z=surfc, colorscale='Viridis')]

    layout = go.Layout(
        width=800,
        height=700,
        autosize=False,
        title='Cluster surface\n' + Currency + ' - ' + region + ' - ' + industryGroup,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                title='Rating',
                #             ticktext= ['AAA','A','BBB','B', 'C'],
                #                         tickvals= [0, 6, 15, 12, 20],
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                title='Maturity',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                title='log spread',
                ticktext=[''],
                tickvals=[290],
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.25, y=-1.25, z=0.25)
            ),
            aspectmode='manual'
        )
    )

    fig = dict(data=data, layout=layout)

    ChartFolder = 'C:\Users\dantonio\Documents\Projects\NextGenFit\ClusterCharts\\'
    filename = ChartFolder + 'ClusterSurface-' + Currency + '-' + region + '-' + industryGroup + '.html'
    directory = os.path.dirname(ChartFolder)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    plotly.offline.plot({
        'data': data,
        'layout': layout},
        filename=filename, show_link=False, auto_open=False)
    if notebookMode:
        plotly.offline.iplot(fig)

    return 0

def plot_3D_cluster_scatter(ratingscale, tenorGrid, surfcT, dfOAS, Currency, region, industryGroup, notebookMode=False):
    col = cl.scales['7']['qual']['Set1']

    col21 = cl.interp(col, 30)

    dfOAS['w_compressed']  = u.compress_weights(dfOAS.AmtOutstanding, 1.)

    dotsizenorm = 8. / max(dfOAS['AmtOutstanding'])
    dfOAS['plw'] = 10.   #*dfOAS['w_compressed']  #10.*np.log(1+dfOAS['AmtOutstanding'] * dotsizenorm)
    dfOAS = dfOAS.loc[dfOAS.logOAS > -1.]


    data = []
    for r in ratingscale.RatingRank:
        trace1 = go.Scatter3d(
            x=np.ones(len(tenorGrid)) * r,
            y=tenorGrid,
            z=surfcT[r - 1],
            mode='lines',
            line=dict(
                color=col21[r - 1],
                width=2.
            ),
            name='Curve ' + np.array(ratingscale.Rating)[r - 1]
        )
        data.append(trace1)

        tmp = dfOAS[dfOAS.MDRank == r]
        y = np.array(tmp.Effective_Duration)
        z = np.array(tmp.logOAS)
        trace1 = go.Scatter3d(
            x=np.ones(len(y)) * r,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=tmp['plw'],
                line=dict(width=1, color=col21[r - 1]),
                color=col21[r - 1]
            ),
            name=np.array(ratingscale.Rating)[r - 1] + ' ' + 'MD',
            text = tmp.CurveShortName,
            showlegend = False
        )
        data.append(trace1)

        tmp = dfOAS[dfOAS.SPRank == r]
        y = np.array(tmp.Effective_Duration)
        z = np.array(tmp.logOAS)
        trace1 = go.Scatter3d(
            x=np.ones(len(y)) * r,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=tmp['plw'],
                line=dict(width=1, color=col21[r - 1]),
                color=col21[r - 1]
            ),
            name=np.array(ratingscale.Rating)[r - 1] + ' ' + 'SP',
            text = tmp.CurveShortName,
            showlegend=False
        )
        data.append(trace1)

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    layout = go.Layout(
        width=1000,
        height=800,
        autosize=False,
        title='Cluster surface\n' + Currency + '-' + region + '-' + industryGroup,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                title='Rating',
                ticktext=['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'C'],
                tickvals=[1, 3, 6, 9, 12, 15, 18, 21],
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                title='Duration',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                title='log(s)',
                #             ticktext= [''],
                #                         tickvals= [290],
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio=dict(x=0.7, y=1, z=0.7),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.2, y=-1.2, z=0.7)
            ),
            aspectmode='manual'
        )
    )

    fig = dict(data=data, layout=layout)
    #plotly.offline.iplot(fig)
    ChartFolder = 'C:\Users\dantonio\Documents\Projects\NextGenFit\ClusterCharts\\'
    filename = ChartFolder + 'ClusterScatter-' + Currency + '-' + region + '-' + industryGroup + '.html'
    directory = os.path.dirname(ChartFolder)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    plotly.offline.plot({
        'data': data,
        'layout': layout},
        filename=filename, show_link=False, auto_open=False)
    if notebookMode:
        plotly.offline.iplot(fig)

def plot_2D_cluster_scatter(rating, ratingscale, tenorGrid, surfcT, dfOAS,notebookMode=False):
    colsi = cl.scales['9']['seq']['Greys'][3:]
    cols = cl.scales['9']['qual']['Set1']
    col = cl.scales['7']['qual']['Set1']
    col21 = cl.interp(col, 30)
    dotsizenorm = 30. / max(dfOAS['AmtOutstanding'])
    dfOAS['plw'] = dfOAS['AmtOutstanding'] * dotsizenorm #10. * np.log(1.+dfOAS['AmtOutstanding'] * dotsizenorm)
    layout = pl.PlotLayout1('Maturity', 'Log(s)', width=900, title='')
    data = []
    r = ratingscale[ratingscale.Rating == rating].RatingRank.values[0]
    if r==1:
        radj = [2]
    elif r==21:
        radj=[20]
    else:
        radj=[r-1,r+1]
    print r

    trace1 = go.Scatter(
        x=tenorGrid,
        y=surfcT[r - 1],
        mode='lines',
        line=dict(
            color=col21[r - 1],
            width=2.
        ),
        name='Curve ' + np.array(ratingscale.Rating)[r - 1]
    )
    data.append(trace1)
    tmp = dfOAS[dfOAS.MDRank == r]
    x = np.array(tmp.Effective_Duration)
    y = np.array(tmp.logOAS)
    trace1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=tmp['plw'],
            line=dict(width=1, color=col21[r - 1]),
            color=col21[r - 1]
        ),
        name=np.array(ratingscale.Rating)[r - 1] + ' ' + 'MD'
    )
    data.append(trace1)

    tmp = dfOAS[dfOAS.SPRank == r]
    x = np.array(tmp.Effective_Duration)
    y = np.array(tmp.logOAS)
    trace1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=tmp['plw'],
            line=dict(width=1, color=col21[r - 1]),
            color=col21[r - 1]
        ),
        name=np.array(ratingscale.Rating)[r - 1] + ' ' + 'SP'
    )
    data.append(trace1)

## Add adjacent curves
    for r in radj:
        trace1 = go.Scatter(
            x=tenorGrid,
            y=surfcT[r - 1],
            mode='lines',
            line=dict(
                color=col21[r - 1],
                width=2.,
                dash='dash'
            ),
            name='Curve ' + np.array(ratingscale.Rating)[r - 1]
        )
        data.append(trace1)
        tmp = dfOAS[dfOAS.MDRank == r]
        x = np.array(tmp.Effective_Duration)
        y = np.array(tmp.logOAS)
        trace1 = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=tmp['plw'],
                line=dict(width=1, color=col21[r - 1]),
                color=col21[r - 1],
                opacity = 0.2
            ),
            name=np.array(ratingscale.Rating)[r - 1] + ' ' + 'MD'
        )
        data.append(trace1)

        tmp = dfOAS[dfOAS.SPRank == r]
        x = np.array(tmp.Effective_Duration)
        y = np.array(tmp.logOAS)
        trace1 = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=tmp['plw'],
                line=dict(width=1, color=colsi[1]),
                color=col21[r - 1],
                opacity=0.2
            ),
            name=np.array(ratingscale.Rating)[r - 1] + ' ' + 'SP'
        )
        data.append(trace1)

    fig = dict(data=data, layout=layout)
    if notebookMode:
        plotly.offline.iplot(fig)



##### OLD CHARTING CODE
   # if ((-1 in self.debugplot_ixlist) & lastIteration) & (self.debugplotSmoothFit & (-1 in self.debugplotSmoothFit_ixlist)):
                #     self.SmoothFit(df_fit.x, df_fit.y, w=df_fit.w, xout= self.TenorGrid) # only called to debug plot SmoothFit
                # pl_curve = self.specific_curve if not self.debugplot_curve else self.debugplot_curve # can also plot any other 'helper' curve # 'USD-PEGV-SEN'
                # #dist = -curvelevels[np.where(curvelist == self.specific_curve)] #float(curvelevels_prev[np.where(curvelist_prev == self.specific_curve)] - d.loc[d[self.curve_col]==pl_curve, 'curvelevel'])
                # pl_outlierInd = np.array(data_i.index)[np.array((data_i[self.curve_col]==pl_curve) & (data_i.outlier == True))]
                # pl_youtlier = data_i[(data_i[self.curve_col]==pl_curve) & (data_i.outlier==True)]
                # plx = data_i.loc[data_i[self.curve_col]==pl_curve, self.tenor_col]
                # plw = 1. #assetweight[np.array(data_i[self.curve_col]==pl_curve)]
                # plw = plw * 25. #/ max(plw)
                # plxs = np.linspace(0, np.ceil(np.array(plx).max()), 200)
                # plxsc = u.logx1Compress(plxs, smfit['logx1Compress']['xCompressLong'],
                #                           smfit['logx1Compress']['xCompressShort'],
                #                           smfit['logx1Compress']['xLast'],
                #                           smfit['logx1Compress']['xFirst'],
                #                           smfit['logx1Compress']['compressionLong'],
                #                           smfit['logx1Compress']['compressionShort'],
                #                           widthLong=smfit['logx1Compress']['widthLong'],
                #                           widthShort=smfit['logx1Compress']['widthShort']) if (self.smfitLogxScale) else plxs
                # plys1 = splfun(plxsc)
                # plys2 = splfun(plxsc)+adjustment[pl_curve]-specificshift
                # ply = data_i.loc[data_i[self.curve_col]==pl_curve, spread_col]
                # print 'chart 1'
                # plotly.offline.iplot(go.Figure(data=[
                #                         go.Scatter(x=plx, y=ply, mode='markers', name=pl_curve, marker=dict(size=plw)),
                #                         go.Scatter(x=plxs, y=plys1, name='post adjustment'), go.Scatter(x=plxs, y=plys2, name='fitted'),
                #                         go.Scatter(x=pl_youtlier[self.tenor_col], y=pl_youtlier[spread_col], mode='markers', name='outlier', marker=dict(color='red',size=10, opacity=0.4))]
                #                     , layout=go.Layout(title='Iteration %s, %s' % (str(it_ix+1), spread_col) , xaxis=dict(title = 't'), yaxis=dict(title = 'log(s)'))))

            #plotcontext=True
            #     if (plotcontext):
            #         x = data_i[self.tenor_col][data_i[self.curve_col]==self.specific_curve]#MatYears or EffectiveTenor
            #         x_curve = self.TenorGrid[self.TenorGrid <= x.max()]
            #         pldata = []
            #         N = len(data_i.groupby([self.curve_col]))
            #
            #         ### just for color palette
            # #         col= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
            #         import colorlover as cl
            #         #from IPython.display import HTML
            #         #cols = cl.scales['9']['seq']['Greys']
            #         cols = cl.scales['4']['qual']['Set1']
            #         col = cl.interp( cols, np.int_(np.ceil(N/10.)*10)) # multiple of 10s for some reason
            #         kmeds = np.array(data_i['wkil'].groupby(data_i[self.curve_col]).apply(np.median)) # curve-wise median
            #         kmorder = kmeds.argsort() # to reverse: [::-1]
            # #         print kmorder
            # #         print data[curve_col].unique()[kmorder]
            #         ###
            #
            #         dotsizenorm = 30. / max(data_i[self.weight_col][data_i[self.curve_col]==self.specific_curve])
            #         opacitynorm = 1. / np.median(data_i['wkil'][data_i[self.curve_col]==self.specific_curve])
            #
            #         indHier = ['IndustryGroup','Sector','SectorGroup','Market']
            #
            #         indHier = {'IndustryGroup':self.IndustryGroup,'Sector':self.Sector,'SectorGroup':self.SectorGroup,'Market':self.Market}
            #         tmp = data_i.loc[data_i[self.curve_col]==self.specific_curve]
            #         trace0= go.Scatter(
            #                         x= tmp[self.tenor_col],
            #                         y= tmp[spread_col],
            #                         mode= 'markers',
            #                         marker= dict(size= tmp['wkil'] * dotsizenorm,
            #                                     line= dict(width=1),
            #                                      color=cols[0]
            #                                    ),
            #                         name= self.specific_curve,
            #                         text= self.specific_curve) # The hover text goes here...
            #
            #         for ih in [0]:
            #             tmp = data_i.loc[(data_i['IL_dist']==ih) & (data_i[self.curve_col]!=self.specific_curve)]
            #             trace1= go.Scatter(
            #                         x= tmp[self.tenor_col],
            #                         y= tmp[spread_col],
            #                         mode= 'markers',
            #                         marker= dict(size= tmp['wkil'] * dotsizenorm,
            #                                     line= dict(width=1),
            #                                      color=cols[1]
            #                                    ),
            #                         name= self.IndustryGroup,
            #                         text= ih) # The hover text goes here...
            #             pldata.append(trace1)
            #         pldata.append(trace0)
            #
            #
            #
            #         #fitted curve
            #         trace1 = go.Scatter(x=plxs, y=plys2, name='fitted curve',mode='lines')
            #         pldata.append(trace1)
            #         layout = go.Layout(title=self.specific_curve,  hovermode= 'closest', xaxis=dict(title = 'Effective Tenor'), yaxis=dict(title = 'log(spread)'))
            #         print 'chart 4'
            #         plotly.offline.iplot(go.Figure(data=pldata, layout = layout))