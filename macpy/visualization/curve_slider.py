import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot
import plotly_layout as pl
from plotly import tools

from IPython.display import display, HTML
from plotly.offline import init_notebook_mode, iplot
import macpy.next_gen_curve as ngc
import macpy.utils.ngc_utils as u

import pandas as pd
import numpy as np
import colorlover as cl

init_notebook_mode(connected=True)
#dataset = pd.read_csv('cs_data_1.csv')



def curve_slider_sigma(dataset,curvesData,Ticker):
    dataset = dataset.loc[dataset['IL_dist']==0]
    sigmas=['0','0.01','0.02','0.03','0.04','0.05','0.1','0.2','0.3','0.4','0.5']
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': [],
        'config': {'scrollzoom': True}
    }

    # fill in most of layout
    figure['layout']['xaxis'] = {'range': [0., 20.], 'title': 'Effective Tenor'}
    figure['layout']['yaxis'] = {'range': [0., 3.], 'title': 'log(spread)'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['slider'] = {
        'args': [
            'slider.value', {
                'duration': 300,
                'ease': 'cubic-in-out'
            }
        ],
        'initialValue': '0.01',
        'plotlycommand': 'animate',
        'values': sigmas,
        'visible': True
    }
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 1000, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 1000, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                # {
                #     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                #     'transition': {'duration': 0}}],
                #     'label': 'Pause',
                #     'method': 'animate'
                # }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Sigma:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': .9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data
    sigma='0'
    # if float(sigma) > 0:
    #     dataset.loc[:,'kernel'] = dataset.distance.apply(lambda x: u.gaussian_kernel(x,float(sigma)))
    # else:
    #     dataset.loc[:,'kernel'] = dataset.distance.apply(lambda x: 1. if x==0 else 0.)
    # dataset['w'] = dataset.kernel.multiply(dataset.loc[:,'AmtOutstanding'])

    folder = 'C:\\ContentDev-MAC\\macpy\\visualization\\cs_data_sigma\\'
    filename = 'cs_data_sigma_0.csv'
    dataset = pd.read_csv(folder + filename)
    dataset_i = dataset.loc[dataset.TICKER == Ticker]
    dataset_o = dataset.loc[dataset.TICKER != Ticker]
    dotsizenorm = 50. / dataset_i.wkil.max()

    data_dict = {
        'x': list(dataset_o['Effective_Duration']),
        'y': list(dataset_o['y']),
        'mode': 'markers',
        'text': list(dataset_o['TICKER']),
        'marker': {
            'size': list(dataset_o['wkil']*dotsizenorm)
        },
        'name': 'Other SOVRNs'
    }
    figure['data'].append(data_dict)
    data_dict = {
        'x': list(dataset_i['Effective_Duration']),
        'y': list(dataset_i['y']),
        'mode': 'markers',
        'text': list(dataset['TICKER']),
        'marker': {
            'size': list(dataset_i['wkil']*dotsizenorm),
            'color':'green'
        },
        'name': Ticker
    }
    figure['data'].append(data_dict)
    data_dict = {
        'x': list(curvesData['x']),
        'y': list(curvesData[sigma]),
        'mode': 'lines',
                'line': {'color':'black'},
        'name': 'Fitted Curve'
    }
    figure['data'].append(data_dict)

    # make frames
    for sigma in sigmas:
        frame = {'data': [], 'name': str(sigma)}
        # if float(sigma) > 0:
        #     dataset.loc[:,'kernel'] = dataset.distance.apply(lambda x: u.gaussian_kernel(x,float(sigma)))
        # else:
        #     dataset.loc[:,'kernel'] = dataset.distance.apply(lambda x: 1. if x==0 else 0.)
        # dataset['w'] = dataset.kernel.multiply(dataset.loc[:,'AmtOutstanding'])

        filename = 'cs_data_sigma_%s.csv' % sigma
        dataset = pd.read_csv(folder + filename)
        dataset_i = dataset.loc[dataset.TICKER == Ticker]
        dataset_o = dataset.loc[dataset.TICKER != Ticker]

        data_dict = {
            'x': list(dataset_o['Effective_Duration']),
            'y': list(dataset_o['y']),
            'mode': 'markers',
            'text': list(dataset_o['TICKER']),
            'marker': {
                'size': list(dataset_o['wkil']*dotsizenorm)
            }
        }
        frame['data'].append(data_dict)
        data_dict = {
        'x': list(dataset_i['Effective_Duration']),
        'y': list(dataset_i['y']),
        'mode': 'markers',
        'text': list(dataset['TICKER']),
        'marker': {
            'size': list(dataset_i['wkil']*dotsizenorm),
            'color':'green'
        },
        'name': Ticker
        }
        frame['data'].append(data_dict)
        data_dict = {
        'x': list(curvesData['x']),
        'y': list(curvesData[sigma]),
        'mode': 'lines',
                'line': {'color':'black'},
        'name': 'Fitted Curve'
        }
        frame['data'].append(data_dict)

        figure['frames'].append(frame)
        slider_step = {'args': [
            [sigma],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 300}}
         ],
         'label': sigma,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)


    figure['layout']['sliders'] = [sliders_dict]

    plotly.offline.iplot(figure)
    plotly.offline.plot(figure, filename='sigma_slider_%s.html' % Ticker,auto_open=False)


def curve_slider_gamma(dataset,curvesData,Ticker):
    gammas=['0','0.01','0.02','0.03','0.04','0.05','0.1','0.2','0.3','0.4','0.5']
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': [],
        'config': {'scrollzoom': True}
    }

    # fill in most of layout
    figure['layout']['xaxis'] = {'range': [0., 20.], 'title': 'Effective Tenor'}
    figure['layout']['yaxis'] = {'range': [-.5, 3.], 'title': 'log(spread)'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['slider'] = {
        'args': [
            'slider.value', {
                'duration': 300,
                'ease': 'cubic-in-out'
            }
        ],
        'initialValue': '0.01',
        'plotlycommand': 'animate',
        'values': gammas,
        'visible': True
    }
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 1000, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 1000, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                # {
                #     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                #     'transition': {'duration': 0}}],
                #     'label': 'Pause',
                #     'method': 'animate'
                # }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Gamma:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': .9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data
    #year = 1952
    gamma='0'
    dataset.loc[:,'IL_weight'] = dataset['IL_dist'].apply(lambda x: u.ind_level_weight_calc(x,float(gamma)))
    dataset['w'] = dataset.kernel.multiply(dataset.loc[:,'AmtOutstanding']).multiply(dataset.loc[:,'IL_weight'])
    dataset_i = dataset.loc[dataset.TICKER == Ticker]
    dataset_ig = dataset.loc[(dataset.TICKER != Ticker) & (dataset.IndustryGroup == 'HSEPP')]
    dataset_ig = dataset.loc[(dataset.TICKER != Ticker) & (dataset.IndustryGroup == 'HSEPP')]
    dotsizenorm = 100. / dataset_i.w.max()
    print 'dotsize', dotsizenorm
    min =1.

    industries = {0:'Household & PPs',1:'Consumer Staples',2:'Non Financials',3:'Corporates',4:'All'}
    cols = cl.scales['5']['qual']['Set1']
    tmp_x = [1000.]
    tmp_y = [1000.]
    tmp_w = [100000.]

    for il in dataset['IL_dist'].unique():
        dataset_o = dataset.loc[(dataset.TICKER != Ticker) & (dataset['IL_dist'] == il)]
        data_dict = {
            'x': tmp_x + list(dataset_o['Effective_Duration']),
            'y': tmp_y + list(dataset_o['logOAS']),
            'mode': 'markers',
            'text': list(dataset_o['TICKER']),
            'marker': {
                'size': tmp_w + list(dataset_o['w']*dotsizenorm),
                'color':cols[il+1]
            },
            'name': industries[il],
            'showlegend':True
        }
        figure['data'].append(data_dict)

        # ##add points just for legend formatting
        # data_dict = {
        #     'x': [100.0],
        #     'y': [100.0],
        #     'mode': 'markers',
        #     'marker': {
        #         'size': 12.0,
        #         'color':cols[il+1]
        #     },
        #     'name': industries[il]
        # }
        # figure['data'].append(data_dict)
        # ## end add points just for legend formatting


    data_dict = {
        'x': list(dataset_i['Effective_Duration']),
        'y': list(dataset_i['logOAS']),
        'mode': 'markers',
        'text': list(dataset['TICKER']),
        'marker': {
            'size': list(dataset_i['w']*dotsizenorm),
            'color':cols[0]
        },
        'name': Ticker
    }
    figure['data'].append(data_dict)
    data_dict = {
        'x': list(curvesData['x']),
        'y': list(curvesData[gamma]),
        'mode': 'lines',
                'line': {'color':'black'},
        'name': 'Fitted Curve'
    }
    figure['data'].append(data_dict)

    # make frames
    for gamma in gammas:
        frame = {'data': [], 'name': str(gamma)}
        dataset.loc[:,'IL_weight'] = dataset['IL_dist'].apply(lambda x: u.ind_level_weight_calc(x,float(gamma)))
        dataset['w'] = dataset.kernel.multiply(dataset.loc[:,'AmtOutstanding']).multiply(dataset.loc[:,'IL_weight'])
        dataset_i = dataset.loc[dataset.TICKER == Ticker]
        dataset_o = dataset.loc[dataset.TICKER != Ticker]


        for il in dataset['IL_dist'].unique():
            dataset_o = dataset.loc[(dataset.TICKER != Ticker) & (dataset['IL_dist'] == il)]
            data_dict = {
                'x': tmp_x + list(dataset_o['Effective_Duration']),
                'y': tmp_y + list(dataset_o['logOAS']),
                'mode': 'markers',
                'text': list(dataset_o['TICKER']),
                'marker': {
                    'size': tmp_w + list(dataset_o['w'] * dotsizenorm),
                    'color':cols[il+1]
                },
                'name': industries[il],
                'showlegend':True
            }
            frame['data'].append(data_dict)

            # ##add points just for legend formatting
            # data_dict = {
            #     'x': [100.0],
            #     'y': [100.0],
            #     'mode': 'markers',
            #     'marker': {
            #         'size': 12.0,
            #         'color':cols[il+1]
            #     },
            #     'name': industries[il]
            # }
            # frame['data'].append(data_dict)
            # ## end add points just for legend formatting

        data_dict = {
        'x': list(dataset_i['Effective_Duration']),
        'y': list(dataset_i['logOAS']),
        'mode': 'markers',
        'text': list(dataset['TICKER']),
        'marker': {
            'size': list(dataset_i['w']*dotsizenorm),
            'color':cols[0]
        },
        'name': Ticker
        }
        frame['data'].append(data_dict)
        data_dict = {
        'x': list(curvesData['x']),
        'y': list(curvesData[gamma]),
        'mode': 'lines',
                'line': {'color':'black'},
        'name': 'Fitted Curve'
        }
        frame['data'].append(data_dict)

        figure['frames'].append(frame)
        slider_step = {'args': [
            [gamma],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 300}}
         ],
         'label': gamma,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)


    figure['layout']['sliders'] = [sliders_dict]

    plotly.offline.iplot(figure)
    plotly.offline.plot(figure, filename='gamma_slider_%s.html' % Ticker, auto_open=False)


def curve_slider_fit(dataset,curve,Ticker):
    steps=['0','1','2','3','4','5']
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': [],
        'config': {'scrollzoom': True}
    }

    # fill in most of layout
    figure['layout']['xaxis'] = {'range': [0., 20.], 'title': 'Effective Tenor'}
    figure['layout']['yaxis'] = {'range': [0., 4.], 'title': 'log(spread)'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['slider'] = {
        'args': [
            'slider.value', {
                'duration': 300,
                'ease': 'cubic-in-out'
            }
        ],
        'initialValue': '0',
        'plotlycommand': 'animate',
        'values': steps,
        'visible': True
    }
    # figure['layout']['updatemenus'] = [
    #     {
    #         'buttons': [
    #             {
    #                 'args': [None, {'frame': {'duration': 1000, 'redraw': False},
    #                          'fromcurrent': True, 'transition': {'duration': 1000, 'easing': 'quadratic-in-out'}}],
    #                 'label': 'Play',
    #                 'method': 'animate'
    #             },
    #             # {
    #             #     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
    #             #     'transition': {'duration': 0}}],
    #             #     'label': 'Pause',
    #             #     'method': 'animate'
    #             # }
    #         ],
    #         'direction': 'left',
    #         'pad': {'r': 10, 't': 87},
    #         'showactive': False,
    #         'type': 'buttons',
    #         'x': 0.1,
    #         'xanchor': 'right',
    #         'y': 0,
    #         'yanchor': 'top'
    #     }
    # ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Step:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': .8,
        'x': 0.0,
        'y': 0,
        'steps': []
    }

    # make data
    step='0'
    #dataset.loc[:,'IL_weight'] = dataset['IL_dist'].apply(lambda x: ngc.ind_level_weight_calc(x,float(gamma)))
    #dataset['w'] = dataset.kernel.multiply(dataset.loc[:,'AmtOutstanding'])
    dataset_i = dataset.loc[dataset.TICKER == Ticker]
    dotsizenorm = 20. / dataset_i.AmtOutstanding.max()

    cols_o = cl.scales['9']['seq']['Blues'][3:]
    col_o = cl.interp( cols_o, 20) # multiple of 10s for some reason

    cols = cl.scales['5']['qual']['Set1']
    tmp_x = [1000.]
    tmp_y = [1000.]
    tmp_w = [100000.]

    data_dict = {
        'x': list(curve['x']),
        'y': list(curve['y']),
        'mode': 'lines',
                'line': {'color':'rgba(0,0,0,1)'},
        'name': 'Fitted Curve'
    }
    figure['data'].append(data_dict)

    data_dict = {
        'x': list(dataset_i['Effective_Duration']),
        'y': list(dataset_i['logOAS']),
        'mode': 'markers',
        'text': list(dataset['TICKER']),
        'marker': {
            'color':cols[0],
            'size': list(dataset_i['AmtOutstanding']*dotsizenorm)
        },
        'name': Ticker
    }
    figure['data'].append(data_dict)

    i=0
    for t in dataset['TICKER'].unique():
        if t!=Ticker:
            dataset_o = dataset.loc[(dataset.TICKER == t)]
            data_dict = {
                'x': list(dataset_o['Effective_Duration']),
                'y': list(dataset_o['logOAS']),
                'mode': 'markers',
                'text': list(dataset_o['TICKER']),
                'marker': {
                    'size': list(dataset_o['AmtOutstanding']*dotsizenorm),
                    'color':col_o[i],
                    'opacity':0.5
                },
                'name': t,
                'showlegend':True
            }
            figure['data'].append(data_dict)
            i+=1

            #curves for final step
            data_dict = {
            'x': list(curve['x']),
            'y': list(curve['y']),
            'mode': 'lines',
                    'line': {'color':'black','opacity':1},
            'name': 'Fitted Curve',
            'showlegend':False
            }
            figure['data'].append(data_dict)
            #end curves for final step



    # make frames
    for step in steps:
        frame = {'data': [], 'name': str(step)}
        dataset_i = dataset.loc[dataset.TICKER == Ticker]

        data_dict = {
            'x': list(dataset_i['Effective_Duration']),
            'y': list(dataset_i['logOAS']),
            'mode': 'markers',
            'text': list(dataset['TICKER']),
            'marker': {
                'color':cols[0],
                'size': list(dataset_i['AmtOutstanding']*dotsizenorm),
                'line': {'color':cols[0]}
            },
            'name': Ticker
        }
        frame['data'].append(data_dict)

        i=0
        if step=='0':  #issuer data only
            for t in dataset['TICKER'].unique():
                if t!=Ticker:
                    dataset_o = dataset.loc[(dataset.TICKER == t)]
                    data_dict = {
                        'x': list(dataset_o['Effective_Duration']),
                        'y': list(dataset_o['logOAS']),
                        'mode': 'markers',
                        'text': list(dataset_o['TICKER']),
                        'marker': {
                            'size': list(dataset_o['AmtOutstanding']*dotsizenorm),
                            'color':'rgba(0,0,0,0)',
                            'opacity':0.5,
                            'line': {'color':'rgba(0,0,0,0)'}
                        },
                        'name': t,
                        'showlegend':False
                    }
                    frame['data'].append(data_dict)
                    i+=1

                    #curves for final step
                    data_dict = {
                    'x': list(curve['x']),
                    'y': list(curve['y']),
                    'mode': 'lines',
                            'line': {'color':'rgba(0,0,0,0)'},
                    'name': 'Fitted Curve',
                    'showlegend':False
                    }
                    frame['data'].append(data_dict)

            data_dict = {
            'x': list(curve['x']),
            'y': list(curve['y']),
            'mode': 'lines',
                    'line': {'color':'rgba(0,0,0,0)'},
            'name': 'Fitted Curve'
            }
            frame['data'].append(data_dict)

        elif step=='1':  #issuer data with peers
            for t in dataset['TICKER'].unique():
                if t!=Ticker:
                    dataset_o = dataset.loc[(dataset.TICKER == t)]
                    data_dict = {
                        'x': list(dataset_o['Effective_Duration']),
                        'y': list(dataset_o['logOAS']),
                        'mode': 'markers',
                        'text': list(dataset_o['TICKER']),
                        'marker': {
                            'size': list(dataset_o['AmtOutstanding']*dotsizenorm),
                            'color':col_o[i],
                            'opacity':0.5
                        },
                        'name': t,
                        'showlegend':True
                    }
                    frame['data'].append(data_dict)
                    i+=1

                    #curves for final step
                    data_dict = {
                    'x': list(curve['x']),
                    'y': list(curve['y']),
                    'mode': 'lines',
                            'line': {'color':'rgba(0,0,0,0)'},
                    'name': 'Fitted Curve',
                    'showlegend':False
                    }
                    frame['data'].append(data_dict)

            data_dict = {
            'x': list(curve['x']),
            'y': list(curve['y']),
            'mode': 'lines',
                    'line': {'color':'rgba(0,0,0,0)'},
            'name': 'Fitted Curve'
            }
            frame['data'].append(data_dict)

        elif step=='2':  #collapse data to single shape
            for t in dataset['TICKER'].unique():
                if t!=Ticker:
                    dataset_o = dataset.loc[(dataset.TICKER == t)]
                    data_dict = {
                        'x': list(dataset_o['Effective_Duration']),
                        'y': list(dataset_o['y']),
                        'mode': 'markers',
                        'text': list(dataset_o['TICKER']),
                        'marker': {
                            'size': list(dataset_o['AmtOutstanding']*dotsizenorm),
                            'color':col_o[i],
                            'opacity':0.5
                        },
                        'name': t,
                        'showlegend':True
                    }
                    frame['data'].append(data_dict)
                    i+=1

                    #curves for final step
                    data_dict = {
                    'x': list(curve['x']),
                    'y': list(curve['y']),
                    'mode': 'lines',
                            'line': {'color':'rgba(0,0,0,0)'},
                    'name': 'Fitted Curve',
                    'showlegend':False
                    }
                    frame['data'].append(data_dict)

            data_dict = {
            'x': list(curve['x']),
            'y': list(curve['y']),
            'mode': 'lines',
                    'line': {'color':'rgba(0,0,0,0)'},
            'name': 'Fitted Curve'
            }
            frame['data'].append(data_dict)

        elif step=='3':  #fit shape
            for t in dataset['TICKER'].unique():
                if t!=Ticker:
                    dataset_o = dataset.loc[(dataset.TICKER == t)]
                    data_dict = {
                        'x': list(dataset_o['Effective_Duration']),
                        'y': list(dataset_o['y']),
                        'mode': 'markers',
                        'text': list(dataset_o['TICKER']),
                        'marker': {
                            'size': list(dataset_o['AmtOutstanding']*dotsizenorm),
                            'color':col_o[i],
                            'opacity':0.5
                        },
                        'name': t,
                        'showlegend':True
                    }
                    frame['data'].append(data_dict)
                    i+=1

                    #curves for final step
                    data_dict = {
                    'x': list(curve['x']),
                    'y': list(curve['y']),
                    'mode': 'lines',
                            'line': {'color':'rgba(0,0,0,0)'},
                    'name': 'Fitted Curve',
                    'showlegend':False
                    }
                    frame['data'].append(data_dict)

            data_dict = {
                'x': list(curve['x']),
                'y': list(curve['y']),
                'mode': 'lines',
                        'line': {'color':cols[0]},
                'name': 'Fitted Curve'
                }
            frame['data'].append(data_dict)

        elif step=='4': #show seperate issuer curves
            for t in dataset['TICKER'].unique():

                d = dataset.loc[dataset.TICKER == t].distance.values[0]

                if t!=Ticker:
                    dataset_o = dataset.loc[(dataset.TICKER == t)]
                    data_dict = {
                        'x': list(dataset_o['Effective_Duration']),
                        'y': list(dataset_o['logOAS']),
                        'mode': 'markers',
                        'text': list(dataset_o['TICKER']),
                        'marker': {
                            'size': list(dataset_o['AmtOutstanding']*dotsizenorm),
                            'color':col_o[i],
                            'opacity':0.5
                        },
                        'name': t,
                        'showlegend':True
                    }
                    frame['data'].append(data_dict)
                    i+=1

                    #curves for final step
                    data_dict = {
                    'x': list(curve['x']),
                    'y': list(np.array(curve['y'].values)-np.ones(curve.shape[0])*d),
                    'mode': 'lines',
                            'line': {'color':col_o[i]},
                    'name': 'Fitted Curve %s' % t,
                    'showlegend':False
                    }
                    frame['data'].append(data_dict)

            data_dict = {
                'x': list(curve['x']),
                'y': list(curve['y']),
                'mode': 'lines',
                        'line': {'color':cols[0]},
                'name': 'Fitted Curve'
                }
            frame['data'].append(data_dict)

        elif step=='5': #final curve
            for t in dataset['TICKER'].unique():

                d = dataset.loc[dataset.TICKER == t].distance.values[0]

                if t!=Ticker:
                    dataset_o = dataset.loc[(dataset.TICKER == t)]
                    data_dict = {
                        'x': list(dataset_o['Effective_Duration']),
                        'y': list(dataset_o['logOAS']),
                        'mode': 'markers',
                        'text': list(dataset_o['TICKER']),
                        'marker': {
                            'size': list(dataset_o['AmtOutstanding']*dotsizenorm),
                            'color':'rgba(0,0,0,0)',
                            'opacity':0.5,
                            'line': {'color':'rgba(0,0,0,0)'}
                        },
                        'name': t,
                        'showlegend':False
                    }
                    frame['data'].append(data_dict)
                    i+=1

                    #curves for final step
                    data_dict = {
                    'x': list(curve['x']),
                    'y': list(np.array(curve['y'].values)-np.ones(curve.shape[0])*d),
                    'mode': 'lines',
                            'line': {'color':'rgba(0,0,0,0)'},
                    'name': 'Fitted Curve %s' % t,
                    'showlegend':False
                    }
                    frame['data'].append(data_dict)

            data_dict = {
                'x': list(curve['x']),
                'y': list(curve['y']),
                'mode': 'lines',
                        'line': {'color':cols[0]},
                'name': 'Fitted Curve'
                }
            frame['data'].append(data_dict)

        # data_dict = {
        # 'x': list(curvesData['x']),
        # 'y': list(curvesData[gamma]),
        # 'mode': 'lines',
        #         'line': {'color':'black'},
        # 'name': 'Fitted Curve'
        # }
        # frame['data'].append(data_dict)

        figure['frames'].append(frame)
        slider_step = {'args': [
            [step],
            {'frame': {'duration': 300, 'redraw': True},
             'mode': 'immediate',
           'transition': {'duration': 300}}
         ],
         'label': step,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]

    plotly.offline.iplot(figure)
    plotly.offline.plot(figure, filename='fit_slider_%s.html' % Ticker, auto_open=False)


def curve_slider_time(dataset,Ticker):

    alphas=['0','0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95','1']
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': [],
        'config': {'scrollzoom': True}
    }

    # fill in most of layout
    figure['layout']['xaxis'] = {'range': ['2015-01-01', '2015-06-30'], 'title': 'Date'}
    figure['layout']['yaxis'] = {'range': [4.2, 5.4], 'title': 'log(spread)'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['slider'] = {
        'args': [
            'slider.value', {
                'duration': 1,
                'ease': 'cubic-in'
            }
        ],
        'initialValue': '0.0',
        'plotlycommand': 'animate',
        'values': alphas,
        'visible': True
    }
    # figure['layout']['updatemenus'] = [
    #     {
    #         'buttons': [
    #             {
    #                 'args': [None, {'frame': {'duration': 300, 'redraw': False},
    #                          'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'cubic-in'}}],
    #                 'label': 'Play',
    #                 'method': 'animate'
    #             },
    #             # {
    #             #     'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
    #             #     'transition': {'duration': 0}}],
    #             #     'label': 'Pause',
    #             #     'method': 'animate'
    #             # }
    #         ],
    #         'direction': 'left',
    #         'pad': {'r': 10, 't': 87},
    #         'showactive': False,
    #         'type': 'buttons',
    #         'x': 0.1,
    #         'xanchor': 'right',
    #         'y': 0,
    #         'yanchor': 'top'
    #     }
    # ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Alpha:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in'},
        'pad': {'b': 10, 't': 50},
        'len': .9,
        'x': 0.,
        'y': 0,
        'steps': []
    }

    # make data
    alpha='0'
    data_dict = {
        'x': list(dataset['TradeDate']),
        'y': list(dataset['0']),
        'mode': 'lines',
        'line':{'dash':'dot'},
        'name': 'Levels'
    }
    figure['data'].append(data_dict)
    data_dict = {
        'x': list(dataset['TradeDate']),
        'y': list(dataset[alpha]),
        'mode': 'lines',
        'name': 'Levels + Changes'
    }
    figure['data'].append(data_dict)
    data_dict = {
        'x': list(dataset['TradeDate']),
        'y': list(dataset['1']),
        'mode': 'lines',
        'line':{'dash':'dash'},
        'name': 'Changes'
    }
    figure['data'].append(data_dict)

    # make frames
    for alpha in alphas:
        frame = {'data': [], 'name': str(alpha)}

        data_dict = {
        'x': list(dataset['TradeDate']),
        'y': list(dataset['0']),
        'mode': 'lines',
        'line':{'dash':'dot'},
        'name': 'Levels'
        }
        frame['data'].append(data_dict)
        data_dict = {
        'x': list(dataset['TradeDate']),
        'y': list(dataset[alpha]),
        'mode': 'lines',
        'name': 'Levels + Changes'
        }
        frame['data'].append(data_dict)
        data_dict = {
        'x': list(dataset['TradeDate']),
        'y': list(dataset['1']),
        'mode': 'lines',
        'line':{'dash':'dash'},
        'name': 'Changes'
        }
        frame['data'].append(data_dict)

        figure['frames'].append(frame)
        slider_step = {'args': [
            [alpha],
            {'frame': {'duration': 1, 'redraw': False},
             'mode': 'animate',
           'transition': {'duration': 1}}
         ],
         'label': alpha,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)


    figure['layout']['sliders'] = [sliders_dict]

    plotly.offline.iplot(figure)
    plotly.offline.plot(figure, filename='levels_changes_slider_%s.html' % Ticker, auto_open=False)

