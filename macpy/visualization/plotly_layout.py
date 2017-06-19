import plotly.graph_objs as go
import plotly.plotly as py

def print_pdf(filename, data, layout):
    py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG')
    fig = go.Figure(data=data,layout=layout)
    py.image.save_as(fig, filename=filename,format='pdf')

def print_png(filename, data, layout):
    py.sign_in('Axioma01', 'qDOgcN9vocMJRGXbPbGG')
    fig = go.Figure(data=data,layout=layout)
    py.image.save_as(fig, filename=filename,format='png', scale=5)


def PlotLayout1(xTitle,yTitle,width=None,height=None,title=None,ymin=None,ymax=None,FontSize=18,top=80,bottom=80,ShowLegend=True):
    # go here for more options: https://plot.ly/python/axes/
    if width==None:
        width=700
    if height==None:
        height=600

    if ymin==None:

        layout=dict(go.Layout(title=title, width=width,height=height, legend=dict(
            font=dict(
                family='Arial',
                size=FontSize,
            )),titlefont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            ),
            margin=go.Margin(
                l=80,
                r=80,
                b=bottom,
                t=top,
                pad=0
            ),
            xaxis=go.XAxis(title=xTitle,titlefont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            ),
            tickfont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            )),
            yaxis=go.YAxis(title=yTitle,titlefont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            ),
            tickfont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            )),
            showlegend=ShowLegend,
                              hovermode='closest'
                              ))

    else:

        layout=dict(go.Layout(title=title, width=width,height=height, legend=dict(
            font=dict(
                family='Arial',
                size=FontSize,
            )),
            margin=go.Margin(
                l=80,
                r=80,
                b=bottom,
                t=top,
                pad=0
            ),
            xaxis=go.XAxis(title=xTitle,titlefont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            ),
            tickfont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            )),
            yaxis=go.YAxis(title=yTitle,titlefont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            ),
            tickfont=dict(
                family='Arial',
                size=FontSize,
                color='black'
            )),range=[ymin,ymax],
            showlegend=ShowLegend,
                              hovermode='closest'
                              ))

    return layout

def PlotLayoutDualAxis(xTitle,yTitle,width=None,height=None,title=None,ymin=None,ymax=None,y2Title=None):
    # go here for more options: https://plot.ly/python/axes/
    if width==None:
        width=700
    if height==None:
        height=600

    if ymin==None:

        layout=dict(go.Layout(title=title, width=width,height=height, legend=dict(
            font=dict(
                family='Arial',
                size=18,
            ),x=1.2,y=1.1),
            margin=go.Margin(
                l=80,
                r=80,
                b=100,
                t=40,
                pad=0
            ),
            xaxis=go.XAxis(title=xTitle,titlefont=dict(
                family='Arial',
                size=18,
                color='black'
            ),
            tickfont=dict(
                family='Arial',
                size=18,
                color='black'
            )),
            yaxis=go.YAxis(title=yTitle,titlefont=dict(
                family='Arial',
                size=18,
                color='black'
            ),
			rangemode='tozero',
            tickfont=dict(
                family='Arial',
                size=18,
                color='black'
            ),
                           anchor='x'),
            yaxis2=go.YAxis(title=y2Title,titlefont=dict(
                family='Arial',
                size=18,
                color='black'
            ),
			rangemode='tozero',
            tickfont=dict(
                family='Arial',
                size=18,
                color='black'
            ),side='right',
            overlaying='y',
            showgrid=False,
                            anchor='x'),
                              ))

    else:

        layout=dict(go.Layout(title=title, width=width,height=height, legend=dict(
            font=dict(
                family='Arial',
                size=18,
            )),
            margin=go.Margin(
                l=80,
                r=80,
                b=100,
                t=40,
                pad=0
            ),
            xaxis=go.XAxis(title=xTitle,titlefont=dict(
                family='Arial',
                size=18,
                color='black'
            ),
            tickfont=dict(
                family='Arial',
                size=18,
                color='black'
            )),
            yaxis=go.YAxis(title=yTitle,titlefont=dict(
                family='Arial',
                size=18,
                color='black'
            ),
            tickfont=dict(
                family='Arial',
                size=18,
                color='black'
            )),range=[ymin,ymax]))

    return layout